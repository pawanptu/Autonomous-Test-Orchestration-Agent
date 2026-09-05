"""Execution of agent-generated Playwright tests.

Each test runs on its own browser context, so a flow that corrupts state
cannot poison the next one and parallel execution is safe when
``ENABLE_PARALLEL_EXECUTION`` is on.

What is captured for every run, pass or fail:

* status and wall-clock duration;
* the viewport screenshot that the visual-diff stage compares to the baseline;
* on failure: a full-page screenshot, a redacted DOM snapshot, the traceback,
  the final URL and any console/page errors.

That evidence bundle is exactly what the Healer needs to tell a broken locator
apart from a broken application, which is why it is collected eagerly rather
than reconstructed later.

Credentials never appear in a generated test. Tests that need them call
``ctx["secret"]("username")``, which reads from the in-memory
:data:`security.SECRET_BOX` at the moment of use.
"""

from __future__ import annotations

import asyncio
import random
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from browser.sandbox import build_namespace, validate_test_source
from browser.screenshots import (
    capture,
    capture_dom_snippet,
    dynamic_locators,
    freeze_dynamic_rendering,
)
from browser.session import (
    BrowserSession,
    attach_console_capture,
    install_navigation_guard,
    install_third_party_blocker,
)
from config import Settings, get_settings
from graph.state import (
    GeneratedTest,
    RiskLevel,
    TestFlow,
    TestResult,
    TestStatus,
    utcnow_iso,
)
from logging_setup import get_logger
from ops import RateLimiter, effective_parallelism, host_of
from security import SECRET_BOX, redact_text, sanitize_url

log = get_logger("aivor.runner")

ProgressFn = Callable[[str, str, str], None]
"""``(flow_id, summary, detail)`` - called at test start and finish."""


def make_secret_accessor(run_id: str) -> Callable[[str], str]:
    """Build the ``ctx["secret"]`` callable for one run.

    Reads from the process-local secret box at call time. Returns an empty
    string for an unknown or absent key so that a generated test degrades into
    a visible assertion failure rather than a ``KeyError`` that the Healer
    would misclassify as a script bug.
    """

    def secret(name: str) -> str:
        creds = SECRET_BOX.get(run_id)
        if creds is None:
            return ""
        value = {
            "username": creds.username,
            "user": creds.username,
            "email": creds.username,
            "password": creds.password,
            "token": creds.token,
        }.get((name or "").strip().lower())
        return value or ""

    return secret


async def execute_test(
    session: BrowserSession,
    *,
    run_id: str,
    flow: TestFlow,
    test: GeneratedTest,
    screenshot_dir: Path,
    settings: Settings | None = None,
    attempt: int = 1,
) -> TestResult:
    """Run one generated test to completion. Never raises."""
    cfg = settings or get_settings()
    result = TestResult(
        flow_id=flow.id,
        flow_name=flow.name,
        status=TestStatus.ERROR,
        attempts=attempt,
        started_at=utcnow_iso(),
    )

    verdict = validate_test_source(test.source)
    if not verdict.ok:
        result.status = TestStatus.ERROR
        result.error_type = "GeneratedSourceRejected"
        result.error_message = verdict.summary()
        result.traceback = "\n".join(verdict.errors)
        log.warning("flow %s: generated source rejected (%s)", flow.id, verdict.summary())
        return result

    console_errors: list[str] = []
    started = time.monotonic()
    context = None
    page = None

    try:
        context = await session.new_context()
        # Analytics, ads and chat widgets are blocked before the page loads.
        # They are the dominant source of false-positive visual diffs (a
        # rotating ad creative changes every capture) and they are not part of
        # the application under test.
        if cfg.visual_block_third_party:
            await install_third_party_blocker(context)
        # The navigation guard applies here too: a generated test navigates,
        # and the pages it reaches can redirect anywhere.
        await install_navigation_guard(context, settings=cfg, context_label="execution")
        page = await context.new_page()
        attach_console_capture(page, console_errors)

        namespace = build_namespace(
            {
                "FLOW_ID": flow.id,
                "FLOW_NAME": flow.name,
                "BASE_URL": flow.url,
            }
        )
        exec(compile(test.source, f"<generated:{flow.id}>", "exec"), namespace)  # noqa: S102
        test_fn = namespace.get("test_flow")
        if not callable(test_fn):
            result.error_type = "MissingTestFunction"
            result.error_message = "the generated module defines no callable test_flow"
            return result

        test_ctx: dict[str, Any] = {
            "secret": make_secret_accessor(run_id),
            "base_url": flow.url,
            "flow_id": flow.id,
            "flow_name": flow.name,
            "run_id": run_id,
            "timeout_ms": cfg.action_timeout_ms,
        }

        await asyncio.wait_for(test_fn(page, test_ctx), timeout=cfg.test_timeout_s)
        result.status = TestStatus.PASSED

    except asyncio.TimeoutError:
        result.status = TestStatus.FAILED
        result.error_type = "TestTimeout"
        result.error_message = (
            f"the test did not finish within {cfg.test_timeout_s:.0f}s"
        )
        result.traceback = "asyncio.TimeoutError: test wall-clock budget exhausted"
    except AssertionError as exc:
        result.status = TestStatus.FAILED
        result.error_type = "AssertionError"
        result.error_message = redact_text(str(exc))[:2000]
        result.traceback = redact_text(traceback.format_exc())[:6000]
    except Exception as exc:
        result.status = TestStatus.FAILED
        result.error_type = type(exc).__name__
        result.error_message = redact_text(str(exc))[:2000]
        result.traceback = redact_text(traceback.format_exc())[:6000]
    finally:
        result.duration_s = round(time.monotonic() - started, 3)
        result.console_errors = console_errors[:20]
        if page is not None:
            try:
                result.final_url = sanitize_url(page.url or "")
            except Exception:
                result.final_url = None
            # The viewport frame is the visual-diff subject and is always taken.
            # Time and randomness are seeded, and dynamic regions are masked,
            # so that two captures of an unchanged page compare equal rather
            # than differing by a clock tick or a rotating avatar.
            if cfg.visual_freeze_time:
                await freeze_dynamic_rendering(page)
            dynamic_masks = await dynamic_locators(page, cfg.visual_mask_selectors)
            result.screenshot_path = await capture(
                page,
                screenshot_dir / f"{flow.id}__viewport.png",
                full_page=False,
                extra_mask=dynamic_masks,
            )
            if result.status is not TestStatus.PASSED:
                failure_shot = await capture(
                    page,
                    screenshot_dir / f"{flow.id}__failure.png",
                    full_page=True,
                )
                if failure_shot:
                    result.screenshot_path = failure_shot
                result.dom_snippet = await capture_dom_snippet(page)
        if context is not None:
            try:
                await context.close()
            except Exception:  # pragma: no cover
                log.debug("context close failed after %s", flow.id, exc_info=True)

    return result


async def run_suite(
    session: BrowserSession,
    *,
    run_id: str,
    flows: Sequence[TestFlow],
    tests: Sequence[GeneratedTest],
    screenshot_dir: Path,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
    order: Sequence[str] | None = None,
) -> list[TestResult]:
    """Execute the whole suite, sequentially or in parallel.

    ``order`` is the risk-ranked flow-id order produced upstream: high-risk
    flows run first so that the most important signal arrives earliest, which
    matters when a demo is watched live and when a run is cut short.
    """
    cfg = settings or get_settings()
    emit = progress or (lambda flow_id, summary, detail="": None)

    by_id = {flow.id: flow for flow in flows}
    pairs: list[tuple[TestFlow, GeneratedTest]] = []
    for test in tests:
        flow = by_id.get(test.flow_id)
        if flow is not None:
            pairs.append((flow, test))

    if order:
        rank = {flow_id: index for index, flow_id in enumerate(order)}
        pairs.sort(key=lambda pair: rank.get(pair[0].id, len(rank)))

    if not pairs:
        return []

    screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def _one(flow: TestFlow, test: GeneratedTest) -> TestResult:
        emit(flow.id, f"Running {flow.id}: {flow.name}", f"category={_category(flow)}")
        result = await execute_test(
            session,
            run_id=run_id,
            flow=flow,
            test=test,
            screenshot_dir=screenshot_dir,
            settings=cfg,
        )
        emit(
            flow.id,
            f"{flow.id} {result.status.value.upper()} in {result.duration_s:.1f}s"
            + (f" - {result.error_type}" if result.error_type else ""),
            (result.error_message or "")[:300],
        )
        return result

    # Parallelism is decided by policy, not by the flag alone: concurrent flows
    # against a target with no host-level throttle is a load test somebody else
    # did not agree to. See ops.effective_parallelism.
    parallel_limit, parallel_reason = effective_parallelism(cfg)
    if parallel_limit > 1 and len(pairs) > 1:
        semaphore = asyncio.Semaphore(parallel_limit)
        limiter = RateLimiter(cfg.target_rate_limit_per_s)
        target_host = host_of(pairs[0][0].url)

        async def _guarded(flow: TestFlow, test: GeneratedTest) -> TestResult:
            async with semaphore:
                await limiter.acquire(target_host)
                return await _one(flow, test)

        log.info("executing %d tests in parallel - %s", len(pairs), parallel_reason)
        gathered = await asyncio.gather(
            *[_guarded(flow, test) for flow, test in pairs], return_exceptions=True
        )
        results: list[TestResult] = []
        for (flow, _), outcome in zip(pairs, gathered):
            if isinstance(outcome, BaseException):
                results.append(
                    TestResult(
                        flow_id=flow.id,
                        flow_name=flow.name,
                        status=TestStatus.ERROR,
                        error_type=type(outcome).__name__,
                        error_message=redact_text(str(outcome))[:500],
                    )
                )
            else:
                results.append(outcome)
        return results

    if cfg.enable_parallel_execution:
        # The operator asked for parallelism and did not get it. Saying so is
        # the difference between a deliberate safety decision and a silent
        # config bug that makes every run mysteriously slow.
        log.warning("parallel execution requested but not used - %s", parallel_reason)

    results = []
    for flow, test in pairs:
        results.append(await _one(flow, test))
    return results


def should_rerun_for_flake(result: TestResult, risk: RiskLevel, cfg: Settings) -> tuple[bool, str]:
    """Whether a failure justifies one confirmation re-run.

    A re-run is only worth its wall-clock when the failure is of a kind that
    genuinely differs between attempts *and* the flow matters enough to pay for
    the certainty. A clean assertion failure is reproducible by construction and
    is re-run for nothing; a timeout, a navigation error or a detached element
    is exactly the ambiguity a second attempt resolves.

    Re-running everything would double the runtime of every red build and teach
    people to distrust the "flaky" label. Re-running nothing leaves a real
    defect and a slow network indistinguishable.
    """
    if not cfg.rerun_failed_flows:
        return False, "reruns disabled by configuration"
    if result.status not in (TestStatus.FAILED, TestStatus.ERROR):
        return False, "the test did not fail"
    if result.attempts and result.attempts > 1:
        return False, "already re-run once; the cap is one re-run per flow"

    blob = f"{result.error_type} {result.error_message}".lower()
    transient = (
        "timeout",
        "timeouterror",
        "navigation",
        "net::",
        "econnreset",
        "element is not attached",
        "detached from",
        "target closed",
        "context was destroyed",
        "waiting for",
    )
    if any(token in blob for token in transient):
        return True, f"failure looks timing-dependent ({result.error_type or 'unknown'})"
    if risk is RiskLevel.HIGH:
        return True, "high-risk flow: confirm the failure before filing it as a defect"
    return False, "deterministic failure on a non-critical flow; one attempt is conclusive"


async def rerun_for_flake(
    session: BrowserSession,
    *,
    run_id: str,
    flow: TestFlow,
    test: GeneratedTest,
    screenshot_dir: Path,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
) -> TestResult:
    """Re-execute one failed test once, after a jittered pause.

    The pause is randomised rather than fixed so that a failure caused by the
    agent landing on the same point of a target's own rate-limit or animation
    cycle does not reproduce for the same reason twice. The result carries
    ``attempt=2``, which stops any further re-run through
    :func:`should_rerun_for_flake`.
    """
    cfg = settings or get_settings()
    emit = progress or (lambda flow_id, summary, detail="": None)
    delay_s = random.uniform(cfg.rerun_jitter_ms / 2000.0, cfg.rerun_jitter_ms / 1000.0)
    emit(
        flow.id,
        f"{flow.id}: re-running once after {delay_s:.2f}s to confirm the failure",
        "a second attempt separates a genuine defect from a timing flake",
    )
    await asyncio.sleep(delay_s)
    result = await execute_test(
        session,
        run_id=run_id,
        flow=flow,
        test=test,
        screenshot_dir=screenshot_dir,
        settings=cfg,
        attempt=2,
    )
    result.rerun_of_failure = True
    return result


def classify_rerun(first: TestResult, second: TestResult) -> tuple[TestResult, str]:
    """Combine a failure and its confirmation attempt into one verdict.

    A flow that fails then passes is *flaky*, which is a finding in its own
    right and not a pass: reporting it as green would hide instability that a
    user will meet in production. The second result is returned so the evidence
    reflects the most recent attempt, with the disagreement recorded.
    """
    if second.status is TestStatus.PASSED and first.status is not TestStatus.PASSED:
        second.flaky = True
        second.notes.append(
            f"attempt 1 failed with {first.error_type or 'an error'} and attempt 2 passed; "
            "this flow is unstable and is reported as flaky rather than as passing"
        )
        return second, "flaky: passed only on the second attempt"
    if second.status in (TestStatus.FAILED, TestStatus.ERROR):
        second.notes.append("failure reproduced on a second attempt; treated as a real defect")
        return second, "confirmed: the failure reproduced"
    return second, "re-run produced a non-failing status"


async def rerun_single(
    session: BrowserSession,
    *,
    run_id: str,
    flow: TestFlow,
    test: GeneratedTest,
    screenshot_dir: Path,
    settings: Settings | None = None,
) -> TestResult:
    """Re-execute one test after the Healer applied a patch.

    Kept separate from :func:`run_suite` so the attempt counter and the
    ``HEALED`` status are set in exactly one place.
    """
    result = await execute_test(
        session,
        run_id=run_id,
        flow=flow,
        test=test,
        screenshot_dir=screenshot_dir,
        settings=settings,
        attempt=2,
    )
    if result.status is TestStatus.PASSED:
        result.status = TestStatus.HEALED
    return result


def _category(flow: TestFlow) -> str:
    value = getattr(flow.category, "value", flow.category)
    return str(value)


def merge_results(
    original: Sequence[TestResult], updated: Sequence[TestResult]
) -> list[TestResult]:
    """Overlay re-run outcomes onto the first-pass results, preserving order."""
    replacement = {r.flow_id: r for r in updated}
    return [replacement.get(r.flow_id, r) for r in original]
