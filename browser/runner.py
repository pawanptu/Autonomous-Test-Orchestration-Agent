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
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from browser.sandbox import build_namespace, validate_test_source
from browser.screenshots import capture, capture_dom_snippet
from browser.session import BrowserSession, attach_console_capture
from config import Settings, get_settings
from graph.state import GeneratedTest, TestFlow, TestResult, TestStatus, utcnow_iso
from logging_setup import get_logger
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
            result.screenshot_path = await capture(
                page,
                screenshot_dir / f"{flow.id}__viewport.png",
                full_page=False,
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

    if cfg.enable_parallel_execution and len(pairs) > 1:
        semaphore = asyncio.Semaphore(max(1, cfg.max_parallel_flows))

        async def _guarded(flow: TestFlow, test: GeneratedTest) -> TestResult:
            async with semaphore:
                return await _one(flow, test)

        log.info(
            "executing %d tests in parallel (max %d concurrent)",
            len(pairs),
            cfg.max_parallel_flows,
        )
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

    results = []
    for flow, test in pairs:
        results.append(await _one(flow, test))
    return results


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
