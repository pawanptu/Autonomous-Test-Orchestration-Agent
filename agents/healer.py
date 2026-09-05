"""Healer sub-agent: is the test broken, or is the application broken?

This is the judgment the whole pipeline exists to automate. A failing test is
worthless information until someone decides whether it means "fix the locator"
or "file a bug", and that decision is what a human normally spends their
morning on.

How the decision is made
------------------------
1. **Evidence, gathered first.** The error and traceback, the redacted DOM
   snapshot taken at the moment of failure, the console errors, the screenshot,
   and - crucially - a *live re-probe* of the locators the test used, performed
   now, against the page as it currently is. "Is the selector still there?" is
   the single strongest signal available and it cannot be answered from logs.

2. **Two confidences, blended.** The 70B model classifies and self-reports a
   confidence. Separately, :mod:`differentiation.confidence_scorer` computes a
   confidence from the gathered signals using the same published rubric. The
   two are blended, which caps how far a persuasive-sounding rationale can move
   the outcome away from the evidence.

3. **A real branch at 0.6.** At or above the threshold, a SCRIPT_ISSUE gets its
   locator or wait patched and the test is re-run once. Below it, nothing is
   applied: the finding is queued for human review with its evidence, and both
   the UI and the report show it as such.

The rule that is never bent
---------------------------
A patch may only substitute a locator or adjust a wait. Any proposed change
that would delete or weaken an assertion is rejected outright by
:func:`patch_weakens_assertions`, regardless of confidence. Making a red test
green by lowering the bar is the one failure mode that would make this whole
system worse than useless, so it is blocked mechanically rather than by
prompting.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from browser.selectors import reprobe_expression
from browser.session import BrowserSession
from config import CONFIDENCE_AUTO_APPLY_THRESHOLD, Settings, get_settings
from differentiation.confidence_scorer import (
    ConfidenceSignals,
    blend,
    decide,
    score_signals,
)
from graph.runtime import RunContext
from graph.state import (
    DefectClass,
    GeneratedTest,
    HealerAction,
    RiskLevel,
    TestFlow,
    TestResult,
    TestStatus,
)
from llm.client import LLMClient, ModelRole
from llm.json_utils import JSONParseError
from llm.prompts import HEALER_SYSTEM, healer_user
from logging_setup import get_logger
from security import redact_text

log = get_logger("aivor.healer")

LOCATOR_RE = re.compile(
    r"page\.(?:get_by_role|get_by_test_id|get_by_label|get_by_placeholder|"
    r"get_by_alt_text|get_by_title|get_by_text|locator)\([^\n]*?\)(?:\.first)?"
)

ASSERTION_RE = re.compile(r"\bexpect\s*\(")
STRONG_ASSERTIONS = (
    "to_contain_text",
    "to_have_text",
    "to_have_value",
    "to_have_url",
    "to_have_count",
    "to_have_attribute",
    "to_be_checked",
)

CAPTCHA_MARKERS = (
    "recaptcha", "g-recaptcha", "hcaptcha", "cf-challenge", "cf-turnstile",
    "are you a robot", "verify you are human", "captcha",
)
AUTH_WALL_MARKERS = (
    'type="password"', "please log in", "please sign in", "you must be logged in",
    "session expired", "unauthorized", "sign in to continue",
)
SPA_MARKERS = ("data-reactroot", "id=\"root\"", "id=\"app\"", "ng-version", "__next", "data-v-app")
NETWORK_MARKERS = (
    "err_connection", "err_network", "err_name_not_resolved", "net::", "econnreset",
    "econnrefused", "socket hang up", "502 bad gateway", "503 service", "504 gateway",
)


# ==========================================================================
# Evidence
# ==========================================================================
async def gather_evidence(
    session: BrowserSession,
    *,
    flow: TestFlow,
    result: TestResult,
    test: GeneratedTest | None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Re-probe the failing page now, and summarise every other signal.

    Never raises: if the page cannot be reopened, the probe results are simply
    absent and the confidence scorer treats the strongest signal as unavailable
    (which correctly lowers confidence rather than inventing certainty).
    """
    cfg = settings or get_settings()
    expressions = extract_locators(test.source if test else "")
    probes: list[dict[str, Any]] = []
    reachable = False

    url = result.final_url or flow.url
    if url and expressions:
        context = None
        try:
            context = await session.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=cfg.nav_timeout_ms, wait_until="domcontentloaded")
            reachable = True
            for expression in expressions[:10]:
                probes.append(await reprobe_expression(page, expression))
        except Exception as exc:
            log.info("could not re-probe %s for %s: %s", url, flow.id, type(exc).__name__)
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:  # pragma: no cover
                    log.debug("healer probe context close failed", exc_info=True)

    return {
        "page_reachable": reachable,
        "probe_url": url,
        "probes": probes,
        "locator_count": len(expressions),
    }


def extract_locators(source: str) -> list[str]:
    """Every locator expression a generated test uses, in order of appearance."""
    if not source:
        return []
    seen: list[str] = []
    for match in LOCATOR_RE.finditer(source):
        expression = match.group(0).strip()
        if expression not in seen:
            seen.append(expression)
    return seen


def derive_signals(
    *,
    flow: TestFlow,
    result: TestResult,
    evidence: dict[str, Any],
) -> ConfidenceSignals:
    """Compute the deterministic half of the confidence score."""
    error_type = (result.error_type or "").lower()
    message = (result.error_message or "").lower()
    dom = (result.dom_snippet or "").lower()

    if "timeout" in error_type or "timeout" in message:
        failure_kind = "timeout"
    elif "assertion" in error_type:
        failure_kind = "assertion"
    elif any(marker in message for marker in ("navigation", "err_aborted", "goto")):
        failure_kind = "navigation"
    elif error_type:
        failure_kind = "exception"
    else:
        failure_kind = "unknown"

    probes = evidence.get("probes") or []
    if probes:
        selector_present: bool | None = any(p.get("present") for p in probes)
    else:
        selector_present = None

    expected = (flow.expected_outcome or "").strip()
    ambiguous = not expected or not re.search(r"[\"'‘’“”]", expected) or "{" in expected

    return ConfidenceSignals(
        selector_present_in_dom=selector_present,
        failure_kind=failure_kind,
        locator_named_in_error=bool(re.search(r"get_by_|locator\(", result.error_message or "")),
        reproduced_twice=result.attempts >= 2,
        spa_race_suspected=(failure_kind == "timeout" and any(m in dom for m in SPA_MARKERS)),
        captcha_or_bot_wall=any(m in dom for m in CAPTCHA_MARKERS),
        network_flaky=any(m in message or m in dom for m in NETWORK_MARKERS),
        ambiguous_expected_text=ambiguous,
        auth_wall=(
            any(m in dom for m in AUTH_WALL_MARKERS)
            and not flow.requires_auth
            and "log" not in flow.name.lower()
        ),
        console_errors_present=bool(result.console_errors),
    )


# ==========================================================================
# Classification and healing
# ==========================================================================
async def heal_failure(
    ctx: RunContext,
    llm: LLMClient,
    session: BrowserSession,
    *,
    flow: TestFlow,
    result: TestResult,
    test: GeneratedTest | None,
    risk: RiskLevel,
    settings: Settings | None = None,
) -> tuple[HealerAction, GeneratedTest | None]:
    """Diagnose one failure and, when warranted, patch its test.

    Returns the :class:`HealerAction` (always) and a patched
    :class:`GeneratedTest` when - and only when - the auto-apply branch was
    taken. The caller re-runs exactly those tests.
    """
    cfg = settings or get_settings()
    evidence = await gather_evidence(session, flow=flow, result=result, test=test, settings=cfg)
    signals = derive_signals(flow=flow, result=result, evidence=evidence)
    deterministic = score_signals(signals)

    action = HealerAction(
        flow_id=flow.id,
        flow_name=flow.name,
        risk=risk,
        signals=signals.as_dict(),
        evidence_refs=_evidence_refs(result, evidence),
    )

    verdict: dict[str, Any] = {}
    try:
        verdict = await llm.complete_json(
            ModelRole.REASONING,
            [
                {"role": "system", "content": HEALER_SYSTEM},
                {
                    "role": "user",
                    "content": healer_user(
                        flow=flow.model_dump(mode="json"),
                        result=_result_for_prompt(result),
                        dom_snippet=result.dom_snippet or "(no DOM snapshot captured)",
                        candidate_selectors=evidence.get("probes") or [],
                        test_source=test.source if test else "(no source available)",
                        console_errors=result.console_errors,
                    ),
                },
            ],
            task=f"healer:{flow.id}",
            max_tokens=2500,
        ) or {}
    except (JSONParseError, Exception) as exc:  # noqa: B014 - deliberate broad catch
        log.warning("healer classification failed for %s: %s", flow.id, exc)
        action.rationale = (
            f"The model could not be consulted ({type(exc).__name__}); the verdict below "
            "comes from the deterministic evidence rubric alone."
        )

    classification = _coerce_class(verdict.get("classification"))
    model_confidence = verdict.get("confidence")
    proposed = verdict.get("proposed_fix") if isinstance(verdict.get("proposed_fix"), dict) else None

    # When the model was unavailable, fall back to a conservative structural
    # read of the evidence rather than guessing GENUINE_DEFECT.
    if not verdict:
        classification = _fallback_classification(signals)

    action.classification = classification
    action.confidence = blend(model_confidence, deterministic.value)
    model_rationale = str(verdict.get("rationale") or "").strip()
    action.rationale = redact_text(
        " ".join(
            part
            for part in (model_rationale, f"Evidence: {deterministic.rationale()}", action.rationale)
            if part
        )
    )[:900]
    if isinstance(verdict.get("signals"), dict):
        action.signals.update({f"model_{k}": v for k, v in verdict["signals"].items()})

    has_fix = bool(proposed and str(proposed.get("kind", "none")).lower() in ("selector", "wait"))
    decision = decide(
        classification,
        action.confidence,
        has_fix=has_fix,
        threshold=CONFIDENCE_AUTO_APPLY_THRESHOLD,
    )
    action.auto_applied = decision.auto_apply
    action.needs_human_review = decision.needs_human_review
    action.action = decision.action

    if not decision.auto_apply:
        action.patch_summary = decision.reason
        _emit(ctx, action, result)
        return action, None

    patched, summary = apply_fix(test, proposed or {}, result=result)
    if patched is None:
        # The fix could not be applied safely; downgrade to the review queue
        # rather than pretending a patch happened.
        action.auto_applied = False
        action.needs_human_review = True
        action.action = "queue_for_review"
        action.patch_summary = summary
        _emit(ctx, action, result)
        return action, None

    action.patch_summary = summary
    _emit(ctx, action, result)
    return action, patched


def _fallback_classification(signals: ConfidenceSignals) -> DefectClass:
    """Structural classification used when the model is unreachable."""
    if signals.captcha_or_bot_wall or signals.network_flaky or signals.auth_wall:
        return DefectClass.ENVIRONMENT
    if signals.selector_present_in_dom is True and signals.failure_kind == "assertion":
        return DefectClass.GENUINE_DEFECT
    if signals.selector_present_in_dom is False and signals.failure_kind == "timeout":
        return DefectClass.SCRIPT_ISSUE
    return DefectClass.UNKNOWN


def _coerce_class(value: Any) -> DefectClass:
    text = str(value or "").strip().upper().replace(" ", "_")
    for member in DefectClass:
        if member.value == text:
            return member
    if "SCRIPT" in text:
        return DefectClass.SCRIPT_ISSUE
    if "DEFECT" in text or "BUG" in text or "APP" in text:
        return DefectClass.GENUINE_DEFECT
    if "ENV" in text or "FLAK" in text:
        return DefectClass.ENVIRONMENT
    return DefectClass.UNKNOWN


# ==========================================================================
# Patching
# ==========================================================================
def patch_weakens_assertions(old_source: str, new_source: str) -> bool:
    """True if the patch removes or softens verification. Blocks the patch.

    Three ways a "fix" cheats, all caught here:
      * fewer ``expect(...)`` calls than before;
      * a strong assertion (text/value/url/count) replaced by a weak one;
      * an assertion line commented out.
    """
    if not old_source or not new_source:
        return False
    if len(ASSERTION_RE.findall(new_source)) < len(ASSERTION_RE.findall(old_source)):
        return True
    for name in STRONG_ASSERTIONS:
        if new_source.count(name) < old_source.count(name):
            return True
    old_lines = {line.strip() for line in old_source.splitlines() if "expect(" in line}
    new_commented = {
        line.strip().lstrip("# ").strip()
        for line in new_source.splitlines()
        if line.strip().startswith("#") and "expect(" in line
    }
    return bool(old_lines & new_commented)


def apply_fix(
    test: GeneratedTest | None,
    proposed: dict[str, Any],
    *,
    result: TestResult | None = None,
) -> tuple[GeneratedTest | None, str]:
    """Apply a locator or wait fix. Returns ``(patched_test, summary)``.

    Returns ``(None, reason)`` when the patch cannot be applied safely - the
    caller then routes the finding to the human-review queue instead of
    silently doing nothing.
    """
    if test is None or not test.source:
        return None, "no generated source is available to patch"

    kind = str(proposed.get("kind") or "none").lower()
    old_expression = str(proposed.get("old_expression") or "").strip()
    new_expression = str(proposed.get("new_expression") or "").strip()
    explanation = redact_text(str(proposed.get("explanation") or ""))[:300]

    if kind == "selector":
        if not new_expression:
            return None, "a selector fix was proposed without a replacement locator"
        if not _is_safe_locator(new_expression):
            return None, (
                f"the proposed replacement {new_expression!r} is not a recognised "
                "Playwright locator expression, so it was not applied"
            )
        source = test.source
        if old_expression and old_expression in source:
            patched_source = source.replace(old_expression, new_expression)
        else:
            existing = extract_locators(source)
            if not existing:
                return None, "the failing test contains no locator to replace"
            patched_source = source.replace(existing[0], new_expression, 1)
            old_expression = existing[0]
        if patched_source == source:
            return None, "the proposed locator substitution changed nothing"
        if patch_weakens_assertions(source, patched_source):
            return None, (
                "REJECTED: the proposed patch would have removed or weakened an "
                "assertion. Assertions are never softened to make a test pass."
            )
        patched = test.model_copy(deep=True)
        patched.source = patched_source
        patched.warnings = [
            *test.warnings,
            f"healer replaced {old_expression} with {new_expression}",
        ]
        return patched, (
            f"Replaced locator {old_expression} with {new_expression}. {explanation}".strip()
        )

    if kind == "wait":
        source = test.source
        patched_source = _apply_wait_fix(source)
        if patched_source == source:
            return None, "no timing adjustment was possible on this test"
        if patch_weakens_assertions(source, patched_source):
            return None, "REJECTED: the timing patch would have weakened an assertion"
        patched = test.model_copy(deep=True)
        patched.source = patched_source
        patched.warnings = [*test.warnings, "healer relaxed waits and added a load-state barrier"]
        return patched, (
            f"Raised action timeouts to 15s and added a domcontentloaded barrier "
            f"after navigation. {explanation}".strip()
        )

    return None, f"no applicable fix kind ({kind!r})"


def _apply_wait_fix(source: str) -> str:
    """Raise timeouts and insert a load-state barrier after each navigation."""
    patched = re.sub(r"timeout\s*=\s*(\d{3,5})", lambda m: f"timeout={max(int(m.group(1)), 15000)}", source)
    lines = patched.splitlines()
    out: list[str] = []
    for line in lines:
        out.append(line)
        if "page.goto(" in line and "await" in line:
            indent = line[: len(line) - len(line.lstrip())]
            barrier = (
                f'{indent}await page.wait_for_load_state("domcontentloaded", timeout=15000)'
            )
            if barrier not in patched:
                out.append(barrier)
    return "\n".join(out) + ("\n" if source.endswith("\n") else "")


def _is_safe_locator(expression: str) -> bool:
    """Only locator expressions we know how to build are accepted as patches."""
    return bool(
        re.match(
            r"^page\.(get_by_role|get_by_test_id|get_by_label|get_by_placeholder|"
            r"get_by_alt_text|get_by_title|get_by_text|locator)\(.*\)(\.first|\.last|\.nth\(\d+\))?$",
            expression.strip(),
        )
    )


# ==========================================================================
# Reporting helpers
# ==========================================================================
def _evidence_refs(result: TestResult, evidence: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    if result.screenshot_path:
        refs.append(f"screenshot:{result.screenshot_path}")
    if result.dom_snippet:
        refs.append(f"dom_snapshot:{len(result.dom_snippet)} chars captured at failure")
    if result.traceback:
        refs.append("traceback:captured")
    if result.console_errors:
        refs.append(f"console_errors:{len(result.console_errors)}")
    probes = evidence.get("probes") or []
    if probes:
        present = sum(1 for p in probes if p.get("present"))
        refs.append(f"live_reprobe:{present}/{len(probes)} locators still resolve")
    elif evidence.get("locator_count"):
        refs.append("live_reprobe:page could not be reopened")
    return refs


def _result_for_prompt(result: TestResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "duration_s": result.duration_s,
        "attempts": result.attempts,
        "error_type": result.error_type,
        "error_message": (result.error_message or "")[:1200],
        "traceback_tail": (result.traceback or "")[-1200:],
        "final_url": result.final_url,
    }


def _emit(ctx: RunContext, action: HealerAction, result: TestResult) -> None:
    if action.auto_applied:
        summary = (
            f"Healer: {action.confidence:.2f} confidence - script fix auto-applied to "
            f"{action.flow_id}, re-running once"
        )
    elif action.classification is DefectClass.GENUINE_DEFECT:
        summary = (
            f"Healer: {action.flow_id} classified GENUINE_DEFECT at "
            f"{action.confidence:.2f} - routing to the bug packager"
        )
    else:
        summary = (
            f"Healer: {action.confidence:.2f} confidence - NOT auto-applied, "
            f"queued for human review ({action.flow_id})"
        )
    ctx.emit(
        "healer",
        "decision",
        summary,
        detail=f"{action.rationale} | {action.patch_summary}".strip(" |")[:900],
        confidence=action.confidence,
        risk=action.risk.value if action.risk else None,
        flow_id=action.flow_id,
        auto_applied=action.auto_applied,
        needs_human_review=action.needs_human_review,
    )


def summarise_actions(actions: Sequence[HealerAction]) -> dict[str, int]:
    return {
        "total": len(actions),
        "script_issues": sum(1 for a in actions if a.classification is DefectClass.SCRIPT_ISSUE),
        "genuine_defects": sum(1 for a in actions if a.classification is DefectClass.GENUINE_DEFECT),
        "environment": sum(1 for a in actions if a.classification is DefectClass.ENVIRONMENT),
        "unknown": sum(1 for a in actions if a.classification is DefectClass.UNKNOWN),
        "auto_applied": sum(1 for a in actions if a.auto_applied),
        "needs_human_review": sum(1 for a in actions if a.needs_human_review),
    }


def failing_results(results: Sequence[TestResult]) -> list[TestResult]:
    """Results the Healer should look at: failures and errors, not skips."""
    return [r for r in results if r.status in (TestStatus.FAILED, TestStatus.ERROR)]
