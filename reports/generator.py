"""Deterministic assembly and rendering of the final run report.

This module contains **no LLM calls**. The report node asks the model for the
executive summary, then hands the parsed result here as a plain dict. That
split exists for two reasons:

* **Testability.** Ordering, joining and rendering are pure functions of the
  run artifacts, so ``tests/test_report_order.py`` can assert on them without a
  network call, an API key, or a mocked provider.

* **Survivability.** A rate-limited or down provider must not destroy the
  deliverable. When ``synthesis`` is ``None`` we compose an honest,
  deterministic summary from the totals and say plainly that it was written
  without the model, rather than emitting an empty section or pretending a
  model wrote it.

The ordering rule is the point of the report
--------------------------------------------
Rows are sorted by **business risk first**, never by flow index and never by
pass/fail alone. A report ordered by flow index tells a manager nothing; a
report whose first row is the highest-risk failure tells them exactly where to
look. Within a risk band the scariest outcome floats to the top (see
:func:`sort_flows_by_risk`).

Redaction discipline
--------------------
Everything this module writes to disk or returns as text is routed through
:mod:`security`: structured data through :func:`security.redact_secrets`, free
text through :func:`security.redact_text`, and URLs through
:func:`security.sanitize_url`. The renderers redact per field rather than
trusting the caller, because report text is largely model output and page-derived
prose, and a renderer's output is frequently pasted somewhere else.

The module has no import-time side effects: nothing is read, written or created
until a function is called.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from config import CONFIDENCE_AUTO_APPLY_THRESHOLD, Settings, get_settings
from graph.state import (
    CoverageEvaluation,
    DecisionEvent,
    FinalReport,
    FlowReportRow,
    GeneratedTest,
    HealerAction,
    PackagedBug,
    PRDGapItem,
    RiskClassification,
    RiskLevel,
    TestFlow,
    TestPlan,
    TestResult,
    TestStatus,
    VisualFinding,
)
from logging_setup import get_logger
from security import redact_secrets, redact_text, sanitize_url

log = get_logger("aivor.report")

T = TypeVar("T", bound=BaseModel)

# --------------------------------------------------------------------------
# Tunables that belong to presentation rather than to the run itself
# --------------------------------------------------------------------------
MAX_DECISION_LOG_EXCERPT: int = 40
"""Roughly how many decision-log events survive into the report excerpt."""

PRIORITY_EVENT_KINDS: frozenset[str] = frozenset({"decision", "replan", "escalate", "error"})
"""Event kinds that are always kept: these are the agent's actual judgment calls."""

MAX_CELL_CHARS: int = 400
"""Table cells are truncated so one runaway stack trace cannot ruin a table."""

_RISK_ABBREV: dict[str, str] = {"high": "HIGH", "medium": "MED", "low": "LOW"}


# ==========================================================================
# Small, total coercion helpers
#
# Everything reaching this module may arrive as a pydantic model (in-process
# graph state) or as a plain dict (state rehydrated from JSON by the API or a
# resumed run). Every helper below accepts both and never raises: one malformed
# item must cost that item, not the whole report.
# ==========================================================================
def _coerce_model(value: Any, model_cls: type[T]) -> T | None:
    """Return ``value`` as ``model_cls``, or ``None`` if it cannot be parsed."""
    if value is None:
        return None
    if isinstance(value, model_cls):
        return value
    try:
        if isinstance(value, Mapping):
            return model_cls.model_validate(dict(value))
        if isinstance(value, BaseModel):
            return model_cls.model_validate(value.model_dump(mode="json"))
    except ValidationError as exc:
        log.warning("dropping malformed %s in report input: %s", model_cls.__name__, exc)
    except Exception:  # pragma: no cover - defensive, must never break the report
        log.warning("unexpected %s payload in report input", model_cls.__name__, exc_info=True)
    return None


def _coerce_models(values: Any, model_cls: type[T]) -> list[T]:
    """Coerce an iterable into a list of ``model_cls``, skipping bad entries."""
    if not values:
        return []
    if isinstance(values, (str, bytes, Mapping)):
        values = [values]
    out: list[T] = []
    try:
        iterator = list(values)
    except TypeError:  # pragma: no cover - not iterable
        return []
    for item in iterator:
        parsed = _coerce_model(item, model_cls)
        if parsed is not None:
            out.append(parsed)
    return out


def _as_risk(value: Any) -> RiskLevel:
    """Normalise anything risk-shaped to a :class:`RiskLevel`; unknown -> MEDIUM."""
    if isinstance(value, RiskLevel):
        return value
    try:
        return RiskLevel(str(value).strip().lower())
    except (ValueError, TypeError):
        return RiskLevel.MEDIUM


def _as_status(value: Any) -> TestStatus:
    """Normalise anything status-shaped to a :class:`TestStatus`; unknown -> ERROR.

    ERROR rather than PASSED is the safe default: an unreadable status must
    never be reported as a success.
    """
    if isinstance(value, TestStatus):
        return value
    try:
        return TestStatus(str(value).strip().lower())
    except (ValueError, TypeError):
        return TestStatus.ERROR


def _text(value: Any, limit: int = MAX_CELL_CHARS) -> str:
    """Redact, flatten and truncate a free-text value for display."""
    if value is None:
        return ""
    raw = value if isinstance(value, str) else str(value)
    cleaned = redact_text(raw).replace("\r", " ").replace("\n", " ").strip()
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(ratio: Any, digits: int = 1) -> str:
    try:
        return f"{float(ratio) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` and naive values."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _short_time(value: Any) -> str:
    parsed = _parse_ts(value)
    return parsed.strftime("%H:%M:%S") if parsed else _text(value, 24)


def _category_text(flow: TestFlow | None) -> str:
    if flow is None:
        return ""
    category = flow.category
    return category.value if hasattr(category, "value") else str(category)


# ==========================================================================
# Ordering - the graded part
# ==========================================================================
def _risk_rank(risk: Any) -> int:
    """Primary sort key: HIGH(0) < MEDIUM(1) < LOW(2)."""
    return _as_risk(risk).sort_key


def _outcome_rank(row: FlowReportRow) -> int:
    """Secondary sort key inside a risk band: the scariest outcome first.

    The ranks are deliberately fall-through, checked in this order:

    ``0`` failed/error, ``1`` needs human review, ``2`` healed,
    ``3`` visual regression, ``4`` passed, ``5`` anything else (skipped, or a
    status this build does not recognise). Skipped sorts last inside its band
    because it carries no observed signal at all; it is still surfaced loudly
    via its ``outcome_label`` and the ``skipped`` total.
    """
    try:
        status = _as_status(getattr(row, "status", None))
        if status in (TestStatus.FAILED, TestStatus.ERROR):
            return 0
        if bool(getattr(row, "needs_human_review", False)):
            return 1
        if status is TestStatus.HEALED or bool(getattr(row, "healed", False)):
            return 2
        if bool(getattr(row, "visual_regression", False)):
            return 3
        if status is TestStatus.PASSED:
            return 4
    except Exception:  # pragma: no cover - the sort must be total
        log.debug("could not rank a report row; sorting it last", exc_info=True)
        return 5
    return 5


def sort_flows_by_risk(rows: Sequence[FlowReportRow]) -> list[FlowReportRow]:
    """Order report rows by business risk, then by outcome severity, then stably.

    This is the ordering the whole report exists to express, so it is a pure,
    total function with no I/O and no logging in the hot path: given the same
    rows it always returns the same order, and it never raises.

    * PRIMARY: risk band - HIGH, then MEDIUM, then LOW.
    * SECONDARY: :func:`_outcome_rank` - failure, human-review queue, heal,
      visual regression, pass, everything else.
    * TERTIARY: the caller's original index, which keeps the sort stable and
      makes two runs over the same data byte-identical.
    """
    indexed = list(enumerate(rows or []))
    indexed.sort(key=lambda pair: (_risk_rank(pair[1].risk), _outcome_rank(pair[1]), pair[0]))
    return [row for _, row in indexed]


# ==========================================================================
# Row assembly
# ==========================================================================
def _outcome_label(
    *,
    status: TestStatus,
    result: TestResult | None,
    heal: HealerAction | None,
    visual: VisualFinding | None,
    bug_ids: Sequence[str],
) -> str:
    """Short human sentence describing what actually happened to one flow.

    The label is what a reader scans; it must be specific enough to act on
    without opening the JSON. Confidence numbers are quoted verbatim so the
    0.6 auto-apply threshold is visible in the artifact, not just in the code.
    """
    regression = bool(visual is not None and visual.is_regression)
    ratio_text = _fmt_pct(visual.changed_ratio) if visual is not None else ""

    if result is None:
        return "skipped - no test was generated"

    if status is TestStatus.SKIPPED:
        reason = _text(result.error_message, 120)
        return f"skipped - {reason}" if reason else "skipped - the test was not executed"

    if status is TestStatus.HEALED:
        rerun = (result.status.value if isinstance(result.status, TestStatus) else "") or ""
        if heal is not None and heal.rerun_status and heal.rerun_status.lower() not in (
            "passed",
            "healed",
        ):
            return f"healed but the re-run ended {_text(heal.rerun_status, 40)}"
        label = "healed and passed on re-run"
        if regression:
            label += f", visual regression {ratio_text}"
        return label if not rerun else label

    if status in (TestStatus.FAILED, TestStatus.ERROR):
        prefix = "failed" if status is TestStatus.FAILED else "error"
        if heal is not None and heal.needs_human_review and not heal.auto_applied:
            return f"{prefix} - queued for human review (confidence {_fmt_float(heal.confidence)})"
        if bug_ids:
            joined = ", ".join(str(b) for b in bug_ids[:3])
            return f"{prefix} - bug {joined} filed with a repro script"
        detail = _text(result.error_type or result.error_message, 80)
        return f"{prefix} - {detail}" if detail else prefix

    if status is TestStatus.PASSED:
        if regression:
            return f"passed functionally, visual regression {ratio_text}"
        if visual is not None and visual.is_new_baseline:
            return "passed; a visual baseline was captured for the first time"
        return "passed"

    return _text(status.value if isinstance(status, TestStatus) else status, 60) or "unknown"


def build_flow_rows(
    *,
    flows: Sequence[TestFlow | Mapping[str, Any]] | None = None,
    classifications: Sequence[RiskClassification | Mapping[str, Any]] | None = None,
    results: Sequence[TestResult | Mapping[str, Any]] | None = None,
    healer_actions: Sequence[HealerAction | Mapping[str, Any]] | None = None,
    visual_findings: Sequence[VisualFinding | Mapping[str, Any]] | None = None,
    packaged_bugs: Sequence[PackagedBug | Mapping[str, Any]] | None = None,
) -> list[FlowReportRow]:
    """Join every per-flow artifact into one risk-ordered table.

    One row per planned flow, joined on ``flow_id`` and tolerant of missing
    pieces: a flow that was never generated or never executed still gets a row
    (status ``SKIPPED``) so that a coverage hole is visible rather than absent.
    Results whose ``flow_id`` is not in the plan - which should not happen, but
    would silently lose a real failure if it did - are appended as extra rows.

    Returns the rows already sorted by :func:`sort_flows_by_risk`.
    """
    flow_models = _coerce_models(flows, TestFlow)
    risk_models = _coerce_models(classifications, RiskClassification)
    result_models = _coerce_models(results, TestResult)
    heal_models = _coerce_models(healer_actions, HealerAction)
    visual_models = _coerce_models(visual_findings, VisualFinding)
    bug_models = _coerce_models(packaged_bugs, PackagedBug)

    risk_by_flow: dict[str, RiskClassification] = {c.flow_id: c for c in risk_models}
    # Later entries win for results and heals: a healer re-run appends a newer
    # verdict for the same flow and that is the one the report must show.
    result_by_flow: dict[str, TestResult] = {r.flow_id: r for r in result_models}
    heal_by_flow: dict[str, HealerAction] = {h.flow_id: h for h in heal_models}
    visual_by_flow: dict[str, VisualFinding] = {}
    for finding in visual_models:
        visual_by_flow.setdefault(finding.flow_id, finding)
    bugs_by_flow: dict[str, list[str]] = {}
    for bug in bug_models:
        bugs_by_flow.setdefault(bug.flow_id, []).append(bug.bug_id)

    rows: list[FlowReportRow] = []
    seen_ids: set[str] = set()

    for flow in flow_models:
        row = _build_one_row(
            flow_id=flow.id,
            flow_name=flow.name or flow.id,
            category=_category_text(flow),
            classification=risk_by_flow.get(flow.id),
            result=result_by_flow.get(flow.id),
            heal=heal_by_flow.get(flow.id),
            visual=visual_by_flow.get(flow.id),
            bug_ids=bugs_by_flow.get(flow.id, []),
        )
        if row is not None:
            rows.append(row)
            seen_ids.add(flow.id)

    for result in result_models:
        if result.flow_id in seen_ids:
            continue
        seen_ids.add(result.flow_id)
        log.warning(
            "result for flow %s has no matching plan entry; reporting it as an orphan row",
            result.flow_id,
        )
        row = _build_one_row(
            flow_id=result.flow_id,
            flow_name=result.flow_name or result.flow_id,
            category="unplanned",
            classification=risk_by_flow.get(result.flow_id),
            result=result,
            heal=heal_by_flow.get(result.flow_id),
            visual=visual_by_flow.get(result.flow_id),
            bug_ids=bugs_by_flow.get(result.flow_id, []),
        )
        if row is not None:
            rows.append(row)

    return sort_flows_by_risk(rows)


def _build_one_row(
    *,
    flow_id: str,
    flow_name: str,
    category: str,
    classification: RiskClassification | None,
    result: TestResult | None,
    heal: HealerAction | None,
    visual: VisualFinding | None,
    bug_ids: Sequence[str],
) -> FlowReportRow | None:
    """Build a single row, returning ``None`` only if the row itself is invalid."""
    try:
        if classification is not None:
            risk = _as_risk(classification.risk)
            rationale = classification.rationale
        elif heal is not None and heal.risk is not None:
            risk = _as_risk(heal.risk)
            rationale = "Risk carried over from the healer's record for this flow."
        else:
            risk = RiskLevel.MEDIUM
            rationale = "No risk classification was recorded for this flow; defaulted to MEDIUM."

        status = _as_status(result.status) if result is not None else TestStatus.SKIPPED
        healed = status is TestStatus.HEALED or bool(heal is not None and heal.auto_applied)
        needs_review = bool(heal is not None and heal.needs_human_review)

        return FlowReportRow(
            flow_id=flow_id,
            flow_name=flow_name or flow_id,
            category=category,
            risk=risk,
            risk_rationale=_text(rationale, 300),
            status=status,
            outcome_label=_outcome_label(
                status=status, result=result, heal=heal, visual=visual, bug_ids=bug_ids
            ),
            duration_s=round(float(result.duration_s), 2) if result is not None else 0.0,
            healed=healed,
            needs_human_review=needs_review,
            visual_regression=bool(visual is not None and visual.is_regression),
            bug_ids=[str(b) for b in bug_ids],
            error_message=_text(result.error_message, 500) or None if result is not None else None,
        )
    except (ValidationError, TypeError, ValueError):
        log.warning("could not build a report row for flow %s", flow_id, exc_info=True)
        return None


# ==========================================================================
# Decision log excerpt
# ==========================================================================
def _thin_middle(values: Sequence[int], keep: int) -> list[int]:
    """Keep ``keep`` entries from the head and tail, dropping the middle.

    Beginning and end carry the most meaning in a pipeline log (what was
    decided up front, how the run ended), so when we must drop events we drop
    them from the middle rather than truncating one end.
    """
    if keep <= 0:
        return []
    items = list(values)
    if len(items) <= keep:
        return items
    head = (keep + 1) // 2
    tail = keep - head
    return items[:head] + (items[-tail:] if tail else [])


def _decision_log_excerpt(
    events: Sequence[DecisionEvent], cap: int = MAX_DECISION_LOG_EXCERPT
) -> list[DecisionEvent]:
    """Pick the most informative events, in chronological order.

    Kept unconditionally: every ``decision``, ``replan``, ``escalate`` and
    ``error`` event - these are the agent's reasoning, which is the thing a
    reviewer actually wants to audit. Kept if the budget allows: the first and
    last event of each stage, which give the excerpt a skeleton. Dropped first:
    routine ``progress`` chatter.
    """
    if not events:
        return []

    priority = [i for i, event in enumerate(events) if str(event.event) in PRIORITY_EVENT_KINDS]

    first_of_stage: dict[str, int] = {}
    last_of_stage: dict[str, int] = {}
    for index, event in enumerate(events):
        stage = str(event.stage)
        first_of_stage.setdefault(stage, index)
        last_of_stage[stage] = index
    boundaries = sorted((set(first_of_stage.values()) | set(last_of_stage.values())) - set(priority))

    if len(priority) >= cap:
        selected = _thin_middle(priority, cap)
    else:
        selected = sorted(set(priority) | set(_thin_middle(boundaries, cap - len(priority))))

    return [events[i] for i in selected]


# ==========================================================================
# Totals, summary, limitations
# ==========================================================================
def _duration_seconds(state: Mapping[str, Any], results: Sequence[TestResult]) -> float:
    """Wall-clock run duration, degrading to the sum of test durations.

    If ``started_at`` is missing we cannot know the wall clock, so we report
    the summed execution time instead of guessing - and the difference is
    visible because the number is smaller, not because we hid it.
    """
    start = _parse_ts(state.get("started_at"))
    if start is not None:
        end = _parse_ts(state.get("finished_at")) or datetime.now(timezone.utc)
        return round(max(0.0, (end - start).total_seconds()), 2)
    return round(sum(float(r.duration_s or 0.0) for r in results), 2)


def _compute_totals(
    *,
    plan: TestPlan | None,
    rows: Sequence[FlowReportRow],
    generated: Sequence[GeneratedTest],
    results: Sequence[TestResult],
    heals: Sequence[HealerAction],
    visuals: Sequence[VisualFinding],
    bugs: Sequence[PackagedBug],
    replan_count: int,
    duration_s: float,
) -> dict[str, Any]:
    """The header numbers. Every one of them is derived, never reported by a model."""
    statuses = [_as_status(row.status) for row in rows]
    high_rows = [row for row in rows if _as_risk(row.risk) is RiskLevel.HIGH]
    return {
        "flows_planned": len(plan.flows) if plan is not None else len(rows),
        "tests_generated": len(generated),
        "passed": sum(1 for s in statuses if s is TestStatus.PASSED),
        "failed": sum(1 for s in statuses if s in (TestStatus.FAILED, TestStatus.ERROR)),
        "healed": sum(1 for row in rows if row.healed or _as_status(row.status) is TestStatus.HEALED),
        "skipped": sum(1 for s in statuses if s is TestStatus.SKIPPED),
        "needs_human_review": sum(1 for h in heals if h.needs_human_review),
        "bugs_filed": len(bugs),
        "visual_regressions": sum(1 for v in visuals if v.is_regression),
        "high_risk_flows": len(high_rows),
        "high_risk_failures": sum(
            1
            for row in high_rows
            if _as_status(row.status) in (TestStatus.FAILED, TestStatus.ERROR)
        ),
        "replan_count": int(replan_count),
        "duration_s": duration_s,
        "tests_executed": sum(1 for s in statuses if s is not TestStatus.SKIPPED),
    }


def _normalise_limitation(text: str) -> frozenset[str]:
    """Content words of a limitation, used only for duplicate detection."""
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "were", "that",
        "this", "it", "as", "for", "on", "with", "not", "so", "are", "be", "by", "at",
    }
    words = {
        "".join(ch for ch in token.lower() if ch.isalnum())
        for token in text.split()
    }
    return frozenset(word for word in words if word and word not in stop)


def _is_duplicate_limitation(candidate: str, existing: Sequence[str]) -> bool:
    """True when the model already said essentially the same thing.

    Deliberately conservative: a false negative merely repeats a caveat, while
    a false positive silently deletes a computed caveat we are required to
    state. Only near-identical wording is treated as a duplicate.
    """
    candidate_words = _normalise_limitation(candidate)
    if not candidate_words:
        return True
    for item in existing:
        item_words = _normalise_limitation(item)
        if not item_words:
            continue
        overlap = len(candidate_words & item_words)
        union = len(candidate_words | item_words)
        if union and overlap / union >= 0.75:
            return True
        smaller = min(len(candidate_words), len(item_words))
        if smaller >= 4 and overlap == smaller:
            return True
    return False


def _computed_limitations(
    *,
    cfg: Settings,
    totals: Mapping[str, Any],
    rows: Sequence[FlowReportRow],
    heals: Sequence[HealerAction],
    visuals: Sequence[VisualFinding],
    plan: TestPlan | None,
    coverage: CoverageEvaluation | None,
    force_proceeded: bool,
    replan_count: int,
    login_ok: bool | None,
    credentials_present: bool,
    offline: bool,
    node_error_count: int,
) -> list[str]:
    """Caveats this process can prove, independent of anything a model said.

    These exist so the report cannot overclaim. Each one is derived from a fact
    of the run: a force-proceed happened or it did not, the crawl budget is a
    number in the settings, a heal was withheld or it was not.
    """
    items: list[str] = []

    if offline:
        items.append(
            "This run used the deterministic offline stub instead of a real model "
            "(LLM_OFFLINE_MODE). Plans, risk verdicts, defect classifications and this "
            "report's wording are canned placeholders, not model judgment: treat the run as "
            "a plumbing smoke test, not as a test of the target application."
        )

    if force_proceeded:
        items.append(
            f"The coverage gate was never satisfied. After {replan_count} re-plan(s) the "
            "orchestrator force-proceeded, so the plan is known to be incomplete and the gaps "
            "listed under Coverage evaluation were never exercised."
        )

    if login_ok is False:
        items.append(
            "Login did not succeed, so every authenticated area was visited as an anonymous "
            "visitor. Any flow behind the login wall is unverified regardless of its status here."
        )
    elif not credentials_present and (plan is not None and plan.auth_flow_present):
        items.append(
            "The target exposes an authentication flow but no credentials were supplied, so "
            "everything behind the login wall was out of reach and is untested."
        )

    items.append(
        f"Discovery was budget-capped at {cfg.crawl_max_pages} page(s) to depth "
        f"{cfg.crawl_max_depth}"
        + (" on the same origin" if cfg.crawl_same_origin_only else "")
        + ". Anything beyond that budget was never seen, so it is neither covered nor "
        "reported as missing."
    )

    items.append(
        "Coverage is a sample, not a guarantee. A clean run means the flows listed below "
        "behaved as expected on this target at this moment; it does not mean the application "
        "is defect-free, and untested areas are not evidence of correctness."
    )

    withheld = [h for h in heals if h.needs_human_review and not h.auto_applied]
    if withheld:
        items.append(
            f"{len(withheld)} proposed fix(es) scored below the "
            f"{CONFIDENCE_AUTO_APPLY_THRESHOLD:.2f} auto-apply confidence threshold and were "
            "NOT applied. They are listed in the human review queue with their evidence; the "
            "underlying failures remain unresolved."
        )

    if cfg.enable_visual_diff or visuals:
        items.append(
            "Visual comparison is pixel-based against a stored baseline at a "
            f"{cfg.visual_diff_threshold * 100:.1f}% changed-pixel threshold. It detects that "
            "rendering changed, not whether the change is a defect; intentional design changes "
            "register as regressions until the baseline is refreshed."
        )
    else:
        items.append(
            "Visual regression checking was disabled for this run, so rendering changes "
            "that do not break an assertion would not have been detected."
        )

    skipped = int(totals.get("skipped", 0) or 0)
    if skipped:
        items.append(
            f"{skipped} planned flow(s) produced no executable test or never ran, so they "
            "contribute nothing to the pass rate and should be read as gaps, not as passes."
        )

    if coverage is not None and not coverage.passed:
        missing = len(coverage.missing) or len(coverage.failed_requirements())
        if missing:
            items.append(
                f"The coverage rubric still reports {missing} unmet requirement(s); those "
                "areas are outside what this run can speak to."
            )

    if node_error_count:
        items.append(
            f"{node_error_count} stage(s) reported an error and the pipeline continued in "
            "degraded mode. See the Errors section before trusting the totals."
        )

    if not cfg.enable_prd_gap_analysis:
        items.append(
            "PRD gap analysis was disabled, so no claim is made about whether the plan covers "
            "the written requirements."
        )

    return items


def _deterministic_summary(
    *,
    target_url: str,
    totals: Mapping[str, Any],
    rows: Sequence[FlowReportRow],
    force_proceeded: bool,
    login_ok: bool | None,
    offline: bool,
) -> str:
    """Compose an honest executive summary without a model.

    Used when the synthesis call failed or was skipped. It states up front that
    no model wrote it, so nobody mistakes a template for analysis. This is the
    degraded path, and it says so.
    """
    sentences: list[str] = [
        "This summary was composed deterministically from the run totals because the "
        "executive-summary model call was unavailable; no model wrote this text."
    ]
    sentences.append(
        f"{totals.get('flows_planned', 0)} flow(s) were planned for "
        f"{sanitize_url(target_url)} and {totals.get('tests_executed', 0)} executed: "
        f"{totals.get('passed', 0)} passed, {totals.get('failed', 0)} failed, "
        f"{totals.get('healed', 0)} healed, {totals.get('skipped', 0)} never ran."
    )

    worst = next(
        (
            row
            for row in rows
            if _as_risk(row.risk) is RiskLevel.HIGH
            and _as_status(row.status) in (TestStatus.FAILED, TestStatus.ERROR)
        ),
        None,
    )
    if worst is not None:
        sentences.append(
            f"The highest-risk failure is \"{_text(worst.flow_name, 120)}\" "
            f"({_text(worst.outcome_label, 160)}); it is the first row of the results table."
        )
    elif int(totals.get("high_risk_flows", 0) or 0):
        high_risk_rows = [row for row in rows if _as_risk(row.risk) is RiskLevel.HIGH]
        unverified = [
            row
            for row in high_risk_rows
            if _as_status(row.status) not in (TestStatus.PASSED, TestStatus.HEALED)
        ]
        if unverified:
            # A skipped high-risk flow is an open question, not a clean bill of
            # health. Saying "nothing high-risk failed" here would be the exact
            # kind of overstatement this report is supposed to avoid.
            sentences.append(
                f"No high-risk flow failed, but {len(unverified)} of "
                f"{len(high_risk_rows)} high-risk flow(s) were never actually "
                f"verified ({_text(unverified[0].outcome_label, 120)}), so this run "
                "says nothing about them either way."
            )
        else:
            sentences.append(
                f"No high-risk flow failed: all {len(high_risk_rows)} high-risk "
                "flow(s) either passed or were healed."
            )
    else:
        sentences.append("No flow in this run was classified as high risk.")

    sentences.append(
        f"{totals.get('bugs_filed', 0)} defect(s) were packaged with a repro script and "
        f"{totals.get('needs_human_review', 0)} finding(s) were queued for human review; "
        f"{totals.get('visual_regressions', 0)} visual regression(s) were detected."
    )

    degraded: list[str] = []
    if force_proceeded:
        degraded.append("the coverage gate was force-proceeded")
    if login_ok is False:
        degraded.append("authentication failed, so protected areas were not reached")
    if offline:
        degraded.append("the run used the offline stub rather than a real model")
    if degraded:
        sentences.append("Degraded modes in effect: " + "; ".join(degraded) + ".")

    return " ".join(sentences)


def _deterministic_business_impact(totals: Mapping[str, Any]) -> str:
    """A concrete, defensible impact line. Never a dollar figure.

    We can honestly claim what the pipeline produced - risk-ordered flows,
    tickets with runnable repros, withheld low-confidence patches. We cannot
    honestly claim what an hour of triage is worth to this organisation, so we
    do not.
    """
    bugs = int(totals.get("bugs_filed", 0) or 0)
    flows = int(totals.get("flows_planned", 0) or 0)
    review = int(totals.get("needs_human_review", 0) or 0)
    high = int(totals.get("high_risk_flows", 0) or 0)

    parts = [
        f"The agent ordered {flows} flow(s) by business risk ({high} high-risk) so triage "
        "starts at the most expensive failure rather than at flow number one."
    ]
    if bugs:
        parts.append(
            f"It auto-filed {bugs} ticket(s), each carrying a runnable repro script and "
            "captured evidence, so an engineer starts from a reproduction instead of a "
            "screenshot and a guess."
        )
    else:
        parts.append(
            "No defect ticket was filed in this run, so nothing here needs engineering triage."
        )
    if review:
        parts.append(
            f"{review} low-confidence finding(s) were withheld from auto-apply and routed to a "
            "human, which is the difference between an agent that hides failures and one that "
            "escalates them."
        )
    return " ".join(parts)


# ==========================================================================
# Assembly
# ==========================================================================
def assemble_report(
    *,
    run_id: str,
    target_url: str,
    state_like: dict[str, Any],
    synthesis: dict[str, Any] | None,
    llm_provider: str,
    models_used: dict[str, str],
    settings: Settings | None = None,
) -> FinalReport:
    """Turn collected run state into the immutable :class:`FinalReport`.

    ``state_like`` mirrors :class:`graph.state.OrchestrationState` but every key
    is optional: this function is also called on a crashed run, where most of
    the pipeline never produced anything. Missing keys default to empty, never
    to a fabricated value.

    ``synthesis`` is the parsed ``{"executive_summary", "business_impact",
    "limitations"}`` object from the report node, or ``None`` when the model
    call failed. In the ``None`` case the summary is composed deterministically
    and says so in its first sentence.

    Computed limitations are always appended to whatever the model produced,
    de-duplicated against it, because the model cannot be trusted to volunteer
    the caveats that make the report honest.
    """
    cfg = settings or get_settings()
    state: Mapping[str, Any] = state_like or {}

    plan = _coerce_model(state.get("test_plan"), TestPlan)
    coverage = _coerce_model(state.get("coverage_evaluation"), CoverageEvaluation)
    classifications = _coerce_models(state.get("risk_classifications"), RiskClassification)
    generated = _coerce_models(state.get("generated_tests"), GeneratedTest)
    results = _coerce_models(state.get("run_results"), TestResult)
    heals = _coerce_models(state.get("healer_actions"), HealerAction)
    visuals = _coerce_models(state.get("visual_diff_findings"), VisualFinding)
    bugs = _coerce_models(state.get("packaged_bugs"), PackagedBug)
    prd_gaps = _coerce_models(state.get("prd_gaps"), PRDGapItem)
    events = _coerce_models(state.get("decision_log"), DecisionEvent)

    rows = build_flow_rows(
        flows=plan.flows if plan is not None else [],
        classifications=classifications,
        results=results,
        healer_actions=heals,
        visual_findings=visuals,
        packaged_bugs=bugs,
    )

    replan_count = int(state.get("replan_count", 0) or 0)
    force_proceeded = bool(state.get("force_proceeded", False))
    login_ok = state.get("login_ok")
    login_ok = None if login_ok is None else bool(login_ok)
    credentials_present = bool(state.get("credentials_present", False))
    feature_flags = dict(state.get("feature_flags") or cfg.feature_flags())
    node_errors = [str(item) for item in (state.get("node_errors") or [])]
    escalations = [str(item) for item in (state.get("escalations") or [])]

    offline = (
        str(llm_provider).strip().lower() in {"offline-stub", "offline_stub", "offline"}
        or bool(feature_flags.get("LLM_OFFLINE_MODE"))
        or cfg.llm_offline_mode
    )

    totals = _compute_totals(
        plan=plan,
        rows=rows,
        generated=generated,
        results=results,
        heals=heals,
        visuals=visuals,
        bugs=bugs,
        replan_count=replan_count,
        duration_s=_duration_seconds(state, results),
    )

    # ---- narrative ------------------------------------------------------
    synth = synthesis if isinstance(synthesis, Mapping) else {}
    summary_text = _text(synth.get("executive_summary"), 4000)
    if not summary_text:
        summary_text = _deterministic_summary(
            target_url=target_url,
            totals=totals,
            rows=rows,
            force_proceeded=force_proceeded,
            login_ok=login_ok,
            offline=offline,
        )
    impact_text = _text(synth.get("business_impact"), 1200)
    if not impact_text:
        impact_text = _deterministic_business_impact(totals)

    model_limitations: list[str] = []
    raw_limitations = synth.get("limitations")
    if isinstance(raw_limitations, str):
        raw_limitations = [raw_limitations]
    for item in raw_limitations or []:
        cleaned = _text(item, 600)
        if cleaned and cleaned not in model_limitations:
            model_limitations.append(cleaned)

    limitations = list(model_limitations)
    for computed in _computed_limitations(
        cfg=cfg,
        totals=totals,
        rows=rows,
        heals=heals,
        visuals=visuals,
        plan=plan,
        coverage=coverage,
        force_proceeded=force_proceeded,
        replan_count=replan_count,
        login_ok=login_ok,
        credentials_present=credentials_present,
        offline=offline,
        node_error_count=len(node_errors),
    ):
        if not _is_duplicate_limitation(computed, limitations):
            limitations.append(computed)

    # ---- coverage gaps --------------------------------------------------
    coverage_gaps: list[str] = []
    if coverage is not None:
        for gap in list(coverage.missing) + coverage.failed_requirements():
            cleaned = _text(gap, 300)
            if cleaned and cleaned not in coverage_gaps:
                coverage_gaps.append(cleaned)

    # ---- errors ---------------------------------------------------------
    errors: list[str] = []
    run_error = state.get("error")
    if run_error:
        errors.append(f"Run error: {_text(run_error, 500)}")
    login_error = state.get("login_error")
    if login_error:
        errors.append(f"Login: {_text(login_error, 300)}")
    for item in node_errors:
        errors.append(_text(item, 500))
    for item in escalations:
        errors.append(f"Escalation: {_text(item, 500)}")
    for test in generated:
        if not test.valid and test.validation_error:
            errors.append(
                f"Generated test for {test.flow_id} was rejected: "
                f"{_text(test.validation_error, 300)}"
            )
    deduped_errors: list[str] = []
    for item in errors:
        if item and item not in deduped_errors:
            deduped_errors.append(item)

    status = str(state.get("status") or "completed")

    return FinalReport(
        run_id=run_id,
        target_url=sanitize_url(target_url or (plan.target_url if plan else "")),
        status=status,
        executive_summary=summary_text,
        business_impact=impact_text,
        llm_provider=str(llm_provider or "unknown"),
        models_used={str(k): _text(v, 120) for k, v in (models_used or {}).items()},
        totals=totals,
        flows=rows,
        coverage_evaluation=coverage,
        coverage_gaps=coverage_gaps,
        replan_count=replan_count,
        force_proceeded=force_proceeded,
        login_ok=login_ok,
        credentials_present=credentials_present,
        healer_actions=heals,
        needs_human_review=[h for h in heals if h.needs_human_review],
        visual_findings=visuals,
        packaged_bugs=bugs,
        prd_gaps=prd_gaps,
        regression_radar=dict(state.get("regression_radar") or {}),
        decision_log_excerpt=_decision_log_excerpt(events),
        feature_flags={str(k): bool(v) for k, v in feature_flags.items()},
        limitations=limitations,
        errors=deduped_errors,
        llm_cost=dict(state.get("llm_cost") or {}),
        timings=dict(state.get("timings") or {}),
        safety=dict(state.get("safety") or {}),
    )


# ==========================================================================
# Markdown rendering
# ==========================================================================
def _md_cell(value: Any) -> str:
    """One GitHub-flavoured markdown cell: redacted, single-line, pipe-escaped."""
    text = _text(value)
    if not text:
        return "-"
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return lines


def _risk_badge(risk: Any) -> str:
    return _RISK_ABBREV.get(_as_risk(risk).value, "MED")


def _yes_no(value: Any) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def render_markdown(report: FinalReport) -> str:
    """Render the report as GitHub-flavoured markdown, risk-ordered.

    Section order is fixed so two runs are diffable, and every cell goes
    through :func:`_md_cell`, which redacts and escapes pipes. Empty sections
    still render with an explicit "nothing here" line: silence would be
    ambiguous between "clean" and "never ran".
    """
    lines: list[str] = []
    totals = report.totals or {}

    # ---- title + metadata ------------------------------------------------
    lines.append(f"# Autonomous test run report - {_text(report.run_id, 80)}")
    lines.append("")
    lines.extend(
        _md_table(
            ["Field", "Value"],
            [
                ["Target", sanitize_url(report.target_url)],
                ["Run id", report.run_id],
                ["Generated at", report.generated_at],
                ["Status", report.status],
                ["Duration (s)", _fmt_float(totals.get("duration_s"))],
                ["LLM provider", report.llm_provider],
                [
                    "Models",
                    ", ".join(f"{k}={v}" for k, v in sorted(report.models_used.items())) or "-",
                ],
                ["Re-plans used", report.replan_count],
                ["Force-proceeded", _yes_no(report.force_proceeded)],
                ["Login succeeded", _yes_no(report.login_ok)],
                ["Credentials supplied", _yes_no(report.credentials_present)],
                [
                    "Feature flags",
                    ", ".join(
                        f"{k}={'on' if v else 'off'}"
                        for k, v in sorted(report.feature_flags.items())
                    )
                    or "-",
                ],
            ],
        )
    )
    lines.append("")

    # ---- executive summary ----------------------------------------------
    lines.append("## Executive summary")
    lines.append("")
    lines.append(_text(report.executive_summary, 4000) or "_No executive summary was produced._")
    lines.append("")

    # ---- business impact -------------------------------------------------
    lines.append("## Business impact")
    lines.append("")
    lines.append(_text(report.business_impact, 1500) or "_No business impact statement._")
    lines.append("")

    # ---- totals ----------------------------------------------------------
    lines.append("## Totals")
    lines.append("")
    lines.extend(_md_table(["Metric", "Value"], [[k, totals[k]] for k in sorted(totals)]))
    lines.append("")

    # ---- risk-ranked results --------------------------------------------
    lines.append("## Results, ordered by business risk")
    lines.append("")
    lines.append(
        "Rows are sorted high risk first, and within a risk band the most serious "
        "outcome first. The first row is the thing to look at."
    )
    lines.append("")
    if report.flows:
        lines.extend(
            _md_table(
                ["Risk", "Flow", "Category", "Status", "Outcome", "Dur (s)", "Bugs", "Why this risk"],
                [
                    [
                        _risk_badge(row.risk),
                        row.flow_name,
                        row.category,
                        _as_status(row.status).value,
                        row.outcome_label,
                        _fmt_float(row.duration_s),
                        ", ".join(row.bug_ids) or "-",
                        row.risk_rationale,
                    ]
                    for row in report.flows
                ],
            )
        )
    else:
        lines.append("_No flow produced a result row; nothing was executed._")
    lines.append("")

    # ---- coverage --------------------------------------------------------
    lines.append("## Coverage evaluation")
    lines.append("")
    coverage = report.coverage_evaluation
    if coverage is not None:
        lines.append(
            f"Gate {'passed' if coverage.passed else 'FAILED'} - score "
            f"{_fmt_float(coverage.score)}, judge confidence {_fmt_float(coverage.confidence)}, "
            f"evaluated plan revision {coverage.evaluated_revision}."
        )
        lines.append("")
        if coverage.checks:
            lines.extend(
                _md_table(
                    ["Check", "Requirement", "Satisfied", "Evidence"],
                    [
                        [c.id, c.requirement, _yes_no(c.satisfied), c.evidence]
                        for c in coverage.checks
                    ],
                )
            )
            lines.append("")
        if coverage.rationale:
            lines.append(f"Rationale: {_text(coverage.rationale, 1200)}")
            lines.append("")
    else:
        lines.append("_No coverage evaluation was recorded for this run._")
        lines.append("")
    if report.coverage_gaps:
        lines.append("Gaps the gate identified:")
        lines.append("")
        for gap in report.coverage_gaps:
            lines.append(f"- {_text(gap, 300)}")
        lines.append("")

    # ---- healer ----------------------------------------------------------
    lines.append("## Healer actions")
    lines.append("")
    if report.healer_actions:
        lines.append(
            f"A fix is auto-applied only for a SCRIPT_ISSUE at or above "
            f"{CONFIDENCE_AUTO_APPLY_THRESHOLD:.2f} confidence. Everything else is queued."
        )
        lines.append("")
        lines.extend(
            _md_table(
                ["Flow", "Classification", "Conf.", "Auto-applied", "Action", "Re-run", "Rationale"],
                [
                    [
                        h.flow_name or h.flow_id,
                        h.classification.value
                        if hasattr(h.classification, "value")
                        else h.classification,
                        _fmt_float(h.confidence),
                        _yes_no(h.auto_applied),
                        h.action,
                        h.rerun_status or "-",
                        h.rationale,
                    ]
                    for h in report.healer_actions
                ],
            )
        )
    else:
        lines.append("_No test failure required healing analysis._")
    lines.append("")

    # ---- human review queue ---------------------------------------------
    lines.append("## Needs human review")
    lines.append("")
    if report.needs_human_review:
        lines.extend(
            _md_table(
                ["Flow", "Classification", "Conf.", "Why it was not auto-applied", "Evidence"],
                [
                    [
                        h.flow_name or h.flow_id,
                        h.classification.value
                        if hasattr(h.classification, "value")
                        else h.classification,
                        _fmt_float(h.confidence),
                        h.rationale or h.action,
                        ", ".join(sanitize_url(str(r)) for r in h.evidence_refs) or "-",
                    ]
                    for h in report.needs_human_review
                ],
            )
        )
    else:
        lines.append("_Nothing was queued for human review._")
    lines.append("")

    # ---- visual ----------------------------------------------------------
    lines.append("## Visual regression findings")
    lines.append("")
    if report.visual_findings:
        lines.extend(
            _md_table(
                ["Flow", "Viewport", "Changed", "Threshold", "Regression", "Diff image", "Note"],
                [
                    [
                        v.flow_name or v.flow_id,
                        v.viewport,
                        _fmt_pct(v.changed_ratio),
                        _fmt_pct(v.threshold),
                        "new baseline" if v.is_new_baseline else _yes_no(v.is_regression),
                        v.diff_path or "-",
                        v.note,
                    ]
                    for v in report.visual_findings
                ],
            )
        )
    else:
        lines.append("_No visual comparison was performed or none produced a finding._")
    lines.append("")

    # ---- bugs ------------------------------------------------------------
    lines.append("## Packaged bugs")
    lines.append("")
    if report.packaged_bugs:
        lines.extend(
            _md_table(
                ["Bug", "Risk", "Severity", "Flow", "Title", "Repro script", "Evidence"],
                [
                    [
                        b.bug_id,
                        _risk_badge(b.risk),
                        b.severity,
                        b.flow_name or b.flow_id,
                        b.title,
                        b.repro_script_path or "-",
                        b.screenshot_path or b.ticket_path or b.directory or "-",
                    ]
                    for b in report.packaged_bugs
                ],
            )
        )
    else:
        lines.append("_No defect was packaged in this run._")
    lines.append("")

    # ---- PRD gaps (only when present) ------------------------------------
    if report.prd_gaps:
        lines.append("## PRD gap analysis")
        lines.append("")
        lines.extend(
            _md_table(
                ["Requirement", "Covered", "Best matching flow", "Similarity"],
                [
                    [g.requirement, _yes_no(g.covered), g.best_match_flow or "-", _fmt_float(g.similarity)]
                    for g in report.prd_gaps
                ],
            )
        )
        lines.append("")

    # ---- regression radar (only when present) ----------------------------
    if report.regression_radar:
        lines.append("## Regression radar")
        lines.append("")
        lines.extend(
            _md_table(
                ["Signal", "Value"],
                [[key, _stringify(report.regression_radar[key])] for key in sorted(report.regression_radar)],
            )
        )
        lines.append("")

    # ---- decision log ----------------------------------------------------
    lines.append("## Decision log excerpt")
    lines.append("")
    if report.decision_log_excerpt:
        lines.append(
            "Judgment calls, re-plans, escalations and errors are kept in full; routine "
            "progress chatter is dropped."
        )
        lines.append("")
        lines.extend(
            _md_table(
                ["Time", "Stage", "Event", "Summary", "Conf.", "Risk"],
                [
                    [
                        _short_time(e.ts),
                        e.stage,
                        e.event,
                        e.summary,
                        _fmt_float(e.confidence) if e.confidence is not None else "-",
                        _RISK_ABBREV.get(str(e.risk), "-") if e.risk else "-",
                    ]
                    for e in report.decision_log_excerpt
                ],
            )
        )
    else:
        lines.append("_No decision events were recorded._")
    lines.append("")

    # ---- limitations -----------------------------------------------------
    lines.append("## Limitations")
    lines.append("")
    if report.limitations:
        for item in report.limitations:
            lines.append(f"- {_text(item, 600)}")
    else:
        lines.append("_No limitations were recorded, which is itself suspicious._")
    lines.append("")

    # ---- errors ----------------------------------------------------------
    lines.append("## Errors")
    lines.append("")
    if report.errors:
        for item in report.errors:
            lines.append(f"- {_text(item, 600)}")
    else:
        lines.append("_No stage reported an error._")
    lines.append("")

    return "\n".join(lines)


def _stringify(value: Any) -> str:
    """Render an arbitrary radar/flag value compactly for a table cell."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _text(value)
    try:
        return _text(json.dumps(redact_secrets(value), ensure_ascii=False, default=str))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return _text(repr(value))


# ==========================================================================
# HTML rendering
# ==========================================================================
_HTML_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --panel: #f7f8fa;
  --border: #d9dde3;
  --fg: #16191d;
  --muted: #5b6470;
  --accent: #1f5eff;
  --high-bg: #fdecec; --high-fg: #9b1c1c; --high-br: #f2b8b8;
  --med-bg: #fff5e2;  --med-fg: #8a5a00;  --med-br: #f0d5a0;
  --low-bg: #eaf6ee;  --low-fg: #17643a;  --low-br: #b9dfc7;
  --bad: #b42318; --good: #17643a; --warn: #8a5a00;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216;
    --panel: #171b21;
    --border: #2a3038;
    --fg: #e6e9ee;
    --muted: #9aa4b2;
    --accent: #7aa2ff;
    --high-bg: #3a1d1d; --high-fg: #ffb4b4; --high-br: #5e2b2b;
    --med-bg: #3a2f16;  --med-fg: #ffd88a;  --med-br: #5d4a1f;
    --low-bg: #16321f;  --low-fg: #9fe0b7;  --low-br: #23512f;
    --bad: #ff9f9f; --good: #9fe0b7; --warn: #ffd88a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px;
  background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
        sans-serif;
}
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 {
  font-size: 18px; margin: 34px 0 10px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
p { margin: 0 0 12px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.note { color: var(--muted); font-size: 13px; }
.empty { color: var(--muted); font-style: italic; }
.card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; margin: 0 0 12px;
}
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 8px 11px; border-bottom: 1px solid var(--border);
         vertical-align: top; }
th { background: var(--panel); font-weight: 600; white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pill {
  display: inline-block; min-width: 46px; text-align: center;
  padding: 2px 9px; border-radius: 999px;
  font-size: 11.5px; font-weight: 700; letter-spacing: 0.04em;
  border: 1px solid transparent;
}
.pill-high { background: var(--high-bg); color: var(--high-fg); border-color: var(--high-br); }
.pill-med  { background: var(--med-bg);  color: var(--med-fg);  border-color: var(--med-br); }
.pill-low  { background: var(--low-bg);  color: var(--low-fg);  border-color: var(--low-br); }
.st-failed, .st-error { color: var(--bad); font-weight: 600; }
.st-passed { color: var(--good); font-weight: 600; }
.st-healed, .st-skipped { color: var(--warn); font-weight: 600; }
.kpis { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 14px; padding: 0; list-style: none; }
.kpis li {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px 14px; min-width: 116px;
}
.kpis .k { display: block; font-size: 11.5px; color: var(--muted); text-transform: uppercase;
           letter-spacing: 0.05em; }
.kpis .v { display: block; font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
ul.plain { margin: 0 0 12px; padding-left: 20px; }
ul.plain li { margin-bottom: 7px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
"""


def _h(value: Any) -> str:
    """Redact then HTML-escape. Order matters: escaping first would hide patterns."""
    return html.escape(_text(value, 4000), quote=True)


def _pill(risk: Any) -> str:
    level = _as_risk(risk)
    css = {"high": "pill-high", "medium": "pill-med", "low": "pill-low"}[level.value]
    return f'<span class="pill {css}">{_RISK_ABBREV[level.value]}</span>'


def _status_html(status: Any) -> str:
    value = _as_status(status).value
    return f'<span class="st-{html.escape(value)}">{html.escape(value)}</span>'


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Build a table whose cells are ALREADY escaped/marked-up HTML fragments."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        '<div class="tablewrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _empty(message: str) -> str:
    return f'<p class="empty">{html.escape(message)}</p>'


def render_html(report: FinalReport) -> str:
    """Render a self-contained HTML report: one style block, no assets, no JS.

    No CDN, no external font, no script tag - the file must open from a
    filesystem on a machine with no network, because that is where a report
    artifact usually gets read. Colours come from CSS custom properties that
    are redefined under ``prefers-color-scheme: dark``, so the same file is
    readable in either theme. Every interpolated string is redacted and then
    HTML-escaped: this document is mostly model output and page-derived text.
    """
    totals = report.totals or {}
    parts: list[str] = []

    parts.append(f"<style>{_HTML_STYLE}</style>")
    parts.append("<main>")
    parts.append(f"<h1>Autonomous test run report</h1>")
    parts.append(
        '<p class="sub">'
        f"{_h(sanitize_url(report.target_url))} &middot; run <code>{_h(report.run_id)}</code> "
        f"&middot; {_h(report.generated_at)} &middot; status {_h(report.status)}"
        "</p>"
    )

    kpi_keys = [
        ("high_risk_failures", "High-risk fails"),
        ("failed", "Failed"),
        ("passed", "Passed"),
        ("healed", "Healed"),
        ("needs_human_review", "For review"),
        ("bugs_filed", "Bugs filed"),
        ("visual_regressions", "Visual regs"),
        ("skipped", "Skipped"),
    ]
    kpis = "".join(
        f'<li><span class="k">{html.escape(label)}</span>'
        f'<span class="v">{_h(totals.get(key, 0))}</span></li>'
        for key, label in kpi_keys
        if key in totals
    )
    if kpis:
        parts.append(f'<ul class="kpis">{kpis}</ul>')

    # ---- run metadata ----------------------------------------------------
    meta_rows = [
        ["Target", _h(sanitize_url(report.target_url))],
        ["Run id", f"<code>{_h(report.run_id)}</code>"],
        ["Generated at", _h(report.generated_at)],
        ["Status", _h(report.status)],
        ["Duration (s)", _h(_fmt_float(totals.get("duration_s")))],
        ["LLM provider", _h(report.llm_provider)],
        ["Models", _h(", ".join(f"{k}={v}" for k, v in sorted(report.models_used.items())) or "-")],
        ["Re-plans used", _h(report.replan_count)],
        ["Force-proceeded", _h(_yes_no(report.force_proceeded))],
        ["Login succeeded", _h(_yes_no(report.login_ok))],
        ["Credentials supplied", _h(_yes_no(report.credentials_present))],
        [
            "Feature flags",
            _h(
                ", ".join(
                    f"{k}={'on' if v else 'off'}" for k, v in sorted(report.feature_flags.items())
                )
                or "-"
            ),
        ],
    ]
    parts.append("<h2>Run metadata</h2>")
    parts.append(_html_table(["Field", "Value"], meta_rows))

    # ---- narrative -------------------------------------------------------
    parts.append("<h2>Executive summary</h2>")
    parts.append(
        f'<div class="card">{_h(report.executive_summary)}</div>'
        if report.executive_summary
        else _empty("No executive summary was produced.")
    )
    parts.append("<h2>Business impact</h2>")
    parts.append(
        f'<div class="card">{_h(report.business_impact)}</div>'
        if report.business_impact
        else _empty("No business impact statement.")
    )

    # ---- totals ----------------------------------------------------------
    parts.append("<h2>Totals</h2>")
    parts.append(
        _html_table(
            ["Metric", "Value"],
            [[_h(key), f'<span class="num">{_h(totals[key])}</span>'] for key in sorted(totals)],
        )
    )

    # ---- risk-ranked results --------------------------------------------
    parts.append("<h2>Results, ordered by business risk</h2>")
    parts.append(
        '<p class="note">Highest business risk first; within a risk band, the most serious '
        "outcome first. The first row is the thing to look at.</p>"
    )
    if report.flows:
        parts.append(
            _html_table(
                ["Risk", "Flow", "Category", "Status", "Outcome", "Dur (s)", "Bugs", "Why this risk"],
                [
                    [
                        _pill(row.risk),
                        _h(row.flow_name),
                        _h(row.category),
                        _status_html(row.status),
                        _h(row.outcome_label),
                        f'<span class="num">{_h(_fmt_float(row.duration_s))}</span>',
                        _h(", ".join(row.bug_ids) or "-"),
                        _h(row.risk_rationale),
                    ]
                    for row in report.flows
                ],
            )
        )
    else:
        parts.append(_empty("No flow produced a result row; nothing was executed."))

    # ---- coverage --------------------------------------------------------
    parts.append("<h2>Coverage evaluation</h2>")
    coverage = report.coverage_evaluation
    if coverage is not None:
        parts.append(
            '<div class="card">'
            f"Gate {'passed' if coverage.passed else '<strong>FAILED</strong>'} &middot; score "
            f"{_h(_fmt_float(coverage.score))} &middot; judge confidence "
            f"{_h(_fmt_float(coverage.confidence))} &middot; plan revision "
            f"{_h(coverage.evaluated_revision)}"
            + (f"<br>{_h(coverage.rationale)}" if coverage.rationale else "")
            + "</div>"
        )
        if coverage.checks:
            parts.append(
                _html_table(
                    ["Check", "Requirement", "Satisfied", "Evidence"],
                    [
                        [_h(c.id), _h(c.requirement), _h(_yes_no(c.satisfied)), _h(c.evidence)]
                        for c in coverage.checks
                    ],
                )
            )
    else:
        parts.append(_empty("No coverage evaluation was recorded for this run."))
    if report.coverage_gaps:
        parts.append(
            '<ul class="plain">'
            + "".join(f"<li>{_h(gap)}</li>" for gap in report.coverage_gaps)
            + "</ul>"
        )

    # ---- healer ----------------------------------------------------------
    parts.append("<h2>Healer actions</h2>")
    if report.healer_actions:
        parts.append(
            f'<p class="note">A fix is auto-applied only for a SCRIPT_ISSUE at or above '
            f"{CONFIDENCE_AUTO_APPLY_THRESHOLD:.2f} confidence. Everything else is queued for a "
            "human.</p>"
        )
        parts.append(
            _html_table(
                ["Flow", "Classification", "Conf.", "Auto-applied", "Action", "Re-run", "Rationale"],
                [
                    [
                        _h(h.flow_name or h.flow_id),
                        _h(getattr(h.classification, "value", h.classification)),
                        f'<span class="num">{_h(_fmt_float(h.confidence))}</span>',
                        _h(_yes_no(h.auto_applied)),
                        _h(h.action),
                        _h(h.rerun_status or "-"),
                        _h(h.rationale),
                    ]
                    for h in report.healer_actions
                ],
            )
        )
    else:
        parts.append(_empty("No test failure required healing analysis."))

    # ---- review queue ----------------------------------------------------
    parts.append("<h2>Needs human review</h2>")
    if report.needs_human_review:
        parts.append(
            _html_table(
                ["Flow", "Classification", "Conf.", "Why it was not auto-applied", "Evidence"],
                [
                    [
                        _h(h.flow_name or h.flow_id),
                        _h(getattr(h.classification, "value", h.classification)),
                        f'<span class="num">{_h(_fmt_float(h.confidence))}</span>',
                        _h(h.rationale or h.action),
                        _h(", ".join(sanitize_url(str(r)) for r in h.evidence_refs) or "-"),
                    ]
                    for h in report.needs_human_review
                ],
            )
        )
    else:
        parts.append(_empty("Nothing was queued for human review."))

    # ---- visual ----------------------------------------------------------
    parts.append("<h2>Visual regression findings</h2>")
    if report.visual_findings:
        parts.append(
            _html_table(
                ["Flow", "Viewport", "Changed", "Threshold", "Regression", "Diff image", "Note"],
                [
                    [
                        _h(v.flow_name or v.flow_id),
                        _h(v.viewport),
                        f'<span class="num">{_h(_fmt_pct(v.changed_ratio))}</span>',
                        f'<span class="num">{_h(_fmt_pct(v.threshold))}</span>',
                        _h("new baseline" if v.is_new_baseline else _yes_no(v.is_regression)),
                        f"<code>{_h(v.diff_path or '-')}</code>",
                        _h(v.note),
                    ]
                    for v in report.visual_findings
                ],
            )
        )
    else:
        parts.append(_empty("No visual comparison was performed or none produced a finding."))

    # ---- bugs ------------------------------------------------------------
    parts.append("<h2>Packaged bugs</h2>")
    if report.packaged_bugs:
        parts.append(
            _html_table(
                ["Bug", "Risk", "Severity", "Flow", "Title", "Repro script", "Evidence"],
                [
                    [
                        f"<code>{_h(b.bug_id)}</code>",
                        _pill(b.risk),
                        _h(b.severity),
                        _h(b.flow_name or b.flow_id),
                        _h(b.title),
                        f"<code>{_h(b.repro_script_path or '-')}</code>",
                        f"<code>{_h(b.screenshot_path or b.ticket_path or b.directory or '-')}</code>",
                    ]
                    for b in report.packaged_bugs
                ],
            )
        )
    else:
        parts.append(_empty("No defect was packaged in this run."))

    # ---- PRD gaps (only when present) ------------------------------------
    if report.prd_gaps:
        parts.append("<h2>PRD gap analysis</h2>")
        parts.append(
            _html_table(
                ["Requirement", "Covered", "Best matching flow", "Similarity"],
                [
                    [
                        _h(g.requirement),
                        _h(_yes_no(g.covered)),
                        _h(g.best_match_flow or "-"),
                        f'<span class="num">{_h(_fmt_float(g.similarity))}</span>',
                    ]
                    for g in report.prd_gaps
                ],
            )
        )

    # ---- regression radar (only when present) ----------------------------
    if report.regression_radar:
        parts.append("<h2>Regression radar</h2>")
        parts.append(
            _html_table(
                ["Signal", "Value"],
                [
                    [_h(key), _h(_stringify(report.regression_radar[key]))]
                    for key in sorted(report.regression_radar)
                ],
            )
        )

    # ---- decision log ----------------------------------------------------
    parts.append("<h2>Decision log excerpt</h2>")
    if report.decision_log_excerpt:
        parts.append(
            '<p class="note">Judgment calls, re-plans, escalations and errors are kept in '
            "full; routine progress chatter is dropped.</p>"
        )
        parts.append(
            _html_table(
                ["Time", "Stage", "Event", "Summary", "Conf.", "Risk"],
                [
                    [
                        _h(_short_time(e.ts)),
                        _h(e.stage),
                        _h(e.event),
                        _h(e.summary),
                        f'<span class="num">'
                        f"{_h(_fmt_float(e.confidence)) if e.confidence is not None else '-'}"
                        f"</span>",
                        _pill(e.risk) if e.risk else "-",
                    ]
                    for e in report.decision_log_excerpt
                ],
            )
        )
    else:
        parts.append(_empty("No decision events were recorded."))

    # ---- limitations -----------------------------------------------------
    parts.append("<h2>Limitations</h2>")
    if report.limitations:
        parts.append(
            '<ul class="plain">'
            + "".join(f"<li>{_h(item)}</li>" for item in report.limitations)
            + "</ul>"
        )
    else:
        parts.append(_empty("No limitations were recorded, which is itself suspicious."))

    # ---- errors ----------------------------------------------------------
    parts.append("<h2>Errors</h2>")
    if report.errors:
        parts.append(
            '<ul class="plain">'
            + "".join(f"<li>{_h(item)}</li>" for item in report.errors)
            + "</ul>"
        )
    else:
        parts.append(_empty("No stage reported an error."))

    parts.append("</main>")
    return "\n".join(parts)


# ==========================================================================
# Persistence
# ==========================================================================
def write_report(run_directory: Path, report: FinalReport) -> dict[str, str]:
    """Write ``report.json``, ``report.md`` and ``report.html`` into the run directory.

    Returns ``{"json": ..., "markdown": ..., "html": ...}``. All three keys are
    always present; a value is the empty string when that particular write
    failed, which is logged as an error. Partial success is reported honestly
    rather than raising and losing the two artifacts that did land.

    ``report.artifacts`` is updated in place with the same mapping before the
    JSON is serialised, so the machine-readable artifact names its own siblings.
    Everything written passes through :func:`security.redact_secrets` (JSON) or
    the redacting renderers plus a final :func:`security.redact_text` sweep
    (Markdown, HTML). ``encoding="utf-8"`` everywhere, so a non-ASCII page title
    does not explode on a Windows console codepage.
    """
    directory = Path(run_directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("could not create run directory %s: %s", directory, exc)
        return {"json": "", "markdown": "", "html": ""}

    targets = {
        "json": directory / "report.json",
        "markdown": directory / "report.md",
        "html": directory / "report.html",
    }
    try:
        report.artifacts.update({key: str(path) for key, path in targets.items()})
    except Exception:  # pragma: no cover - defensive
        log.debug("could not record artifact paths on the report", exc_info=True)

    written: dict[str, str] = {"json": "", "markdown": "", "html": ""}

    # JSON first: it is the machine artifact the API and the UI read back.
    try:
        payload = redact_secrets(report.model_dump(mode="json"))
        targets["json"].write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        written["json"] = str(targets["json"])
    except (OSError, TypeError, ValueError) as exc:
        log.error("could not write report.json: %s", exc)

    for key, renderer in (("markdown", render_markdown), ("html", render_html)):
        try:
            body = redact_text(renderer(report))
            targets[key].write_text(body, encoding="utf-8")
            written[key] = str(targets[key])
        except (OSError, TypeError, ValueError) as exc:
            log.error("could not write report.%s: %s", "md" if key == "markdown" else key, exc)
        except Exception:  # pragma: no cover - a renderer bug must not lose the JSON
            log.error("unexpected failure rendering %s", key, exc_info=True)

    return written
