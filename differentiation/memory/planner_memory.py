"""Planner memory: what has already been covered, and how deep the Planner has gone.

This is the memory namespace that delivers the headline behaviour: a second
run against the *same URL* fetches the flows already proved to pass and spends
its budget on the next unsatisfied depth level instead of re-planning the same
happy paths from zero.

What is deliberately not stored - no DOM, no screenshots, no raw error text,
no step values, no full URLs, no credentials, no generated source. Flows are
keyed by :func:`differentiation.memory.keys.flow_key`, never by the per-run
``flow_id`` or the model-authored ``name`` - both drift across runs. Only a
redacted ``name_hint`` is kept, purely so a human reading the memory file or
the report can tell what a hash refers to.

Depth ladder, and a stated assumption
--------------------------------------
The depth ladder (L1 smoke .. L6 deep_crawl) has six levels, but
:class:`graph.state.FlowCategory` has only three (``happy_path``,
``edge_case``, ``error_state``) because that enum is shared with the coverage
gate and changing it would ripple through code this feature must not touch.
The mapping is therefore a heuristic, stated here rather than left implicit:

* ``L1 smoke``      - a ``happy_path`` flow with two steps or fewer whose
                       business hints mention navigation, or whose name reads
                       like a smoke check.
* ``L2 happy_path``  - every other ``happy_path`` flow.
* ``L3 edge_case``   - every ``edge_case`` flow.
* ``L4 error_state``  - every ``error_state`` flow.
* ``L5 cross_flow``  - any flow (of any category) whose steps navigate to more
                       than one distinct page path - a multi-page journey.
* ``L6 deep_crawl``  - a flow whose page path was not among the pages known to
                       memory *before* this run's crawl updated it.

A flow's depth level is assigned once, when its :class:`FlowMemory` entry is
first created, and never reclassified on a later run - otherwise a flow could
oscillate between levels and ``next_depth`` would never settle.

A level is satisfied only when every flow at that level exists, is fresh
(within ``MEMORY_TTL_DAYS``) and its last status is ``passed`` or ``healed``.
A level holding a ``failed``, ``error``, ``not_run`` or **stale** flow is
unsatisfied, so a regression is always re-planned before the ladder advances -
"regression before expansion". A stale-but-previously-passing flow still
suppresses duplicate planning (see :func:`compute_directive`); it just stops
counting as *verified* coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from differentiation.memory.keys import flow_key as compute_flow_key
from differentiation.memory.store import (
    DEFAULT_MAX_RUNS,
    DEFAULT_TTL_DAYS,
    MEMORY_VERSION,
    append_run_ref,
    is_stale,
    namespace_path,
    read_json,
    utcnow_iso,
    write_atomic,
)
from graph.state import CarriedFlow, SiteMap, TestFlow, TestPlan, TestResult
from logging_setup import get_logger
from security import redact_text, sanitize_url

if TYPE_CHECKING:
    pass

log = get_logger("aivor.memory.planner")

_PASSING_STATUSES = frozenset({"passed", "healed"})

DepthLevel = str
"""``"L1"`` .. ``"L6"``, kept as a plain string rather than a new enum so it
can be dropped straight into JSON without a converter."""

DEPTH_LEVELS: list[tuple[str, str]] = [
    ("L1", "smoke"),
    ("L2", "happy_path"),
    ("L3", "edge_case"),
    ("L4", "error_state"),
    ("L5", "cross_flow"),
    ("L6", "deep_crawl"),
]
_NAME_BY_CODE: dict[str, str] = dict(DEPTH_LEVELS)
_CODE_BY_NAME: dict[str, str] = {name: code for code, name in DEPTH_LEVELS}

MAX_PROMPT_ENTRIES: int = 40
"""Cap on how many covered-flow lines the planner prompt lists, so memory can
never push a prompt past ``llm_max_tokens``."""

L6_MAX_DEPTH: int = 5
L6_MAX_PAGES: int = 40


def _url_path(url: str) -> str:
    try:
        parts = urlsplit(url or "")
        return (parts.path or "/").lower()
    except ValueError:
        return (url or "").lower()


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
class FlowMemory(BaseModel):
    """One flow's cumulative cross-run record."""

    flow_key: str
    name_hint: str = ""
    category: str = "happy_path"
    depth_level: str = "smoke"
    risk: str = "medium"
    last_status: str = "not_run"
    last_run_id: str = ""
    first_seen_at: str = ""
    last_seen_at: str = ""
    times_run: int = 0
    times_passed: int = 0
    consecutive_failures: int = 0
    steps_count: int = 0
    match_kind: str = "fingerprint"
    """``"fingerprint"`` (the normal case) or ``"name"`` when a fingerprint
    miss fell back to a normalised-name lookup - see :func:`_match_existing`."""


class PlannerMemory(BaseModel):
    """The planner namespace: ``reports/baselines/_memory/<host>/planner.json``."""

    version: int = MEMORY_VERSION
    target: str = ""
    updated_at: str = ""
    depth_satisfied: str = ""
    known_pages: list[str] = Field(default_factory=list)
    flows: dict[str, FlowMemory] = Field(default_factory=dict)
    runs: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Load / record
# --------------------------------------------------------------------------
def load_planner_memory(target_url: str) -> PlannerMemory:
    """Load the planner namespace, or an empty one. Never raises."""
    raw = read_json(namespace_path(target_url, "planner"))
    if not raw:
        return PlannerMemory(target=sanitize_url(target_url or ""))
    try:
        return PlannerMemory.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a corrupt file is "no memory"
        log.debug("planner.json for %s failed validation (%s); starting fresh", target_url, exc)
        return PlannerMemory(target=sanitize_url(target_url or ""))


def _match_existing(memory: PlannerMemory, key: str, name_hint: str) -> FlowMemory | None:
    """Prefer the fingerprint match; fall back to a normalised-name match so a
    small step edit does not orphan an entry."""
    existing = memory.flows.get(key)
    if existing is not None:
        return existing
    normalised = redact_text(name_hint or "").strip().lower()
    if not normalised:
        return None
    for candidate in memory.flows.values():
        if candidate.name_hint.strip().lower() == normalised:
            return candidate
    return None


def _depth_level_for(flow: TestFlow, *, known_pages_before: frozenset[str], next_level_hint: str) -> str:
    """Heuristic depth-level classification - see the module docstring."""
    category = str(getattr(flow.category, "value", flow.category))
    path = _url_path(flow.url)
    goto_targets = {
        _url_path(step.target) for step in flow.steps if step.action == "goto" and step.target
    }
    if len(goto_targets) > 1:
        return "cross_flow"
    if known_pages_before and path not in known_pages_before and next_level_hint == "deep_crawl":
        return "deep_crawl"
    if category == "edge_case":
        return "edge_case"
    if category == "error_state":
        return "error_state"
    name = flow.name.lower()
    hints = " ".join(flow.business_hints).lower()
    if len(flow.steps) <= 2 and ("navigation" in hints or "smoke" in name or "load" in name):
        return "smoke"
    return "happy_path"


def record_planner_memory(
    *,
    target_url: str,
    run_id: str,
    plan: TestPlan | None,
    results: Sequence[TestResult],
    site_map: SiteMap | None,
    risk_lookup: dict[str, str] | None = None,
    enabled: bool = True,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> dict[str, Any]:
    """Update the planner namespace with this run's outcomes. Never raises."""
    if not enabled:
        return {"enabled": False, "persisted": False}
    try:
        memory = load_planner_memory(target_url)
        known_pages_before = frozenset(memory.known_pages)
        status_by_flow_id = {r.flow_id: str(getattr(r.status, "value", r.status)) for r in results or ()}
        now = utcnow_iso()
        next_level_hint = _NAME_BY_CODE[next_depth(memory)]

        for flow in (plan.flows if plan else []):
            name_hint = redact_text(flow.name)[:120]
            key = compute_flow_key(flow)
            existing = _match_existing(memory, key, name_hint)
            match_kind = "fingerprint"
            if existing is not None and existing.flow_key != key:
                match_kind = "name"
                # Re-key under the new fingerprint; the name match was a
                # fallback to avoid orphaning history after a small edit.
                memory.flows.pop(existing.flow_key, None)

            status = status_by_flow_id.get(flow.id, "not_run")
            if existing is None:
                depth_level = _depth_level_for(
                    flow, known_pages_before=known_pages_before, next_level_hint=next_level_hint
                )
                existing = FlowMemory(
                    flow_key=key,
                    category=str(getattr(flow.category, "value", flow.category)),
                    depth_level=depth_level,
                    first_seen_at=now,
                )

            existing.flow_key = key
            existing.name_hint = name_hint
            existing.risk = (risk_lookup or {}).get(flow.id, existing.risk)
            existing.last_status = status
            existing.last_run_id = run_id
            existing.last_seen_at = now
            existing.times_run += 1
            existing.steps_count = len(flow.steps)
            existing.match_kind = match_kind
            if status in _PASSING_STATUSES:
                existing.times_passed += 1
                existing.consecutive_failures = 0
            elif status in ("failed", "error"):
                existing.consecutive_failures += 1
            memory.flows[key] = existing

        if site_map is not None:
            paths = sorted({_url_path(page.url) for page in site_map.pages})
            memory.known_pages = paths[:200]

        satisfied = [name for code, name in DEPTH_LEVELS if level_satisfied(memory, name)]
        memory.depth_satisfied = ",".join(satisfied)
        memory.target = sanitize_url(target_url or "")
        memory.updated_at = now
        memory.runs = append_run_ref(memory.runs, run_id, max_runs)

        persisted = write_atomic(namespace_path(target_url, "planner"), memory.model_dump(mode="json"))
        return {"enabled": True, "persisted": persisted, "depth_satisfied": memory.depth_satisfied}
    except Exception as exc:  # noqa: BLE001 - memory must never fail a run
        log.warning("planner memory record failed for %s: %s", target_url, exc)
        return {"enabled": True, "persisted": False}


# --------------------------------------------------------------------------
# Depth ladder
# --------------------------------------------------------------------------
def _flows_at_level(memory: PlannerMemory, level_name: str) -> list[FlowMemory]:
    return [f for f in memory.flows.values() if f.depth_level == level_name]


def level_satisfied(memory: PlannerMemory, level_name: str, *, ttl_days: int = DEFAULT_TTL_DAYS) -> bool:
    """True only when every flow at this level exists, is fresh, and last
    passed or was healed. Empty (no flow yet planned at this level) is
    unsatisfied - there is nothing to expand from."""
    flows = _flows_at_level(memory, level_name)
    if not flows:
        return False
    for flow in flows:
        if flow.last_status not in _PASSING_STATUSES:
            return False
        if is_stale(flow.last_seen_at, ttl_days):
            return False
    return True


def next_depth(memory: PlannerMemory, *, ttl_days: int = DEFAULT_TTL_DAYS) -> str:
    """The lowest unsatisfied depth level code, e.g. ``"L3"``."""
    for code, name in DEPTH_LEVELS:
        if not level_satisfied(memory, name, ttl_days=ttl_days):
            return code
    return DEPTH_LEVELS[-1][0]


# --------------------------------------------------------------------------
# Directive: what generate_plan and the coverage gate consume
# --------------------------------------------------------------------------
@dataclass
class PlannerDirective:
    """What memory tells this run's Planner and coverage gate.

    ``covered_keys`` is used to drop model-proposed duplicates in
    :func:`agents.planner.generate_plan` and is intentionally broader than
    "fresh and verified" - a stale-but-previously-passing flow still
    suppresses duplicate planning, it just does not count toward
    ``target_level`` (see the module docstring).
    """

    enabled: bool = False
    first_run: bool = True
    target_level: str = "L1"
    target_level_name: str = "smoke"
    satisfied_labels: list[str] = field(default_factory=list)
    covered_keys: frozenset[str] = field(default_factory=frozenset)
    covered_display: list[dict[str, str]] = field(default_factory=list)
    needs_reverification: list[dict[str, str]] = field(default_factory=list)
    unexercised_pages: list[str] = field(default_factory=list)
    carried: list[CarriedFlow] = field(default_factory=list)
    crawl_max_depth_override: int | None = None
    crawl_max_pages_override: int | None = None

    def summary_line(self, run_hint: str = "") -> str:
        if not self.enabled or self.first_run:
            return "Memory: first run for this target"
        carried_n = len(self.carried)
        satisfied = ", ".join(self.satisfied_labels) or "none"
        origin = run_hint or (self.carried[0].origin_run_id if self.carried else "")
        origin_txt = f" from {origin}" if origin else ""
        return (
            f"Memory: {carried_n} flow(s) carried forward{origin_txt}; "
            f"{satisfied} satisfied; planning {self.target_level} {self.target_level_name}"
        )


def compute_directive(
    memory: PlannerMemory,
    *,
    ttl_days: int = DEFAULT_TTL_DAYS,
    enabled: bool = True,
) -> PlannerDirective:
    """Turn stored memory into the directive this run's Planner acts on."""
    if not enabled or not memory.flows:
        return PlannerDirective(enabled=enabled, first_run=True)

    target_code = next_depth(memory, ttl_days=ttl_days)
    satisfied = [name for code, name in DEPTH_LEVELS if level_satisfied(memory, name, ttl_days=ttl_days)]

    covered_keys: set[str] = set()
    covered_display: list[dict[str, str]] = []
    needs_reverification: list[dict[str, str]] = []
    carried: list[CarriedFlow] = []

    # Longest-uncovered-area first is approximated by category diversity: list
    # non-happy_path categories first so truncation keeps the harder-won ones.
    ordered_items = sorted(
        memory.flows.items(),
        key=lambda item: (item[1].category == "happy_path", item[0]),
    )

    for key, flow in ordered_items:
        if flow.last_status in _PASSING_STATUSES:
            covered_keys.add(key)
            if len(covered_display) < MAX_PROMPT_ENTRIES:
                covered_display.append({"category": flow.category, "name_hint": flow.name_hint})
            carried.append(
                CarriedFlow(
                    flow_key=key,
                    name_hint=flow.name_hint,
                    category=flow.category,
                    origin_run_id=flow.last_run_id,
                    risk=flow.risk,
                )
            )
        else:
            needs_reverification.append({"category": flow.category, "name_hint": flow.name_hint})

    directive = PlannerDirective(
        enabled=True,
        first_run=False,
        target_level=target_code,
        target_level_name=_NAME_BY_CODE[target_code],
        satisfied_labels=satisfied,
        covered_keys=frozenset(covered_keys),
        covered_display=covered_display,
        needs_reverification=needs_reverification,
        unexercised_pages=list(memory.known_pages),
        carried=carried,
    )

    if target_code == "L6":
        directive.crawl_max_depth_override = L6_MAX_DEPTH
        directive.crawl_max_pages_override = L6_MAX_PAGES

    return directive


def flows_should_drop(directive: PlannerDirective, flows: Sequence[TestFlow]) -> tuple[list[TestFlow], int]:
    """Split ``flows`` into (kept, dropped_count) against the covered set."""
    if not directive.enabled or not directive.covered_keys:
        return list(flows), 0
    kept: list[TestFlow] = []
    dropped = 0
    for flow in flows:
        if compute_flow_key(flow) in directive.covered_keys:
            dropped += 1
        else:
            kept.append(flow)
    return kept, dropped
