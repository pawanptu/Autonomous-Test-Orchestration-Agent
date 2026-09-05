"""Typed domain models and the LangGraph orchestration state.

Everything that crosses a node boundary is declared here. Two rules govern
this module:

* **No raw credentials.** The state carries ``credentials_present: bool`` and
  ``login_ok: bool | None``. The actual values live in
  :data:`security.SECRET_BOX`, keyed by ``run_id``, and are wiped when the run
  ends. Anything in this file may be serialised to disk.

* **Everything is typed.** Nodes exchange pydantic models, not loose dicts, so
  a malformed LLM response is rejected at the parse boundary rather than three
  stages later.

The LangGraph state itself is a ``TypedDict``. ``decision_log`` uses an
additive reducer so that every node can append events without having to
re-emit the whole history.
"""

from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
Stage = Literal[
    "orchestrator",
    "planner",
    "coverage_gate",
    "risk_ranking",
    "generator",
    "runner",
    "healer",
    "visual_diff",
    "bug_packager",
    "report",
]

EventKind = Literal[
    "start",
    "progress",
    "decision",
    "replan",
    "escalate",
    "complete",
    "error",
]

RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class FlowCategory(str, Enum):
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    ERROR_STATE = "error_state"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def sort_key(self) -> int:
        """0 for HIGH so that ``sorted`` puts the scariest flow first."""
        return {"high": 0, "medium": 1, "low": 2}[self.value]


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    HEALED = "healed"
    """Failed on the first attempt, passed after an auto-applied script fix."""


class DefectClass(str, Enum):
    SCRIPT_ISSUE = "SCRIPT_ISSUE"
    GENUINE_DEFECT = "GENUINE_DEFECT"
    ENVIRONMENT = "ENVIRONMENT"
    """Neither the app nor the script: captcha, network, rate limit, auth wall."""
    UNKNOWN = "UNKNOWN"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Live-visibility event
# --------------------------------------------------------------------------
class DecisionEvent(BaseModel):
    """One line of the agent's visible reasoning.

    Every node emits at least ``start`` and ``complete``; nodes that make a
    routing choice also emit ``decision`` with a confidence and a rationale.
    The Streamlit UI renders this list verbatim, and it is appended to
    ``reports/runs/<run_id>/events.jsonl`` the moment it is created.
    """

    model_config = ConfigDict(use_enum_values=True)

    ts: str = Field(default_factory=utcnow_iso)
    stage: Stage
    event: EventKind
    summary: str
    detail: str = ""
    confidence: float | None = None
    risk: Literal["high", "medium", "low"] | None = None
    flow_id: str | None = None
    auto_applied: bool | None = None
    needs_human_review: bool = False

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(1.0, float(v)))


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
class TestStep(BaseModel):
    """A single action inside a flow.

    ``action`` is a small verb vocabulary the Generator knows how to translate
    into Playwright calls. ``target`` is a natural-language description of the
    element ("the email input"), never a raw selector: real selectors are
    resolved later against the live DOM by :mod:`browser.selectors`.
    """

    action: Literal[
        "goto",
        "click",
        "fill",
        "select",
        "check",
        "press",
        "wait_for",
        "assert_text",
        "assert_visible",
        "assert_url",
        "assert_not_visible",
        "screenshot",
    ] = "click"
    target: str = ""
    value: str | None = None
    description: str = ""

    @field_validator("value")
    @classmethod
    def _no_none_string(cls, v: str | None) -> str | None:
        return None if v in ("null", "None") else v


class TestFlow(BaseModel):
    """One end-to-end user journey the agent decided is worth testing."""

    id: str
    name: str
    category: FlowCategory = FlowCategory.HAPPY_PATH
    steps: list[TestStep] = Field(default_factory=list)
    expected_outcome: str = ""
    url: str = ""
    business_hints: list[str] = Field(default_factory=list)
    requires_auth: bool = False

    model_config = ConfigDict(use_enum_values=False)


class DiscoveredPage(BaseModel):
    """A page the crawler reached, plus the interactive surface it exposes."""

    url: str
    title: str = ""
    depth: int = 0
    status: int | None = None
    is_protected: bool = False
    forms: list[dict[str, Any]] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    buttons: list[str] = Field(default_factory=list)
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    text_excerpt: str = ""


class SiteMap(BaseModel):
    """Everything the crawler learned about the target."""

    target_url: str
    pages: list[DiscoveredPage] = Field(default_factory=list)
    login_detected: bool = False
    login_url: str | None = None
    ecommerce_signals: list[str] = Field(default_factory=list)
    auth_blocked: bool = False
    notes: list[str] = Field(default_factory=list)

    def area_names(self) -> list[str]:
        """Coarse functional areas, used by the coverage rubric."""
        seen: list[str] = []
        for page in self.pages:
            label = (page.title or page.url).strip()
            if label and label not in seen:
                seen.append(label)
        return seen


class TestPlan(BaseModel):
    """The Planner's output: the contract the Generator compiles."""

    target_url: str
    summary: str = ""
    flows: list[TestFlow] = Field(default_factory=list)
    discovered_areas: list[str] = Field(default_factory=list)
    auth_flow_present: bool = False
    ecommerce_like: bool = False
    revision: int = 0
    notes: list[str] = Field(default_factory=list)

    def by_category(self, category: FlowCategory) -> list[TestFlow]:
        return [f for f in self.flows if f.category == category]

    def category_counts(self) -> dict[str, int]:
        return {
            c.value: len([f for f in self.flows if f.category == c])
            for c in FlowCategory
        }


# --------------------------------------------------------------------------
# Coverage gate
# --------------------------------------------------------------------------
class RubricCheck(BaseModel):
    """One line of the coverage rubric with the judge's verdict."""

    id: str
    requirement: str
    satisfied: bool
    evidence: str = ""


class CoverageEvaluation(BaseModel):
    """Verdict of the pre-generation coverage gate."""

    passed: bool
    score: float = 0.0
    confidence: float = 0.5
    checks: list[RubricCheck] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    feedback: str = ""
    rationale: str = ""
    evaluated_revision: int = 0

    def failed_requirements(self) -> list[str]:
        return [c.requirement for c in self.checks if not c.satisfied]


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------
class RiskClassification(BaseModel):
    """Risk verdict for one flow, with a rubric citation."""

    flow_id: str
    flow_name: str = ""
    risk: RiskLevel = RiskLevel.MEDIUM
    rationale: str = ""
    rubric_cite: str = ""
    confidence: float = 0.5
    source: Literal["llm", "rubric-fallback", "default"] = "llm"


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
class SelectorCandidate(BaseModel):
    strategy: Literal["role", "label", "placeholder", "text", "testid", "css", "alt", "title"]
    expression: str
    match_count: int = 0
    visible: bool = False
    note: str = ""


class SelectorValidation(BaseModel):
    """Record of resolving one natural-language target against the live DOM."""

    step_index: int
    intent: str
    page_url: str = ""
    candidates: list[SelectorCandidate] = Field(default_factory=list)
    chosen: str | None = None
    chosen_strategy: str | None = None
    valid: bool = False
    note: str = ""


class GeneratedTest(BaseModel):
    """One executable Playwright test file produced by the agent."""

    flow_id: str
    flow_name: str = ""
    file_path: str = ""
    module_name: str = ""
    source: str = ""
    selector_validations: list[SelectorValidation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_by_model: str = ""
    valid: bool = True
    validation_error: str | None = None
    repair_attempts: int = 0


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
class TestResult(BaseModel):
    """Outcome of executing one generated test."""

    flow_id: str
    flow_name: str = ""
    status: TestStatus = TestStatus.ERROR
    duration_s: float = 0.0
    attempts: int = 1
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    dom_snippet: str | None = None
    screenshot_path: str | None = None
    final_url: str | None = None
    console_errors: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utcnow_iso)

    @property
    def failed(self) -> bool:
        return self.status in (TestStatus.FAILED, TestStatus.ERROR)


# --------------------------------------------------------------------------
# Healing
# --------------------------------------------------------------------------
class HealerAction(BaseModel):
    """What the Healer concluded about one failing test, and what it did.

    ``auto_applied`` is only ever True when ``classification`` is
    ``SCRIPT_ISSUE`` *and* ``confidence >= CONFIDENCE_AUTO_APPLY_THRESHOLD``.
    A genuine defect is never "fixed" by weakening an assertion.
    """

    flow_id: str
    flow_name: str = ""
    classification: DefectClass = DefectClass.UNKNOWN
    confidence: float = 0.5
    rationale: str = ""
    action: str = ""
    auto_applied: bool = False
    needs_human_review: bool = False
    patch_summary: str = ""
    rerun_status: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)
    risk: RiskLevel | None = None


# --------------------------------------------------------------------------
# Visual regression
# --------------------------------------------------------------------------
class VisualFinding(BaseModel):
    """Pixel-level comparison against the stored baseline for a flow."""

    flow_id: str
    flow_name: str = ""
    viewport: str = "1280x900"
    baseline_path: str | None = None
    current_path: str | None = None
    diff_path: str | None = None
    changed_ratio: float = 0.0
    threshold: float = 0.02
    is_regression: bool = False
    is_new_baseline: bool = False
    note: str = ""
    risk: RiskLevel | None = None


# --------------------------------------------------------------------------
# Bugs
# --------------------------------------------------------------------------
class PackagedBug(BaseModel):
    """A ticket-ready defect artifact written to disk."""

    bug_id: str
    flow_id: str
    flow_name: str = ""
    title: str
    description: str
    classification: DefectClass = DefectClass.GENUINE_DEFECT
    confidence: float = 0.5
    risk: RiskLevel = RiskLevel.MEDIUM
    severity: Literal["critical", "major", "minor"] = "major"
    steps_to_reproduce: list[str] = Field(default_factory=list)
    expected: str = ""
    actual: str = ""
    labels: list[str] = Field(default_factory=list)
    directory: str = ""
    repro_script_path: str | None = None
    screenshot_path: str | None = None
    ticket_path: str | None = None
    created_at: str = Field(default_factory=utcnow_iso)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
class FlowReportRow(BaseModel):
    """One row of the risk-ordered results table."""

    flow_id: str
    flow_name: str
    category: str
    risk: RiskLevel
    risk_rationale: str = ""
    status: TestStatus
    outcome_label: str = ""
    duration_s: float = 0.0
    healed: bool = False
    needs_human_review: bool = False
    visual_regression: bool = False
    bug_ids: list[str] = Field(default_factory=list)
    error_message: str | None = None


class PRDGapItem(BaseModel):
    requirement: str
    covered: bool
    best_match_flow: str | None = None
    similarity: float = 0.0


class FinalReport(BaseModel):
    """The synthesised deliverable: machine JSON + rendered Markdown/HTML."""

    run_id: str
    target_url: str
    generated_at: str = Field(default_factory=utcnow_iso)
    status: str = "completed"
    executive_summary: str = ""
    business_impact: str = ""
    llm_provider: str = "groq"
    models_used: dict[str, str] = Field(default_factory=dict)

    totals: dict[str, Any] = Field(default_factory=dict)
    flows: list[FlowReportRow] = Field(default_factory=list)
    coverage_evaluation: CoverageEvaluation | None = None
    coverage_gaps: list[str] = Field(default_factory=list)
    replan_count: int = 0
    force_proceeded: bool = False
    login_ok: bool | None = None
    credentials_present: bool = False

    healer_actions: list[HealerAction] = Field(default_factory=list)
    needs_human_review: list[HealerAction] = Field(default_factory=list)
    visual_findings: list[VisualFinding] = Field(default_factory=list)
    packaged_bugs: list[PackagedBug] = Field(default_factory=list)
    prd_gaps: list[PRDGapItem] = Field(default_factory=list)
    regression_radar: dict[str, Any] = Field(default_factory=dict)

    decision_log_excerpt: list[DecisionEvent] = Field(default_factory=list)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# LangGraph state
# --------------------------------------------------------------------------
class OrchestrationState(TypedDict, total=False):
    """The graph's shared state.

    ``total=False`` because LangGraph merges partial dicts returned by nodes.
    ``decision_log`` uses ``operator.add`` so nodes return only their *new*
    events; every other list field is owned by exactly one node and is
    overwritten wholesale.
    """

    # ---- inputs ----------------------------------------------------------
    run_id: str
    target_url: str
    prd_text: str | None
    user_intent: str | None
    credentials_present: bool
    login_ok: bool | None
    login_error: str | None

    # ---- planning --------------------------------------------------------
    site_map: SiteMap | None
    test_plan: TestPlan | None
    coverage_evaluation: CoverageEvaluation | None
    coverage_feedback: str | None
    replan_count: int
    force_proceeded: bool

    # ---- risk ------------------------------------------------------------
    risk_classifications: list[RiskClassification]

    # ---- generation / execution -----------------------------------------
    generated_tests: list[GeneratedTest]
    run_results: list[TestResult]
    heal_pass_count: int

    # ---- differentiation -------------------------------------------------
    healer_actions: list[HealerAction]
    visual_diff_findings: list[VisualFinding]
    packaged_bugs: list[PackagedBug]
    prd_gaps: list[PRDGapItem]
    regression_radar: dict[str, Any]

    # ---- bookkeeping -----------------------------------------------------
    decision_log: Annotated[list[DecisionEvent], operator.add]
    current_stage: str
    status: RunStatus
    error: str | None
    node_errors: list[str]
    escalations: list[str]
    feature_flags: dict[str, bool]
    started_at: str
    finished_at: str | None
    final_report: FinalReport | None


def initial_state(
    *,
    run_id: str,
    target_url: str,
    prd_text: str | None,
    user_intent: str | None,
    credentials_present: bool,
    feature_flags: dict[str, bool],
) -> OrchestrationState:
    """Build the state a run starts from. Contains no secret material."""
    return OrchestrationState(
        run_id=run_id,
        target_url=target_url,
        prd_text=prd_text,
        user_intent=user_intent,
        credentials_present=credentials_present,
        login_ok=None,
        login_error=None,
        site_map=None,
        test_plan=None,
        coverage_evaluation=None,
        coverage_feedback=None,
        replan_count=0,
        force_proceeded=False,
        risk_classifications=[],
        generated_tests=[],
        run_results=[],
        heal_pass_count=0,
        healer_actions=[],
        visual_diff_findings=[],
        packaged_bugs=[],
        prd_gaps=[],
        regression_radar={},
        decision_log=[],
        current_stage="orchestrator",
        status="running",
        error=None,
        node_errors=[],
        escalations=[],
        feature_flags=feature_flags,
        started_at=utcnow_iso(),
        finished_at=None,
        final_report=None,
    )
