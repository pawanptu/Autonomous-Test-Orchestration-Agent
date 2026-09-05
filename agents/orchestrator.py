"""Meta-orchestrator: the agent that coordinates the other agents.

This module owns the three decisions that make the system autonomous rather
than merely automated:

* **Is this plan good enough to spend generation on?** The coverage gate runs
  *before* the Generator, applies the published rubric, and either passes the
  plan through or sends the Planner specific, actionable feedback.

* **Re-plan, proceed, or escalate?** Every routing choice carries a confidence
  and a rationale that lands in the decision log. The re-plan budget is two;
  on exhaustion the pipeline force-proceeds with the best plan it has and says
  so loudly rather than looping or giving up.

* **What does all of this mean?** Final synthesis: totals, risk ordering,
  limitations, and an executive summary that leads with what is broken.

It also owns the run lifecycle - browser, LLM client, secret custody - so that
:func:`execute_run` is the single entry point the API calls and the single
place credentials are wiped.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Sequence

from browser.session import BrowserSession
from config import REPLAN_CAP, Settings, ensure_dirs, get_settings, run_dir
from safe_actions import SafetyPolicy, describe_policy
from differentiation.prd_gap import analyse_prd_gaps, summarise_gaps
from differentiation.regression_radar import record as record_radar
from graph.runtime import RunContext, create_context, drop_context
from graph.state import (
    CoverageEvaluation,
    DefectClass,
    FinalReport,
    RubricCheck,
    RunStatus,
    SiteMap,
    TestPlan,
    TestStatus,
    initial_state,
    utcnow_iso,
)
from llm.client import LLMClient, LLMError, ModelRole
from llm.json_utils import JSONParseError
from llm.prompts import (
    COVERAGE_RUBRIC,
    COVERAGE_SYSTEM,
    ORCHESTRATOR_ROUTE_SYSTEM,
    REPORT_SYNTHESIS_SYSTEM,
    coverage_user,
    orchestrator_route_user,
    report_synthesis_user,
)
from logging_setup import attach_run_log, configure_logging, detach_run_log, get_logger
from reports.generator import assemble_report, write_report
from security import SECRET_BOX, redact_text, sanitize_url

log = get_logger("aivor.orchestrator")


# ==========================================================================
# Coverage gate
# ==========================================================================
def deterministic_coverage_check(plan: TestPlan, site_map: SiteMap | None) -> list[RubricCheck]:
    """Apply the coverage rubric mechanically.

    This runs alongside the model's judgment, not instead of it. Its job is to
    catch the case where a model waves a plan through that plainly lacks an
    error state, which is the most expensive kind of gate failure because it is
    invisible until the report.
    """
    flows = plan.flows
    categories = {getattr(f.category, "value", f.category) for f in flows}
    blob = " ".join(
        f"{f.name} {f.expected_outcome} {' '.join(h for h in f.business_hints)}".lower()
        for f in flows
    )
    login_exists = bool(site_map and site_map.login_detected) or plan.auth_flow_present
    commerce = bool(site_map and site_map.ecommerce_signals) or plan.ecommerce_like

    def check(rid: str, satisfied: bool, evidence: str) -> RubricCheck:
        requirement = next((text for cid, text in COVERAGE_RUBRIC if cid == rid), rid)
        return RubricCheck(id=rid, requirement=requirement, satisfied=satisfied, evidence=evidence)

    areas = list(plan.discovered_areas or (site_map.area_names()[:5] if site_map else []))
    happy = [f for f in flows if getattr(f.category, "value", f.category) == "happy_path"]

    checks = [
        check(
            "C1",
            bool(happy) and (not areas or len(happy) >= min(2, len(areas))),
            f"{len(happy)} happy-path flow(s) across {len(areas) or 'unknown'} discovered area(s)",
        ),
        check(
            "C2",
            "edge_case" in categories,
            "an edge_case flow is present" if "edge_case" in categories else "no edge_case flow",
        ),
        check(
            "C3",
            "error_state" in categories,
            "an error_state flow is present" if "error_state" in categories else "no error_state flow",
        ),
        check(
            "C4",
            (not login_exists)
            or (
                any(k in blob for k in ("log in", "login", "sign in", "authenticat"))
                and any(k in blob for k in ("invalid", "incorrect", "wrong", "reject", "fail"))
            ),
            "not applicable: no login UI discovered"
            if not login_exists
            else (
                "auth happy path and an invalid-credential flow both appear in the plan"
                if (
                    any(k in blob for k in ("log in", "login", "sign in", "authenticat"))
                    and any(k in blob for k in ("invalid", "incorrect", "wrong", "reject", "fail"))
                )
                else "a login UI exists but the plan lacks an authentication and/or an "
                "invalid-credential flow"
            ),
        ),
        check(
            "C5",
            (not commerce)
            or any(k in blob for k in ("cart", "basket", "checkout", "order", "payment")),
            "not applicable: no cart/checkout surface discovered"
            if not commerce
            else (
                "commerce flows appear in the plan"
                if any(k in blob for k in ("cart", "basket", "checkout", "order", "payment"))
                else "a cart/checkout/payment surface was discovered but no flow covers it"
            ),
        ),
        check(
            "C6",
            bool(flows) and all(f.expected_outcome.strip() for f in flows),
            f"{sum(1 for f in flows if f.expected_outcome.strip())}/{len(flows)} flows state an "
            "expected outcome",
        ),
    ]
    return checks


async def evaluate_coverage(
    ctx: RunContext,
    llm: LLMClient,
    *,
    plan: TestPlan,
    site_map: SiteMap | None,
) -> CoverageEvaluation:
    """Judge the plan against the rubric before any code is generated.

    The model's verdict is authoritative on the qualitative checks; the
    deterministic checker can only make the gate *stricter*, never more
    permissive. A model outage degrades to the deterministic result rather than
    letting an unreviewed plan through.
    """
    mechanical = deterministic_coverage_check(plan, site_map)
    mechanical_by_id = {c.id: c for c in mechanical}

    evaluation: CoverageEvaluation | None = None
    try:
        payload = await llm.complete_json(
            ModelRole.REASONING,
            [
                {"role": "system", "content": COVERAGE_SYSTEM},
                {
                    "role": "user",
                    "content": coverage_user(
                        plan=plan.model_dump(mode="json"),
                        login_detected=bool(site_map and site_map.login_detected),
                        ecommerce_signals=(site_map.ecommerce_signals if site_map else []),
                        discovered_areas=plan.discovered_areas,
                    ),
                },
            ],
            task=f"coverage_gate:rev{plan.revision}",
            max_tokens=2500,
        )
        if isinstance(payload, dict):
            checks = [
                RubricCheck(
                    id=str(c.get("id") or "?")[:8],
                    requirement=str(c.get("requirement") or "")[:300],
                    satisfied=bool(c.get("satisfied")),
                    evidence=str(c.get("evidence") or "")[:300],
                )
                for c in (payload.get("checks") or [])
                if isinstance(c, dict)
            ]
            evaluation = CoverageEvaluation(
                passed=bool(payload.get("passed")),
                score=_clamp(payload.get("score"), 0.0),
                confidence=_clamp(payload.get("confidence"), 0.5),
                checks=checks or mechanical,
                missing=[str(m)[:300] for m in (payload.get("missing") or [])][:10],
                feedback=str(payload.get("feedback") or "")[:1500],
                rationale=str(payload.get("rationale") or "")[:800],
                evaluated_revision=plan.revision,
            )
    except (JSONParseError, LLMError, Exception) as exc:  # noqa: B014
        log.warning("coverage gate LLM call failed: %s", exc)
        ctx.emit(
            "coverage_gate",
            "error",
            "Coverage judge unavailable - falling back to the deterministic rubric",
            detail=f"{type(exc).__name__}: {exc}",
            confidence=0.4,
        )

    if evaluation is None:
        failed = [c for c in mechanical if not c.satisfied]
        evaluation = CoverageEvaluation(
            passed=not failed,
            score=round(sum(1 for c in mechanical if c.satisfied) / len(mechanical), 2),
            confidence=0.5,
            checks=mechanical,
            missing=[c.requirement for c in failed],
            feedback=_compose_feedback(failed, plan, site_map),
            rationale="Evaluated by the deterministic rubric only; the judge model was unavailable.",
            evaluated_revision=plan.revision,
        )
        return evaluation

    # The deterministic checker can only tighten the gate.
    overrides: list[str] = []
    for check in evaluation.checks:
        mech = mechanical_by_id.get(check.id)
        if mech is not None and check.satisfied and not mech.satisfied:
            check.satisfied = False
            check.evidence = f"{check.evidence} | overridden: {mech.evidence}".strip(" |")
            overrides.append(check.id)
    for mech in mechanical:
        if mech.id not in {c.id for c in evaluation.checks}:
            evaluation.checks.append(mech)
            if not mech.satisfied:
                overrides.append(mech.id)

    if overrides:
        failed = [c for c in evaluation.checks if not c.satisfied]
        evaluation.passed = False
        evaluation.missing = sorted(set(evaluation.missing + [c.requirement for c in failed]))
        if not evaluation.feedback.strip():
            evaluation.feedback = _compose_feedback(failed, plan, site_map)
        evaluation.rationale = (
            f"{evaluation.rationale} Deterministic rubric overrode the judge on "
            f"{', '.join(sorted(set(overrides)))}."
        ).strip()
        ctx.emit(
            "coverage_gate",
            "decision",
            f"Deterministic rubric overrode the judge on {', '.join(sorted(set(overrides)))}",
            detail="the mechanical check found a rubric line the judge marked satisfied",
            confidence=0.9,
        )

    evaluation.score = round(
        sum(1 for c in evaluation.checks if c.satisfied) / max(len(evaluation.checks), 1), 2
    )
    return evaluation


def _compose_feedback(
    failed: Sequence[RubricCheck], plan: TestPlan, site_map: SiteMap | None
) -> str:
    """Turn failed rubric lines into instructions the Planner can act on."""
    if not failed:
        return ""
    lines = ["The plan was rejected by the coverage gate. Add the following:"]
    login_url = (site_map.login_url if site_map else None) or plan.target_url
    for check in failed:
        if check.id == "C1":
            lines.append(
                "- A happy-path flow for each primary area listed in discovered_areas that "
                "does not already have one."
            )
        elif check.id == "C2":
            lines.append(
                "- At least one edge_case flow: submit a required field empty, or use a "
                "boundary value (0, a very long string, a special character), and assert "
                "the specific validation message or the unchanged state."
            )
        elif check.id == "C3":
            lines.append(
                f"- At least one error_state flow: request a URL that does not exist under "
                f"{sanitize_url(plan.target_url)} and assert that a not-found page renders, "
                "or submit a form with invalid data and assert the inline error text."
            )
        elif check.id == "C4":
            lines.append(
                f"- Both an authentication happy path AND an invalid-credential error flow "
                f"against the login form at {sanitize_url(login_url)}. The negative flow must "
                "use obviously fake values and assert the visible error message."
            )
        elif check.id == "C5":
            lines.append(
                "- Flows covering the cart/checkout surface that the crawl discovered: add an "
                "item, verify the cart contents persist, and start checkout."
            )
        elif check.id == "C6":
            lines.append(
                "- A concrete expected_outcome for every flow. Describe the observable result "
                "(text that appears, URL that changes, element that disappears), not the click."
            )
        else:
            lines.append(f"- {check.requirement}")
    return "\n".join(lines)


# ==========================================================================
# Routing
# ==========================================================================
async def route_after_coverage(
    ctx: RunContext,
    llm: LLMClient,
    *,
    evaluation: CoverageEvaluation,
    replan_count: int,
) -> dict[str, Any]:
    """Decide replan / proceed / escalate, with a confidence and a rationale.

    The budget is enforced here in code, not left to the model: a model that
    wants a third re-plan gets ``proceed`` with ``force_proceeded`` recorded.
    """
    if evaluation.passed:
        return {
            "action": "proceed",
            "confidence": max(0.6, evaluation.confidence),
            "rationale": (
                f"Coverage gate passed with score {evaluation.score:.2f}; "
                "every applicable rubric line is satisfied."
            ),
            "feedback": "",
        }

    if replan_count >= REPLAN_CAP:
        return {
            "action": "proceed",
            "confidence": 0.45,
            "rationale": (
                f"The re-plan budget of {REPLAN_CAP} is exhausted and the gate still fails on "
                f"{len(evaluation.failed_requirements())} rubric line(s). Force-proceeding with "
                "the best available plan; the remaining gaps are recorded in the report as "
                "untested-flow risk."
            ),
            "feedback": "",
            "forced": True,
        }

    # Ask the meta-agent whether these specific gaps are worth a re-plan cycle.
    decision = {
        "action": "replan",
        "confidence": 0.7,
        "rationale": (
            f"Gate failed on {', '.join(evaluation.failed_requirements()[:3]) or 'coverage'}; "
            f"a re-plan is affordable ({replan_count + 1}/{REPLAN_CAP})."
        ),
        "feedback": evaluation.feedback,
    }

    # Deterministic short-circuit. The model's only remaining freedom here is
    # "replan" versus "escalate" - a failed gate can never be waved through, as
    # the coercion below enforces. On any re-plan except the last affordable
    # one, replanning is unambiguously correct: budget remains, and escalating
    # early would surrender coverage the agent is still able to win. Spending a
    # 70B call to confirm that is pure cost, so the call is skipped and the
    # deterministic route stands.
    if replan_count < REPLAN_CAP - 1:
        decision["rationale"] += " Routed deterministically; no model call was needed."
        ctx.emit(
            "coverage_gate",
            "decision",
            f"Re-plan {replan_count + 1}/{REPLAN_CAP} routed deterministically",
            detail="the gate failed with budget remaining, which admits only one action",
            confidence=decision["confidence"],
        )
        if not str(decision["feedback"]).strip():
            decision["feedback"] = evaluation.feedback or "Add edge-case and error-state coverage."
        return decision

    try:
        payload = await llm.complete_json(
            ModelRole.REASONING,
            [
                {"role": "system", "content": ORCHESTRATOR_ROUTE_SYSTEM},
                {
                    "role": "user",
                    "content": orchestrator_route_user(
                        stage="coverage_gate",
                        situation={
                            "passed": evaluation.passed,
                            "score": evaluation.score,
                            "missing": evaluation.missing,
                            "failed_checks": evaluation.failed_requirements(),
                            "feedback_draft": evaluation.feedback,
                        },
                        replan_count=replan_count,
                        replan_cap=REPLAN_CAP,
                    ),
                },
            ],
            task="orchestrator:route_after_coverage",
            max_tokens=800,
        )
        if isinstance(payload, dict):
            action = str(payload.get("action") or "replan").strip().lower()
            # A failed gate with budget remaining may be answered with "replan"
            # or "escalate" - never with "proceed". Letting the model wave a
            # failing plan through would make the gate decorative, which is the
            # single most important thing it must not be.
            if action == "proceed":
                log.info(
                    "routing model proposed 'proceed' on a FAILED gate with "
                    "%d/%d re-plans used; coercing to 'replan'",
                    replan_count,
                    REPLAN_CAP,
                )
                action = "replan"
            if action in ("replan", "escalate"):
                decision["action"] = action
            decision["confidence"] = _clamp(payload.get("confidence"), decision["confidence"])
            decision["rationale"] = str(payload.get("rationale") or decision["rationale"])[:600]
            feedback = str(payload.get("feedback") or "").strip()
            if action == "replan":
                decision["feedback"] = feedback or evaluation.feedback
    except (JSONParseError, LLMError, Exception) as exc:  # noqa: B014
        log.warning("routing model unavailable, using the deterministic route: %s", exc)

    if decision["action"] == "replan" and not str(decision["feedback"]).strip():
        decision["feedback"] = evaluation.feedback or "Add edge-case and error-state coverage."
    return decision


# ==========================================================================
# Synthesis
# ==========================================================================
async def synthesise_report(
    ctx: RunContext,
    llm: LLMClient | None,
    state: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> FinalReport:
    """Compose the final report: facts assembled locally, prose from the model.

    If the model is unavailable the report is still produced in full; only the
    executive prose degrades, and the report says which path it took.
    """
    cfg = settings or get_settings()
    facts = _report_facts(state)

    synthesis: dict[str, Any] | None = None
    if llm is not None:
        try:
            payload = await llm.complete_json(
                ModelRole.REASONING,
                [
                    {"role": "system", "content": REPORT_SYNTHESIS_SYSTEM},
                    {"role": "user", "content": report_synthesis_user(facts=facts)},
                ],
                task="report:synthesis",
                max_tokens=1500,
            )
            if isinstance(payload, dict):
                synthesis = {
                    "executive_summary": str(payload.get("executive_summary") or "")[:2000],
                    "business_impact": str(payload.get("business_impact") or "")[:800],
                    "limitations": [
                        str(item)[:300] for item in (payload.get("limitations") or [])
                    ][:10],
                }
        except (JSONParseError, LLMError, Exception) as exc:  # noqa: B014
            log.warning("report synthesis failed, using the deterministic summary: %s", exc)
            ctx.emit(
                "report",
                "error",
                "Executive summary model call failed - using the computed summary",
                detail=f"{type(exc).__name__}: {exc}",
            )

    # Fold the cost, timing and safety accounting into the state the report is
    # assembled from. Doing it here rather than in the assembler keeps the
    # assembler a pure projection of state, which is what makes it testable.
    enriched = dict(state)
    if llm is not None:
        enriched["llm_cost"] = llm.cost_summary()
    enriched["safety"] = _safety_summary(state, cfg)

    return assemble_report(
        run_id=ctx.run_id,
        target_url=state.get("target_url", ""),
        state_like=enriched,
        synthesis=synthesis,
        llm_provider=(llm.primary_name if llm else "unavailable"),
        models_used=(llm.models_used() if llm else {}),
        settings=cfg,
    )


def _safety_summary(state: dict[str, Any], cfg: Settings) -> dict[str, Any]:
    """The policy the run executed under, and everything it refused.

    A report that omits what was blocked is claiming coverage it does not have:
    a checkout flow that safe mode reduced to a raised assertion is not a
    tested checkout flow, and the reader has to be told so.
    """
    site_map = state.get("site_map")
    blocked_navigations = list(getattr(site_map, "blocked_targets", []) or [])

    blocked_actions: list[dict[str, Any]] = []
    for test in state.get("generated_tests") or []:
        for entry in getattr(test, "blocked_actions", []) or []:
            blocked_actions.append({"flow_id": getattr(test, "flow_id", ""), **entry})

    return {
        "target_policy": {
            "allow_private_targets": cfg.allow_private_targets,
            "allow_insecure_tls": cfg.allow_insecure_tls,
            "allowlist": list(cfg.target_allowlist),
            "resolve_dns": cfg.target_resolve_dns,
        },
        "safe_mode": describe_policy(SafetyPolicy.from_settings(cfg)),
        "blocked_navigations": blocked_navigations[:50],
        "blocked_actions": blocked_actions[:50],
        "blocked_navigation_count": len(blocked_navigations),
        "blocked_action_count": len(blocked_actions),
    }


def _report_facts(state: dict[str, Any]) -> dict[str, Any]:
    """The pre-computed facts the summariser is told not to contradict."""
    results = state.get("run_results") or []
    healer_actions = state.get("healer_actions") or []
    bugs = state.get("packaged_bugs") or []
    visual = state.get("visual_diff_findings") or []
    risks = {r.flow_id: r.risk.value for r in (state.get("risk_classifications") or [])}
    plan = state.get("test_plan")

    failing_high = [
        r.flow_name or r.flow_id
        for r in results
        if r.status in (TestStatus.FAILED, TestStatus.ERROR) and risks.get(r.flow_id) == "high"
    ]
    return {
        "target_url": sanitize_url(state.get("target_url", "")),
        "flows_planned": len(plan.flows) if plan else 0,
        "tests_generated": len([t for t in (state.get("generated_tests") or []) if t.valid]),
        "passed": sum(1 for r in results if r.status is TestStatus.PASSED),
        "healed": sum(1 for r in results if r.status is TestStatus.HEALED),
        "failed": sum(1 for r in results if r.status in (TestStatus.FAILED, TestStatus.ERROR)),
        "high_risk_failures": failing_high,
        "risk_counts": {
            level: sum(1 for value in risks.values() if value == level)
            for level in ("high", "medium", "low")
        },
        "bugs_filed": len(bugs),
        "bug_titles": [b.title for b in bugs][:8],
        "needs_human_review": sum(1 for a in healer_actions if a.needs_human_review),
        "auto_applied_heals": sum(1 for a in healer_actions if a.auto_applied),
        "genuine_defects": sum(
            1 for a in healer_actions if a.classification is DefectClass.GENUINE_DEFECT
        ),
        "visual_regressions": sum(1 for v in visual if v.is_regression),
        "coverage_gaps": (
            state["coverage_evaluation"].missing if state.get("coverage_evaluation") else []
        ),
        "replan_count": state.get("replan_count", 0),
        "force_proceeded": state.get("force_proceeded", False),
        "login_ok": state.get("login_ok"),
        "credentials_present": state.get("credentials_present", False),
        "node_errors": state.get("node_errors") or [],
        "escalations": state.get("escalations") or [],
    }


# ==========================================================================
# Run lifecycle
# ==========================================================================
async def execute_run(
    *,
    run_id: str,
    target_url: str,
    user_intent: str | None = None,
    prd_text: str | None = None,
    settings: Settings | None = None,
) -> FinalReport:
    """Run the whole pipeline for one target. The single entry point.

    Credentials, if any, were placed in :data:`security.SECRET_BOX` by the API
    before this was called and are wiped here in ``finally`` regardless of how
    the run ends. This function never raises: a catastrophic failure still
    yields a report describing what happened.
    """
    cfg = settings or get_settings()
    configure_logging(cfg.log_level)
    ensure_dirs()

    ctx = create_context(run_id, cfg)
    attach_run_log(run_id, ctx.dir / "agent.log")
    started = time.monotonic()

    llm: LLMClient | None = None
    session: BrowserSession | None = None
    state: dict[str, Any] = {}

    try:
        ctx.set_progress(status="running", current_stage="orchestrator")
        ctx.emit(
            "orchestrator",
            "start",
            f"Run started against {sanitize_url(target_url)}",
            detail=(
                f"credentials_present={ctx.credentials_present} "
                f"intent={'yes' if user_intent else 'no'} "
                f"prd={'yes' if prd_text else 'no'} "
                f"flags={cfg.feature_flags()}"
            ),
            confidence=1.0,
        )

        llm = LLMClient(cfg)
        ctx.llm = llm
        if not llm.providers:
            raise RuntimeError(
                "No LLM provider is configured. Set GROQ_API_KEY in .env (free key at "
                "https://console.groq.com), or set LLM_OFFLINE_MODE=true to smoke-test "
                "the plumbing without a model."
            )

        session = BrowserSession(cfg)
        await session.start()
        ctx.scratch()["session"] = session

        from graph.graph import build_graph  # local import: avoids a cycle at import time

        app = build_graph()
        state = dict(
            initial_state(
                run_id=run_id,
                target_url=target_url,
                prd_text=prd_text,
                user_intent=user_intent,
                credentials_present=ctx.credentials_present,
                feature_flags=cfg.feature_flags(),
            )
        )
        state = await app.ainvoke(state, config={"recursion_limit": 64})

    except Exception as exc:
        detail = redact_text(traceback.format_exc())[:4000]
        log.error("run %s failed: %s", run_id, exc)
        ctx.emit(
            "orchestrator",
            "error",
            f"Run failed: {type(exc).__name__}: {exc}",
            detail=detail,
            needs_human_review=True,
        )
        state = dict(state or {})
        state.setdefault("target_url", target_url)
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["node_errors"] = [*(state.get("node_errors") or []), state["error"]]

    # ---- always produce a report, even from a partial state --------------
    report: FinalReport
    try:
        state.setdefault("run_id", run_id)
        state.setdefault("target_url", target_url)
        state.setdefault("credentials_present", ctx.credentials_present)
        state["decision_log"] = ctx.events.snapshot()
        state["finished_at"] = utcnow_iso()
        state.setdefault("started_at", ctx.started_at)

        if cfg.enable_prd_gap_analysis and prd_text and not state.get("prd_gaps"):
            gaps, method = analyse_prd_gaps(
                prd_text=prd_text, plan=state.get("test_plan"), enabled=True
            )
            state["prd_gaps"] = gaps
            ctx.emit(
                "report",
                "progress",
                f"PRD gap analysis: {summarise_gaps(gaps, method=method)}",
                detail=f"method={method}",
            )

        if cfg.enable_regression_radar and not state.get("regression_radar"):
            state["regression_radar"] = record_radar(
                target_url=target_url,
                run_id=run_id,
                plan=state.get("test_plan"),
                results=state.get("run_results") or [],
                classifications=state.get("risk_classifications") or [],
                enabled=True,
            )

        report = await synthesise_report(ctx, llm, state, settings=cfg)
        paths = write_report(ctx.dir, report)
        report.artifacts.update(paths)
        report.artifacts["events"] = str(ctx.dir / "events.jsonl")
        report.artifacts["generated_tests"] = str(ctx.dir / "generated_tests")
        report.artifacts["run_directory"] = str(ctx.dir)
        # Rewrite the JSON with the artifact paths included.
        write_report(ctx.dir, report)

        status: RunStatus = "failed" if state.get("error") else "completed"
        ctx.set_progress(
            status=status,
            current_stage="report",
            finished_at=state["finished_at"],
            error=state.get("error"),
            # Numeric entries only: the live snapshot is rendered as metric
            # tiles, and a list or a dict in there would break the UI.
            counts={
                key: value
                for key, value in (report.totals or {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
        )
        ctx.emit(
            "report",
            "complete",
            (
                f"Report ready: {report.totals.get('passed', 0)} passed, "
                f"{report.totals.get('failed', 0)} failed, "
                f"{report.totals.get('bugs_filed', 0)} bug(s) filed, "
                f"{report.totals.get('needs_human_review', 0)} queued for review"
            ),
            detail=f"artifacts written to {ctx.dir}",
            confidence=1.0,
        )
    except Exception as exc:  # pragma: no cover - reporting must not fail the run
        log.error("report synthesis failed for %s: %s", run_id, exc, exc_info=True)
        report = FinalReport(
            run_id=run_id,
            target_url=sanitize_url(target_url),
            status="failed",
            executive_summary=(
                "The run could not be completed and the report could not be fully "
                f"assembled: {type(exc).__name__}: {exc}"
            ),
            errors=[redact_text(str(exc))[:500]],
            decision_log_excerpt=ctx.events.tail(40),
        )
        ctx.set_progress(status="failed", error=str(exc))

    finally:
        duration = time.monotonic() - started
        log.info("run %s finished in %.1fs", run_id, duration)
        if session is not None:
            await session.stop()
        if llm is not None:
            await llm.aclose()
        # Credentials leave the process here, whatever happened above.
        SECRET_BOX.wipe(run_id)
        detach_run_log(run_id)

    return report


def cleanup_run(run_id: str) -> None:
    """Release a finished run's context and any residual secret material."""
    SECRET_BOX.wipe(run_id)
    drop_context(run_id)


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
