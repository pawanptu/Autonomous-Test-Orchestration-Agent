"""The LangGraph: explicit nodes, explicit conditional edges.

This is deliberately not one long prompt chain. Each stage is a node with a
single responsibility, and the interesting behaviour lives in the *edges*:

    planner ─▶ coverage_gate ─┬─(replan, budget 2)─▶ planner
                              ├─(coverage_ok)──────▶ risk_ranking
                              └─(budget spent)─────▶ risk_ranking  [force_proceeded]

    risk_ranking ─▶ generator ─▶ runner ─┬─(failures, first pass)─▶ healer
                                         └─(clean or already healed)─▶ visual_diff

    healer ─┬─(a patch was auto-applied)─▶ runner   [re-run once]
            └─(nothing to re-run)────────▶ visual_diff

    visual_diff ─▶ bug_packager ─▶ report ─▶ END

Two invariants make the loops safe:

* ``replan_count`` is capped at :data:`config.REPLAN_CAP`. On exhaustion the
  gate routes forward and stamps ``force_proceeded``.
* ``heal_pass_count`` is capped at :data:`config.HEAL_RERUN_CAP`, so the
  runner ↔ healer cycle can execute at most once.

Every node is wrapped by :func:`node`, which emits ``start``/``complete``
events, keeps the live progress snapshot current, and converts an exception
into a recorded ``error`` plus a routing hint - so a node failure degrades the
run into a partial report instead of killing the process.
"""

from __future__ import annotations

import functools
import traceback
from typing import Any, Awaitable, Callable

from langgraph.graph import END, StateGraph

from config import HEAL_RERUN_CAP, REPLAN_CAP, get_settings
from graph.runtime import RunContext, get_context
from graph.state import (
    DefectClass,
    OrchestrationState,
    RiskLevel,
    Stage,
    TestStatus,
    utcnow_iso,
)
from logging_setup import get_logger
from security import redact_text, sanitize_url

log = get_logger("aivor.graph")

NodeFn = Callable[[OrchestrationState], Awaitable[dict[str, Any]]]


# ==========================================================================
# Node wrapper
# ==========================================================================
def node(stage: Stage) -> Callable[[NodeFn], NodeFn]:
    """Instrument a node: lifecycle events, progress snapshot, error capture.

    A node that raises does not propagate. The exception is redacted, recorded
    in ``node_errors``, and the state is marked so the conditional edges can
    route to the report. The orchestrator then still produces a partial report,
    which is the whole point of catching here rather than at ``ainvoke``.
    """

    def decorate(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: OrchestrationState) -> dict[str, Any]:
            ctx = get_context(state["run_id"])
            ctx.set_progress(current_stage=stage)
            ctx.emit(stage, "start", f"{stage} started")
            try:
                update = await fn(state)
            except Exception as exc:
                detail = redact_text(traceback.format_exc())[:3000]
                log.error("node %s failed: %s", stage, exc)
                ctx.emit(
                    stage,
                    "error",
                    f"{stage} failed: {type(exc).__name__}: {exc}",
                    detail=detail,
                    needs_human_review=True,
                )
                message = f"{stage}: {type(exc).__name__}: {exc}"
                return {
                    "current_stage": stage,
                    "node_errors": [*(state.get("node_errors") or []), message],
                    "error": state.get("error") or message,
                }
            update.setdefault("current_stage", stage)
            ctx.emit(stage, "complete", f"{stage} complete")
            return update

        return wrapper

    return decorate


def _session(ctx: RunContext) -> Any:
    session = ctx.scratch().get("session")
    if session is None:
        raise RuntimeError("no BrowserSession registered for this run")
    return session


def _stage_failed(state: OrchestrationState, stage: str) -> bool:
    """True when the named stage recorded an error in this run."""
    return any(err.startswith(f"{stage}:") for err in (state.get("node_errors") or []))


# ==========================================================================
# Nodes
# ==========================================================================
@node("planner")
async def planner_node(state: OrchestrationState) -> dict[str, Any]:
    """Crawl on the first pass, then (re-)plan.

    The crawl is expensive and the application does not change between
    re-plans, so it happens once and the site map is reused. A re-plan is a
    pure re-prompt with the gate's feedback attached.
    """
    from agents.planner import describe_plan, generate_plan, explore_target, renumber_flows

    ctx = get_context(state["run_id"])
    cfg = get_settings()
    update: dict[str, Any] = {}

    site_map = state.get("site_map")
    login_ok = state.get("login_ok")

    if site_map is None:
        site_map, login_result = await explore_target(
            ctx, _session(ctx), target_url=state["target_url"], settings=cfg
        )
        update["site_map"] = site_map
        if login_result is not None:
            login_ok = login_result.ok
            update["login_ok"] = login_ok
            update["login_error"] = login_result.error
            ctx.set_progress(login_ok=login_ok)
        ctx.emit(
            "planner",
            "progress",
            f"Crawl complete: {len(site_map.pages)} page(s), "
            f"login_detected={site_map.login_detected}, "
            f"ecommerce_signals={len(site_map.ecommerce_signals)}",
            detail="; ".join(site_map.notes[-2:]),
        )
        if not site_map.pages:
            raise RuntimeError(
                f"the crawl reached no pages on {sanitize_url(state['target_url'])}; "
                "the target may be unreachable or blocking automation"
            )

    revision = int(state.get("replan_count", 0))
    feedback = state.get("coverage_feedback")
    if feedback:
        ctx.emit(
            "planner",
            "replan",
            f"Re-planning with coverage feedback (revision {revision})",
            detail=feedback[:600],
            confidence=0.7,
        )

    plan = await generate_plan(
        ctx,
        ctx.llm,
        site_map=site_map,
        user_intent=state.get("user_intent"),
        prd_text=state.get("prd_text"),
        coverage_feedback=feedback,
        revision=revision,
        settings=cfg,
    )
    plan = renumber_flows(plan)

    ctx.emit("planner", "decision", describe_plan(plan), detail=plan.summary[:400], confidence=0.75)
    ctx.set_progress(counts={**ctx.progress.get("counts", {}), "flows": len(plan.flows)})

    update["test_plan"] = plan
    update["coverage_feedback"] = None
    return update


@node("coverage_gate")
async def coverage_gate_node(state: OrchestrationState) -> dict[str, Any]:
    """Judge the plan before a single line of test code is generated."""
    from agents.orchestrator import evaluate_coverage, route_after_coverage

    ctx = get_context(state["run_id"])
    plan = state.get("test_plan")
    if plan is None:
        raise RuntimeError("coverage gate reached with no plan to evaluate")

    evaluation = await evaluate_coverage(
        ctx, ctx.llm, plan=plan, site_map=state.get("site_map")
    )
    replan_count = int(state.get("replan_count", 0))
    decision = await route_after_coverage(
        ctx, ctx.llm, evaluation=evaluation, replan_count=replan_count
    )

    update: dict[str, Any] = {"coverage_evaluation": evaluation}

    if decision["action"] == "replan":
        update["replan_count"] = replan_count + 1
        update["coverage_feedback"] = decision["feedback"]
        ctx.emit(
            "coverage_gate",
            "replan",
            (
                f"Coverage gate: missing {_first_gap(evaluation)} - sending back to "
                f"Planner (replan {replan_count + 1}/{REPLAN_CAP})"
            ),
            detail=decision["rationale"],
            confidence=decision["confidence"],
        )
        ctx.set_progress(replan_count=replan_count + 1)
        return update

    if decision.get("forced"):
        update["force_proceeded"] = True
        ctx.emit(
            "coverage_gate",
            "escalate",
            (
                f"Re-plan budget exhausted ({REPLAN_CAP}/{REPLAN_CAP}) - force-proceeding "
                f"with {len(evaluation.failed_requirements())} unmet rubric line(s)"
            ),
            detail=decision["rationale"],
            confidence=decision["confidence"],
            needs_human_review=True,
        )
        ctx.set_progress(force_proceeded=True)
    elif decision["action"] == "escalate":
        update["escalations"] = [
            *(state.get("escalations") or []),
            f"coverage_gate: {decision['rationale']}",
        ]
        ctx.emit(
            "coverage_gate",
            "escalate",
            "Coverage gate escalated - continuing in degraded mode",
            detail=decision["rationale"],
            confidence=decision["confidence"],
            needs_human_review=True,
        )
    elif evaluation.passed:
        ctx.emit(
            "coverage_gate",
            "decision",
            f"Coverage gate PASSED (score {evaluation.score:.2f}) - proceeding to risk ranking",
            detail=decision["rationale"],
            confidence=decision["confidence"],
        )
    else:
        # Reached when the gate failed but the orchestrator chose to proceed
        # anyway without exhausting the budget. Say so plainly rather than
        # letting the log imply a pass.
        ctx.emit(
            "coverage_gate",
            "decision",
            (
                f"Coverage gate FAILED (score {evaluation.score:.2f}) but the orchestrator "
                f"chose to proceed with {len(evaluation.failed_requirements())} unmet "
                "rubric line(s)"
            ),
            detail=decision["rationale"],
            confidence=decision["confidence"],
            needs_human_review=True,
        )
    return update


@node("risk_ranking")
async def risk_ranking_node(state: OrchestrationState) -> dict[str, Any]:
    """Classify every flow so the rest of the pipeline can act on priority."""
    from differentiation.risk_ranking import rank_flows, risk_summary

    ctx = get_context(state["run_id"])
    plan = state.get("test_plan")
    if plan is None or not plan.flows:
        return {"risk_classifications": []}

    def emit(summary: str, detail: str = "", risk: str | None = None, confidence: float | None = None) -> None:
        ctx.emit("risk_ranking", "decision", summary, detail=detail, risk=risk, confidence=confidence)

    classifications = await rank_flows(ctx.llm, plan.flows, emit=emit)
    counts = risk_summary(classifications)
    ctx.emit(
        "risk_ranking",
        "progress",
        f"Risk profile: {counts['high']} high, {counts['medium']} medium, {counts['low']} low",
        detail="the final report and the execution order both follow this ranking",
    )
    ctx.set_progress(
        risk_classifications=[c.model_dump(mode="json") for c in classifications],
        counts={**ctx.progress.get("counts", {}), **{f"risk_{k}": v for k, v in counts.items()}},
    )
    return {"risk_classifications": classifications}


@node("generator")
async def generator_node(state: OrchestrationState) -> dict[str, Any]:
    """Resolve selectors live, then write one executable test per flow."""
    from agents.generator import generate_test, resolve_flow_selectors, write_tests
    from differentiation.risk_ranking import risk_map, risk_order

    ctx = get_context(state["run_id"])
    cfg = get_settings()
    plan = state.get("test_plan")
    if plan is None or not plan.flows:
        return {"generated_tests": []}

    classifications = state.get("risk_classifications") or []
    risks = risk_map(classifications)
    order = risk_order(classifications)
    by_id = {f.id: f for f in plan.flows}
    ordered_flows = [by_id[fid] for fid in order if fid in by_id] or list(plan.flows)

    session = _session(ctx)
    site_map = state.get("site_map")
    tests = []

    for position, flow in enumerate(ordered_flows, start=1):
        risk = risks.get(flow.id, RiskLevel.MEDIUM)
        ctx.emit(
            "generator",
            "progress",
            f"Generating {flow.id} ({position}/{len(ordered_flows)}): {flow.name}",
            detail=f"{len(flow.steps)} step(s), validating selectors against the live page",
            risk=risk.value,
            flow_id=flow.id,
        )
        try:
            validations = await resolve_flow_selectors(
                session, ctx=ctx, flow=flow, site_map=site_map, settings=cfg
            )
        except Exception as exc:
            log.warning("selector walk failed for %s: %s", flow.id, exc)
            ctx.emit(
                "generator",
                "error",
                f"{flow.id}: selector walk failed ({type(exc).__name__})",
                detail=str(exc)[:300],
                flow_id=flow.id,
            )
            validations = []

        resolved = sum(1 for v in validations if v.valid)
        ctx.emit(
            "generator",
            "progress",
            f"{flow.id}: {resolved}/{len(validations)} selectors resolved against the live DOM",
            detail="; ".join(
                f"step {v.step_index}: {v.chosen_strategy or 'unresolved'}"
                for v in validations[:5]
            ),
            flow_id=flow.id,
        )

        test = await generate_test(
            ctx,
            ctx.llm,
            flow=flow,
            validations=validations,
            risk=risk,
            base_url=plan.target_url,
            settings=cfg,
        )
        tests.append(test)
        if test.valid:
            ctx.emit(
                "generator",
                "decision",
                f"{flow.id}: test generated by {test.generated_by_model}",
                detail="; ".join(test.warnings[:2]) or "no warnings",
                risk=risk.value,
                flow_id=flow.id,
                confidence=0.8 if test.repair_attempts == 0 else 0.6,
            )
        else:
            ctx.emit(
                "generator",
                "decision",
                f"{flow.id}: NOT generated - {test.validation_error}",
                detail=(
                    "reported as a remaining coverage gap rather than emitting a test "
                    "that would not really exercise the flow"
                ),
                risk=risk.value,
                flow_id=flow.id,
                needs_human_review=True,
            )

    write_tests(ctx.dir / "generated_tests", tests, plan.flows, risks)
    valid = [t for t in tests if t.valid]
    ctx.emit(
        "generator",
        "progress",
        f"Generated {len(valid)}/{len(tests)} executable tests",
        detail=f"written to {ctx.dir / 'generated_tests'}",
    )
    ctx.set_progress(counts={**ctx.progress.get("counts", {}), "tests_generated": len(valid)})
    return {"generated_tests": tests}


@node("runner")
async def runner_node(state: OrchestrationState) -> dict[str, Any]:
    """Execute the suite, or re-run just the tests the Healer patched."""
    from browser.runner import (
        classify_rerun,
        merge_results,
        rerun_for_flake,
        rerun_single,
        run_suite,
        should_rerun_for_flake,
    )
    from differentiation.risk_ranking import risk_map, risk_order

    ctx = get_context(state["run_id"])
    cfg = get_settings()
    plan = state.get("test_plan")
    tests = [t for t in (state.get("generated_tests") or []) if t.valid and t.source]
    if plan is None or not tests:
        ctx.emit("runner", "progress", "No executable tests to run", detail="nothing was generated")
        return {"run_results": state.get("run_results") or []}

    session = _session(ctx)
    screenshots = ctx.dir / "screenshots"
    pending = ctx.scratch().pop("pending_rerun", None)

    if pending:
        # Second pass: only the flows whose tests were auto-patched.
        by_id = {f.id: f for f in plan.flows}
        results = list(state.get("run_results") or [])
        rerun: list[Any] = []
        for patched in pending:
            flow = by_id.get(patched.flow_id)
            if flow is None:
                continue
            ctx.emit(
                "runner",
                "progress",
                f"Re-running {flow.id} after the healer patch",
                flow_id=flow.id,
            )
            outcome = await rerun_single(
                session,
                run_id=ctx.run_id,
                flow=flow,
                test=patched,
                screenshot_dir=screenshots,
                settings=cfg,
            )
            rerun.append(outcome)
            ctx.emit(
                "runner",
                "decision",
                f"{flow.id} after heal: {outcome.status.value.upper()}",
                detail=(outcome.error_message or "the patched test passed")[:300],
                flow_id=flow.id,
                auto_applied=True,
            )
        merged = merge_results(results, rerun)
        _record_run_progress(ctx, merged)
        return {"run_results": merged, "heal_pass_count": int(state.get("heal_pass_count", 0))}

    def progress(flow_id: str, summary: str, detail: str = "") -> None:
        ctx.emit("runner", "progress", summary, detail=detail, flow_id=flow_id)

    mode = "parallel" if cfg.enable_parallel_execution else "sequential"
    ctx.emit(
        "runner",
        "progress",
        f"Executing {len(tests)} tests ({mode}, highest risk first)",
        detail=f"timeout {cfg.test_timeout_s:.0f}s per test",
    )
    results = await run_suite(
        session,
        run_id=ctx.run_id,
        flows=plan.flows,
        tests=tests,
        screenshot_dir=screenshots,
        settings=cfg,
        progress=progress,
        order=risk_order(state.get("risk_classifications") or []),
    )

    # Bounded flake confirmation. A failure that looks timing-dependent, or one
    # on a high-risk flow, is re-run exactly once with jitter before it is
    # believed. A flow that then passes is reported FLAKY, not green: an
    # intermittent failure is one a user will eventually meet.
    risks = risk_map(state.get("risk_classifications") or [])
    by_flow = {test.flow_id: test for test in tests}
    flows_by_id = {flow.id: flow for flow in plan.flows}
    confirmed: list[Any] = []
    for result in results:
        risk = risks.get(result.flow_id, RiskLevel.MEDIUM)
        should, why = should_rerun_for_flake(result, risk, cfg)
        flow = flows_by_id.get(result.flow_id)
        test = by_flow.get(result.flow_id)
        if not should or flow is None or test is None:
            confirmed.append(result)
            continue
        ctx.emit(
            "runner",
            "decision",
            f"{result.flow_id}: re-running once to confirm the failure",
            detail=why,
            flow_id=result.flow_id,
        )
        second = await rerun_for_flake(
            session,
            run_id=ctx.run_id,
            flow=flow,
            test=test,
            screenshot_dir=screenshots,
            settings=cfg,
            progress=progress,
        )
        final, verdict = classify_rerun(result, second)
        ctx.emit(
            "runner",
            "decision",
            f"{result.flow_id}: {verdict}",
            detail="; ".join(final.notes)[:300],
            flow_id=result.flow_id,
            needs_human_review=final.flaky,
        )
        confirmed.append(final)
    results = confirmed

    _record_run_progress(ctx, results)
    return {"run_results": results}


def _record_run_progress(ctx: RunContext, results: list[Any]) -> None:
    counts = {
        "passed": sum(1 for r in results if r.status is TestStatus.PASSED),
        "healed": sum(1 for r in results if r.status is TestStatus.HEALED),
        "failed": sum(1 for r in results if r.status in (TestStatus.FAILED, TestStatus.ERROR)),
    }
    ctx.emit(
        "runner",
        "decision",
        f"Suite result: {counts['passed']} passed, {counts['failed']} failed"
        + (f", {counts['healed']} healed" if counts["healed"] else ""),
        detail="failures are handed to the healer for classification",
    )
    ctx.set_progress(counts={**ctx.progress.get("counts", {}), **counts})


@node("healer")
async def healer_node(state: OrchestrationState) -> dict[str, Any]:
    """Classify each failure and take the confidence-appropriate branch."""
    from agents.healer import failing_results, heal_failure, summarise_actions
    from differentiation.risk_ranking import risk_map

    ctx = get_context(state["run_id"])
    cfg = get_settings()
    plan = state.get("test_plan")
    results = state.get("run_results") or []
    failures = failing_results(results)
    if plan is None or not failures:
        return {"healer_actions": state.get("healer_actions") or [], "heal_pass_count": 1}

    risks = risk_map(state.get("risk_classifications") or [])
    flows = {f.id: f for f in plan.flows}
    tests = {t.flow_id: t for t in (state.get("generated_tests") or [])}
    session = _session(ctx)

    ctx.emit(
        "healer",
        "progress",
        f"Analysing {len(failures)} failing test(s)",
        detail="gathering DOM snapshots, console errors and a live locator re-probe",
    )

    actions = list(state.get("healer_actions") or [])
    to_rerun: list[Any] = []
    for result in failures:
        flow = flows.get(result.flow_id)
        if flow is None:
            continue
        action, patched = await heal_failure(
            ctx,
            ctx.llm,
            session,
            flow=flow,
            result=result,
            test=tests.get(result.flow_id),
            risk=risks.get(result.flow_id, RiskLevel.MEDIUM),
            settings=cfg,
        )
        actions.append(action)
        if patched is not None:
            to_rerun.append(patched)

    summary = summarise_actions(actions)
    ctx.emit(
        "healer",
        "progress",
        (
            f"Healer summary: {summary['script_issues']} script issue(s), "
            f"{summary['genuine_defects']} genuine defect(s), "
            f"{summary['auto_applied']} auto-applied, "
            f"{summary['needs_human_review']} queued for human review"
        ),
        detail="patches are only applied at or above 0.60 confidence",
    )
    ctx.set_progress(
        healer_actions=[a.model_dump(mode="json") for a in actions],
        counts={**ctx.progress.get("counts", {}), **{f"heal_{k}": v for k, v in summary.items()}},
    )

    update: dict[str, Any] = {"healer_actions": actions, "heal_pass_count": 1}
    if to_rerun:
        # Patched sources replace the originals so the report shows what ran.
        patched_by_id = {t.flow_id: t for t in to_rerun}
        update["generated_tests"] = [
            patched_by_id.get(t.flow_id, t) for t in (state.get("generated_tests") or [])
        ]
        ctx.scratch()["pending_rerun"] = to_rerun
    return update


@node("visual_diff")
async def visual_diff_node(state: OrchestrationState) -> dict[str, Any]:
    """Compare each flow's frame against its stored baseline."""
    from differentiation.risk_ranking import risk_map
    from differentiation.visual_diff import analyse, summarise

    ctx = get_context(state["run_id"])
    cfg = get_settings()
    if not cfg.enable_visual_diff:
        ctx.emit(
            "visual_diff",
            "progress",
            "Visual diff disabled (ENABLE_VISUAL_DIFF=false)",
            detail="set the flag to compare screenshots against stored baselines",
        )
        return {"visual_diff_findings": []}

    plan = state.get("test_plan")
    findings = analyse(
        run_directory=ctx.dir,
        target_url=state["target_url"],
        results=state.get("run_results") or [],
        flows=(plan.flows if plan else []),
        risk_lookup=risk_map(state.get("risk_classifications") or []),
        settings=cfg,
    )
    for finding in findings:
        if finding.is_regression:
            ctx.emit(
                "visual_diff",
                "decision",
                (
                    f"Visual diff: {finding.changed_ratio * 100:.1f}% pixels changed on "
                    f"{finding.flow_name or finding.flow_id} vs baseline"
                ),
                detail=finding.note,
                risk=finding.risk.value if finding.risk else None,
                flow_id=finding.flow_id,
                needs_human_review=True,
            )
        elif finding.is_new_baseline:
            ctx.emit(
                "visual_diff",
                "progress",
                f"Visual baseline established for {finding.flow_id}",
                detail=finding.note,
                flow_id=finding.flow_id,
            )

    stats = summarise(findings)
    ctx.emit(
        "visual_diff",
        "progress",
        (
            f"Visual check: {stats['checked']} frame(s), {stats['regressions']} regression(s), "
            f"{stats['new_baselines']} new baseline(s)"
        ),
        detail="visual findings are reported separately from functional failures",
    )
    ctx.set_progress(
        visual_findings=[f.model_dump(mode="json") for f in findings],
        counts={
            **ctx.progress.get("counts", {}),
            "visual_regressions": stats["regressions"],
        },
    )
    return {"visual_diff_findings": findings}


@node("bug_packager")
async def bug_packager_node(state: OrchestrationState) -> dict[str, Any]:
    """Turn every confirmed defect into a ticket-ready artifact on disk."""
    from differentiation.bug_packager import package_bugs
    from differentiation.risk_ranking import risk_map

    ctx = get_context(state["run_id"])
    plan = state.get("test_plan")
    actions = state.get("healer_actions") or []
    defects = [a for a in actions if a.classification is DefectClass.GENUINE_DEFECT]
    if not defects:
        ctx.emit(
            "bug_packager",
            "progress",
            "No genuine application defects to package",
            detail=f"{len(actions)} healer action(s) reviewed",
        )
        return {"packaged_bugs": []}

    def emit(summary: str, detail: str = "", risk: str | None = None, confidence: float | None = None) -> None:
        ctx.emit("bug_packager", "decision", summary, detail=detail, risk=risk, confidence=confidence)

    bugs = await package_bugs(
        llm=ctx.llm,
        run_directory=ctx.dir,
        run_id=ctx.run_id,
        flows=(plan.flows if plan else []),
        results=state.get("run_results") or [],
        healer_actions=actions,
        risk_lookup=risk_map(state.get("risk_classifications") or []),
        tests=state.get("generated_tests") or [],
        base_url=state.get("target_url", ""),
        emit=emit,
    )
    ctx.set_progress(
        packaged_bugs=[b.model_dump(mode="json") for b in bugs],
        counts={**ctx.progress.get("counts", {}), "bugs_filed": len(bugs)},
    )
    return {"packaged_bugs": bugs}


@node("report")
async def report_node(state: OrchestrationState) -> dict[str, Any]:
    """Terminal node. The orchestrator does the synthesis and file writing."""
    ctx = get_context(state["run_id"])
    ctx.emit(
        "report",
        "progress",
        "Synthesising the final risk-ranked report",
        detail="totals, risk ordering, coverage gaps, heals, visual findings and bugs",
    )
    return {
        "status": "failed" if state.get("error") else "completed",
        "finished_at": utcnow_iso(),
    }


# ==========================================================================
# Conditional edges
# ==========================================================================
def route_after_coverage(state: OrchestrationState) -> str:
    """``planner`` (re-plan), ``risk_ranking`` (proceed) or ``report`` (abort)."""
    if _stage_failed(state, "coverage_gate") and state.get("test_plan") is None:
        return "report"
    evaluation = state.get("coverage_evaluation")
    replan_count = int(state.get("replan_count", 0))
    if evaluation is None:
        return "risk_ranking" if state.get("test_plan") else "report"
    if state.get("coverage_feedback") and replan_count <= REPLAN_CAP:
        return "planner"
    return "risk_ranking"


def route_after_planner(state: OrchestrationState) -> str:
    """A planner that produced nothing goes straight to the report."""
    plan = state.get("test_plan")
    if plan is None or not plan.flows:
        return "report"
    return "coverage_gate"


def route_after_generator(state: OrchestrationState) -> str:
    """Skip execution entirely when nothing executable was produced."""
    tests = [t for t in (state.get("generated_tests") or []) if t.valid and t.source]
    return "runner" if tests else "report"


def route_after_run(state: OrchestrationState) -> str:
    """Heal on the first pass only; a healed pass goes on to the visual check."""
    if int(state.get("heal_pass_count", 0)) >= HEAL_RERUN_CAP:
        return "visual_diff"
    failures = [
        r
        for r in (state.get("run_results") or [])
        if r.status in (TestStatus.FAILED, TestStatus.ERROR)
    ]
    return "healer" if failures else "visual_diff"


def route_after_heal(state: OrchestrationState) -> str:
    """Re-run once if a patch was actually applied, else move on."""
    ctx = get_context(state["run_id"])
    return "runner" if ctx.scratch().get("pending_rerun") else "visual_diff"


def _first_gap(evaluation: Any) -> str:
    missing = list(getattr(evaluation, "missing", []) or [])
    if missing:
        return missing[0][:90]
    failed = evaluation.failed_requirements() if hasattr(evaluation, "failed_requirements") else []
    return failed[0][:90] if failed else "coverage"


# ==========================================================================
# Assembly
# ==========================================================================
def build_graph() -> Any:
    """Compile the orchestration graph. Safe to call at import time."""
    builder = StateGraph(OrchestrationState)

    builder.add_node("planner", planner_node)
    builder.add_node("coverage_gate", coverage_gate_node)
    builder.add_node("risk_ranking", risk_ranking_node)
    builder.add_node("generator", generator_node)
    builder.add_node("runner", runner_node)
    builder.add_node("healer", healer_node)
    builder.add_node("visual_diff", visual_diff_node)
    builder.add_node("bug_packager", bug_packager_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"coverage_gate": "coverage_gate", "report": "report"},
    )
    builder.add_conditional_edges(
        "coverage_gate",
        route_after_coverage,
        {"planner": "planner", "risk_ranking": "risk_ranking", "report": "report"},
    )
    builder.add_edge("risk_ranking", "generator")
    builder.add_conditional_edges(
        "generator",
        route_after_generator,
        {"runner": "runner", "report": "report"},
    )
    builder.add_conditional_edges(
        "runner",
        route_after_run,
        {"healer": "healer", "visual_diff": "visual_diff"},
    )
    builder.add_conditional_edges(
        "healer",
        route_after_heal,
        {"runner": "runner", "visual_diff": "visual_diff"},
    )
    builder.add_edge("visual_diff", "bug_packager")
    builder.add_edge("bug_packager", "report")
    builder.add_edge("report", END)

    return builder.compile()


def render_mermaid() -> str:
    """The pipeline as Mermaid, for the README and for debugging."""
    return """flowchart TD
    START([URL + optional creds/PRD/intent]) --> planner
    planner{{Planner<br/>login - crawl - plan}} --> gate{{Coverage gate<br/>rubric C1-C6}}
    gate -- gaps, budget left --> planner
    gate -- gaps, budget spent --> risk[force_proceeded]
    gate -- passed --> risk{{Risk ranking<br/>HIGH / MED / LOW}}
    risk --> gen{{Generator<br/>live selector validation}}
    gen --> run{{Runner}}
    run -- failures --> heal{{Healer<br/>SCRIPT_ISSUE vs GENUINE_DEFECT}}
    run -- all green --> vis{{Visual diff}}
    heal -- confidence >= 0.6 --> run
    heal -- confidence < 0.6 --> vis
    vis --> bugs{{Bug packager}}
    bugs --> report{{Risk-ranked report}}
    report --> END([JSON + Markdown + HTML])
"""
