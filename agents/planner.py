"""Planner sub-agent: explore the live application, then propose a test plan.

Two responsibilities, deliberately kept separate:

* :func:`explore_target` drives a real browser. It logs in first when
  credentials exist (so protected pages return real content), crawls
  breadth-first within the configured budget, and produces a
  :class:`graph.state.SiteMap`. This is the agent's only source of ground
  truth; nothing downstream is allowed to invent a page or an element.

* :func:`generate_plan` turns that map into a :class:`graph.state.TestPlan`
  using the reasoning model. Planning is a judgment task - deciding what is
  worth testing and what "correct" means for each flow - so it gets the large
  model, not the cheap one.

Re-planning is the same function with ``coverage_feedback`` supplied. The
Planner is never told "try harder"; it is told exactly which rubric line failed
and what the missing flow should assert, because vague feedback burns a
re-plan cycle from a budget of two.
"""

from __future__ import annotations

from typing import Any, Sequence

from browser.crawler import crawl, summarise_for_prompt
from browser.login import LoginResult, apply_token_header, perform_login, save_storage_state
from browser.session import BrowserSession
from config import Settings, get_settings
from differentiation.pattern_library import hints_for_prompt
from graph.runtime import RunContext
from graph.state import FlowCategory, SiteMap, TestFlow, TestPlan, TestStep
from llm.client import LLMClient, ModelRole
from llm.json_utils import JSONParseError, coerce_list
from llm.prompts import PLANNER_SYSTEM, planner_user
from logging_setup import get_logger
from security import sanitize_url

log = get_logger("aivor.planner")

VALID_ACTIONS = {
    "goto", "click", "fill", "select", "check", "press", "wait_for",
    "assert_text", "assert_visible", "assert_url", "assert_not_visible", "screenshot",
}

# Loose synonyms small models reach for, mapped onto the supported vocabulary.
ACTION_ALIASES = {
    "navigate": "goto", "open": "goto", "visit": "goto", "browse": "goto",
    "type": "fill", "enter": "fill", "input": "fill", "set": "fill",
    "tap": "click", "submit": "click", "press_button": "click", "choose": "select",
    "verify": "assert_visible", "expect": "assert_visible", "assert": "assert_visible",
    "check_text": "assert_text", "see": "assert_text", "contains": "assert_text",
    "wait": "wait_for", "sleep": "wait_for", "capture": "screenshot",
}

CATEGORY_ALIASES = {
    "happy": FlowCategory.HAPPY_PATH,
    "happy_path": FlowCategory.HAPPY_PATH,
    "happypath": FlowCategory.HAPPY_PATH,
    "positive": FlowCategory.HAPPY_PATH,
    "edge": FlowCategory.EDGE_CASE,
    "edge_case": FlowCategory.EDGE_CASE,
    "edgecase": FlowCategory.EDGE_CASE,
    "boundary": FlowCategory.EDGE_CASE,
    "error": FlowCategory.ERROR_STATE,
    "error_state": FlowCategory.ERROR_STATE,
    "errorstate": FlowCategory.ERROR_STATE,
    "negative": FlowCategory.ERROR_STATE,
}


# ==========================================================================
# Exploration
# ==========================================================================
async def explore_target(
    ctx: RunContext,
    session: BrowserSession,
    *,
    target_url: str,
    settings: Settings | None = None,
) -> tuple[SiteMap, LoginResult | None]:
    """Authenticate (if we can) and crawl. Never raises.

    Returns the site map plus the login result, or ``(SiteMap, None)`` when no
    credentials were supplied. A failed login does not abort the run: the crawl
    continues over whatever is publicly reachable and the map is flagged
    ``auth_blocked`` so the coverage gate and the report can say so.
    """
    cfg = settings or get_settings()
    creds = ctx.credentials
    login_result: LoginResult | None = None

    context = await session.new_context(use_storage_state=False)
    page = await context.new_page()

    try:
        if creds is not None and creds.token:
            await apply_token_header(context, creds.token)
            ctx.emit(
                "planner",
                "progress",
                "Bearer token applied to the browser context",
                detail=(
                    "Assumption: the target accepts 'Authorization: Bearer <token>'. "
                    "Targets expecting a cookie or localStorage entry will not be "
                    "authenticated by this and the run will report login_ok=False."
                ),
            )

        try:
            await page.goto(
                target_url, timeout=cfg.nav_timeout_ms, wait_until="domcontentloaded"
            )
        except Exception as exc:
            ctx.emit(
                "planner",
                "error",
                f"Could not open the target: {type(exc).__name__}",
                detail=f"target={sanitize_url(target_url)}",
                needs_human_review=True,
            )
            return SiteMap(target_url=target_url, notes=[f"target unreachable: {type(exc).__name__}"]), None

        if creds is not None and (creds.username or creds.password):
            ctx.emit("planner", "progress", "Credentials supplied - authenticating before the crawl")
            login_result = await perform_login(
                page,
                creds,
                base_url=target_url,
                action_timeout_ms=cfg.action_timeout_ms,
                nav_timeout_ms=cfg.nav_timeout_ms,
            )
            if login_result.ok:
                state_path = await save_storage_state(
                    context, ctx.dir.parent / f".auth_state_{ctx.run_id}.json"
                )
                if state_path:
                    ctx.storage_state_path = ctx.dir.parent / f".auth_state_{ctx.run_id}.json"
                    session.storage_state_path = state_path
                ctx.emit(
                    "planner",
                    "decision",
                    "Authenticated: protected pages are in scope",
                    detail="; ".join(login_result.evidence[:3]),
                    confidence=0.85,
                )
            else:
                ctx.emit(
                    "planner",
                    "escalate",
                    "AUTH BLOCKED - continuing with public pages only",
                    detail=(
                        f"{login_result.error}. Exploration of protected pages is "
                        "stopped; the report will list this as a coverage limitation."
                    ),
                    confidence=0.9,
                    needs_human_review=True,
                )
        elif creds is not None and creds.token:
            login_result = LoginResult(ok=True, method="bearer-token", login_url=target_url)

        auth_blocked = bool(login_result is not None and not login_result.ok)

        def _progress(summary: str, detail: str = "") -> None:
            ctx.emit("planner", "progress", summary, detail=detail)

        site_map = await crawl(
            page,
            start_url=target_url,
            settings=cfg,
            progress=_progress,
            auth_blocked=auth_blocked,
        )
        if login_result is not None and login_result.ok and not site_map.login_url:
            site_map.login_url = login_result.login_url
            site_map.login_detected = True
        return site_map, login_result
    finally:
        try:
            await context.close()
        except Exception:  # pragma: no cover
            log.debug("crawl context close failed", exc_info=True)


# ==========================================================================
# Planning
# ==========================================================================
async def generate_plan(
    ctx: RunContext,
    llm: LLMClient,
    *,
    site_map: SiteMap,
    user_intent: str | None,
    prd_text: str | None,
    coverage_feedback: str | None,
    revision: int,
    settings: Settings | None = None,
) -> TestPlan:
    """Ask the reasoning model for a plan, then enforce structural invariants.

    Raises :class:`llm.json_utils.JSONParseError` (or an ``LLMError``) if the
    model cannot produce usable JSON after its one repair attempt. The caller -
    the planner node - turns that into an error event and a partial report
    rather than a crashed process.
    """
    cfg = settings or get_settings()
    summary = summarise_for_prompt(site_map)

    intent = user_intent if (cfg.enable_intent_bias and user_intent) else None
    if user_intent and not cfg.enable_intent_bias:
        ctx.emit(
            "planner",
            "progress",
            "Intent supplied but ENABLE_INTENT_BIAS is off - planning unbiased",
            detail="set ENABLE_INTENT_BIAS=true to let the intent steer scope",
        )

    # Pattern hints are suggestions drawn from a small library of canonical
    # flow shapes; they bias the model toward the non-happy-path coverage the
    # gate will demand, without dictating the plan.
    blob = " ".join(
        [summary.get("target_url", "")]
        + [str(p.get("title", "")) + " " + str(p.get("text_excerpt", "")) for p in summary.get("pages", [])]
    )
    try:
        summary["pattern_hints"] = hints_for_prompt(blob)
    except Exception:  # pragma: no cover - the library is advisory only
        log.debug("pattern hints unavailable", exc_info=True)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": planner_user(
                site_map=summary,
                user_intent=intent,
                prd_excerpt=prd_text,
                coverage_feedback=coverage_feedback,
                revision=revision,
                max_flows=cfg.max_flows_to_generate,
                credentials_present=ctx.credentials_present,
            ),
        },
    ]

    payload = await llm.complete_json(
        ModelRole.REASONING,
        messages,
        task=f"planner:rev{revision}",
        max_tokens=6000,
    )
    return build_plan_from_payload(payload, site_map=site_map, revision=revision, settings=cfg)


def build_plan_from_payload(
    payload: Any,
    *,
    site_map: SiteMap,
    revision: int,
    settings: Settings | None = None,
) -> TestPlan:
    """Convert raw model output into a validated :class:`TestPlan`.

    Per-flow validation is isolated: one malformed flow is dropped with a note
    rather than discarding an otherwise good plan. Flow ids are renumbered so
    that duplicates from the model cannot collide downstream.
    """
    cfg = settings or get_settings()
    data = payload if isinstance(payload, dict) else {"flows": coerce_list(payload, "flows")}

    flows: list[TestFlow] = []
    notes: list[str] = []
    for index, raw in enumerate(coerce_list(data.get("flows"), "flows")):
        if not isinstance(raw, dict):
            notes.append(f"dropped flow #{index}: not an object")
            continue
        try:
            flows.append(_coerce_flow(raw, index=len(flows) + 1, default_url=site_map.target_url))
        except Exception as exc:
            notes.append(f"dropped flow #{index} ({raw.get('name', 'unnamed')!r}): {exc}")
            log.warning("dropping malformed flow #%d: %s", index, exc)

    if len(flows) > cfg.max_flows_to_generate:
        notes.append(
            f"plan truncated from {len(flows)} to {cfg.max_flows_to_generate} flows "
            "by MAX_FLOWS_TO_GENERATE"
        )
        flows = flows[: cfg.max_flows_to_generate]

    plan = TestPlan(
        target_url=site_map.target_url,
        summary=str(data.get("summary") or "")[:2000],
        flows=flows,
        discovered_areas=[
            str(a)[:120] for a in coerce_list(data.get("discovered_areas")) if a
        ] or site_map.area_names()[:8],
        auth_flow_present=bool(data.get("auth_flow_present", site_map.login_detected)),
        ecommerce_like=bool(data.get("ecommerce_like", bool(site_map.ecommerce_signals))),
        revision=revision,
        notes=notes,
    )
    return plan


def _coerce_flow(raw: dict[str, Any], *, index: int, default_url: str) -> TestFlow:
    """Normalise one flow dict into a :class:`TestFlow`."""
    steps: list[TestStep] = []
    for raw_step in coerce_list(raw.get("steps"), "steps"):
        if isinstance(raw_step, str):
            steps.append(TestStep(action="click", target=raw_step, description=raw_step))
            continue
        if not isinstance(raw_step, dict):
            continue
        action = str(raw_step.get("action") or "click").strip().lower().replace(" ", "_")
        action = ACTION_ALIASES.get(action, action)
        if action not in VALID_ACTIONS:
            action = "assert_visible" if "assert" in action or "verify" in action else "click"
        value = raw_step.get("value")
        steps.append(
            TestStep(
                action=action,  # type: ignore[arg-type]
                target=str(raw_step.get("target") or raw_step.get("selector") or "")[:300],
                value=None if value in (None, "", "null") else str(value)[:500],
                description=str(raw_step.get("description") or "")[:300],
            )
        )

    if not steps:
        raise ValueError("flow has no usable steps")

    category = CATEGORY_ALIASES.get(
        str(raw.get("category") or "happy_path").strip().lower().replace(" ", "_"),
        FlowCategory.HAPPY_PATH,
    )
    url = str(raw.get("url") or default_url).strip() or default_url

    # Guarantee the flow starts somewhere concrete; a plan whose first step is a
    # click on an unopened page is not executable.
    if steps[0].action != "goto":
        steps.insert(
            0,
            TestStep(action="goto", target=url, description="open the starting page for this flow"),
        )

    return TestFlow(
        id=str(raw.get("id") or f"F{index:03d}").strip()[:12] or f"F{index:03d}",
        name=str(raw.get("name") or f"Flow {index}")[:200],
        category=category,
        steps=steps[:20],
        expected_outcome=str(raw.get("expected_outcome") or "")[:500],
        url=url,
        business_hints=[str(h)[:60] for h in coerce_list(raw.get("business_hints"))][:8],
        requires_auth=bool(raw.get("requires_auth", False)),
    )


def renumber_flows(plan: TestPlan) -> TestPlan:
    """Force unique, ordered flow ids. Models reuse ids across a re-plan."""
    seen: set[str] = set()
    for position, flow in enumerate(plan.flows, start=1):
        candidate = flow.id or f"F{position:03d}"
        if candidate in seen:
            candidate = f"F{position:03d}"
        while candidate in seen:
            position += 1
            candidate = f"F{position:03d}"
        flow.id = candidate
        seen.add(candidate)
    return plan


def describe_plan(plan: TestPlan) -> str:
    """The one-line summary the decision log shows after planning."""
    counts = plan.category_counts()
    return (
        f"Planner found {len(plan.flows)} flows "
        f"({counts['happy_path']} happy, {counts['edge_case']} edge, "
        f"{counts['error_state']} error)"
    )


def flows_needing_auth(plan: TestPlan) -> Sequence[TestFlow]:
    return [f for f in plan.flows if f.requires_auth]
