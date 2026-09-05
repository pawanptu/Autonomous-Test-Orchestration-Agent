"""Streamlit front end: watch the agent think.

The central requirement here is not a form and a result page - it is the live
decision log. While a run is in flight this app polls
``GET /run/{id}/status`` every 1.5 seconds and renders every
:class:`graph.state.DecisionEvent` the agent has emitted, with its stage,
confidence, risk and whether a heal was auto-applied. A spinner would hide
exactly the thing worth showing.

This process never imports the agent and never calls an LLM. It is an HTTP
client for the FastAPI backend, which keeps the browser automation, the model
calls and the credential custody in one place.

Credentials typed into the login expander are POSTed once to start the run and
are then cleared from ``st.session_state``. Nothing echoes them back, and the
backend never returns them in any response.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

try:
    from config import get_settings

    _DEFAULTS = get_settings()
    DEFAULT_API = _DEFAULTS.api_base_url
    DEFAULT_TARGET = _DEFAULTS.default_target_url
except Exception:  # pragma: no cover - the UI must start even without config
    DEFAULT_API = "http://127.0.0.1:8000"
    DEFAULT_TARGET = "https://books.toscrape.com/"

POLL_SECONDS = 1.5
REQUEST_TIMEOUT = 20

STAGES = [
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

EVENT_ICON = {
    "start": "▶️",
    "progress": "·",
    "decision": "🧠",
    "replan": "🔁",
    "escalate": "🚨",
    "complete": "✅",
    "error": "❌",
}

# Session-state keys owned by the login widgets. They are wiped by
# clear_credential_state(), which must run *before* those widgets are
# instantiated on a given rerun: Streamlit raises StreamlitAPIException on any
# assignment to a key a live widget owns.
#
# NOTE: use comments, not bare string literals, for annotations in this file.
# Streamlit's "magic" renders a bare expression at module level as page content,
# so a docstring-style constant annotation would show up in the UI.
CRED_KEYS = ("cred_username", "cred_password", "cred_token", "cred_login_url")

RISK_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}
STATUS_ICON = {
    "passed": "✅",
    "healed": "🩹",
    "failed": "❌",
    "error": "💥",
    "skipped": "⏭️",
}

st.set_page_config(
    page_title="Autonomous Test Orchestration Agent",
    page_icon="🧪",
    layout="wide",
)


# ==========================================================================
# HTTP helpers - every call returns (ok, payload_or_error)
# ==========================================================================
def api_get(base: str, path: str, *, timeout: int = REQUEST_TIMEOUT) -> tuple[bool, Any]:
    try:
        response = requests.get(f"{base.rstrip('/')}{path}", timeout=timeout)
    except requests.RequestException as exc:
        return False, f"could not reach the API at {base}: {type(exc).__name__}"
    if response.status_code >= 400:
        return False, _error_text(response)
    try:
        return True, response.json()
    except ValueError:
        return True, response.text


def api_get_text(base: str, path: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base.rstrip('/')}{path}", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return False, f"could not reach the API: {type(exc).__name__}"
    if response.status_code >= 400:
        return False, _error_text(response)
    return True, response.text


def api_post(base: str, path: str, payload: dict[str, Any]) -> tuple[bool, Any]:
    try:
        response = requests.post(
            f"{base.rstrip('/')}{path}", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        return False, f"could not reach the API at {base}: {type(exc).__name__}"
    if response.status_code >= 400:
        return False, _error_text(response)
    try:
        return True, response.json()
    except ValueError:
        return False, "the API returned a non-JSON response"


def _error_text(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = body.get("detail", body)
    except ValueError:
        detail = response.text[:400]
    if isinstance(detail, list):  # pydantic validation errors
        detail = "; ".join(
            f"{'.'.join(str(p) for p in item.get('loc', []))}: {item.get('msg', '')}"
            for item in detail
        )
    return f"HTTP {response.status_code}: {detail}"


# ==========================================================================
# Sidebar
# ==========================================================================
def render_sidebar() -> str:
    st.sidebar.title("🧪 Test Orchestration Agent")
    base = st.sidebar.text_input("API base URL", value=st.session_state.get("api_base", DEFAULT_API))
    st.session_state["api_base"] = base

    if st.sidebar.button("Check connection", use_container_width=True):
        ok, payload = api_get(base, "/health", timeout=8)
        if not ok:
            st.sidebar.error(payload)
        else:
            st.session_state["health"] = payload

    health = st.session_state.get("health")
    if isinstance(health, dict):
        provider = health.get("llm_provider", "none")
        if health.get("llm_configured"):
            st.sidebar.success(f"API up · provider: **{provider}**")
        else:
            st.sidebar.error("API up, but no LLM provider is configured")
        if provider == "offline-stub":
            st.sidebar.warning(
                "LLM_OFFLINE_MODE is on. Responses come from a deterministic stub, "
                "not a model. Runs made now are for plumbing checks only."
            )
        if not health.get("playwright_ready", True):
            st.sidebar.warning(health.get("note") or "Playwright browsers are not installed.")
        models = health.get("models") or {}
        st.sidebar.caption(
            f"reasoning: `{models.get('reasoning', '?')}`\n\n"
            f"codegen: `{models.get('codegen', '?')}`"
        )
        flags = health.get("feature_flags") or {}
        with st.sidebar.expander("Feature flags"):
            for name, value in flags.items():
                st.write(f"{'✅' if value else '⬜'} `{name}`")
        st.sidebar.caption(f"Active runs: {health.get('active_runs', 0)}")

    st.sidebar.divider()
    st.sidebar.subheader("Security")
    st.sidebar.caption(
        "Use a **throwaway test account**. Credentials are held in memory for the "
        "duration of one run, are never logged, never written into a report, never "
        "embedded in a generated test, and are wiped when the run ends. Point the "
        "agent at staging - generation clicks and fills real forms."
    )
    return base


# ==========================================================================
# Submission form
# ==========================================================================
def render_form(base: str) -> None:
    st.title("Autonomous Test Orchestration Agent")
    st.caption(
        "Give it a URL. It crawls, plans, gates its own coverage, ranks business risk, "
        "writes Playwright tests against live selectors, runs them, heals or files bugs, "
        "diffs screenshots, and reports — with no human step in between."
    )

    with st.form("start_run", clear_on_submit=False):
        url = st.text_input("Target URL *", value=DEFAULT_TARGET, placeholder="https://example.com")
        intent = st.text_area(
            "What should it focus on? (optional)",
            placeholder="focus on checkout and authentication flows",
            height=80,
        )
        prd_file = st.file_uploader(
            "Product requirements document (optional, .txt or .md)", type=["txt", "md"]
        )

        with st.expander("Login (optional) — only if the target needs it"):
            st.caption(
                "Sent once to start the run, then cleared from this browser session. "
                "Never echoed back, never stored, never written to disk."
            )
            col_a, col_b = st.columns(2)
            username = col_a.text_input("Username or email", key=CRED_KEYS[0])
            password = col_b.text_input("Password", type="password", key=CRED_KEYS[1])
            col_c, col_d = st.columns(2)
            token = col_c.text_input("Bearer token", type="password", key=CRED_KEYS[2])
            login_url = col_d.text_input(
                "Login URL", key=CRED_KEYS[3], placeholder="auto-detect"
            )

        submitted = st.form_submit_button("🚀 Start autonomous run", use_container_width=True)

    if not submitted:
        return

    if not url.strip():
        st.error("A target URL is required.")
        return

    prd_text = None
    if prd_file is not None:
        try:
            prd_text = prd_file.getvalue().decode("utf-8", errors="replace")
        except Exception as exc:
            st.warning(f"Could not read the uploaded PRD ({exc}); continuing without it.")

    payload: dict[str, Any] = {"url": url.strip()}
    if intent.strip():
        payload["intent"] = intent.strip()
    if prd_text:
        payload["prd_text"] = prd_text
    credentials = {
        k: v
        for k, v in {
            "username": username or None,
            "password": password or None,
            "token": token or None,
            "login_url": login_url or None,
        }.items()
        if v
    }
    if credentials:
        payload["credentials"] = credentials

    ok, response = api_post(base, "/run", payload)
    if not ok:
        st.error(response)
        return

    st.session_state["run_id"] = response.get("run_id")
    st.session_state["polling"] = True
    # The credential widgets are wiped by clear_credential_state() at the top of
    # the next run, before any widget is instantiated. Clearing them here would
    # raise StreamlitAPIException, because Streamlit forbids assigning to a
    # session_state key that a live widget owns.
    st.rerun()


# ==========================================================================
# Live view
# ==========================================================================
def render_stage_strip(current: str, status: str) -> None:
    try:
        index = STAGES.index(current)
    except ValueError:
        index = -1
    chunks = []
    for position, stage in enumerate(STAGES):
        label = stage.replace("_", " ")
        if status in ("completed", "failed") or position < index:
            chunks.append(f"✅ {label}")
        elif position == index:
            chunks.append(f"**🔵 {label}**")
        else:
            chunks.append(f"◽ {label}")
    st.markdown(" &nbsp;→&nbsp; ".join(chunks))


def render_metrics(payload: dict[str, Any]) -> None:
    counts = payload.get("counts") or {}
    started = payload.get("started_at") or ""
    elapsed = ""
    try:
        from datetime import datetime, timezone

        begin = datetime.fromisoformat(started)
        end = (
            datetime.fromisoformat(payload["finished_at"])
            if payload.get("finished_at")
            else datetime.now(timezone.utc)
        )
        elapsed = f"{(end - begin).total_seconds():.0f}s"
    except Exception:
        elapsed = "—"

    row1 = st.columns(5)
    row1[0].metric("Stage", (payload.get("current_stage") or "—").replace("_", " "))
    row1[1].metric("Elapsed", elapsed)
    row1[2].metric("Re-plans", f"{payload.get('replan_count', 0)} / 2")
    row1[3].metric("Flows", counts.get("flows", 0))
    row1[4].metric("Tests", counts.get("tests_generated", 0))

    row2 = st.columns(5)
    row2[0].metric("Passed", counts.get("passed", 0))
    row2[1].metric("Failed", counts.get("failed", 0))
    row2[2].metric("Healed", counts.get("healed", 0))
    row2[3].metric("Bugs filed", counts.get("bugs_filed", 0))
    row2[4].metric("Visual regressions", counts.get("visual_regressions", 0))

    if payload.get("force_proceeded"):
        st.warning(
            "The re-plan budget was exhausted and the orchestrator force-proceeded. "
            "Remaining coverage gaps are listed in the report."
        )
    if payload.get("credentials_present"):
        login_ok = payload.get("login_ok")
        if login_ok is True:
            st.success("Authenticated — protected pages were in scope.")
        elif login_ok is False:
            st.error(
                "AUTH BLOCKED — login failed, so only publicly reachable pages were explored."
            )


def render_decision_log(events: list[dict[str, Any]]) -> None:
    st.subheader("Agent decision log")
    st.caption(
        "Every stage emits an event the moment it starts, decides and finishes. "
        "This is the agent's reasoning, live."
    )
    if not events:
        st.info("Waiting for the first event…")
        return

    with st.container(height=460):
        for event in events:
            icon = EVENT_ICON.get(event.get("event", ""), "·")
            stage = (event.get("stage") or "").replace("_", " ")
            summary = event.get("summary") or ""
            badges: list[str] = []
            confidence = event.get("confidence")
            if confidence is not None:
                badges.append(f"`conf {float(confidence):.2f}`")
            risk = event.get("risk")
            if risk:
                badges.append(f"{RISK_ICON.get(risk, '')} `{risk.upper()}`")
            if event.get("auto_applied") is True:
                badges.append("`AUTO-APPLIED`")
            elif event.get("auto_applied") is False:
                badges.append("`NOT APPLIED`")
            if event.get("needs_human_review"):
                badges.append("**`NEEDS HUMAN REVIEW`**")

            line = f"{icon} **{stage}** — {summary}"
            if badges:
                line += "  " + " ".join(badges)
            if event.get("event") == "error":
                st.error(line)
            elif event.get("event") in ("replan", "escalate"):
                st.warning(line)
            else:
                st.markdown(line)
            detail = (event.get("detail") or "").strip()
            if detail:
                st.caption(detail[:600])


def render_live(base: str, run_id: str) -> None:
    ok, payload = api_get(base, f"/run/{run_id}/status")
    if not ok:
        st.error(payload)
        if st.button("Stop polling"):
            st.session_state["polling"] = False
        return

    status = payload.get("status", "running")
    header = st.columns([3, 1])
    header[0].markdown(f"### Run `{run_id}` — **{status}**")
    if header[1].button("⏹ Cancel run", disabled=status not in ("queued", "running")):
        api_delete_ok, _ = api_get(base, f"/run/{run_id}/status")  # keep the record fresh
        try:
            requests.delete(f"{base.rstrip('/')}/run/{run_id}", timeout=10)
        except requests.RequestException as exc:
            st.error(f"Could not cancel: {type(exc).__name__}")
        st.session_state["polling"] = False
        st.rerun()

    render_stage_strip(payload.get("current_stage", ""), status)
    render_metrics(payload)
    st.divider()
    render_decision_log(payload.get("decision_log") or [])

    if status in ("completed", "failed", "cancelled"):
        st.session_state["polling"] = False
        if payload.get("error"):
            st.error(f"Run error: {payload['error']}")
        render_results(base, run_id)
        return

    time.sleep(POLL_SECONDS)
    st.rerun()


# ==========================================================================
# Final results
# ==========================================================================
def render_results(base: str, run_id: str) -> None:
    ok, report = api_get(base, f"/run/{run_id}/report")
    if not ok:
        st.warning(f"Report not available yet — {report}")
        return

    st.divider()
    st.header("Report")

    summary = report.get("executive_summary") or ""
    if summary:
        st.markdown(f"#### Executive summary\n{summary}")
    impact = report.get("business_impact") or ""
    if impact:
        st.success(f"**Business impact.** {impact}")

    _render_flow_table(report)
    _render_coverage(report)
    _render_review_queue(report)
    _render_healer_table(report)
    _render_visual(report)
    _render_bugs(base, run_id, report)
    _render_prd_and_radar(report)
    _render_limitations(report)
    _render_downloads(base, run_id)


def _render_flow_table(report: dict[str, Any]) -> None:
    rows = report.get("flows") or []
    st.subheader("Risk-ranked results")
    st.caption(
        "Ordered by business risk first, then by how bad the outcome was — not by flow index."
    )
    if not rows:
        st.info("No flows were executed.")
        return
    table = [
        {
            "Risk": f"{RISK_ICON.get(str(r.get('risk')), '')} {str(r.get('risk', '')).upper()}",
            "Flow": r.get("flow_name", ""),
            "Category": str(r.get("category", "")).replace("_", " "),
            "Status": f"{STATUS_ICON.get(str(r.get('status')), '')} {r.get('status', '')}",
            "Outcome": r.get("outcome_label", ""),
            "Duration": f"{float(r.get('duration_s', 0) or 0):.1f}s",
            "Bugs": ", ".join(r.get("bug_ids") or []),
        }
        for r in rows
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_coverage(report: dict[str, Any]) -> None:
    evaluation = report.get("coverage_evaluation") or {}
    gaps = report.get("coverage_gaps") or []
    with st.expander(
        f"Coverage gate — score {evaluation.get('score', 0)} · "
        f"{'PASSED' if evaluation.get('passed') else 'FAILED'} · "
        f"{report.get('replan_count', 0)} re-plan(s)"
    ):
        checks = evaluation.get("checks") or []
        if checks:
            st.dataframe(
                [
                    {
                        "": "✅" if c.get("satisfied") else "❌",
                        "ID": c.get("id"),
                        "Requirement": c.get("requirement"),
                        "Evidence": c.get("evidence"),
                    }
                    for c in checks
                ],
                use_container_width=True,
                hide_index=True,
            )
        if evaluation.get("rationale"):
            st.caption(evaluation["rationale"])
        if gaps:
            st.warning("**Remaining coverage gaps**\n\n" + "\n".join(f"- {g}" for g in gaps))


def _render_review_queue(report: dict[str, Any]) -> None:
    queue = report.get("needs_human_review") or []
    st.subheader(f"Needs human review ({len(queue)})")
    if not queue:
        st.caption("Nothing was left for a human: every finding was confidently classified.")
        return
    st.caption(
        "These failures scored **below the 0.60 confidence threshold**, so no patch was "
        "applied. They are queued with their evidence rather than silently changed."
    )
    for action in queue:
        with st.container(border=True):
            cols = st.columns([3, 1, 1])
            cols[0].markdown(f"**{action.get('flow_name') or action.get('flow_id')}**")
            cols[1].metric("Confidence", f"{float(action.get('confidence', 0)):.2f}")
            cols[2].markdown(f"`{action.get('classification', '')}`")
            st.write(action.get("rationale", ""))
            if action.get("patch_summary"):
                st.caption(action["patch_summary"])
            refs = action.get("evidence_refs") or []
            if refs:
                st.caption("Evidence: " + " · ".join(str(r) for r in refs))


def _render_healer_table(report: dict[str, Any]) -> None:
    actions = report.get("healer_actions") or []
    if not actions:
        return
    with st.expander(f"All healer actions ({len(actions)})"):
        st.dataframe(
            [
                {
                    "Flow": a.get("flow_name") or a.get("flow_id"),
                    "Classification": a.get("classification"),
                    "Confidence": round(float(a.get("confidence", 0) or 0), 2),
                    "Auto-applied": "✅" if a.get("auto_applied") else "—",
                    "Re-run": a.get("rerun_status") or "—",
                    "Action": a.get("action"),
                }
                for a in actions
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_visual(report: dict[str, Any]) -> None:
    findings = report.get("visual_findings") or []
    regressions = [f for f in findings if f.get("is_regression")]
    st.subheader(f"Visual regression ({len(regressions)} of {len(findings)} frames)")
    st.caption(
        "Pixel comparison against the stored baseline. Reported separately from "
        "functional failures: a flow can pass every assertion while the layout breaks."
    )
    if not findings:
        st.caption("No frames were compared.")
        return
    for finding in findings:
        if not finding.get("is_regression") and not finding.get("is_new_baseline"):
            continue
        label = (
            f"{'🔺' if finding.get('is_regression') else '🆕'} "
            f"{finding.get('flow_name') or finding.get('flow_id')} — "
            f"{float(finding.get('changed_ratio', 0)) * 100:.1f}% changed"
        )
        with st.expander(label, expanded=bool(finding.get("is_regression"))):
            st.caption(finding.get("note", ""))
            columns = st.columns(3)
            for column, key, caption in (
                (columns[0], "baseline_path", "Baseline"),
                (columns[1], "current_path", "Current"),
                (columns[2], "diff_path", "Diff"),
            ):
                path = finding.get(key)
                if path and Path(path).is_file():
                    column.image(str(path), caption=caption, use_container_width=True)
                else:
                    column.caption(f"{caption}: not available locally")


def _render_bugs(base: str, run_id: str, report: dict[str, Any]) -> None:
    bugs = report.get("packaged_bugs") or []
    st.subheader(f"Packaged bugs ({len(bugs)})")
    if not bugs:
        st.caption("No genuine application defects were confirmed in this run.")
        return
    st.caption(
        "Each is a distinct artifact on disk: a standalone repro script, a screenshot "
        "and a paste-ready ticket. No credentials appear in any of them."
    )
    for bug in bugs:
        bug_id = bug.get("bug_id", "")
        risk = str(bug.get("risk", "medium"))
        with st.expander(
            f"{RISK_ICON.get(risk, '')} **{bug_id}** — {bug.get('title', '')} "
            f"({bug.get('severity', 'major')})"
        ):
            ok, artifact = api_get(base, f"/run/{run_id}/bugs/{bug_id}")
            if not ok:
                st.error(artifact)
                continue
            st.markdown(artifact.get("description", ""))
            shot = artifact.get("screenshot_base64")
            if shot:
                try:
                    st.image(base64.b64decode(shot), caption="Failure screenshot")
                except Exception:
                    st.caption("Screenshot could not be decoded.")
            elif artifact.get("note"):
                st.caption(artifact["note"])

            columns = st.columns(2)
            if artifact.get("repro_script"):
                columns[0].download_button(
                    "⬇ repro.py",
                    artifact["repro_script"],
                    file_name=f"{bug_id}_repro.py",
                    mime="text/x-python",
                    use_container_width=True,
                    key=f"repro_{bug_id}",
                )
            if artifact.get("ticket_markdown"):
                columns[1].download_button(
                    "⬇ ticket.md",
                    artifact["ticket_markdown"],
                    file_name=f"{bug_id}_ticket.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"ticket_{bug_id}",
                )
            with st.popover("View repro script"):
                st.code(artifact.get("repro_script", ""), language="python")


def _render_prd_and_radar(report: dict[str, Any]) -> None:
    gaps = report.get("prd_gaps") or []
    if gaps:
        uncovered = [g for g in gaps if not g.get("covered")]
        with st.expander(f"PRD gap analysis — {len(uncovered)} of {len(gaps)} uncovered"):
            st.dataframe(
                [
                    {
                        "": "✅" if g.get("covered") else "❌",
                        "Requirement": g.get("requirement"),
                        "Best matching flow": g.get("best_match_flow") or "—",
                        "Similarity": round(float(g.get("similarity", 0) or 0), 2),
                    }
                    for g in gaps
                ],
                use_container_width=True,
                hide_index=True,
            )

    radar = report.get("regression_radar") or {}
    if radar and radar.get("enabled") is not False and not radar.get("first_run"):
        with st.expander(f"Regression radar — {radar.get('summary', '')}"):
            st.json(radar)


def _render_limitations(report: dict[str, Any]) -> None:
    limitations = report.get("limitations") or []
    errors = report.get("errors") or []
    if limitations:
        with st.expander("Limitations — what you must not conclude from this run", expanded=False):
            for item in limitations:
                st.markdown(f"- {item}")
    if errors:
        st.error("**Errors during the run**\n\n" + "\n".join(f"- {e}" for e in errors))


def _render_downloads(base: str, run_id: str) -> None:
    st.divider()
    columns = st.columns(3)
    ok_md, markdown = api_get_text(base, f"/run/{run_id}/report.md")
    if ok_md:
        columns[0].download_button(
            "⬇ report.md", markdown, file_name=f"{run_id}_report.md",
            mime="text/markdown", use_container_width=True,
        )
    ok_json, payload = api_get_text(base, f"/run/{run_id}/report")
    if ok_json:
        columns[1].download_button(
            "⬇ report.json", payload, file_name=f"{run_id}_report.json",
            mime="application/json", use_container_width=True,
        )
    ok_events, events = api_get_text(base, f"/run/{run_id}/events.jsonl")
    if ok_events:
        columns[2].download_button(
            "⬇ events.jsonl", events, file_name=f"{run_id}_events.jsonl",
            mime="application/x-ndjson", use_container_width=True,
        )


# ==========================================================================
# Entry point
# ==========================================================================
def clear_credential_state() -> None:
    """Wipe the login widgets' session state.

    Called at the top of every rerun in which a run is in progress - that is,
    on the path where the form is *not* rendered - so the assignment happens
    before any of those widgets exist. Once a run has been submitted the
    browser session has no further use for the values, and this is the only
    point at which Streamlit permits removing them.
    """
    for key in CRED_KEYS:
        st.session_state.pop(key, None)


def main() -> None:
    run_id = st.session_state.get("run_id")
    if run_id:
        clear_credential_state()

    base = render_sidebar()

    if not run_id:
        render_form(base)
        _render_recent(base)
        return

    if st.button("← Start another run"):
        for key in ("run_id", "polling"):
            st.session_state.pop(key, None)
        st.rerun()

    if st.session_state.get("polling"):
        render_live(base, run_id)
    else:
        ok, payload = api_get(base, f"/run/{run_id}/status")
        if ok:
            render_stage_strip(payload.get("current_stage", ""), payload.get("status", ""))
            render_metrics(payload)
            st.divider()
            render_decision_log(payload.get("decision_log") or [])
            render_results(base, run_id)
        else:
            st.error(payload)


def _render_recent(base: str) -> None:
    ok, runs = api_get(base, "/runs?limit=10", timeout=8)
    if not ok or not runs:
        return
    with st.expander("Recent runs"):
        for record in runs:
            columns = st.columns([3, 1, 1])
            columns[0].write(
                f"`{record.get('run_id')}` — {record.get('target_url')} "
                f"({record.get('status')})"
            )
            columns[1].write(f"{record.get('bug_count', 0)} bug(s)")
            if columns[2].button("Open", key=f"open_{record.get('run_id')}"):
                st.session_state["run_id"] = record.get("run_id")
                st.session_state["polling"] = record.get("status") in ("queued", "running")
                st.rerun()


main()
