"""FastAPI surface for the Autonomous Test Orchestration Agent.

``POST /run`` with a URL is the whole interface: everything after that is
autonomous. The rest of the endpoints exist so a human can *watch* -
``GET /run/{id}/status`` returns the full decision log accumulated so far, which
is what makes the Streamlit UI a live view of the agent's reasoning rather than
a spinner.

Credential handling
-------------------
``POST /run`` is the only endpoint that accepts credentials, and this module is
the only place they enter the process. They go straight into
:data:`security.SECRET_BOX` keyed by run id, are never written to the store,
never returned by any endpoint, and are wiped by the orchestrator when the run
ends (and again on shutdown, and again on cancellation). Every outbound payload
passes through :func:`security.redact_secrets`.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from api.models import (
    BugArtifactResponse,
    BugSummaryOut,
    ErrorResponse,
    HealthResponse,
    RunAccepted,
    RunRequest,
    RunStatusResponse,
    RunSummaryOut,
)
from api.store import STORE, live_context, snapshot_for_status
from config import ensure_dirs, get_settings, run_dir
from graph.state import new_run_id, utcnow_iso
from logging_setup import configure_logging, get_logger
from safe_actions import SafetyPolicy, parse_categories
from security import SECRET_BOX, Credentials, redact_secrets, redact_text, sanitize_url
from target_policy import TargetPolicy, evaluate_target, log_decision

log = get_logger("aivor.api")

VERSION = "1.0.0"
MAX_SCREENSHOT_B64_BYTES = 4 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    ensure_dirs()
    log.info(
        "API ready on %s:%s (llm_configured=%s, offline=%s)",
        settings.api_host,
        settings.api_port,
        bool(settings.groq_api_key),
        settings.llm_offline_mode,
    )
    try:
        yield
    finally:
        for record in STORE.list_runs(limit=200):
            if not record.terminal and record.task is not None:
                record.task.cancel()
        # Nothing secret survives the process, even on an abrupt shutdown.
        SECRET_BOX.wipe_all()
        log.info("API shut down; secret box wiped")


app = FastAPI(
    title="Autonomous Test Orchestration Agent",
    description=(
        "Give it a URL. It crawls, plans, gates its own coverage, ranks risk, "
        "generates Playwright tests with live selector validation, runs them, "
        "heals or files bugs, diffs screenshots, and reports - with no human "
        "step in between."
    ),
    version=VERSION,
    lifespan=lifespan,
)

# Local-demo CORS. The Streamlit UI runs on a different port on the same
# machine; this is not intended to be exposed publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Errors
# ==========================================================================
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail=redact_text(f"{type(exc).__name__}: {exc}")[:500]
        ).model_dump(),
    )


def _require_record(run_id: str) -> Any:
    record = STORE.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id {run_id!r}")
    return record


# ==========================================================================
# Run submission
# ==========================================================================
@app.post("/run", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
async def start_run(request: RunRequest) -> RunAccepted:
    """Start an autonomous run. Returns immediately with a run id.

    The URL is the only required input. Credentials, an intent sentence and a
    PRD are all optional and all change what the Planner does.
    """
    settings = get_settings()

    if not settings.llm_available:
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM provider is configured. Put a free Groq key in .env as "
                "GROQ_API_KEY (https://console.groq.com, no card required), or set "
                "LLM_OFFLINE_MODE=true to smoke-test the plumbing without a model."
            ),
        )
    # Full target admission, including name resolution. The request validator
    # already applied the syntactic half; this is the part that needs DNS, run
    # off the event loop so a slow resolver cannot stall the whole service.
    #
    # A client may ask for the private-target override but can never grant it:
    # the request is intersected with the server's own policy, so an exposed
    # instance cannot be turned into an SSRF proxy by anyone who can POST to it.
    server_allows_private = settings.allow_private_targets
    policy = TargetPolicy(
        allow_private=server_allows_private and request.allow_private_target,
        allow_insecure_tls=settings.allow_insecure_tls,
        allowlist=tuple(settings.target_allowlist),
        resolve_dns=settings.target_resolve_dns,
    )
    decision = await asyncio.to_thread(evaluate_target, request.url, policy)
    log_decision(decision, context="POST /run")
    if not decision.allowed:
        if (
            decision.category in ("loopback", "private", "link-local")
            and server_allows_private
            and not request.allow_private_target
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{decision.detail} This service permits private targets, but the "
                    'request did not ask for one: resend with "allow_private_target": true.'
                ),
            )
        raise HTTPException(status_code=400, detail=decision.detail)

    # Destructive-action authorisation is intersected the same way.
    requested = parse_categories(request.authorize_destructive)
    server_authorized = SafetyPolicy.from_settings(settings).authorized
    granted = requested & server_authorized
    refused = sorted(c.value for c in (requested - server_authorized))
    if refused:
        raise HTTPException(
            status_code=400,
            detail=(
                f"this service does not authorise the destructive categor(y/ies) {refused}. "
                "Add them to AUTHORIZED_DESTRUCTIVE_ACTIONS on the server, and only against "
                "a throwaway environment."
            ),
        )

    if STORE.active_count() >= settings.max_concurrent_runs:
        raise HTTPException(
            status_code=429,
            detail=(
                f"{STORE.active_count()} run(s) already in flight and the limit is "
                f"{settings.max_concurrent_runs}. Each run drives a real browser; "
                "raise MAX_CONCURRENT_RUNS only if the machine can take it."
            ),
        )

    run_id = new_run_id()
    credentials_present = False

    if request.credentials is not None and request.credentials.present():
        # The one and only door credentials come through.
        SECRET_BOX.put(
            run_id,
            Credentials(
                username=request.credentials.username,
                password=request.credentials.password,
                token=request.credentials.token,
                login_url=request.credentials.login_url,
            ),
        )
        credentials_present = True

    STORE.create(run_id, request.url, credentials_present=credentials_present)
    STORE.prune(max_runs=50)

    async def _runner() -> None:
        # Imported lazily: the agent stack pulls in Playwright and LangGraph,
        # and a circular import would otherwise form via the orchestrator.
        from agents.orchestrator import execute_run

        try:
            STORE.update(run_id, status="running")
            report = await execute_run(
                run_id=run_id,
                target_url=request.url,
                user_intent=request.intent,
                prd_text=request.prd_text,
            )
            ctx = live_context(run_id)
            STORE.update(
                run_id,
                status="failed" if report.status == "failed" else "completed",
                current_stage="report",
                finished_at=utcnow_iso(),
                final_report=report,
                counts={k: v for k, v in report.totals.items() if isinstance(v, int)},
                login_ok=report.login_ok,
                replan_count=report.replan_count,
                force_proceeded=report.force_proceeded,
                error=(report.errors[0] if report.errors else None),
                risk_classifications=(ctx.progress.get("risk_classifications") if ctx else []) or [],
                healer_actions=[a.model_dump(mode="json") for a in report.healer_actions],
                visual_findings=[v.model_dump(mode="json") for v in report.visual_findings],
                packaged_bugs=[b.model_dump(mode="json") for b in report.packaged_bugs],
            )
        except asyncio.CancelledError:
            STORE.update(
                run_id,
                status="cancelled",
                finished_at=utcnow_iso(),
                error="the run was cancelled",
            )
            raise
        except Exception as exc:  # pragma: no cover - execute_run already guards
            log.error("run %s crashed: %s", run_id, exc, exc_info=True)
            STORE.update(
                run_id,
                status="failed",
                finished_at=utcnow_iso(),
                error=redact_text(f"{type(exc).__name__}: {exc}")[:500],
            )
        finally:
            SECRET_BOX.wipe(run_id)

    STORE.set_task(run_id, asyncio.create_task(_runner(), name=f"run-{run_id}"))

    return RunAccepted(
        run_id=run_id,
        status="queued",
        message=(
            "Run accepted. Poll the status URL for the live decision log; the agent "
            "needs no further input."
        ),
        status_url=f"/run/{run_id}/status",
        report_url=f"/run/{run_id}/report",
    )


# ==========================================================================
# Live status
# ==========================================================================
@app.get("/run/{run_id}/status", response_model=RunStatusResponse)
async def get_status(run_id: str) -> RunStatusResponse:
    """Current stage plus the full decision log so far. Never any credential."""
    record = _require_record(run_id)
    payload = snapshot_for_status(record, live_context(run_id))
    return RunStatusResponse.model_validate(payload)


@app.get("/run/{run_id}/events.jsonl", response_class=PlainTextResponse)
async def get_events(run_id: str) -> PlainTextResponse:
    """Raw append-only event stream, exactly as written to disk."""
    _require_record(run_id)
    path = run_dir(run_id) / "events.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no events have been written yet")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="application/x-ndjson")


@app.get("/runs", response_model=list[RunSummaryOut])
async def list_runs(limit: int = 25) -> list[RunSummaryOut]:
    return [RunSummaryOut.model_validate(r.summary()) for r in STORE.list_runs(limit=limit)]


@app.delete("/run/{run_id}")
async def cancel_run(run_id: str) -> dict[str, Any]:
    """Best-effort cancellation. Wipes the run's credentials immediately."""
    record = _require_record(run_id)
    if record.terminal:
        return {"run_id": run_id, "cancelled": False, "status": record.status}
    cancelled = STORE.cancel(run_id)
    SECRET_BOX.wipe(run_id)
    return {"run_id": run_id, "cancelled": cancelled, "status": STORE.require(run_id).status}


# ==========================================================================
# Report
# ==========================================================================
@app.get("/run/{run_id}/report")
async def get_report(run_id: str) -> dict[str, Any]:
    """The final JSON report, ordered by risk, plus links to the rendered forms."""
    record = _require_record(run_id)
    directory = run_dir(run_id)
    json_path = directory / "report.json"

    if record.final_report is None and not json_path.exists():
        if not record.terminal:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} is still {record.status} (stage: {record.current_stage}). "
                    f"Poll /run/{run_id}/status until it completes."
                ),
            )
        raise HTTPException(status_code=404, detail="no report was produced for this run")

    if record.final_report is not None:
        payload: dict[str, Any] = record.final_report.model_dump(mode="json")
    else:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"report on disk is unreadable: {exc}")

    payload["markdown_url"] = f"/run/{run_id}/report.md"
    payload["html_url"] = f"/run/{run_id}/report.html"
    payload["events_url"] = f"/run/{run_id}/events.jsonl"
    return redact_secrets(payload)


@app.get("/run/{run_id}/report.md", response_class=PlainTextResponse)
async def get_report_markdown(run_id: str) -> PlainTextResponse:
    _require_record(run_id)
    return PlainTextResponse(_read_artifact(run_id, "report.md"), media_type="text/markdown")


@app.get("/run/{run_id}/report.html", response_class=HTMLResponse)
async def get_report_html(run_id: str) -> HTMLResponse:
    _require_record(run_id)
    return HTMLResponse(_read_artifact(run_id, "report.html"))


def _read_artifact(run_id: str, filename: str) -> str:
    path = run_dir(run_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} has not been written yet")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not read {filename}: {exc}")


# ==========================================================================
# Bugs
# ==========================================================================
@app.get("/run/{run_id}/bugs", response_model=list[BugSummaryOut])
async def list_bugs(run_id: str) -> list[BugSummaryOut]:
    """Every packaged defect for a run, highest risk first."""
    _require_record(run_id)
    summaries: list[BugSummaryOut] = []
    for bug in _load_bugs(run_id):
        directory = Path(bug.get("directory") or "")
        summaries.append(
            BugSummaryOut(
                bug_id=bug.get("bug_id", "?"),
                flow_id=bug.get("flow_id", ""),
                flow_name=bug.get("flow_name", ""),
                title=bug.get("title", ""),
                classification=str(bug.get("classification", "")),
                confidence=float(bug.get("confidence", 0.0) or 0.0),
                risk=str(bug.get("risk", "medium")),
                severity=str(bug.get("severity", "major")),
                labels=list(bug.get("labels") or []),
                has_screenshot=bool(bug.get("screenshot_path"))
                or (directory / "screenshot.png").exists(),
                has_repro_script=bool(bug.get("repro_script_path"))
                or (directory / "repro.py").exists(),
                created_at=str(bug.get("created_at", "")),
                detail_url=f"/run/{run_id}/bugs/{bug.get('bug_id', '')}",
            )
        )
    return summaries


@app.get("/run/{run_id}/bugs/{bug_id}", response_model=BugArtifactResponse)
async def get_bug(run_id: str, bug_id: str) -> BugArtifactResponse:
    """The complete packaged artifact: ticket, repro script and screenshot."""
    _require_record(run_id)
    bug = next((b for b in _load_bugs(run_id) if b.get("bug_id") == bug_id), None)
    if bug is None:
        raise HTTPException(status_code=404, detail=f"unknown bug_id {bug_id!r} for run {run_id}")

    directory = Path(bug.get("directory") or (run_dir(run_id) / "bugs" / bug_id))
    repro = _safe_read(Path(bug.get("repro_script_path") or (directory / "repro.py")))
    ticket = _safe_read(Path(bug.get("ticket_path") or (directory / "ticket.md")))

    note = ""
    screenshot_b64: str | None = None
    screenshot_path = bug.get("screenshot_path") or str(directory / "screenshot.png")
    shot = Path(screenshot_path)
    if shot.exists():
        size = shot.stat().st_size
        if size <= MAX_SCREENSHOT_B64_BYTES:
            try:
                screenshot_b64 = base64.b64encode(shot.read_bytes()).decode("ascii")
            except OSError as exc:
                note = f"screenshot could not be read: {exc}"
        else:
            note = (
                f"screenshot is {size // 1024} KB, above the {MAX_SCREENSHOT_B64_BYTES // 1024} KB "
                "inline limit; read it from screenshot_path instead"
            )
    else:
        screenshot_path = None
        note = "no screenshot was captured for this defect"

    return BugArtifactResponse(
        bug_id=bug_id,
        flow_id=bug.get("flow_id", ""),
        title=bug.get("title", ""),
        description=bug.get("description", ""),
        classification=str(bug.get("classification", "")),
        confidence=float(bug.get("confidence", 0.0) or 0.0),
        risk=str(bug.get("risk", "medium")),
        severity=str(bug.get("severity", "major")),
        steps_to_reproduce=list(bug.get("steps_to_reproduce") or []),
        expected=bug.get("expected", ""),
        actual=bug.get("actual", ""),
        labels=list(bug.get("labels") or []),
        repro_script=repro,
        screenshot_base64=screenshot_b64,
        screenshot_path=screenshot_path,
        ticket_markdown=ticket,
        created_at=str(bug.get("created_at", "")),
        note=note,
    )


def _load_bugs(run_id: str) -> list[dict[str, Any]]:
    """Bugs from the in-memory record if present, else from disk."""
    record = STORE.get(run_id)
    if record is not None and record.packaged_bugs:
        return list(record.packaged_bugs)

    bugs: list[dict[str, Any]] = []
    bugs_dir = run_dir(run_id) / "bugs"
    if not bugs_dir.is_dir():
        return bugs
    for child in sorted(bugs_dir.iterdir()):
        manifest = child / "bug.json"
        if not manifest.is_file():
            continue
        try:
            bugs.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skipping unreadable bug manifest %s: %s", manifest, exc)
    return bugs


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError as exc:  # pragma: no cover
        log.warning("could not read %s: %s", path, exc)
        return ""


# ==========================================================================
# Health
# ==========================================================================
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Cheap readiness probe. Never launches a browser."""
    settings = get_settings()
    playwright_ready, note = _playwright_ready()

    provider = "none"
    if settings.llm_offline_mode:
        provider = "offline-stub"
    elif settings.groq_api_key:
        provider = "groq"

    return HealthResponse(
        status="ok",
        version=VERSION,
        llm_configured=settings.llm_available,
        llm_provider=provider,
        models={
            "reasoning": settings.model_reasoning,
            "codegen": settings.model_codegen,
        },
        feature_flags=settings.feature_flags(),
        active_runs=STORE.active_count(),
        playwright_ready=playwright_ready,
        note=note,
    )


def _playwright_ready() -> tuple[bool, str]:
    """Check the package imports and a browser bundle exists on disk.

    Deliberately does not launch anything: a health probe that starts Chromium
    would take seconds and could exhaust memory under polling.
    """
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False, "playwright is not installed: pip install -r requirements.txt"

    import os

    candidates = [
        Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")) if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") else None,
        Path.home() / "AppData" / "Local" / "ms-playwright",
        Path.home() / ".cache" / "ms-playwright",
        Path.home() / "Library" / "Caches" / "ms-playwright",
    ]
    for candidate in candidates:
        if candidate and candidate.is_dir() and any(candidate.glob("chromium*")):
            return True, ""
    return False, "no Chromium bundle found: run `playwright install chromium`"


@app.get("/")
async def root() -> dict[str, Any]:
    settings = get_settings()
    return {
        "name": "Autonomous Test Orchestration Agent",
        "version": VERSION,
        "docs": "/docs",
        "start_a_run": {
            "method": "POST",
            "path": "/run",
            "minimal_body": {"url": sanitize_url(settings.default_target_url)},
        },
        "endpoints": [
            "POST   /run",
            "GET    /run/{id}/status",
            "GET    /run/{id}/report",
            "GET    /run/{id}/report.md",
            "GET    /run/{id}/report.html",
            "GET    /run/{id}/bugs",
            "GET    /run/{id}/bugs/{bug_id}",
            "GET    /run/{id}/events.jsonl",
            "DELETE /run/{id}",
            "GET    /runs",
            "GET    /health",
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    cfg = get_settings()
    uvicorn.run("api.app:app", host=cfg.api_host, port=cfg.api_port, reload=False)
