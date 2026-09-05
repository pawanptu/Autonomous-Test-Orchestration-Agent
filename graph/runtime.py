"""Per-run runtime services: the event sink and the run context.

LangGraph state is data. Everything that is *not* data - the HTTP client, the
open log file, the run directory, the redaction context - lives in a
:class:`RunContext` held in a process-local registry keyed by ``run_id``.
Nodes look up their context by the ``run_id`` already present in state, which
keeps the serialisable state free of handles and secrets.

The :class:`EventSink` is the backbone of the live-visibility requirement. A
node calls ``ctx.emit(...)`` the instant something happens; the event is
appended to an in-memory list (read by ``GET /run/{id}/status``), written to
``reports/runs/<run_id>/events.jsonl`` and mirrored into the run log. Nothing
is buffered until the end of a stage.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

from config import Settings, get_settings, run_dir
from graph.state import DecisionEvent, EventKind, Stage, utcnow_iso
from logging_setup import get_logger
from security import SECRET_BOX, redact_secrets, redact_text

log = get_logger("aivor.runtime")


class EventSink:
    """Append-only, thread-safe decision log with immediate durability."""

    def __init__(self, run_id: str, events_path: Path) -> None:
        self.run_id = run_id
        self.events_path = events_path
        self._events: list[DecisionEvent] = []
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[DecisionEvent], None]] = []
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------
    def emit(
        self,
        stage: Stage,
        event: EventKind,
        summary: str,
        *,
        detail: str = "",
        confidence: float | None = None,
        risk: str | None = None,
        flow_id: str | None = None,
        auto_applied: bool | None = None,
        needs_human_review: bool = False,
    ) -> DecisionEvent:
        """Create, persist and broadcast one decision event.

        ``summary`` and ``detail`` are redacted before they are stored, so a
        node that accidentally interpolates a credential into a message still
        cannot leak it to the UI, the report or the JSONL file.
        """
        record = DecisionEvent(
            ts=utcnow_iso(),
            stage=stage,
            event=event,
            summary=redact_text(str(summary))[:500],
            detail=redact_text(str(detail))[:2000],
            confidence=confidence,
            risk=risk if risk in ("high", "medium", "low") else None,
            flow_id=flow_id,
            auto_applied=auto_applied,
            needs_human_review=needs_human_review,
        )
        with self._lock:
            self._events.append(record)
            try:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")
            except OSError as exc:  # pragma: no cover - disk full / permissions
                log.warning("could not append to events.jsonl: %s", exc)
            subscribers = list(self._subscribers)

        level = log.error if event == "error" else log.info
        level(
            "[%s] %s %s",
            record.stage,
            record.event,
            record.summary,
            extra={"run_id": self.run_id, "stage": record.stage},
        )
        for callback in subscribers:
            try:
                callback(record)
            except Exception:  # pragma: no cover - a bad subscriber must not break the run
                log.debug("event subscriber raised", exc_info=True)
        return record

    # -- reading -----------------------------------------------------------
    def snapshot(self) -> list[DecisionEvent]:
        with self._lock:
            return list(self._events)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self.snapshot()]

    def tail(self, n: int = 25) -> list[DecisionEvent]:
        with self._lock:
            return list(self._events[-n:])

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def subscribe(self, callback: Callable[[DecisionEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)


class RunContext:
    """Everything a node needs that is not serialisable state."""

    def __init__(self, run_id: str, settings: Settings | None = None) -> None:
        self.run_id = run_id
        self.settings = settings or get_settings()
        self.dir: Path = run_dir(run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "screenshots").mkdir(exist_ok=True)
        (self.dir / "bugs").mkdir(exist_ok=True)
        (self.dir / "generated_tests").mkdir(exist_ok=True)
        self.events = EventSink(run_id, self.dir / "events.jsonl")
        self.started_at = utcnow_iso()
        self.storage_state_path: Path | None = None
        self.llm: Any | None = None  # set by the orchestrator (llm.client.LLMClient)
        self.cancelled = threading.Event()
        self._scratch: dict[str, Any] = {}
        self.progress: dict[str, Any] = {
            "status": "queued",
            "current_stage": "orchestrator",
            "started_at": self.started_at,
            "finished_at": None,
            "replan_count": 0,
            "force_proceeded": False,
            "login_ok": None,
            "counts": {},
            "risk_classifications": [],
            "healer_actions": [],
            "visual_findings": [],
            "packaged_bugs": [],
            "error": None,
        }
        """Live, API-visible snapshot of the run.

        Nodes write into this as they go, so ``GET /run/{id}/status`` can report
        risk verdicts, heals and bugs *while the graph is still executing*
        rather than only once the final state is returned. It is deliberately
        plain JSON-able data and it never contains a credential.
        """

    # -- convenience -------------------------------------------------------
    def emit(self, stage: Stage, event: EventKind, summary: str, **kwargs: Any) -> DecisionEvent:
        return self.events.emit(stage, event, summary, **kwargs)

    @property
    def credentials(self):
        """Live credentials for this run, or ``None``. Never persist these."""
        return SECRET_BOX.get(self.run_id)

    @property
    def credentials_present(self) -> bool:
        return SECRET_BOX.present(self.run_id)

    def screenshot_path(self, name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        return self.dir / "screenshots" / safe

    def write_json(self, filename: str, payload: Any) -> Path:
        """Write a redacted JSON artifact into the run directory."""
        path = self.dir / filename
        path.write_text(
            json.dumps(redact_secrets(payload), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path

    def set_progress(self, **fields: Any) -> None:
        """Merge fields into the live snapshot, redacting on the way in."""
        for key, value in fields.items():
            self.progress[key] = redact_secrets(value)

    def scratch(self) -> dict[str, Any]:
        """Free-form per-run scratch space shared between nodes.

        Used for non-serialisable handoffs, e.g. the resolved selector map the
        Generator produces and the Healer wants to inspect.
        """
        return self._scratch

    def close(self) -> None:
        """Release per-run resources. Credentials are wiped by the caller."""
        if self.storage_state_path is not None:
            try:
                self.storage_state_path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                log.debug("could not remove storage state file", exc_info=True)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
_contexts: dict[str, RunContext] = {}
_contexts_lock = threading.RLock()


def create_context(run_id: str, settings: Settings | None = None) -> RunContext:
    ctx = RunContext(run_id, settings)
    with _contexts_lock:
        _contexts[run_id] = ctx
    return ctx


def get_context(run_id: str) -> RunContext:
    with _contexts_lock:
        ctx = _contexts.get(run_id)
    if ctx is None:
        raise KeyError(f"no RunContext registered for run_id={run_id!r}")
    return ctx


def try_get_context(run_id: str) -> RunContext | None:
    with _contexts_lock:
        return _contexts.get(run_id)


def drop_context(run_id: str) -> None:
    with _contexts_lock:
        ctx = _contexts.pop(run_id, None)
    if ctx is not None:
        ctx.close()


def active_run_ids() -> list[str]:
    with _contexts_lock:
        return list(_contexts)


def emit_many(ctx: RunContext, events: Iterable[tuple[Stage, EventKind, str]]) -> list[DecisionEvent]:
    """Emit a batch of simple events (used by the offline smoke path)."""
    return [ctx.emit(stage, kind, summary) for stage, kind, summary in events]
