"""In-memory run registry for the API layer.

Deliberately not a database. A run is a live, in-process activity with an open
browser and an open HTTP client attached; persisting its record would create
the illusion that a process restart could resume it. What *is* durable lives on
disk under ``reports/runs/<run_id>/`` - events, screenshots, generated tests,
bugs and the report - and is served from there.

Security boundary
-----------------
No credential ever enters this module. A record carries
``credentials_present: bool`` and nothing else; the values live only in
:data:`security.SECRET_BOX`, keyed by run id, and are wiped when the run ends.

Live status
-----------
The decision log and the incremental risk/heal/bug snapshots are read from the
run's :class:`graph.runtime.RunContext` rather than from the record, so
``GET /run/{id}/status`` reflects what the agent has done *right now* instead of
what it had done when a node last returned.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from graph.runtime import RunContext, try_get_context
from graph.state import FinalReport, RunStatus, utcnow_iso
from logging_setup import get_logger
from security import redact_secrets, sanitize_url

log = get_logger("aivor.store")

# Fields ``update()`` will accept. Anything else is ignored with a debug log,
# so a typo in a caller cannot silently invent state.
MUTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "status",
        "current_stage",
        "finished_at",
        "login_ok",
        "replan_count",
        "force_proceeded",
        "error",
        "final_report",
        "risk_classifications",
        "healer_actions",
        "visual_findings",
        "packaged_bugs",
        "counts",
    }
)


@dataclass
class RunRecord:
    """Everything the API knows about one run, minus anything secret."""

    run_id: str
    target_url: str
    status: RunStatus = "queued"
    current_stage: str = "orchestrator"
    started_at: str = field(default_factory=utcnow_iso)
    finished_at: str | None = None
    credentials_present: bool = False
    login_ok: bool | None = None
    replan_count: int = 0
    force_proceeded: bool = False
    error: str | None = None
    final_report: FinalReport | None = None

    counts: dict[str, int] = field(default_factory=dict)
    risk_classifications: list[dict[str, Any]] = field(default_factory=list)
    healer_actions: list[dict[str, Any]] = field(default_factory=list)
    visual_findings: list[dict[str, Any]] = field(default_factory=list)
    packaged_bugs: list[dict[str, Any]] = field(default_factory=list)

    task: Any = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def summary(self) -> dict[str, Any]:
        """Fields for the ``/runs`` listing. Never includes a credential."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "target_url": sanitize_url(self.target_url),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "credentials_present": self.credentials_present,
            "replan_count": self.replan_count,
            "force_proceeded": self.force_proceeded,
            "bug_count": len(self.packaged_bugs),
            "error": self.error,
        }


class RunStore:
    """Thread-safe registry of runs. One instance per process (:data:`STORE`)."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------
    def create(self, run_id: str, target_url: str, credentials_present: bool = False) -> RunRecord:
        record = RunRecord(
            run_id=run_id,
            target_url=target_url,
            credentials_present=credentials_present,
            status="queued",
        )
        with self._lock:
            self._records[run_id] = record
        log.info("run %s registered (credentials_present=%s)", run_id, credentials_present)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def require(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def update(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                log.debug("update for unknown run %s ignored", run_id)
                return
            for key, value in fields.items():
                if key in MUTABLE_FIELDS:
                    setattr(record, key, value)
                else:
                    log.debug("ignoring unknown record field %r", key)

    def set_task(self, run_id: str, task: Any) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is not None:
                record.task = task

    def cancel(self, run_id: str) -> bool:
        """Best-effort cancellation. Returns True if a task was signalled."""
        with self._lock:
            record = self._records.get(run_id)
            if record is None or record.task is None or record.terminal:
                return False
            task = record.task
        cancelled = bool(task.cancel())
        if cancelled:
            self.update(
                run_id,
                status="cancelled",
                finished_at=utcnow_iso(),
                error="the run was cancelled by an API request",
            )
        return cancelled

    # -- queries -----------------------------------------------------------
    def list_runs(self, limit: int = 25) -> list[RunRecord]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r.started_at, reverse=True)
        return records[: max(1, limit)]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if not r.terminal)

    def prune(self, max_runs: int = 50) -> None:
        """Drop the oldest terminal records once the registry grows large."""
        with self._lock:
            if len(self._records) <= max_runs:
                return
            terminal = sorted(
                (r for r in self._records.values() if r.terminal),
                key=lambda r: r.finished_at or r.started_at,
            )
            for record in terminal[: len(self._records) - max_runs]:
                self._records.pop(record.run_id, None)
                log.debug("pruned run record %s", record.run_id)

    async def acreate(self, run_id: str, target_url: str, credentials_present: bool = False) -> RunRecord:
        async with self._async_lock:
            return self.create(run_id, target_url, credentials_present)


STORE = RunStore()
"""Process-wide run registry. Import this; do not construct another."""


def snapshot_for_status(record: RunRecord, ctx: RunContext | None = None) -> dict[str, Any]:
    """Merge the record with the run's live context into a status payload.

    The context wins wherever it has fresher information, because it is written
    by the nodes as they execute. Everything is passed through
    :func:`security.redact_secrets` on the way out: this payload is the single
    most likely place for an accidental leak, since it carries model-authored
    prose and page-derived text.
    """
    live = ctx.progress if ctx is not None else {}
    events = ctx.events.snapshot() if ctx is not None else []

    def pick(key: str, fallback: Any) -> Any:
        value = live.get(key)
        if value in (None, [], {}) and key != "error":
            return fallback
        return value if value is not None else fallback

    status = record.status
    live_status = live.get("status")
    if live_status and not record.terminal:
        status = live_status

    payload = {
        "run_id": record.run_id,
        "status": status,
        "current_stage": pick("current_stage", record.current_stage),
        "started_at": record.started_at,
        "finished_at": record.finished_at or live.get("finished_at"),
        "target_url": sanitize_url(record.target_url),
        "credentials_present": record.credentials_present,
        "login_ok": (
            record.login_ok if record.login_ok is not None else live.get("login_ok")
        ),
        "replan_count": int(pick("replan_count", record.replan_count) or 0),
        "force_proceeded": bool(pick("force_proceeded", record.force_proceeded)),
        "counts": pick("counts", record.counts) or {},
        "decision_log": [event.model_dump(mode="json") for event in events],
        "risk_classifications": pick("risk_classifications", record.risk_classifications) or [],
        "healer_actions": pick("healer_actions", record.healer_actions) or [],
        "visual_findings": pick("visual_findings", record.visual_findings) or [],
        "packaged_bugs": pick("packaged_bugs", record.packaged_bugs) or [],
        "error": record.error or live.get("error"),
        "event_count": len(events),
    }
    return redact_secrets(payload)


def live_context(run_id: str) -> RunContext | None:
    """The run's context if it is still registered, else ``None``."""
    return try_get_context(run_id)


def counts_from_report(report: FinalReport | None) -> dict[str, int]:
    """Totals for a finished run, for the listing endpoint."""
    if report is None:
        return {}
    return {k: v for k, v in (report.totals or {}).items() if isinstance(v, int)}


def iter_terminal(records: Iterable[RunRecord]) -> list[RunRecord]:
    return [record for record in records if record.terminal]
