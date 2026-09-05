"""Cross-run regression radar, behind ``ENABLE_REGRESSION_RADAR``.

A single run can only tell you what is broken *now*. The sentence a team
actually acts on is "three flows changed since the last run, and checkout went
from passing to failing". That requires memory, so this module keeps a compact
per-target history on disk and diffs the current run against the most recent
previous one.

What is stored, and what is deliberately not
--------------------------------------------
Only what a diff needs: per flow its name, category, risk and final status,
plus a handful of counts. No DOM, no screenshots, no error text, no URLs beyond
the sanitised target. Two reasons: the file is read on every run so it must stay
small, and the less that is written the less there is to leak. The whole payload
still passes through :func:`security.redact_secrets`, and the target is stored
via :func:`security.sanitize_url`, so a history file can never contain
``https://user:pass@host``.

Crash safety
------------
The history file is the only piece of cross-run state the agent owns; a
half-written one would poison every later run. Writes therefore go to a
temporary file in the same directory, are flushed and fsynced, and are then
moved into place with :func:`os.replace`, which is atomic on Windows and POSIX
alike. A corrupt or unreadable file is treated as "no history" rather than as an
error - a regression radar is a nice-to-have and must never fail a run.

Known limitation
----------------
History is keyed by host, as ``reports/baselines/_radar/<host>.json``, so two
different applications served from the same host share one file. Each entry
records its own sanitised target and :func:`record` prefers the most recent
entry with a matching target, falling back to the most recent entry overall;
the entry actually compared against is reported in ``compared_with``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from config import BASELINES_DIR
from graph.state import RiskClassification, TestPlan, TestResult, utcnow_iso
from logging_setup import get_logger
from security import redact_secrets, redact_text, sanitize_url

log = get_logger("aivor.radar")

RADAR_DIRNAME: str = "_radar"
HISTORY_VERSION: int = 1
DEFAULT_MAX_HISTORY: int = 20

NOT_RUN: str = "not_run"
"""Status recorded for a planned flow that never produced a result, so that a
flow disappearing from execution is visible as a change rather than as a
removal."""

FAILING_STATUSES: frozenset[str] = frozenset({"failed", "error"})
PASSING_STATUSES: frozenset[str] = frozenset({"passed", "healed"})

_SAFE_SLUG_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789.-"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _host_slug(target_url: str) -> str:
    """Filesystem-safe host label for ``target_url``.

    The URL is sanitised first, so any embedded userinfo is gone before the
    hostname is read. A URL with no parseable host degrades to a short digest of
    the sanitised URL, which keeps distinct targets in distinct files instead of
    piling them all into one.
    """
    clean = sanitize_url(target_url or "")
    host = ""
    port: int | None = None
    try:
        parts = urlsplit(clean if "//" in clean else f"//{clean}")
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        host = ""

    if host:
        slug = "".join(ch if ch in _SAFE_SLUG_CHARS else "_" for ch in host).strip("._-")
        if port:
            slug = f"{slug}_{port}"
    else:
        digest = hashlib.sha1(clean.encode("utf-8", "replace")).hexdigest()[:12]
        slug = f"target-{digest}"
    return slug[:80] or "unknown-target"


def history_path(target_url: str) -> Path:
    """Where this target's run history lives."""
    return BASELINES_DIR / RADAR_DIRNAME / f"{_host_slug(target_url)}.json"


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def load_history(target_url: str) -> dict[str, Any]:
    """Load the stored history, or an empty one.

    Never raises and never propagates a corrupt file: a missing, unreadable or
    malformed history is indistinguishable from "this is the first run", which
    is exactly how the caller should treat it. The failure is logged and noted
    in the returned dict so the report can say so honestly.
    """
    fallback: dict[str, Any] = {
        "version": HISTORY_VERSION,
        "target": sanitize_url(target_url or ""),
        "runs": [],
    }
    path = history_path(target_url)
    if not path.exists():
        return fallback

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("regression radar history at %s is unreadable (%s); starting fresh",
                    path.name, exc)
        fallback["note"] = "the previous history file could not be read and was ignored"
        return fallback

    if not isinstance(raw, dict):
        fallback["note"] = "the previous history file had an unexpected shape and was ignored"
        return fallback

    runs = raw.get("runs")
    return {
        "version": int(raw.get("version") or HISTORY_VERSION),
        "target": str(raw.get("target") or fallback["target"]),
        "runs": [entry for entry in runs if isinstance(entry, dict)] if isinstance(runs, list)
        else [],
    }


# --------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------
def _enum_value(value: Any, default: str = "") -> str:
    """Value of an enum member, or the string form of a plain value."""
    if value is None:
        return default
    return str(getattr(value, "value", value))


def snapshot(
    plan: TestPlan | None,
    results: Sequence[TestResult],
    classifications: Sequence[RiskClassification],
) -> dict[str, Any]:
    """Compact, diffable picture of one run.

    Keyed by flow id because that is the only identifier stable across runs -
    flow *names* are model output and drift between plans. Flows that appear in
    the results but not in the plan are still recorded, so a defensive gap in an
    upstream node cannot silently shrink the history.
    """
    risk_by_id: dict[str, str] = {}
    for item in classifications or ():
        try:
            risk_by_id[item.flow_id] = _enum_value(item.risk, "medium")
        except AttributeError:  # pragma: no cover - defensive against a loose dict
            log.debug("skipping a malformed risk classification", exc_info=True)

    status_by_id: dict[str, str] = {}
    name_by_id: dict[str, str] = {}
    for result in results or ():
        try:
            status_by_id[result.flow_id] = _enum_value(result.status, "error")
            if result.flow_name:
                name_by_id[result.flow_id] = result.flow_name
        except AttributeError:  # pragma: no cover - defensive
            log.debug("skipping a malformed test result", exc_info=True)

    flows: dict[str, dict[str, str]] = {}
    for flow in (plan.flows if plan is not None else []):
        try:
            flows[flow.id] = {
                "name": redact_text(flow.name or flow.id),
                "category": _enum_value(flow.category, "happy_path"),
                "risk": risk_by_id.get(flow.id, "medium"),
                "status": status_by_id.get(flow.id, NOT_RUN),
            }
        except AttributeError:  # pragma: no cover - defensive
            log.debug("skipping a malformed flow while snapshotting", exc_info=True)

    for flow_id, status in status_by_id.items():
        if flow_id in flows:
            continue
        flows[flow_id] = {
            "name": redact_text(name_by_id.get(flow_id, flow_id)),
            "category": "unknown",
            "risk": risk_by_id.get(flow_id, "medium"),
            "status": status,
        }

    counts: dict[str, int] = {
        "flows": len(flows),
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "healed": 0,
        NOT_RUN: 0,
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
    }
    for entry in flows.values():
        status_key = entry.get("status", "")
        if status_key in counts:
            counts[status_key] += 1
        risk_key = f"{entry.get('risk', 'medium')}_risk"
        if risk_key in counts:
            counts[risk_key] += 1

    return {"flows": flows, "counts": counts}


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def _flow_ref(flow_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "flow_id": flow_id,
        "name": entry.get("name", flow_id),
        "risk": entry.get("risk", "medium"),
        "status": entry.get("status", NOT_RUN),
    }


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def compare(previous: dict | None, current: dict) -> dict[str, Any]:
    """Diff two snapshots into the sentence a human wants to read.

    A previous snapshot that is missing or holds no flows is reported as
    ``first_run``: calling every flow "added" on the first run would be
    technically true and completely useless. ``newly_failing`` and
    ``newly_passing`` are restricted to flows present in *both* runs, because a
    brand new flow has no previous state to have changed from; those show up
    under ``flows_added`` instead.
    """
    current_flows = current.get("flows") if isinstance(current, dict) else {}
    if not isinstance(current_flows, dict):
        current_flows = {}
    previous_flows = previous.get("flows") if isinstance(previous, dict) else None
    if not isinstance(previous_flows, dict):
        previous_flows = {}

    if not previous_flows:
        return {
            "first_run": True,
            "flows_added": [],
            "flows_removed": [],
            "status_changed": [],
            "newly_failing": [],
            "newly_passing": [],
            "changed_count": 0,
            "summary": (
                f"First recorded run for this target ({_plural(len(current_flows), 'flow')} "
                "captured); there is no previous run to compare against."
            ),
        }

    added = [
        _flow_ref(flow_id, entry)
        for flow_id, entry in current_flows.items()
        if isinstance(entry, dict) and flow_id not in previous_flows
    ]
    removed = [
        _flow_ref(flow_id, entry)
        for flow_id, entry in previous_flows.items()
        if isinstance(entry, dict) and flow_id not in current_flows
    ]

    status_changed: list[dict[str, Any]] = []
    newly_failing: list[dict[str, Any]] = []
    newly_passing: list[dict[str, Any]] = []
    for flow_id, entry in current_flows.items():
        if not isinstance(entry, dict):
            continue
        before = previous_flows.get(flow_id)
        if not isinstance(before, dict):
            continue
        old_status = str(before.get("status", NOT_RUN))
        new_status = str(entry.get("status", NOT_RUN))
        if old_status == new_status:
            continue
        change = {
            "flow_id": flow_id,
            "name": entry.get("name", flow_id),
            "from": old_status,
            "to": new_status,
            "risk": entry.get("risk", "medium"),
        }
        status_changed.append(change)
        if old_status not in FAILING_STATUSES and new_status in FAILING_STATUSES:
            newly_failing.append(change)
        elif old_status in FAILING_STATUSES and new_status in PASSING_STATUSES:
            newly_passing.append(change)

    changed_count = len(added) + len(removed) + len(status_changed)
    if changed_count == 0:
        summary = "No flows changed since the previous run."
    else:
        summary = f"{_plural(changed_count, 'flow')} changed since the previous run"
        details: list[str] = []
        if newly_failing:
            details.append(f"{len(newly_failing)} newly failing")
        if newly_passing:
            details.append(f"{len(newly_passing)} newly passing")
        if added:
            details.append(f"{len(added)} added")
        if removed:
            details.append(f"{len(removed)} removed")
        if details:
            summary += " (" + ", ".join(details) + ")"
        summary += "."

    return {
        "first_run": False,
        "flows_added": added,
        "flows_removed": removed,
        "status_changed": status_changed,
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "changed_count": changed_count,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Serialise ``payload`` and move it into place atomically.

    The temporary file is created in the destination directory so that
    :func:`os.replace` stays a same-volume rename, which is the only form that
    is guaranteed atomic on Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(redact_secrets(payload), indent=2, ensure_ascii=False, default=str)
    handle_fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort cleanup
            log.debug("could not remove the temporary radar file", exc_info=True)
        raise


def _most_recent(runs: Sequence[dict[str, Any]], target: str) -> dict[str, Any] | None:
    """The entry to compare against: same target if we have one, else the last.

    See the module docstring - history is keyed by host, so one file can hold
    runs against several targets on that host.
    """
    for entry in reversed(list(runs)):
        if str(entry.get("target", "")) == target:
            return entry
    return runs[-1] if runs else None


def record(
    *,
    target_url: str,
    run_id: str,
    plan: TestPlan | None,
    results: Sequence[TestResult],
    classifications: Sequence[RiskClassification],
    enabled: bool = True,
    max_history: int = DEFAULT_MAX_HISTORY,
) -> dict[str, Any]:
    """Compare this run against the stored history, then append it.

    Returns the comparison from :func:`compare` plus ``enabled``, the ids of
    this run and of the run it was compared with, and ``persisted``, which is
    ``False`` when the diff succeeded but the history could not be written. When
    ``enabled`` is ``False`` nothing is read, nothing is written, and
    ``{"enabled": False}`` is returned.

    This function never raises. The radar is an enhancement layered on top of a
    finished run; taking that run's report down because a JSON file was locked
    would be a strictly worse outcome than losing one data point.
    """
    if not enabled:
        return {"enabled": False}

    try:
        target = sanitize_url(target_url or "")
        current = snapshot(plan, results, classifications)
        history = load_history(target_url)
        runs = [entry for entry in history.get("runs", []) if isinstance(entry, dict)]
        previous = _most_recent(runs, target)

        comparison = compare(previous.get("snapshot") if previous else None, current)
        comparison["enabled"] = True
        comparison["target"] = target
        comparison["run_id"] = run_id
        comparison["compared_with"] = str(previous.get("run_id", "")) if previous else None
        comparison["history_runs"] = len(runs) + 1
        if history.get("note"):
            comparison["note"] = str(history["note"])

        runs.append(
            {
                "run_id": run_id,
                "ts": utcnow_iso(),
                "target": target,
                "snapshot": current,
            }
        )
        if max_history > 0:
            runs = runs[-max_history:]

        payload = {
            "version": HISTORY_VERSION,
            "target": target,
            "updated_at": utcnow_iso(),
            "runs": runs,
        }
        try:
            _write_atomic(history_path(target_url), payload)
            comparison["persisted"] = True
        except OSError as exc:
            log.warning("could not persist the regression radar history: %s", exc)
            comparison["persisted"] = False
            comparison["note"] = (
                "the comparison is valid but this run could not be appended to the "
                "history, so the next run will not see it"
            )
        return comparison
    except Exception as exc:  # the radar must never fail a finished run
        log.warning("regression radar failed: %s", exc, exc_info=True)
        return {
            "enabled": True,
            "first_run": True,
            "flows_added": [],
            "flows_removed": [],
            "status_changed": [],
            "newly_failing": [],
            "newly_passing": [],
            "changed_count": 0,
            "persisted": False,
            "error": f"{type(exc).__name__}: {exc}",
            "summary": "The regression radar could not run; no cross-run comparison is available.",
        }
