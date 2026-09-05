"""Shared substrate for agent memory: paths, atomic IO, fingerprinting, TTL.

Disk layout, one directory per target (mirrors ``differentiation.regression_radar``):

    reports/baselines/_memory/<host-slug>/meta.json
    reports/baselines/_memory/<host-slug>/planner.json
    reports/baselines/_memory/<host-slug>/generator.json
    reports/baselines/_memory/<host-slug>/healer.json

What is deliberately not stored, here or in any namespace built on top of this
module: no DOM, no screenshots, no raw error text, no step values, no full
URLs, no credentials, no generated source. Every string that reaches
:func:`write_atomic` is passed through :func:`security.redact_secrets`, and any
URL is passed through :func:`security.sanitize_url` before it is stored. These
files are read on every run, so they stay small, and the less they hold the
less there is to leak.

Crash safety copies :mod:`differentiation.regression_radar` exactly: a write
goes to a temporary file in the destination directory, is flushed and fsynced,
and is moved into place with :func:`os.replace`, which is atomic on Windows and
POSIX alike. A corrupt or unreadable file is treated as "no memory" rather than
as an error - memory is an enhancement layered on a working pipeline, and a
locked or garbage JSON file must degrade to today's behaviour, never fail a
run. Every function in this module that touches disk therefore never raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from config import BASELINES_DIR
from graph.state import SiteMap
from logging_setup import get_logger
from security import redact_secrets, sanitize_url

log = get_logger("aivor.memory.store")

MEMORY_DIRNAME: str = "_memory"
MEMORY_VERSION: int = 1
DEFAULT_MAX_RUNS: int = 20
DEFAULT_TTL_DAYS: int = 14
SITE_PATH_LOSS_THRESHOLD: float = 0.3
"""Fraction of previously-known pages that must go missing before a site is
considered changed enough to distrust prior verification (see the module
docstring on 'shared invalidation' in the feature spec)."""

_SAFE_SLUG_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789.-"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _host_slug(target_url: str) -> str:
    """Filesystem-safe host label for ``target_url``.

    Mirrors :func:`differentiation.regression_radar._host_slug` exactly, on
    purpose: the two caches should key identically so a reader never has to
    wonder why a target lives in one history but not the other.
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


def host_slug(target_url: str) -> str:
    """Public accessor used by the API to compute the slug for a request."""
    return _host_slug(target_url)


def is_valid_host_slug(candidate: str) -> bool:
    """True when ``candidate`` is safe to join onto a filesystem path.

    Used by the API before it touches disk on a caller-supplied host: rejects
    anything containing a path separator or a parent-directory reference,
    independent of what :func:`host_slug` would ever itself produce.
    """
    if not candidate or "/" in candidate or "\\" in candidate or ".." in candidate:
        return False
    return all(ch in _SAFE_SLUG_CHARS or ch == "_" for ch in candidate.lower())


def memory_dir(target_url: str) -> Path:
    """The directory holding every namespace file for this target."""
    return BASELINES_DIR / MEMORY_DIRNAME / _host_slug(target_url)


def memory_dir_for_slug(slug: str) -> Path:
    """Directory for an already-computed, already-validated host slug."""
    return BASELINES_DIR / MEMORY_DIRNAME / slug


def namespace_path(target_url: str, name: str) -> Path:
    return memory_dir(target_url) / f"{name}.json"


def meta_path(target_url: str) -> Path:
    return memory_dir(target_url) / "meta.json"


# --------------------------------------------------------------------------
# Atomic IO
# --------------------------------------------------------------------------
def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file. Never raises; a missing/corrupt file reads as ``{}``."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.debug("memory file %s is unreadable (%s); treated as empty", path, exc)
        return {}
    if not isinstance(raw, dict):
        log.debug("memory file %s has an unexpected shape; treated as empty", path)
        return {}
    return raw


def write_atomic(path: Path, payload: dict[str, Any]) -> bool:
    """Serialise and move ``payload`` into place atomically.

    Returns ``True`` on success and ``False`` on any failure - permissions, a
    full disk, a locked file on Windows - without raising. The caller reports
    this as ``persisted: False`` rather than failing the run.
    """
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(redact_secrets(payload), indent=2, ensure_ascii=False, default=str)
        handle_fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.stem}-", suffix=".tmp"
        )
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        return True
    except Exception as exc:  # noqa: BLE001 - persistence must never fail a run
        log.warning("could not persist memory file %s: %s", path, exc)
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort cleanup
                log.debug("could not remove the temporary memory file", exc_info=True)
        return False


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_run_ref(runs: Sequence[str], run_id: str, max_runs: int) -> list[str]:
    """Append ``run_id`` to a capped, de-duplicated run-reference list."""
    out = [r for r in runs if r and r != run_id]
    out.append(run_id)
    return out[-max(1, max_runs):]


# --------------------------------------------------------------------------
# TTL
# --------------------------------------------------------------------------
def is_stale(timestamp: str | None, ttl_days: int) -> bool:
    """True when ``timestamp`` (ISO 8601) is older than ``ttl_days``.

    A missing or unparsable timestamp is treated as stale: an entry with no
    reliable age must not silently count as fresh evidence.
    """
    if not timestamp:
        return True
    try:
        seen = datetime.fromisoformat(timestamp)
    except ValueError:
        return True
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - seen > timedelta(days=max(0, ttl_days))


# --------------------------------------------------------------------------
# Site fingerprint / staleness
# --------------------------------------------------------------------------
def _url_path(url: str) -> str:
    try:
        parts = urlsplit(url or "")
        return (parts.path or "/").lower()
    except ValueError:
        return (url or "").lower()


def site_shape(site_map: SiteMap | None) -> dict[str, str]:
    """Path -> compact structural shape, the input to both the fingerprint and
    the staleness diff. Paths only: no query string, no host, no page text."""
    if site_map is None:
        return {}
    shape: dict[str, str] = {}
    for page in site_map.pages:
        path = _url_path(page.url)
        shape[path] = f"forms={len(page.forms)}|inputs={len(page.inputs)}|buttons={len(page.buttons)}"
    return shape


def site_fingerprint(site_map: SiteMap | None) -> str:
    """Short, stable digest of the crawl's structural shape."""
    shape = site_shape(site_map)
    if not shape:
        return ""
    blob = "\n".join(f"{path}:{shape[path]}" for path in sorted(shape))
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:24]


def evaluate_site_change(
    previous_shape: dict[str, str],
    current_shape: dict[str, str],
    *,
    path_loss_threshold: float = SITE_PATH_LOSS_THRESHOLD,
) -> tuple[bool, str]:
    """True + a human-readable reason when the site changed enough to distrust
    prior selector/flow verification.

    Triggers on either more than ``path_loss_threshold`` of previously-known
    paths going missing, or any shared page's form/input/button shape
    changing. An empty ``previous_shape`` (first run) is never a change.
    """
    if not previous_shape:
        return False, ""
    missing = [p for p in previous_shape if p not in current_shape]
    ratio = len(missing) / len(previous_shape)
    if ratio > path_loss_threshold:
        return True, (
            f"{len(missing)}/{len(previous_shape)} previously known page(s) "
            f"({ratio:.0%}) are no longer present in this crawl"
        )
    changed = [
        p for p in previous_shape
        if p in current_shape and previous_shape[p] != current_shape[p]
    ]
    if changed:
        return True, f"{len(changed)} previously known page(s) changed form/input/button shape"
    return False, ""


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
class MemoryMeta(BaseModel):
    """Cross-namespace bookkeeping: version, site fingerprint, staleness, runs."""

    version: int = MEMORY_VERSION
    target: str = ""
    updated_at: str = ""
    fingerprint: str = ""
    page_shape: dict[str, str] = Field(default_factory=dict)
    stale: bool = False
    stale_reason: str = ""
    runs: list[str] = Field(default_factory=list)


def load_meta(target_url: str) -> MemoryMeta:
    """Load ``meta.json``, or a fresh default. Never raises."""
    raw = read_json(meta_path(target_url))
    if not raw:
        return MemoryMeta(target=sanitize_url(target_url or ""))
    try:
        return MemoryMeta.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a corrupt meta file is "no meta"
        log.debug("meta.json for %s failed validation (%s); using defaults", target_url, exc)
        return MemoryMeta(target=sanitize_url(target_url or ""))


def save_meta(target_url: str, meta: MemoryMeta) -> bool:
    return write_atomic(meta_path(target_url), meta.model_dump(mode="json"))
