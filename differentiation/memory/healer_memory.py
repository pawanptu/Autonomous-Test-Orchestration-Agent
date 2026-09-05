"""Healer memory: what this failure turned out to be last time.

The Healer's decision is the one the entire pipeline exists to automate, so
memory is deliberately built to *inform* it, never to *hijack* it. Every bound
below is enforced in code, not left to a prompt:

* Memory may shift the blended confidence by at most
  :data:`MEMORY_CONFIDENCE_INFLUENCE` (0.10). Without a clamp, one wrong
  classification would reinforce itself forever - it raises the confidence,
  which gets stored, which raises it again next time.
* Memory may **never** move a decision across the auto-apply threshold on its
  own (:func:`apply_memory_prior`): if the live evidence lands below the
  threshold and only memory pushes it above, the decision stays below.
* Memory never touches :func:`agents.healer.patch_weakens_assertions`. That
  guard is mechanical and absolute regardless of how many prior runs a patch
  appeared to work in - this module does not call it and does not export
  anything that could bypass it.
* A stale or TTL-expired prior is shown to the model as history (see
  :func:`prior_for_prompt`) but contributes zero to the numeric blend.

Failure identity comes from :func:`differentiation.memory.keys.failure_signature`
- a *kind* of failure, not one occurrence - so the same recurring defect
matches across runs even when the error message carries a fresh timestamp.

What is deliberately not stored: no DOM, no screenshots, no raw error text
(only a normalised failure *kind*), no step values, no full URLs, no
credentials, no generated source.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from config import CONFIDENCE_AUTO_APPLY_THRESHOLD
from differentiation.memory.store import (
    DEFAULT_MAX_RUNS,
    DEFAULT_TTL_DAYS,
    MEMORY_VERSION,
    append_run_ref,
    is_stale,
    namespace_path,
    read_json,
    utcnow_iso,
    write_atomic,
)
from logging_setup import get_logger
from security import redact_text, sanitize_url

log = get_logger("aivor.memory.healer")

MEMORY_CONFIDENCE_INFLUENCE: float = 0.10
"""Named constant, tested directly: the maximum amount a memory prior may move
the Healer's blended confidence in either direction."""

FLAKY_SCORE_THRESHOLD: float = 0.5
"""Above this, the report surfaces a signature as 'known flaky' rather than a
straightforward defect or script issue."""

_FLAKY_INCREMENT: float = 0.25
_FLAKY_DECAY: float = 0.15


class FailureMemory(BaseModel):
    """One failure signature's cumulative cross-run record."""

    signature: str
    flow_key: str = ""
    name_hint: str = ""
    failure_kind: str = "unknown"
    classification: str = "UNKNOWN"
    blended_confidence: float = 0.5
    times_seen: int = 0
    first_seen_at: str = ""
    first_seen_run_id: str = ""
    last_seen_at: str = ""
    last_seen_run_id: str = ""
    patch_kind_tried: str = "none"
    patch_worked: bool | None = None
    rerun_status: str = ""
    needs_human_review: bool = False
    packaged_bug_id: str | None = None
    resolved_at: str | None = None
    flaky_score: float = 0.0


class HealerMemory(BaseModel):
    """The healer namespace: ``reports/baselines/_memory/<host>/healer.json``."""

    version: int = MEMORY_VERSION
    target: str = ""
    updated_at: str = ""
    failures: dict[str, FailureMemory] = Field(default_factory=dict)
    runs: list[str] = Field(default_factory=list)


def load_healer_memory(target_url: str) -> HealerMemory:
    """Load the healer namespace, or an empty one. Never raises."""
    raw = read_json(namespace_path(target_url, "healer"))
    if not raw:
        return HealerMemory(target=sanitize_url(target_url or ""))
    try:
        return HealerMemory.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a corrupt file is "no memory"
        log.debug("healer.json for %s failed validation (%s); starting fresh", target_url, exc)
        return HealerMemory(target=sanitize_url(target_url or ""))


def get_prior(memory: HealerMemory, signature: str) -> FailureMemory | None:
    return memory.failures.get(signature)


# --------------------------------------------------------------------------
# Confidence blending - the bounded part
# --------------------------------------------------------------------------
def apply_memory_prior(
    base_confidence: float,
    prior: FailureMemory | None,
    *,
    ttl_days: int = DEFAULT_TTL_DAYS,
    threshold: float = CONFIDENCE_AUTO_APPLY_THRESHOLD,
    max_influence: float = MEMORY_CONFIDENCE_INFLUENCE,
) -> tuple[float, str]:
    """Blend a memory prior into ``base_confidence`` under the mandated bounds.

    Returns ``(adjusted_confidence, note)``. ``note`` is empty when there was
    no prior to apply, so the caller can tell "no memory" apart from "memory
    applied but changed nothing".
    """
    if prior is None:
        return round(max(0.0, min(1.0, base_confidence)), 3), ""
    if is_stale(prior.last_seen_at, ttl_days):
        return (
            round(max(0.0, min(1.0, base_confidence)), 3),
            "a prior classification exists but is stale/expired and contributed nothing to the blend",
        )

    raw_shift = prior.blended_confidence - base_confidence
    shift = max(-max_influence, min(max_influence, raw_shift))
    adjusted = max(0.0, min(1.0, base_confidence + shift))

    # Memory alone must never carry a decision across the threshold.
    if (base_confidence < threshold) != (adjusted < threshold):
        adjusted = (threshold - 1e-6) if base_confidence < threshold else threshold

    note = (
        f"prior run(s) classified this failure signature {prior.classification} at confidence "
        f"{prior.blended_confidence:.2f} (seen {prior.times_seen}x, first in {prior.first_seen_run_id or 'an earlier run'})"
    )
    return round(adjusted, 3), note


def prior_for_prompt(prior: FailureMemory | None, *, ttl_days: int = DEFAULT_TTL_DAYS) -> str:
    """The ``PRIOR RUNS`` block for :func:`llm.prompts.healer_user`.

    Empty string when there is nothing to say - the prompt then omits the
    section entirely rather than printing a hollow header.
    """
    if prior is None or prior.times_seen == 0:
        return ""
    stale_note = " (stale/expired - not counted in the confidence)" if is_stale(prior.last_seen_at, ttl_days) else ""
    lines = [
        "PRIOR RUNS - this failure signature has been seen before:",
        f"  seen {prior.times_seen} time(s), first in {prior.first_seen_run_id or 'an earlier run'}{stale_note}",
        f"  previously classified {prior.classification} at confidence {prior.blended_confidence:.2f}",
    ]
    if prior.patch_kind_tried != "none":
        outcome = "did NOT fix it" if prior.patch_worked is False else (
            "fixed it" if prior.patch_worked else "outcome unknown"
        )
        lines.append(f"  a {prior.patch_kind_tried} was applied once and {outcome}")
    lines.append(
        "Treat this as one more piece of evidence, not as a verdict. The live "
        "probe and the DOM snapshot above are the current facts."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Patch repetition guard
# --------------------------------------------------------------------------
def patch_kind_should_be_skipped(prior: FailureMemory | None, kind: str) -> bool:
    """True when memory says this exact patch kind already failed for this
    signature - :func:`agents.healer.apply_fix` must not retry it."""
    return bool(prior and prior.patch_kind_tried == kind and prior.patch_worked is False)


# --------------------------------------------------------------------------
# Recurring bug id
# --------------------------------------------------------------------------
def prior_bug_id(prior: FailureMemory | None) -> str | None:
    return prior.packaged_bug_id if prior and prior.packaged_bug_id else None


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------
def note_failure_outcome(
    memory: HealerMemory,
    signature: str,
    *,
    flow_key: str,
    name_hint: str,
    failure_kind: str,
    classification: str,
    blended_confidence: float,
    run_id: str,
    patch_kind_tried: str = "none",
    patch_worked: bool | None = None,
    rerun_status: str = "",
    needs_human_review: bool = False,
    packaged_bug_id: str | None = None,
) -> FailureMemory:
    """Update (or create) the failure record for this signature this run."""
    now = utcnow_iso()
    entry = memory.failures.get(signature)
    was_flaky_candidate = bool(entry and entry.rerun_status == "failed" and rerun_status == "passed")
    if entry is None:
        entry = FailureMemory(signature=signature, flow_key=flow_key, first_seen_at=now, first_seen_run_id=run_id)

    entry.name_hint = redact_text(name_hint)[:120]
    entry.failure_kind = failure_kind
    entry.classification = classification
    entry.blended_confidence = round(max(0.0, min(1.0, blended_confidence)), 3)
    entry.times_seen += 1
    entry.last_seen_at = now
    entry.last_seen_run_id = run_id
    entry.needs_human_review = needs_human_review
    if patch_kind_tried != "none":
        entry.patch_kind_tried = patch_kind_tried
        entry.patch_worked = patch_worked
    if rerun_status:
        entry.rerun_status = rerun_status
    if packaged_bug_id and not entry.packaged_bug_id:
        entry.packaged_bug_id = packaged_bug_id
    if was_flaky_candidate:
        entry.flaky_score = round(min(1.0, entry.flaky_score + _FLAKY_INCREMENT), 3)
    memory.failures[signature] = entry
    return entry


def note_failure_resolved(memory: HealerMemory, signature: str, *, run_id: str) -> None:
    """A previously-recorded failure's flow now passes clean; mark it resolved."""
    entry = memory.failures.get(signature)
    if entry is None:
        return
    entry.resolved_at = utcnow_iso()
    entry.last_seen_run_id = run_id
    entry.flaky_score = round(max(0.0, entry.flaky_score - _FLAKY_DECAY), 3)


def record_healer_memory(
    *,
    target_url: str,
    memory: HealerMemory,
    run_id: str,
    enabled: bool = True,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> dict[str, Any]:
    """Persist the (already-mutated) healer namespace. Never raises."""
    if not enabled:
        return {"enabled": False, "persisted": False}
    try:
        memory.target = sanitize_url(target_url or "")
        memory.updated_at = utcnow_iso()
        memory.runs = append_run_ref(memory.runs, run_id, max_runs)
        persisted = write_atomic(namespace_path(target_url, "healer"), memory.model_dump(mode="json"))
        return {"enabled": True, "persisted": persisted}
    except Exception as exc:  # noqa: BLE001 - memory must never fail a run
        log.warning("healer memory record failed for %s: %s", target_url, exc)
        return {"enabled": True, "persisted": False}
