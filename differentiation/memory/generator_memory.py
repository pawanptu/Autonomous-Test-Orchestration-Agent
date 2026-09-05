"""Generator memory: which selector already worked, and which flows the model
cannot author.

The Generator's expensive step is the live walk in
:func:`agents.generator.resolve_flow_selectors` - opening a browser context and
searching the DOM for every step's target. The same button on the same page
resolves the same way next run, so that result is highly reusable.

A remembered selector is never trusted blind. It is always re-verified against
the live page (the same cheap presence check the Healer uses,
:func:`browser.selectors.reprobe_expression`) before it is used - a stale
selector that silently "resolved" from memory would generate a test that
cannot possibly pass, which is worse than the search it replaced. The
re-verification itself happens in :mod:`agents.generator`; this module only
holds the record and the eviction bookkeeping.

What is deliberately not stored: no DOM, no screenshots, no raw error text, no
step values, no full URLs, no credentials, and - this module's own rule - no
generated test *source*. A selector expression is stored only after it has
passed the same credential-literal scan the Generator applies to model output,
because a resolved selector can quote page text and page text can contain an
email or a token.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from differentiation.memory.store import (
    DEFAULT_MAX_RUNS,
    MEMORY_VERSION,
    append_run_ref,
    namespace_path,
    read_json,
    utcnow_iso,
    write_atomic,
)
from logging_setup import get_logger
from security import assert_no_secret_literals, redact_text, sanitize_url

log = get_logger("aivor.memory.generator")

CONSECUTIVE_MISS_EVICTION: int = 2
"""A remembered selector that fails to re-verify this many times in a row is
dropped rather than kept as a false lead."""

CONSECUTIVE_MODEL_FAILURE_THRESHOLD: int = 2
"""After this many consecutive model-authoring failures for a flow, the third
attempt skips the model round-trip entirely and goes straight to the
deterministic compiler."""


class SelectorMemory(BaseModel):
    """One selector resolution, remembered by page + action + intent."""

    selector_key: str
    page_path: str = ""
    action: str = ""
    intent_hint: str = ""
    expression: str = ""
    strategy: str = ""
    match_count: int = 0
    last_verified_at: str = ""
    last_run_id: str = ""
    hits: int = 0
    misses: int = 0
    consecutive_misses: int = 0


class AuthoringMemory(BaseModel):
    """One flow's model-authoring track record."""

    flow_key: str
    name_hint: str = ""
    module_name: str = ""
    last_generated_by_model: str = ""
    repair_attempts: int = 0
    fallback_used: bool = False
    consecutive_model_failures: int = 0
    last_validation_error_kind: str = ""
    last_run_id: str = ""


class GeneratorMemory(BaseModel):
    """The generator namespace: ``reports/baselines/_memory/<host>/generator.json``."""

    version: int = MEMORY_VERSION
    target: str = ""
    updated_at: str = ""
    selectors: dict[str, SelectorMemory] = Field(default_factory=dict)
    authoring: dict[str, AuthoringMemory] = Field(default_factory=dict)
    runs: list[str] = Field(default_factory=list)


def load_generator_memory(target_url: str) -> GeneratorMemory:
    """Load the generator namespace, or an empty one. Never raises."""
    raw = read_json(namespace_path(target_url, "generator"))
    if not raw:
        return GeneratorMemory(target=sanitize_url(target_url or ""))
    try:
        return GeneratorMemory.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a corrupt file is "no memory"
        log.debug("generator.json for %s failed validation (%s); starting fresh", target_url, exc)
        return GeneratorMemory(target=sanitize_url(target_url or ""))


# --------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------
def get_selector(memory: GeneratorMemory, key: str) -> SelectorMemory | None:
    return memory.selectors.get(key)


def note_selector_hit(
    memory: GeneratorMemory,
    key: str,
    *,
    page_path: str,
    action: str,
    intent_hint: str,
    expression: str,
    strategy: str,
    match_count: int,
    run_id: str,
) -> bool:
    """Record a selector that resolved (fresh search or re-verified memory).

    Returns ``False`` (and stores nothing) when ``expression`` looks like it
    could carry a credential literal - the same guard the Generator applies to
    model-authored code applies here, because a resolved selector can quote
    live page text.
    """
    if expression and assert_no_secret_literals(expression):
        log.warning("refusing to store a selector expression that looks like a credential literal")
        return False
    entry = memory.selectors.get(key) or SelectorMemory(selector_key=key, page_path=page_path, action=action)
    entry.page_path = page_path
    entry.action = action
    entry.intent_hint = redact_text(intent_hint)[:160]
    entry.expression = expression
    entry.strategy = strategy
    entry.match_count = match_count
    entry.last_verified_at = utcnow_iso()
    entry.last_run_id = run_id
    entry.hits += 1
    entry.consecutive_misses = 0
    memory.selectors[key] = entry
    return True


def note_selector_miss(memory: GeneratorMemory, key: str) -> None:
    """Record that a remembered selector no longer resolves; evict after
    :data:`CONSECUTIVE_MISS_EVICTION` consecutive misses."""
    entry = memory.selectors.get(key)
    if entry is None:
        return
    entry.misses += 1
    entry.consecutive_misses += 1
    if entry.consecutive_misses >= CONSECUTIVE_MISS_EVICTION:
        memory.selectors.pop(key, None)


# --------------------------------------------------------------------------
# Authoring
# --------------------------------------------------------------------------
def should_skip_model(memory: GeneratorMemory, key: str, *, threshold: int = CONSECUTIVE_MODEL_FAILURE_THRESHOLD) -> bool:
    entry = memory.authoring.get(key)
    return bool(entry and entry.consecutive_model_failures >= threshold)


def module_name_for(memory: GeneratorMemory, key: str, default: str) -> str:
    """Reuse a stable module name across runs so a regenerated suite diffs
    cleanly against the previous one."""
    entry = memory.authoring.get(key)
    if entry and entry.module_name:
        return entry.module_name
    return default


def note_authoring_outcome(
    memory: GeneratorMemory,
    key: str,
    *,
    name_hint: str,
    module_name: str,
    generated_by_model: str,
    repair_attempts: int,
    fallback_used: bool,
    validation_error_kind: str,
    run_id: str,
    model_success: bool,
) -> None:
    entry = memory.authoring.get(key) or AuthoringMemory(flow_key=key, module_name=module_name)
    entry.name_hint = redact_text(name_hint)[:120]
    entry.module_name = entry.module_name or module_name
    entry.last_generated_by_model = generated_by_model
    entry.repair_attempts = repair_attempts
    entry.last_validation_error_kind = validation_error_kind
    entry.last_run_id = run_id
    entry.fallback_used = fallback_used
    if model_success:
        entry.consecutive_model_failures = 0
    elif fallback_used:
        entry.consecutive_model_failures += 1
    memory.authoring[key] = entry


def record_generator_memory(
    *,
    target_url: str,
    memory: GeneratorMemory,
    run_id: str,
    enabled: bool = True,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> dict[str, Any]:
    """Persist the (already-mutated) generator namespace. Never raises."""
    if not enabled:
        return {"enabled": False, "persisted": False}
    try:
        memory.target = sanitize_url(target_url or "")
        memory.updated_at = utcnow_iso()
        memory.runs = append_run_ref(memory.runs, run_id, max_runs)
        persisted = write_atomic(namespace_path(target_url, "generator"), memory.model_dump(mode="json"))
        return {"enabled": True, "persisted": persisted}
    except Exception as exc:  # noqa: BLE001 - memory must never fail a run
        log.warning("generator memory record failed for %s: %s", target_url, exc)
        return {"enabled": True, "persisted": False}


def drop_selectors_for_pages(memory: GeneratorMemory, changed_paths: set[str]) -> int:
    """Evict every selector entry for a page whose structural fingerprint
    changed. Returns the number of entries dropped."""
    to_drop = [key for key, entry in memory.selectors.items() if entry.page_path in changed_paths]
    for key in to_drop:
        memory.selectors.pop(key, None)
    return len(to_drop)
