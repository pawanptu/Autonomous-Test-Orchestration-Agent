"""Stable cross-run identity for flows, selectors and failures.

Flow ids and names are not stable across runs and must never be used as memory
keys: :func:`agents.planner.renumber_flows` reassigns ``F001, F002, ...`` on
every plan, and flow *names* are model output that drifts between runs even
when the underlying flow is semantically identical. Everything in this module
hashes normalised *semantics* instead - category, page path, and step shape -
so the same flow matches itself across runs regardless of how the model
labelled it this time.

Nothing here ever hashes a step's actual ``value``: only its coarse kind
(``empty``, ``text``, ``email``, ``number``, ``long``). That means a password
change cannot alter a key, and - just as importantly - a password can never be
recovered from one.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from differentiation.confidence_scorer import ConfidenceSignals
    from graph.state import TestFlow, TestResult

_ID_SHAPED_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

KEY_LENGTH: int = 16


def _url_path(url: str) -> str:
    """URL path only, lowercased - host and query are dropped so a session id
    or a locale subdomain cannot key the memory."""
    try:
        parts = urlsplit(url or "")
        return (parts.path or "/").lower()
    except ValueError:
        return (url or "").lower()


def normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, replace id-shaped digit runs with ``#``.

    The digit substitution means "item 42" and "item 7" normalise identically,
    so a step referencing a different row/id on an otherwise-identical page
    does not spuriously mint a new key.
    """
    cleaned = _ID_SHAPED_RE.sub("#", (text or "").strip().lower())
    return _WS_RE.sub(" ", cleaned).strip()


def value_kind(value: str | None) -> str:
    """Coarse shape of a step value - never the value itself."""
    v = (value or "").strip()
    if not v:
        return "empty"
    if _EMAIL_RE.match(v):
        return "email"
    if _NUMBER_RE.match(v):
        return "number"
    if len(v) > 40:
        return "long"
    return "text"


def _digest(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:KEY_LENGTH]


def flow_key(flow: "TestFlow") -> str:
    """Stable cross-run identity for a flow: sha256 over normalised semantics.

    Built from ``category`` + the URL *path* of ``flow.url`` + the ordered step
    shape, where each step contributes ``(action, normalised target, value
    kind)``. Never hashes a step's value.
    """
    category = str(getattr(flow.category, "value", flow.category))
    path = _url_path(flow.url)
    step_shape = "|".join(
        f"{step.action}:{normalise_text(step.target)}:{value_kind(step.value)}"
        for step in flow.steps
    )
    return _digest(f"{category}::{path}::{step_shape}")


def selector_key(*, page_path: str, action: str, intent: str) -> str:
    """Stable identity for one selector resolution on one page."""
    return _digest(f"{_url_path(page_path)}::{action}::{normalise_text(intent)}")


def failure_signature(*, flow_key: str, signals: "ConfidenceSignals", result: "TestResult") -> str:
    """Stable identity for a *kind* of failure, not one occurrence.

    Deliberately excludes the error *message* text: a timestamp or an element
    index embedded in the message must not change which bucket a recurring
    defect falls into. Built from the flow's key, the failure kind, a
    normalised error *type* name, whether the error named a locator, and which
    marker classes fired.
    """
    error_type = normalise_text(result.error_type or "")
    markers = "|".join(
        name
        for name, flag in (
            ("captcha", bool(signals.captcha_or_bot_wall)),
            ("auth_wall", bool(signals.auth_wall)),
            ("network_flaky", bool(signals.network_flaky)),
            ("spa_race", bool(signals.spa_race_suspected)),
        )
        if flag
    )
    blob = (
        f"{flow_key}::{signals.failure_kind}::{error_type}::"
        f"{bool(signals.locator_named_in_error)}::{markers}"
    )
    return _digest(blob)
