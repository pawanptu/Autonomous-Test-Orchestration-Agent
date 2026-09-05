"""Credential custody and redaction.

This module is the single place in the repository that is allowed to hold a
raw credential, and the single place that knows how to scrub one out of
arbitrary data.

Two guarantees are implemented here:

1. **Custody.** Raw credentials live in :class:`SecretBox`, an in-process,
   per-run container. They are *never* placed in LangGraph state, on disk, in
   a report, in a prompt, or in an API response. The box is wiped when the run
   finishes (success, failure or cancellation).

2. **Redaction.** :func:`redact_secrets` walks any JSON-ish structure and
   replaces (a) every registered secret *value*, (b) every value under a
   sensitive *key name*, and (c) a set of well-known credential *patterns*
   (bearer tokens, Groq keys, ``https://user:pass@host`` URLs) with the
   constant ``***REDACTED***``.

The logging filter, the FastAPI response models, the Streamlit UI, the report
generator and the healer evidence collector all route through :func:`redact_secrets`.

Why value-based redaction has a minimum length
----------------------------------------------
Registering a two-character password would cause the redactor to punch holes
through unrelated prose. Values shorter than :data:`MIN_REDACTABLE_LENGTH` are
therefore not registered for substring matching; they are still protected by
key-name redaction and by never being serialised in the first place.
"""

from __future__ import annotations

import ipaddress
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

REDACTED: str = "***REDACTED***"
MIN_REDACTABLE_LENGTH: int = 4

# Key names whose *values* are always redacted, regardless of content.
SENSITIVE_KEY_PARTS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_key",
        "authorization",
        "auth",
        "cookie",
        "cookies",
        "session_id",
        "credential",
        "credentials",
        "private_key",
        "client_secret",
        "storage_state",
        "bearer",
        "otp",
    }
)

# Key names that merely *look* sensitive but are structural booleans/labels we
# explicitly want to keep visible in the report and UI.
SENSITIVE_KEY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "credentials_present",
        "has_credentials",
        "requires_auth",
        "auth_blocked",
        "token_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "auth_flow_present",
        "login_ok",
        "needs_human_review",
    }
)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization: Bearer <token>  /  "Bearer <token>"
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-+/=]{8,}"), r"\1 " + REDACTED),
    # Groq / OpenAI style keys.
    (re.compile(r"\bgsk_[A-Za-z0-9]{8,}"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), REDACTED),
    # Basic-auth credentials embedded in a URL.
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1" + REDACTED + "@"),
    # password=... / token=... inside a query string or form body.
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret)"
            r"(\s*[=:]\s*|%3D)"
            r"(\"[^\"]*\"|'[^']*'|[^\s&;,}\]\"']+)"
        ),
        lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}",
    ),
)


# --------------------------------------------------------------------------
# Registry of live secret values
# --------------------------------------------------------------------------
_registry_lock = threading.RLock()
_run_secrets: dict[str, set[str]] = {}


def register_secret_values(run_id: str, values: Iterable[str | None]) -> None:
    """Register secret values so the redactor can scrub them everywhere.

    Values shorter than :data:`MIN_REDACTABLE_LENGTH` are ignored for
    substring matching (see the module docstring).
    """
    with _registry_lock:
        bucket = _run_secrets.setdefault(run_id, set())
        for value in values:
            if isinstance(value, str) and len(value.strip()) >= MIN_REDACTABLE_LENGTH:
                bucket.add(value.strip())


def unregister_run(run_id: str) -> None:
    """Forget every secret value registered for ``run_id``."""
    with _registry_lock:
        _run_secrets.pop(run_id, None)


def active_secret_values() -> tuple[str, ...]:
    """Snapshot of every registered secret value, longest first.

    Longest-first ordering matters: if a password happens to be a prefix of a
    token, replacing the longer value first avoids leaving a suffix behind.
    """
    with _registry_lock:
        values: set[str] = set()
        for bucket in _run_secrets.values():
            values |= bucket
    return tuple(sorted(values, key=len, reverse=True))


def clear_all_secrets() -> None:
    """Drop every registered value. Used by tests and on interpreter shutdown."""
    with _registry_lock:
        _run_secrets.clear()


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------
def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    if lowered in SENSITIVE_KEY_ALLOWLIST:
        return False
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def redact_text(text: str, extra_values: Iterable[str] = ()) -> str:
    """Scrub a single string: registered values first, then patterns."""
    if not text:
        return text
    out = text
    candidates = list(active_secret_values()) + [
        v for v in extra_values if isinstance(v, str) and len(v) >= MIN_REDACTABLE_LENGTH
    ]
    for value in sorted(set(candidates), key=len, reverse=True):
        if value and value in out:
            out = out.replace(value, REDACTED)
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact_secrets(obj: Any, extra_values: Iterable[str] = ()) -> Any:
    """Recursively redact secrets from any JSON-ish structure.

    Handles ``str``, ``Mapping``, ``list``/``tuple``/``set``, and objects that
    expose ``model_dump()`` (pydantic) or ``__dict__``. Scalars that cannot
    contain a secret (int, float, bool, None) are returned untouched.

    The function never raises: redaction failing closed on an unexpected type
    would be worse than returning a best-effort ``repr``.
    """
    try:
        return _redact(obj, tuple(extra_values), depth=0)
    except Exception:  # pragma: no cover - defensive, must never break logging
        return REDACTED


_MAX_DEPTH = 20


def _redact(obj: Any, extra: tuple[str, ...], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return "***TRUNCATED***"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return redact_text(obj, extra)
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, Mapping):
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            if _is_sensitive_key(key):
                out[key] = REDACTED if value not in (None, "", [], {}) else value
            else:
                out[key] = _redact(value, extra, depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact(item, extra, depth + 1) for item in obj]
    if isinstance(obj, set):
        return [_redact(item, extra, depth + 1) for item in sorted(obj, key=repr)]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _redact(dump(mode="json"), extra, depth + 1)
        except TypeError:
            return _redact(dump(), extra, depth + 1)
    if hasattr(obj, "__dict__"):
        return _redact(vars(obj), extra, depth + 1)
    return redact_text(str(obj), extra)


def sanitize_url(url: str) -> str:
    """Strip userinfo and sensitive query parameters from a URL.

    Screenshots, reports and the decision log all carry URLs; a URL is a very
    common accidental credential channel (``?token=...``,
    ``https://user:pass@host``).
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return redact_text(url)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = REDACTED + "@" + netloc.rsplit("@", 1)[1]
    query = parts.query
    if query:
        kept: list[str] = []
        for chunk in query.split("&"):
            key = chunk.split("=", 1)[0]
            kept.append(f"{key}={REDACTED}" if _is_sensitive_key(key) else chunk)
        query = "&".join(kept)
    return redact_text(urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment)))


# --------------------------------------------------------------------------
# Transport security for credentials
# --------------------------------------------------------------------------
_LOOPBACK_HOSTNAMES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _is_local_dev_host(hostname: str) -> bool:
    """True for loopback/private/link-local hosts, where a bare HTTP dev
    server is normal and the traffic never leaves the machine or LAN."""
    hostname = (hostname or "").strip().lower().strip("[]")
    if not hostname:
        return False
    if hostname in _LOOPBACK_HOSTNAMES or hostname.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def insecure_for_credentials(url: str) -> bool:
    """True when *url* would carry a login credential over an unencrypted wire.

    Plain HTTP is fine for anonymous crawling, but a username, password or
    bearer token must never be submitted to a plain-HTTP endpoint unless the
    host is a loopback/private/link-local address the operator is
    deliberately pointing at (e.g. ``http://localhost:3000`` during local
    development). An unparsable URL fails closed (treated as insecure).
    """
    if not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return True
    if parts.scheme.lower() != "http":
        return False
    return not _is_local_dev_host(parts.hostname or "")


# --------------------------------------------------------------------------
# SecretBox
# --------------------------------------------------------------------------
@dataclass
class Credentials:
    """Raw credentials for one run. Never serialised, never logged.

    ``login_url`` is not itself a secret, but it travels with the credentials
    so that the login helper has everything it needs in one object.
    """

    username: str | None = None
    password: str | None = None
    token: str | None = None
    login_url: str | None = None

    def present(self) -> bool:
        return bool(self.username or self.password or self.token)

    def secret_values(self) -> list[str]:
        """The values that must be redacted from all output."""
        return [v for v in (self.username, self.password, self.token) if v]

    def describe(self) -> dict[str, Any]:
        """Safe, boolean-only description for logs and the report."""
        return {
            "has_username": bool(self.username),
            "has_password": bool(self.password),
            "has_token": bool(self.token),
            "login_url_provided": bool(self.login_url),
        }

    # Make accidental leakage loud and impossible.
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Credentials({self.describe()})"

    __str__ = __repr__

    def __reduce__(self):  # pragma: no cover - defensive
        raise TypeError("Credentials are deliberately not picklable or serialisable")


class SecretBox:
    """Process-local, run-scoped credential store.

    The orchestrator puts credentials in at the start of a run and calls
    :meth:`wipe` in a ``finally`` block. LangGraph state only ever carries
    ``credentials_present: bool``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[str, Credentials] = {}

    def put(self, run_id: str, creds: Credentials) -> None:
        with self._lock:
            self._store[run_id] = creds
        register_secret_values(run_id, creds.secret_values())

    def get(self, run_id: str) -> Credentials | None:
        with self._lock:
            return self._store.get(run_id)

    def present(self, run_id: str) -> bool:
        creds = self.get(run_id)
        return bool(creds and creds.present())

    def wipe(self, run_id: str) -> None:
        """Remove the credentials and de-register their values.

        Best-effort overwrite of the string fields first, so that a heap dump
        taken later is less likely to contain the value. Python strings are
        immutable so this only drops the reference - documented honestly
        rather than dressed up as secure erasure.
        """
        with self._lock:
            creds = self._store.pop(run_id, None)
        if creds is not None:
            creds.username = None
            creds.password = None
            creds.token = None
        unregister_run(run_id)

    def wipe_all(self) -> None:
        with self._lock:
            run_ids = list(self._store)
        for run_id in run_ids:
            self.wipe(run_id)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        with self._lock:
            return f"SecretBox(runs={len(self._store)})"


SECRET_BOX = SecretBox()
"""The single process-wide secret box. Import this, do not instantiate another."""


def assert_no_secret_literals(text: str, run_id: str | None = None) -> list[str]:
    """Return the names of any registered secrets found verbatim in ``text``.

    Used by the Generator to hard-fail on LLM output that embedded a real
    password into a test file, and by the bug packager before writing a repro
    script to disk. An empty list means the text is clean.
    """
    hits: list[str] = []
    if not text:
        return hits
    if run_id is not None:
        creds = SECRET_BOX.get(run_id)
        if creds is not None:
            for label, value in (
                ("username", creds.username),
                ("password", creds.password),
                ("token", creds.token),
            ):
                if value and len(value) >= MIN_REDACTABLE_LENGTH and value in text:
                    hits.append(label)
            return hits
    for value in active_secret_values():
        if value in text:
            hits.append("secret")
    return hits
