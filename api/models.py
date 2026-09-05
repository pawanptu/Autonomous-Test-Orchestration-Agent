"""Request and response schemas for the HTTP surface.

The single most important rule in this module is the *separation of the request
and response type hierarchies*. ``POST /run`` is the only endpoint that accepts
a credential, and :class:`CredentialsIn` is the only model in the repository
that can hold one. No response model declares a credential field, so there is
no serialisation path from a submitted secret back out to a client - not via
``/status``, ``/report``, ``/bugs``, ``/health``, nor via an error body.

Three further defences are layered on top of that separation:

* :class:`CredentialsIn` marks ``password`` and ``token`` with ``repr=False``
  *and* overrides ``__repr__``/``__str__``, so an accidental f-string
  interpolation of the whole model (the most common real-world leak) prints a
  boolean description instead of the values.
* Input models set ``extra="forbid"``. A client that posts an unexpected field
  gets a 422 rather than having it silently swallowed, which keeps the audited
  input surface exactly as wide as this file says it is.
* Response models are deliberately tolerant about *enumerations* (plain ``str``
  where the domain model uses a ``Literal``). A response model must never
  refuse to render data it is merely echoing; rejecting an unknown stage name
  would turn a cosmetic mismatch into an outage of the live status endpoint.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from security import Credentials, insecure_for_credentials
from target_policy import TargetPolicy, evaluate_target

# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})


class CredentialsIn(BaseModel):
    """Optional login material for the target application.

    Values live on this object only for the few milliseconds between FastAPI
    parsing the body and the handler moving them into
    :data:`security.SECRET_BOX`. They are never copied into the run store, the
    LangGraph state, a prompt, a generated test, a report or a log line.

    ``username`` is not marked ``repr=False``: it is frequently needed in
    operator-facing diagnostics ("login failed for the supplied user") and it
    is registered with the redactor anyway. ``password`` and ``token`` are
    hidden from ``repr`` unconditionally.
    """

    model_config = ConfigDict(extra="forbid")

    username: str | None = Field(default=None, max_length=512)
    password: str | None = Field(default=None, max_length=512, repr=False)
    token: str | None = Field(default=None, max_length=8192, repr=False)
    login_url: str | None = Field(
        default=None,
        max_length=2048,
        description="Explicit sign-in page. When omitted the crawler tries to find one.",
    )

    # -- introspection -----------------------------------------------------
    def present(self) -> bool:
        """True when at least one usable secret was supplied.

        A body carrying only ``login_url`` is *not* credentials: there is
        nothing to sign in with, so the run is treated as anonymous.
        """
        return bool(
            (self.username or "").strip()
            or (self.password or "").strip()
            or (self.token or "").strip()
        )

    def describe(self) -> dict[str, bool]:
        """Boolean-only description that is safe to log or return."""
        return {
            "has_username": bool(self.username),
            "has_password": bool(self.password),
            "has_token": bool(self.token),
            "login_url_provided": bool(self.login_url),
        }

    def to_credentials(self) -> Credentials:
        """Convert into the custody object owned by :mod:`security`.

        Kept here so the API handler never has to touch the individual secret
        attributes: it calls this once and hands the result straight to
        ``SECRET_BOX.put``.
        """
        return Credentials(
            username=(self.username or None),
            password=(self.password or None),
            token=(self.token or None),
            login_url=(self.login_url or None),
        )

    # -- leak-proof rendering ---------------------------------------------
    def __repr__(self) -> str:
        return f"CredentialsIn({self.describe()})"

    __str__ = __repr__


class RunRequest(BaseModel):
    """Body of ``POST /run``.

    ``extra="forbid"`` is a security control, not tidiness: it guarantees that
    the only credential-shaped data the service will ever accept is the nested
    :class:`CredentialsIn`, so there is no undeclared field for a secret to
    arrive in and get echoed back by a permissive handler.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        ...,
        max_length=2048,
        description="Target application URL. Must be http:// or https://.",
    )
    intent: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional natural-language steer, e.g. 'focus on checkout'.",
    )
    prd_text: str | None = Field(
        default=None,
        max_length=200_000,
        description="Optional requirements text used by the PRD gap analysis.",
    )
    credentials: CredentialsIn | None = Field(default=None, repr=False)
    allow_private_target: bool = Field(
        default=False,
        description=(
            "Request that a loopback/private/link-local target be permitted. Only "
            "honoured when the service is itself configured with "
            "ALLOW_PRIVATE_TARGETS=true; a client can never widen the server policy."
        ),
    )
    authorize_destructive: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Destructive action categories to authorise for this run (payment, "
            "checkout, delete, account_cancellation, password_reset, email_send, "
            "irreversible_submit). Only honoured for categories the service already "
            "authorises via AUTHORIZED_DESTRUCTIVE_ACTIONS."
        ),
    )

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, value: Any) -> str:
        """Strip whitespace and reject anything that is not http/https.

        The rejection message deliberately never quotes the submitted value: a
        URL is a common accidental credential channel (``?token=...``,
        ``https://user:pass@host``) and validation errors are rendered before
        the value has been registered with the redactor.

        This validator performs the *syntactic* half of the target admission
        policy - scheme, host, metadata hostnames and IP-literal address class -
        so that an obviously inadmissible URL is a fast 422. The half that needs
        name resolution runs in the request handler, which can await it without
        blocking the event loop, and again on every redirect hop in
        :mod:`browser.session`. See :mod:`target_policy`.
        """
        if not isinstance(value, str):
            raise ValueError("url must be a string beginning with http:// or https://")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("url must not be empty")
        try:
            parts = urlsplit(cleaned)
        except ValueError as exc:  # malformed IPv6 literal and friends
            raise ValueError("url could not be parsed") from exc
        if parts.scheme.lower() not in ALLOWED_URL_SCHEMES:
            raise ValueError("url must use the http or https scheme")
        if not parts.netloc:
            raise ValueError("url must include a host, for example https://example.com")

        # Syntactic policy pass: no DNS, so an IP literal or a known metadata
        # hostname is rejected here while a name is deferred to the handler.
        # ``allow_private`` is deliberately True at this stage: whether a private
        # target is permitted is a server-configuration question the handler
        # answers, and deciding it here would reject localhost before the
        # operator's own override could be consulted.
        decision = evaluate_target(
            cleaned, TargetPolicy(allow_private=True, resolve_dns=False)
        )
        if not decision.allowed:
            raise ValueError(decision.detail)
        return cleaned

    @field_validator("authorize_destructive", mode="before")
    @classmethod
    def _normalize_categories(cls, value: Any) -> Any:
        """Accept a comma-separated string as well as a list."""
        if isinstance(value, str):
            return [chunk.strip() for chunk in value.replace(";", ",").split(",") if chunk.strip()]
        return value

    @field_validator("intent", "prd_text", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """Treat an empty or whitespace-only string as "not supplied"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _reject_insecure_credential_transport(self) -> "RunRequest":
        """Refuse to submit a credential over plain HTTP to a public host.

        A username, password or bearer token is about to be typed into a real
        login form or sent as an Authorization header; doing that over
        unencrypted HTTP would expose it to anyone on the network path. Plain
        HTTP is still allowed for anonymous crawling, and for credentials
        against a loopback/private host (``http://localhost:3000``) since
        that traffic never leaves the machine or LAN.
        """
        if self.credentials is not None and self.credentials.present():
            for label, target in (
                ("url", self.url),
                ("credentials.login_url", self.credentials.login_url),
            ):
                if target and insecure_for_credentials(target):
                    raise ValueError(
                        f"refusing to send credentials to {label!r} over plain http://; "
                        "use https://, or target a loopback/private host such as "
                        "http://localhost:3000 for local development"
                    )
        return self


# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------
class RunAccepted(BaseModel):
    """202 body for ``POST /run``.

    The URLs are relative on purpose: the service is routinely reached through
    a different host than it binds to (Streamlit sidecar, port forward, reverse
    proxy), and a relative path stays correct in every one of those cases.
    """

    run_id: str
    status: str = "queued"
    message: str = ""
    status_url: str = ""
    report_url: str = ""


class DecisionEventOut(BaseModel):
    """One line of the agent's visible reasoning, mirroring
    :class:`graph.state.DecisionEvent` field for field.

    It is a separate class rather than a re-export so that the wire format can
    stay stable if the internal model gains a field that should not be public,
    and so that ``stage``/``event`` can be widened to ``str`` (see the module
    docstring).
    """

    ts: str = ""
    stage: str = ""
    event: str = ""
    summary: str = ""
    detail: str = ""
    confidence: float | None = None
    risk: str | None = None
    flow_id: str | None = None
    auto_applied: bool | None = None
    needs_human_review: bool = False

    @field_validator("risk", mode="before")
    @classmethod
    def _known_risk_only(cls, value: Any) -> str | None:
        text = str(value or "").strip().lower()
        return text if text in ("high", "medium", "low") else None

    @classmethod
    def from_event(cls, event: Any) -> "DecisionEventOut":
        """Build from a :class:`graph.state.DecisionEvent` or a plain mapping.

        Tolerant by design: the live status endpoint must keep streaming the
        decision log even if one event was produced by an older schema.
        """
        if isinstance(event, Mapping):
            payload: dict[str, Any] = dict(event)
        else:
            dump = getattr(event, "model_dump", None)
            payload = dump(mode="json") if callable(dump) else {"summary": str(event)}
        return cls.model_validate(payload)


class RunStatusResponse(BaseModel):
    """Live view of one run.

    There is deliberately **no** credentials field on this model, and none on
    any model it nests. ``credentials_present`` is a boolean and is the only
    thing a client ever learns about the login material.
    """

    run_id: str
    status: str = "queued"
    current_stage: str = ""
    started_at: str = ""
    finished_at: str | None = None
    target_url: str = ""
    """Already passed through :func:`security.sanitize_url`."""
    credentials_present: bool = False
    login_ok: bool | None = None
    replan_count: int = 0
    force_proceeded: bool = False
    counts: dict[str, int | float] = Field(default_factory=dict)
    """Live tallies (flows, tests, passed, failed, heals, bugs, ...).

    Values are ``int`` except for elapsed-time entries such as ``duration_s``,
    which is why the union is here rather than a bare ``int``. The union order
    matters: pydantic's smart mode keeps whole numbers as ``int`` so the UI
    renders "3" rather than "3.0".
    """
    decision_log: list[DecisionEventOut] = Field(default_factory=list)
    risk_classifications: list[dict[str, Any]] = Field(default_factory=list)
    healer_actions: list[dict[str, Any]] = Field(default_factory=list)
    visual_findings: list[dict[str, Any]] = Field(default_factory=list)
    packaged_bugs: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    event_count: int = 0


class RunSummaryOut(BaseModel):
    """One row of ``GET /runs``. A deliberately small projection of a record."""

    run_id: str
    status: str = "queued"
    current_stage: str = ""
    target_url: str = ""
    started_at: str = ""
    finished_at: str | None = None
    credentials_present: bool = False
    replan_count: int = 0
    force_proceeded: bool = False
    bug_count: int = 0
    error: str | None = None


class BugSummaryOut(BaseModel):
    """One row of ``GET /run/{id}/bugs``: enough to render a list, no payloads."""

    bug_id: str
    flow_id: str = ""
    flow_name: str = ""
    title: str = ""
    classification: str = ""
    confidence: float = 0.0
    risk: str = "medium"
    severity: str = "major"
    labels: list[str] = Field(default_factory=list)
    has_screenshot: bool = False
    has_repro_script: bool = False
    created_at: str = ""
    detail_url: str = ""


class BugArtifactResponse(BaseModel):
    """A single packaged defect, with its artifacts inlined for the UI.

    ``screenshot_base64`` is capped (see :data:`api.app.MAX_SCREENSHOT_B64_CHARS`).
    When the image is larger than the cap the field is ``None``, ``note``
    explains why, and ``screenshot_path`` still points at the file on disk -
    an honest degraded response rather than a silent truncation that would
    render as a corrupt image.
    """

    bug_id: str
    flow_id: str = ""
    title: str = ""
    description: str = ""
    classification: str = ""
    confidence: float = 0.0
    risk: str = "medium"
    severity: str = "major"
    steps_to_reproduce: list[str] = Field(default_factory=list)
    expected: str = ""
    actual: str = ""
    labels: list[str] = Field(default_factory=list)
    repro_script: str = ""
    screenshot_base64: str | None = None
    screenshot_path: str | None = None
    ticket_markdown: str = ""
    created_at: str = ""
    note: str = ""


class HealthResponse(BaseModel):
    """Cheap liveness and configuration probe.

    ``playwright_ready`` is answered by a filesystem check only. Launching a
    browser to answer a health check would make the probe cost seconds and
    could exhaust the process's browser budget, so the honest, cheap signal is
    "the package imports and a chromium build directory exists"; ``note``
    carries the caveat when it is False.
    """

    status: str = "ok"
    version: str = ""
    llm_configured: bool = False
    llm_provider: str = "none"
    models: dict[str, str] = Field(default_factory=dict)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    active_runs: int = 0
    playwright_ready: bool = False
    note: str = ""


class ErrorResponse(BaseModel):
    """Uniform error body. ``detail`` is always redacted before it is built."""

    detail: str


__all__ = [
    "ALLOWED_URL_SCHEMES",
    "BugArtifactResponse",
    "BugSummaryOut",
    "CredentialsIn",
    "DecisionEventOut",
    "ErrorResponse",
    "HealthResponse",
    "RunAccepted",
    "RunRequest",
    "RunStatusResponse",
    "RunSummaryOut",
]
