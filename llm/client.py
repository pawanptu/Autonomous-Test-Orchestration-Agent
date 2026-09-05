"""Provider-agnostic chat-completion client with retry and model routing.

Design
------
*Roles, not model names.* Callers ask for :attr:`ModelRole.REASONING` or
:attr:`ModelRole.CODEGEN` and the client picks the concrete model for whichever
provider is answering. That is what makes the fallback seam real: adding the
Gemini free tier is a provider registration, not a change to any caller.

*Rate-limit awareness.* Groq's free tier is generous but finite. The 70B
reasoning model is gated behind a per-role minimum interval and the mechanical
plan-to-code work is routed to the 8B model. Every 429 is retried with
exponential backoff plus jitter, honouring ``Retry-After`` when present.

*One JSON repair.* :meth:`LLMClient.complete_json` parses defensively, and on
failure re-asks exactly once with the parser error attached before giving up.
The calling node then records an error event instead of crashing the run.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel

from config import Settings, get_settings
from llm.json_utils import (
    JSONParseError,
    build_retry_messages,
    loads_lenient,
)
from logging_setup import get_logger
from security import redact_text

log = get_logger("aivor.llm")

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]


class ModelRole(str, Enum):
    """What the call is for, which decides which model tier answers it."""

    REASONING = "reasoning"
    """Judgment: coverage evaluation, risk ranking, defect classification,
    confidence scoring, orchestrator routing, report synthesis. Large model."""

    CODEGEN = "codegen"
    """Mechanical translation: plan steps to Playwright calls, ticket prose.
    Small, fast, cheap model. Never the 70B."""


class LLMError(RuntimeError):
    """Base class for every failure the LLM layer surfaces."""


class LLMAuthError(LLMError):
    """Bad or missing API key. Never retried."""


class LLMRateLimitError(LLMError):
    """429 that survived the whole retry budget."""


class LLMUnavailableError(LLMError):
    """No provider is configured, or every provider failed."""


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_s: float = 0.0
    attempts: int = 1
    used_fallback: bool = False


class Provider(Protocol):
    """The seam every backend implements."""

    name: str

    def resolve_model(self, role: ModelRole) -> str: ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------
# OpenAI-compatible HTTP provider (Groq today, Gemini free tier tomorrow)
# --------------------------------------------------------------------------
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class OpenAICompatibleProvider:
    """Any backend that speaks ``POST /chat/completions``.

    Groq and Google's Gemini OpenAI-compatibility endpoint both do, so one
    implementation covers the required provider and the optional free-tier
    fallback without a second HTTP client.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        settings: Settings,
        model_map: dict[ModelRole, str],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.settings = settings
        self.model_map = model_map
        self._extra_headers = extra_headers or {}
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    def resolve_model(self, role: ModelRole) -> str:
        return self.model_map[role]

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self.base_url,
                        timeout=httpx.Timeout(self.settings.llm_timeout_s),
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                            **self._extra_headers,
                        },
                    )
        return self._client

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        client = await self._http()
        started = time.monotonic()
        last_error: Exception | None = None
        allow_json_mode = json_mode

        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                response = await client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                log.warning(
                    "%s transport error on attempt %d/%d: %s",
                    self.name,
                    attempt,
                    self.settings.llm_max_retries,
                    exc,
                )
                await self._sleep_backoff(attempt)
                continue

            if response.status_code in (401, 403):
                raise LLMAuthError(
                    f"{self.name} rejected the API key (HTTP {response.status_code}). "
                    "Check GROQ_API_KEY."
                )

            if response.status_code == 400 and allow_json_mode:
                # Some models reject response_format. Drop it and retry once.
                body = redact_text(response.text)[:300]
                log.warning("%s rejected json_mode, retrying without it: %s", self.name, body)
                payload.pop("response_format", None)
                allow_json_mode = False
                continue

            if response.status_code in _RETRYABLE_STATUS:
                last_error = LLMError(
                    f"{self.name} HTTP {response.status_code}: {redact_text(response.text)[:300]}"
                )
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                log.warning(
                    "%s HTTP %s on attempt %d/%d (retry_after=%s)",
                    self.name,
                    response.status_code,
                    attempt,
                    self.settings.llm_max_retries,
                    retry_after,
                )
                await self._sleep_backoff(attempt, retry_after)
                continue

            if response.status_code >= 400:
                raise LLMError(
                    f"{self.name} HTTP {response.status_code}: "
                    f"{redact_text(response.text)[:500]}"
                )

            data = response.json()
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"{self.name} returned an unexpected payload shape") from exc

            usage_raw = data.get("usage") or {}
            return LLMResponse(
                text=text,
                model=data.get("model", model),
                provider=self.name,
                usage=LLMUsage(
                    prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
                    completion_tokens=int(usage_raw.get("completion_tokens", 0)),
                    total_tokens=int(usage_raw.get("total_tokens", 0)),
                ),
                latency_s=time.monotonic() - started,
                attempts=attempt,
            )

        message = f"{self.name} exhausted {self.settings.llm_max_retries} attempts: {last_error}"
        if isinstance(last_error, LLMError) and "429" in str(last_error):
            raise LLMRateLimitError(message)
        raise LLMError(message)

    async def _sleep_backoff(self, attempt: int, retry_after: float | None = None) -> None:
        """Exponential backoff with full jitter, floored by ``Retry-After``."""
        base = self.settings.llm_backoff_base_s * (2 ** (attempt - 1))
        delay = min(base, self.settings.llm_backoff_max_s)
        delay = random.uniform(delay * 0.5, delay)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.settings.llm_backoff_max_s))
        await asyncio.sleep(delay)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class _RoleThrottle:
    """Minimum spacing between calls of a given role.

    Cheap insurance against the free-tier requests-per-minute ceiling. It is a
    floor on spacing, not a token-bucket: the retry loop handles real 429s.
    """

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval_s - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class LLMClient:
    """Routes prompts to the right model tier, with retry and one JSON repair."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.providers: list[Provider] = _build_providers(self.settings)
        self.usage_by_model: dict[str, LLMUsage] = {}
        self.call_count = 0
        self.failure_count = 0
        self._throttles = {
            ModelRole.REASONING: _RoleThrottle(float(os.getenv("REASONING_MIN_INTERVAL_S", "1.2"))),
            ModelRole.CODEGEN: _RoleThrottle(float(os.getenv("CODEGEN_MIN_INTERVAL_S", "0.35"))),
        }

    # -- introspection -----------------------------------------------------
    @property
    def primary_name(self) -> str:
        return self.providers[0].name if self.providers else "none"

    def models_used(self) -> dict[str, str]:
        primary = self.providers[0] if self.providers else None
        if primary is None:
            return {}
        return {
            "reasoning": primary.resolve_model(ModelRole.REASONING),
            "codegen": primary.resolve_model(ModelRole.CODEGEN),
        }

    def usage_summary(self) -> dict[str, Any]:
        total = LLMUsage()
        for usage in self.usage_by_model.values():
            total.add(usage)
        return {
            "calls": self.call_count,
            "failures": self.failure_count,
            "total_tokens": total.total_tokens,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
            "by_model": {
                model: {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                }
                for model, u in self.usage_by_model.items()
            },
        }

    # -- core --------------------------------------------------------------
    async def complete(
        self,
        role: ModelRole,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        task: str = "",
    ) -> LLMResponse:
        """Run one completion, falling through the provider chain on failure."""
        if not self.providers:
            raise LLMUnavailableError(
                "No LLM provider configured. Set GROQ_API_KEY in .env, or set "
                "LLM_OFFLINE_MODE=true for a plumbing-only smoke run."
            )
        if temperature is None:
            temperature = (
                self.settings.llm_temperature_reasoning
                if role is ModelRole.REASONING
                else self.settings.llm_temperature_codegen
            )
        max_tokens = max_tokens or self.settings.llm_max_tokens

        await self._throttles[role].acquire()

        errors: list[str] = []
        for index, provider in enumerate(self.providers):
            model = provider.resolve_model(role)
            try:
                response = await provider.complete(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except LLMAuthError:
                raise
            except LLMError as exc:
                errors.append(f"{provider.name}: {exc}")
                self.failure_count += 1
                log.warning("provider %s failed for task=%s: %s", provider.name, task, exc)
                continue
            response.used_fallback = index > 0
            self.call_count += 1
            self.usage_by_model.setdefault(response.model, LLMUsage()).add(response.usage)
            log.debug(
                "llm task=%s role=%s model=%s tokens=%s latency=%.2fs",
                task,
                role.value,
                response.model,
                response.usage.total_tokens,
                response.latency_s,
            )
            return response

        raise LLMUnavailableError("all providers failed -> " + " | ".join(errors))

    async def complete_json(
        self,
        role: ModelRole,
        messages: Sequence[Message],
        *,
        model_cls: Type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str = "",
    ) -> Any:
        """Complete and parse JSON, with exactly one repair round-trip.

        Returns a validated ``model_cls`` instance when one is supplied, else
        the parsed Python object. Raises :class:`llm.json_utils.JSONParseError`
        if the second attempt is still unusable.
        """
        base_messages = list(messages)
        response = await self.complete(
            role,
            base_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            task=task,
        )
        try:
            return _validate(response.text, model_cls)
        except JSONParseError as first_error:
            log.warning(
                "task=%s produced unparsable JSON, retrying once: %s",
                task,
                first_error,
            )
            retry_messages = build_retry_messages(
                base_messages, response.text, str(first_error)
            )
            retry = await self.complete(
                role,
                retry_messages,
                temperature=0.0,
                max_tokens=max_tokens,
                json_mode=True,
                task=f"{task}:json-repair",
            )
            try:
                return _validate(retry.text, model_cls)
            except JSONParseError as second_error:
                self.failure_count += 1
                raise JSONParseError(
                    f"task={task} returned invalid JSON twice. "
                    f"first={first_error}; second={second_error}",
                    raw=retry.text,
                ) from second_error

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()


def _validate(text: str, model_cls: Type[T] | None) -> Any:
    data = loads_lenient(text)
    if model_cls is None:
        return data
    try:
        return model_cls.model_validate(data)
    except Exception as exc:  # pydantic ValidationError and friends
        raise JSONParseError(f"schema validation failed: {exc}", text) from exc


# --------------------------------------------------------------------------
# Provider registry
# --------------------------------------------------------------------------
def _build_providers(settings: Settings) -> list[Provider]:
    """Assemble the provider chain: primary first, fallbacks after.

    Offline mode short-circuits the whole chain with the deterministic stub so
    that the graph, API and UI can be exercised without any key at all.
    """
    if settings.llm_offline_mode:
        from llm.offline_stub import OfflineStubProvider

        log.warning(
            "LLM_OFFLINE_MODE is enabled: responses come from a deterministic "
            "stub, NOT from a model. Reports produced in this mode are labelled."
        )
        return [OfflineStubProvider()]

    providers: list[Provider] = []
    if settings.groq_api_key:
        providers.append(
            OpenAICompatibleProvider(
                name="groq",
                base_url=settings.groq_base_url,
                api_key=settings.groq_api_key,
                settings=settings,
                model_map={
                    ModelRole.REASONING: settings.model_reasoning,
                    ModelRole.CODEGEN: settings.model_codegen,
                },
            )
        )

    # Optional free-tier fallback. Google exposes an OpenAI-compatible
    # endpoint, so the same provider class covers it with no caller changes.
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        providers.append(
            OpenAICompatibleProvider(
                name="gemini",
                base_url=os.getenv(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai",
                ),
                api_key=gemini_key,
                settings=settings,
                model_map={
                    ModelRole.REASONING: os.getenv("GEMINI_MODEL_REASONING", "gemini-2.0-flash"),
                    ModelRole.CODEGEN: os.getenv("GEMINI_MODEL_CODEGEN", "gemini-2.0-flash-lite"),
                },
            )
        )
    return providers
