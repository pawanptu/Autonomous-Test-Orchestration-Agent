"""Operational limits and observability for a service that drives real browsers.

Every run in this system costs a browser process, a share of a rate-limited LLM
tier, and - most importantly - traffic against somebody else's application. The
controls here exist so that those costs stay bounded and attributable:

* :class:`TargetQueue` admits work per *host*, not just globally, so two runs
  never hammer one target concurrently.
* :class:`RateLimiter` spaces navigations against a host, so a crawl reads as
  browsing rather than as a load test.
* :class:`Budget` gives a run a wall-clock ceiling that is checked at stage
  boundaries and propagates cancellation into the browser.
* :class:`SpanRecorder` times each stage and each sub-operation, so "the run
  took nine minutes" becomes "healing took seven of them".

Errors are surfaced, never swallowed
------------------------------------
Nothing in this module catches an exception to keep a run alive. A span records
the failure and re-raises; the queue releases its slot in a ``finally`` and lets
the error propagate. The one deliberate exception is
:meth:`SpanRecorder.span`'s bookkeeping, which must not mask the original error
with an accounting bug.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from logging_setup import get_logger

log = get_logger("aivor.ops")


def host_of(url: str) -> str:
    """Host key used for per-target limits. Port included, userinfo dropped."""
    try:
        parts = urlsplit(url or "")
        host = (parts.hostname or "").lower()
        return f"{host}:{parts.port}" if parts.port else host or "unknown"
    except ValueError:
        return "unknown"


class BudgetExceeded(RuntimeError):
    """A run exceeded its wall-clock budget and was stopped."""


@dataclass
class Budget:
    """Wall-clock ceiling for one run.

    Checked at stage boundaries rather than enforced by a timer, so that a stage
    is never killed halfway through writing an artifact. A run that blows the
    budget stops cleanly with the stages it completed recorded.
    """

    limit_s: float
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.limit_s - self.elapsed_s)

    @property
    def exhausted(self) -> bool:
        return self.elapsed_s >= self.limit_s

    def check(self, stage: str) -> None:
        """Raise :class:`BudgetExceeded` when the run has run out of time."""
        if self.exhausted:
            raise BudgetExceeded(
                f"run exceeded its {self.limit_s:.0f}s time budget at stage {stage!r} "
                f"(elapsed {self.elapsed_s:.0f}s); raise RUN_TIME_BUDGET_S to allow longer runs"
            )


class RateLimiter:
    """Minimum spacing between operations against one host.

    A simple spacing floor rather than a token bucket: the goal is to look like
    a person browsing, and a bucket would permit an initial burst that is
    exactly what a target's own rate limiter reacts to.
    """

    def __init__(self, per_second: float) -> None:
        self.min_interval_s = 1.0 / per_second if per_second > 0 else 0.0
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, host: str) -> float:
        """Wait until another operation against ``host`` is allowed.

        Returns how long the caller was delayed, which the span recorder
        attributes so that "slow crawl" can be told apart from "throttled crawl".
        """
        if self.min_interval_s <= 0:
            return 0.0
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval_s - (now - self._last.get(host, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
                self._last[host] = time.monotonic()
                return wait
            self._last[host] = now
            return 0.0


class TargetQueue:
    """Admission control keyed by target host as well as globally.

    Two limits apply to every run: a global ceiling on concurrent runs (the
    machine only has so many browsers in it) and a per-host ceiling (somebody
    else's application only has so much patience). Both are released in a
    ``finally`` so a crashing run cannot leak a slot.
    """

    def __init__(self, *, global_limit: int, per_target_limit: int) -> None:
        self.global_limit = max(1, int(global_limit))
        self.per_target_limit = max(1, int(per_target_limit))
        self._global = asyncio.Semaphore(self.global_limit)
        self._per_host: dict[str, asyncio.Semaphore] = {}
        self._waiting: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def _host_semaphore(self, host: str) -> asyncio.Semaphore:
        async with self._lock:
            if host not in self._per_host:
                self._per_host[host] = asyncio.Semaphore(self.per_target_limit)
            return self._per_host[host]

    @asynccontextmanager
    async def admit(self, url: str) -> AsyncIterator[str]:
        """Acquire a global slot and a per-host slot for the duration of a run.

        The global slot is taken first and released last. Acquiring in a
        consistent order across all callers is what makes this deadlock-free.
        """
        host = host_of(url)
        semaphore = await self._host_semaphore(host)
        async with self._lock:
            self._waiting[host] = self._waiting.get(host, 0) + 1
        try:
            await self._global.acquire()
            try:
                await semaphore.acquire()
                try:
                    log.debug("admitted run against %s", host)
                    yield host
                finally:
                    semaphore.release()
            finally:
                self._global.release()
        finally:
            async with self._lock:
                self._waiting[host] = max(0, self._waiting.get(host, 1) - 1)

    def snapshot(self) -> dict[str, Any]:
        """Queue depth, for ``/health`` and the operator UI."""
        return {
            "global_limit": self.global_limit,
            "per_target_limit": self.per_target_limit,
            "waiting_by_host": {h: n for h, n in self._waiting.items() if n},
        }


@dataclass
class Span:
    """One timed operation."""

    name: str
    stage: str
    duration_s: float
    ok: bool
    error: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "duration_s": round(self.duration_s, 3),
            "ok": self.ok,
            "error": self.error,
            **({"attributes": self.attributes} if self.attributes else {}),
        }


class SpanRecorder:
    """Collects timings for one run.

    Deliberately not OpenTelemetry: the repository has no tracing dependency and
    adding one for a handful of durations would be a poor trade. The shape is
    span-compatible, so exporting later is a mapping rather than a rewrite.
    """

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def span(
        self, name: str, *, stage: str = "", **attributes: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Time a block, recording success or the exception that ended it.

        The exception is always re-raised. This records what happened; it does
        not decide what happens next, and it never converts a failure into a
        silent success.
        """
        started = time.monotonic()
        extra: dict[str, Any] = dict(attributes)
        try:
            yield extra
        except BaseException as exc:
            self.spans.append(
                Span(
                    name=name,
                    stage=stage or name,
                    duration_s=time.monotonic() - started,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    attributes=extra,
                )
            )
            raise
        else:
            self.spans.append(
                Span(
                    name=name,
                    stage=stage or name,
                    duration_s=time.monotonic() - started,
                    ok=True,
                    attributes=extra,
                )
            )

    def summary(self) -> dict[str, Any]:
        """Per-stage roll-up plus the slowest individual operations."""
        by_stage: dict[str, dict[str, Any]] = {}
        for span in self.spans:
            entry = by_stage.setdefault(
                span.stage, {"stage": span.stage, "count": 0, "duration_s": 0.0, "failures": 0}
            )
            entry["count"] += 1
            entry["duration_s"] += span.duration_s
            if not span.ok:
                entry["failures"] += 1
        stages = sorted(by_stage.values(), key=lambda e: -float(e["duration_s"]))
        for entry in stages:
            entry["duration_s"] = round(float(entry["duration_s"]), 3)
        slowest = sorted(self.spans, key=lambda s: -s.duration_s)[:10]
        return {
            "by_stage": stages,
            "slowest": [s.as_dict() for s in slowest],
            "total_spans": len(self.spans),
            "failed_spans": sum(1 for s in self.spans if not s.ok),
        }


class ContextPool:
    """Ceiling on browser contexts open at once within a single run.

    Each context is a browser profile with its own memory. Parallel execution
    without this bound is the fastest way to make the machine swap and then to
    blame the target application for the resulting timeouts.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._semaphore = asyncio.Semaphore(self.limit)
        self.peak_in_use = 0
        self._in_use = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[None]:
        await self._semaphore.acquire()
        async with self._lock:
            self._in_use += 1
            self.peak_in_use = max(self.peak_in_use, self._in_use)
        try:
            yield
        finally:
            async with self._lock:
                self._in_use -= 1
            self._semaphore.release()


def effective_parallelism(settings: Any) -> tuple[int, str]:
    """How many flows may execute at once, and the reason for that number.

    Parallel execution stays off unless host-level rate limiting is actually in
    force. Running flows concurrently against an unthrottled target converts a
    test run into a denial-of-service attempt on somebody else's staging box,
    so the flag alone is not sufficient authority.
    """
    if not getattr(settings, "enable_parallel_execution", False):
        return 1, "sequential: ENABLE_PARALLEL_EXECUTION is off"
    rate = float(getattr(settings, "target_rate_limit_per_s", 0.0) or 0.0)
    if rate <= 0:
        return 1, (
            "sequential: parallel execution was requested but TARGET_RATE_LIMIT_PER_S "
            "is unset, so there is no host-level throttle to keep concurrent flows civil"
        )
    limit = max(1, int(getattr(settings, "max_parallel_flows", 1)))
    return limit, f"parallel: up to {limit} flows, throttled to {rate:.1f} navigations/s per host"
