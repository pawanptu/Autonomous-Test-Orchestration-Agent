"""Content-addressed cache for model answers about a page.

The expensive, repeated question in this pipeline is "what is on this page, and
which locator addresses this element". Two things make that answer cacheable:

* it is a pure function of the page's *interactive structure*, not of the
  wall-clock time at which it was asked; and
* the crawler already computes a fingerprint of exactly that structure
  (:func:`browser.crawler.fingerprint_page`).

So the cache key is the canonical URL plus that DOM fingerprint plus the task
name. A re-run against an unchanged page is free; a page that grew a form field
gets a new fingerprint and is re-asked, which is the behaviour a caching layer
for a *testing* tool must have - a stale locator would be reported as a passing
test against markup that no longer exists.

Scope
-----
Process-local and bounded. There is deliberately no disk persistence: a cache
that survived a restart would let a locator validated against last week's DOM
be trusted today, and the failure mode (a green test that never touched the
application) is worse than the cost it saves.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any

from logging_setup import get_logger

log = get_logger("aivor.llm.cache")


def make_key(task: str, url: str, fingerprint: str, extra: Any = None) -> str:
    """Build the cache key for one question about one page state.

    ``extra`` covers anything else that changes the answer - the step being
    resolved, the model role - and is serialised deterministically so that two
    equal payloads produce one key.
    """
    payload = json.dumps(
        {"task": task, "url": url, "fp": fingerprint, "extra": extra},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:32]


@dataclass
class CacheStats:
    """Hit/miss accounting, reported per run so the saving is measurable."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.lookups) if self.lookups else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 3),
        }


class ResponseCache:
    """Bounded, thread-safe, in-process cache of model answers.

    Eviction is first-in-first-out rather than least-recently-used: entries are
    keyed by a DOM fingerprint that changes when the page changes, so the value
    of an entry does not decay with time and recency carries no signal worth the
    extra bookkeeping.
    """

    def __init__(self, max_entries: int = 512, *, enabled: bool = True) -> None:
        self.max_entries = max(1, int(max_entries))
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._order: list[str] = []

    def get(self, key: str) -> tuple[bool, Any]:
        """Return ``(hit, value)``.

        A two-tuple rather than ``None``-means-miss because a cached answer is
        legitimately allowed to be ``None`` or an empty list.
        """
        if not self.enabled:
            return False, None
        with self._lock:
            if key in self._data:
                self.stats.hits += 1
                return True, self._data[key]
            self.stats.misses += 1
            return False, None

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            if key not in self._data:
                self._order.append(key)
                self.stats.stores += 1
            self._data[key] = value
            while len(self._order) > self.max_entries:
                oldest = self._order.pop(0)
                self._data.pop(oldest, None)
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._order.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


@dataclass
class StageMetrics:
    """Per-stage cost and latency accounting.

    Recorded for every pipeline stage - crawl, planning, coverage, generation,
    execution, healing, visual comparison - so that "LLM cost went up" can be
    attributed to a stage instead of guessed at.
    """

    stage: str
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


# Published per-million-token prices for the models this agent routes to.
# Groq's free tier bills nothing, so these produce an *equivalent* cost: what
# the same traffic would cost at list price. That is the number worth watching,
# because it is what the pipeline would cost if promoted off the free tier.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model substring -> (prompt, completion)
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.10, 0.40),
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}

DEFAULT_PRICE_USD_PER_MTOK: tuple[float, float] = (0.20, 0.60)
"""Fallback used for an unrecognised model, so an unknown model still produces
a number rather than silently costing zero."""


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """List-price equivalent cost of one call."""
    lowered = (model or "").lower()
    prices = DEFAULT_PRICE_USD_PER_MTOK
    for fragment, values in MODEL_PRICES_USD_PER_MTOK.items():
        if fragment in lowered:
            prices = values
            break
    return (prompt_tokens / 1_000_000) * prices[0] + (completion_tokens / 1_000_000) * prices[1]


class MetricsRecorder:
    """Collects :class:`StageMetrics` for one run.

    The LLM client writes into this through :meth:`record_call`; stages that
    avoid a call entirely record that through :meth:`record_cache_hit`, which is
    what makes the saving visible rather than merely absent from the totals.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stages: dict[str, StageMetrics] = {}

    def _stage(self, stage: str) -> StageMetrics:
        key = stage or "unattributed"
        if key not in self._stages:
            self._stages[key] = StageMetrics(stage=key)
        return self._stages[key]

    def record_call(
        self,
        stage: str,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
        attempts: int = 1,
        failed: bool = False,
    ) -> None:
        with self._lock:
            metrics = self._stage(stage)
            metrics.calls += 1
            metrics.prompt_tokens += max(0, prompt_tokens)
            metrics.completion_tokens += max(0, completion_tokens)
            metrics.latency_s += max(0.0, latency_s)
            metrics.retries += max(0, attempts - 1)
            if failed:
                metrics.failures += 1
            metrics.estimated_cost_usd += estimate_cost_usd(
                model, prompt_tokens, completion_tokens
            )

    def record_cache_hit(self, stage: str) -> None:
        with self._lock:
            self._stage(stage).cache_hits += 1

    def record_failure(self, stage: str) -> None:
        with self._lock:
            self._stage(stage).failures += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            stages = [m.as_dict() for m in self._stages.values()]
        stages.sort(key=lambda entry: -float(entry["estimated_cost_usd"]))
        return {
            "by_stage": stages,
            "totals": {
                "calls": sum(int(s["calls"]) for s in stages),
                "cache_hits": sum(int(s["cache_hits"]) for s in stages),
                "retries": sum(int(s["retries"]) for s in stages),
                "failures": sum(int(s["failures"]) for s in stages),
                "total_tokens": sum(int(s["total_tokens"]) for s in stages),
                "latency_s": round(sum(float(s["latency_s"]) for s in stages), 3),
                "estimated_cost_usd": round(
                    sum(float(s["estimated_cost_usd"]) for s in stages), 6
                ),
            },
        }
