"""LLM caching, cost accounting and deterministic routing.

Three separate savings are pinned here, because each fails differently:

* **Caching** must key on the page's *structure*, so an unchanged page is free
  and a changed page is re-asked. A cache that ignored the change would hand
  back a locator for markup that no longer exists, and the resulting green test
  would never have touched the application.
* **Accounting** must attribute cost to a pipeline stage, so a regression can be
  found rather than guessed at.
* **Deterministic routing** must replace only the decisions that were never
  judgment calls in the first place.
"""

from __future__ import annotations

import pytest

from llm.cache import (
    MetricsRecorder,
    ResponseCache,
    estimate_cost_usd,
    make_key,
)
from llm.client import stage_of


class TestCacheKeys:
    def test_same_question_about_the_same_page_state_is_one_key(self):
        a = make_key("selector", "https://ex.com/login", "fp1", {"step": 2})
        b = make_key("selector", "https://ex.com/login", "fp1", {"step": 2})
        assert a == b

    def test_a_changed_fingerprint_produces_a_different_key(self):
        """A page that grew a form field must be re-asked, not served stale.

        This is the assertion that keeps the cache honest: serving a stale
        locator would produce a passing test against markup that is gone.
        """
        before = make_key("selector", "https://ex.com/login", "fp1")
        after = make_key("selector", "https://ex.com/login", "fp2")
        assert before != after

    def test_different_urls_produce_different_keys(self):
        assert make_key("selector", "https://ex.com/a", "fp") != make_key(
            "selector", "https://ex.com/b", "fp"
        )

    def test_different_tasks_produce_different_keys(self):
        assert make_key("selector", "u", "fp") != make_key("discovery", "u", "fp")

    def test_extra_payload_ordering_does_not_matter(self):
        assert make_key("t", "u", "fp", {"a": 1, "b": 2}) == make_key(
            "t", "u", "fp", {"b": 2, "a": 1}
        )


class TestResponseCache:
    def test_a_stored_value_is_returned_on_the_next_lookup(self):
        cache = ResponseCache()
        key = make_key("t", "u", "fp")
        assert cache.get(key) == (False, None)
        cache.put(key, {"locator": "page.get_by_test_id('x')"})
        hit, value = cache.get(key)
        assert hit is True
        assert value["locator"] == "page.get_by_test_id('x')"

    def test_hit_and_miss_are_counted(self):
        cache = ResponseCache()
        key = make_key("t", "u", "fp")
        cache.get(key)
        cache.put(key, 1)
        cache.get(key)
        stats = cache.stats.as_dict()
        assert stats["hits"] == 1 and stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_a_cached_none_is_distinguishable_from_a_miss(self):
        """``None`` is a legitimate answer, which is why get returns a pair."""
        cache = ResponseCache()
        key = make_key("t", "u", "fp")
        cache.put(key, None)
        assert cache.get(key) == (True, None)

    def test_the_cache_is_bounded(self):
        cache = ResponseCache(max_entries=3)
        for index in range(6):
            cache.put(make_key("t", "u", f"fp{index}"), index)
        assert len(cache) == 3
        assert cache.stats.evictions == 3

    def test_disabling_the_cache_makes_every_lookup_a_miss(self):
        cache = ResponseCache(enabled=False)
        key = make_key("t", "u", "fp")
        cache.put(key, 1)
        assert cache.get(key) == (False, None)

    def test_clear_empties_the_cache(self):
        cache = ResponseCache()
        cache.put(make_key("t", "u", "fp"), 1)
        cache.clear()
        assert len(cache) == 0


class TestCostEstimation:
    def test_known_models_are_priced(self):
        assert estimate_cost_usd("openai/gpt-oss-120b", 1_000_000, 0) == pytest.approx(0.15)

    def test_completion_tokens_cost_more_than_prompt_tokens(self):
        prompt_only = estimate_cost_usd("openai/gpt-oss-120b", 1_000_000, 0)
        completion_only = estimate_cost_usd("openai/gpt-oss-120b", 0, 1_000_000)
        assert completion_only > prompt_only

    def test_an_unknown_model_still_produces_a_number(self):
        """Silently costing zero would hide an expensive misrouted call."""
        assert estimate_cost_usd("some-new-model", 1_000_000, 1_000_000) > 0

    def test_zero_tokens_cost_nothing(self):
        assert estimate_cost_usd("openai/gpt-oss-20b", 0, 0) == 0.0


class TestStageAttribution:
    @pytest.mark.parametrize(
        "task,stage",
        [
            ("generator:F001", "generator"),
            ("coverage_gate:rev0", "coverage_gate"),
            ("report:synthesis", "report"),
            ("planner", "planner"),
        ],
    )
    def test_stage_is_the_task_prefix(self, task, stage):
        assert stage_of(task) == stage

    def test_an_unlabelled_call_is_attributed_not_dropped(self):
        """A missing label must show up as a gap, not vanish from the totals."""
        assert stage_of("") == "unattributed"


class TestMetricsRecorder:
    def test_calls_roll_up_per_stage(self):
        metrics = MetricsRecorder()
        metrics.record_call(
            "generator", model="openai/gpt-oss-20b", prompt_tokens=1000,
            completion_tokens=500, latency_s=1.5,
        )
        metrics.record_call(
            "generator", model="openai/gpt-oss-20b", prompt_tokens=500,
            completion_tokens=250, latency_s=0.5,
        )
        stages = {entry["stage"]: entry for entry in metrics.summary()["by_stage"]}
        assert stages["generator"]["calls"] == 2
        assert stages["generator"]["total_tokens"] == 2250
        assert stages["generator"]["latency_s"] == pytest.approx(2.0)

    def test_retries_are_counted_from_the_attempt_number(self):
        metrics = MetricsRecorder()
        metrics.record_call(
            "planner", model="m", prompt_tokens=10, completion_tokens=10,
            latency_s=0.1, attempts=3,
        )
        stages = {e["stage"]: e for e in metrics.summary()["by_stage"]}
        assert stages["planner"]["retries"] == 2

    def test_cache_hits_are_recorded_where_no_call_happened(self):
        """The saving must be visible, not merely absent from the totals."""
        metrics = MetricsRecorder()
        metrics.record_cache_hit("selectors")
        metrics.record_cache_hit("selectors")
        stages = {e["stage"]: e for e in metrics.summary()["by_stage"]}
        assert stages["selectors"]["cache_hits"] == 2
        assert stages["selectors"]["calls"] == 0

    def test_failures_are_recorded(self):
        metrics = MetricsRecorder()
        metrics.record_failure("healer")
        stages = {e["stage"]: e for e in metrics.summary()["by_stage"]}
        assert stages["healer"]["failures"] == 1

    def test_totals_aggregate_every_stage(self):
        metrics = MetricsRecorder()
        metrics.record_call(
            "a", model="openai/gpt-oss-20b", prompt_tokens=100,
            completion_tokens=100, latency_s=0.1,
        )
        metrics.record_call(
            "b", model="openai/gpt-oss-120b", prompt_tokens=100,
            completion_tokens=100, latency_s=0.2,
        )
        totals = metrics.summary()["totals"]
        assert totals["calls"] == 2
        assert totals["total_tokens"] == 400
        assert totals["estimated_cost_usd"] > 0

    def test_stages_are_ordered_by_cost(self):
        """The report leads with whichever stage is worth optimising."""
        metrics = MetricsRecorder()
        metrics.record_call(
            "cheap", model="openai/gpt-oss-20b", prompt_tokens=10,
            completion_tokens=10, latency_s=0.1,
        )
        metrics.record_call(
            "expensive", model="openai/gpt-oss-120b", prompt_tokens=500_000,
            completion_tokens=500_000, latency_s=0.1,
        )
        assert metrics.summary()["by_stage"][0]["stage"] == "expensive"


class TestDeterministicRouting:
    """The coverage gate must not spend a reasoning call to confirm arithmetic."""

    def test_a_passing_gate_needs_no_model_call(self):
        import inspect

        from agents.orchestrator import route_after_coverage

        source = inspect.getsource(route_after_coverage)
        # The passing branch returns before any provider is consulted.
        assert source.index("if evaluation.passed") < source.index("llm.complete_json")

    def test_an_exhausted_budget_needs_no_model_call(self):
        import inspect

        from agents.orchestrator import route_after_coverage

        source = inspect.getsource(route_after_coverage)
        assert source.index("replan_count >= REPLAN_CAP") < source.index("llm.complete_json")

    def test_an_unambiguous_replan_short_circuits_before_the_model(self):
        """With budget remaining, a failed gate admits exactly one action.

        The model's only freedom is replan-versus-escalate, and it cannot wave a
        failing plan through. Confirming the obvious choice costs a 70B call per
        cycle for no decision.
        """
        import inspect

        from agents.orchestrator import route_after_coverage

        source = inspect.getsource(route_after_coverage)
        assert "if replan_count < REPLAN_CAP - 1" in source
        assert source.index("if replan_count < REPLAN_CAP - 1") < source.index(
            "llm.complete_json"
        )
