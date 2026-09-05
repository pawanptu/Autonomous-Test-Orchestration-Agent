"""Queue limits, rate limiting, cancellation and span instrumentation.

Every run costs a browser process, a share of a rate-limited model tier, and
traffic against somebody else's application. The last of those is the one that
matters most: an unthrottled parallel run against a staging box is
indistinguishable from a denial-of-service attempt, and it is not the target
owner's job to absorb that.
"""

from __future__ import annotations

import asyncio

import pytest

from config import Settings
from ops import (
    Budget,
    BudgetExceeded,
    ContextPool,
    RateLimiter,
    SpanRecorder,
    TargetQueue,
    effective_parallelism,
    host_of,
)


class TestHostKey:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://Example.COM/a", "example.com"),
            ("https://example.com:8443/a", "example.com:8443"),
            ("http://user:pass@example.com/a", "example.com"),
        ],
    )
    def test_host_key_normalises(self, url, expected):
        assert host_of(url) == expected

    def test_credentials_never_appear_in_the_host_key(self):
        """The key reaches logs and metrics labels."""
        assert "pass" not in host_of("http://user:pass@example.com/")

    def test_unparsable_url_gets_a_sentinel(self):
        assert host_of("") == "unknown"


class TestTargetQueue:
    async def test_runs_against_one_host_are_serialised(self):
        """Two runs must never hammer a single target concurrently."""
        queue = TargetQueue(global_limit=4, per_target_limit=1)
        concurrent = 0
        peak = 0

        async def run() -> None:
            nonlocal concurrent, peak
            async with queue.admit("https://one.test/"):
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0.02)
                concurrent -= 1

        await asyncio.gather(*(run() for _ in range(4)))
        assert peak == 1

    async def test_different_hosts_run_concurrently(self):
        """The per-target limit must not become a global bottleneck."""
        queue = TargetQueue(global_limit=4, per_target_limit=1)
        concurrent = 0
        peak = 0

        async def run(host: str) -> None:
            nonlocal concurrent, peak
            async with queue.admit(f"https://{host}/"):
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0.03)
                concurrent -= 1

        await asyncio.gather(*(run(f"h{i}.test") for i in range(3)))
        assert peak > 1

    async def test_global_limit_caps_total_concurrency(self):
        queue = TargetQueue(global_limit=2, per_target_limit=5)
        concurrent = 0
        peak = 0

        async def run(index: int) -> None:
            nonlocal concurrent, peak
            async with queue.admit(f"https://h{index}.test/"):
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0.02)
                concurrent -= 1

        await asyncio.gather(*(run(i) for i in range(6)))
        assert peak <= 2

    async def test_a_failing_run_releases_its_slot(self):
        """A leaked semaphore would wedge the service until restart."""
        queue = TargetQueue(global_limit=1, per_target_limit=1)

        with pytest.raises(ValueError):
            async with queue.admit("https://one.test/"):
                raise ValueError("boom")

        async with queue.admit("https://one.test/"):
            pass  # acquiring again proves the slot came back

    async def test_cancellation_releases_its_slot(self):
        """Cancellation must propagate without stranding capacity."""
        queue = TargetQueue(global_limit=1, per_target_limit=1)
        started = asyncio.Event()

        async def long_run() -> None:
            async with queue.admit("https://one.test/"):
                started.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(long_run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.wait_for(queue.admit("https://one.test/").__aenter__(), timeout=1.0)

    def test_snapshot_reports_configuration(self):
        queue = TargetQueue(global_limit=3, per_target_limit=2)
        snapshot = queue.snapshot()
        assert snapshot["global_limit"] == 3
        assert snapshot["per_target_limit"] == 2


class TestRateLimiter:
    async def test_operations_against_one_host_are_spaced(self):
        limiter = RateLimiter(per_second=50.0)  # 20ms apart
        await limiter.acquire("one.test")
        delayed = await limiter.acquire("one.test")
        assert delayed > 0

    async def test_different_hosts_do_not_throttle_each_other(self):
        limiter = RateLimiter(per_second=50.0)
        await limiter.acquire("one.test")
        assert await limiter.acquire("two.test") == 0.0

    async def test_zero_rate_disables_throttling(self):
        limiter = RateLimiter(per_second=0)
        assert await limiter.acquire("one.test") == 0.0


class TestParallelismPolicy:
    def test_parallel_execution_is_off_by_default(self):
        limit, reason = effective_parallelism(Settings())
        assert limit == 1
        assert "ENABLE_PARALLEL_EXECUTION is off" in reason

    def test_parallelism_requires_host_rate_limiting(self):
        """The flag alone is not authority to hammer someone else's server."""
        settings = Settings(enable_parallel_execution=True, target_rate_limit_per_s=0.0)
        limit, reason = effective_parallelism(settings)
        assert limit == 1
        assert "no host-level throttle" in reason

    def test_parallelism_is_permitted_when_throttled(self):
        settings = Settings(
            enable_parallel_execution=True, target_rate_limit_per_s=4.0, max_parallel_flows=3
        )
        limit, reason = effective_parallelism(settings)
        assert limit == 3
        assert "throttled" in reason


class TestBudget:
    def test_a_fresh_budget_permits_work(self):
        Budget(limit_s=60.0).check("planner")  # must not raise

    def test_an_exhausted_budget_stops_the_run_with_an_actionable_message(self):
        with pytest.raises(BudgetExceeded) as excinfo:
            Budget(limit_s=0.0).check("healer")
        assert "healer" in str(excinfo.value)
        assert "RUN_TIME_BUDGET_S" in str(excinfo.value)

    def test_remaining_never_goes_negative(self):
        assert Budget(limit_s=0.0).remaining_s == 0.0


class TestSpanRecorder:
    async def test_successful_spans_are_timed(self):
        recorder = SpanRecorder()
        async with recorder.span("crawl", stage="discovery"):
            await asyncio.sleep(0.01)
        assert recorder.summary()["total_spans"] == 1
        assert recorder.summary()["failed_spans"] == 0

    async def test_a_failing_span_records_and_re_raises(self):
        """Instrumentation must never swallow the error it is measuring."""
        recorder = SpanRecorder()
        with pytest.raises(ValueError):
            async with recorder.span("generate", stage="generator"):
                raise ValueError("boom")
        summary = recorder.summary()
        assert summary["failed_spans"] == 1
        assert "ValueError" in recorder.spans[0].error

    async def test_cancellation_is_recorded_and_propagated(self):
        recorder = SpanRecorder()
        with pytest.raises(asyncio.CancelledError):
            async with recorder.span("run", stage="orchestrator"):
                raise asyncio.CancelledError()
        assert recorder.summary()["failed_spans"] == 1

    async def test_spans_roll_up_by_stage(self):
        recorder = SpanRecorder()
        async with recorder.span("a", stage="discovery"):
            await asyncio.sleep(0.005)
        async with recorder.span("b", stage="discovery"):
            await asyncio.sleep(0.005)
        async with recorder.span("c", stage="generator"):
            await asyncio.sleep(0.001)
        stages = {entry["stage"]: entry for entry in recorder.summary()["by_stage"]}
        assert stages["discovery"]["count"] == 2
        assert stages["generator"]["count"] == 1


class TestContextPool:
    async def test_open_contexts_are_capped(self):
        pool = ContextPool(limit=2)
        concurrent = 0
        peak = 0

        async def use() -> None:
            nonlocal concurrent, peak
            async with pool.lease():
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0.02)
                concurrent -= 1

        await asyncio.gather(*(use() for _ in range(6)))
        assert peak <= 2
        assert pool.peak_in_use <= 2

    async def test_a_failing_lease_is_released(self):
        pool = ContextPool(limit=1)
        with pytest.raises(RuntimeError):
            async with pool.lease():
                raise RuntimeError("boom")
        async with pool.lease():
            pass
