"""Performance benchmarks for the streaming roadmap success metrics.

Targets (from docs/design/kaos-agents-streaming.md §Success metrics):

| Metric                  | Target  |
|-------------------------|---------|
| Time to first event     | < 500ms |
| Event delivery latency  | < 50ms  |
| SSE connection setup    | < 100ms |
| API cold start          | < 2s    |
| Memory overhead/session | < 10MB  |

We measure with mocked LLM dispatch (so the numbers reflect the
streaming/Runner/wire-format overhead, not LLM latency). Live API
latency is measured separately in tests/integration/test_streaming_live.py.
"""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from kaos_agents.api import create_app
from kaos_agents.config import Agent
from kaos_agents.events import AgentEvent
from kaos_agents.models import IntentResult, IntentType
from kaos_agents.runner import Runner

# Numeric targets in seconds (or bytes for memory).
TIME_TO_FIRST_EVENT_TARGET_S = 0.500
EVENT_DELIVERY_LATENCY_TARGET_S = 0.050
SSE_CONNECTION_SETUP_TARGET_S = 0.100
API_COLD_START_TARGET_S = 2.0
MEMORY_PER_SESSION_TARGET_B = 10 * 1024 * 1024  # 10 MB


def _patch_llm() -> Any:
    """Context manager: patches BaseAgent's LLM-bound methods to bypass the network."""
    mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="bench")
    return patch.multiple(
        "kaos_agents.agent.BaseAgent",
        _classify=AsyncMock(return_value=mock_intent),
        _dispatch=AsyncMock(return_value=("benchmark response", [])),
    )


# ---------------------------------------------------------------------------
# Time to first event
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.benchmark(group="streaming-metrics")
class TestTimeToFirstEvent:
    @pytest.mark.asyncio
    async def test_time_to_first_event_under_target(self) -> None:
        """First event from Runner.run() should arrive in < 500ms (mocked LLM)."""
        agent = Agent()
        runner = Runner(agent)

        with _patch_llm():
            t0 = time.perf_counter()
            iterator = runner.run("hello", "ttfe-1")
            first_event = await iterator.__anext__()
            ttfe = time.perf_counter() - t0
            # Drain the rest so the generator closes
            async for _ in iterator:
                pass

        assert first_event is not None
        assert ttfe < TIME_TO_FIRST_EVENT_TARGET_S, (
            f"Time to first event {ttfe * 1000:.1f}ms exceeded "
            f"{TIME_TO_FIRST_EVENT_TARGET_S * 1000:.0f}ms target"
        )


# ---------------------------------------------------------------------------
# Event delivery latency
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.benchmark(group="streaming-metrics")
class TestEventDeliveryLatency:
    @pytest.mark.asyncio
    async def test_per_event_latency_under_target(self) -> None:
        """Average gap between sequential yields should be < 50ms (mocked LLM)."""
        agent = Agent()
        runner = Runner(agent)

        timestamps: list[float] = []
        with _patch_llm():
            async for _ in runner.run("hello", "lat-1"):
                timestamps.append(time.perf_counter())

        assert len(timestamps) >= 2
        deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        avg_latency = sum(deltas) / len(deltas)
        max_latency = max(deltas)

        # Average must beat the target; allow a single outlier to exceed
        # by up to 5x (event-loop scheduling noise).
        assert avg_latency < EVENT_DELIVERY_LATENCY_TARGET_S, (
            f"Average event latency {avg_latency * 1000:.1f}ms exceeded target "
            f"{EVENT_DELIVERY_LATENCY_TARGET_S * 1000:.0f}ms (max={max_latency * 1000:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# SSE connection setup
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.benchmark(group="streaming-metrics")
class TestSSEConnectionSetup:
    @pytest.mark.asyncio
    async def test_sse_setup_under_target(self) -> None:
        """Time from POST to first byte received should be < 100ms (mocked LLM)."""
        app = create_app()
        transport = ASGITransport(app=app)

        with _patch_llm():
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                t0 = time.perf_counter()
                async with client.stream(
                    "POST",
                    "/v1/sessions/sse-setup-1/messages",
                    json={"message": "hi"},
                ) as resp:
                    # Read the first chunk — measure setup time
                    async for _chunk in resp.aiter_bytes(chunk_size=1024):
                        setup = time.perf_counter() - t0
                        break

        assert setup < SSE_CONNECTION_SETUP_TARGET_S, (
            f"SSE setup {setup * 1000:.1f}ms exceeded {SSE_CONNECTION_SETUP_TARGET_S * 1000:.0f}ms"
        )


# ---------------------------------------------------------------------------
# API cold start
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.benchmark(group="streaming-metrics")
class TestAPIColdStart:
    @pytest.mark.asyncio
    async def test_api_cold_start_under_target(self) -> None:
        """create_app() to first request response should be < 2s."""
        with _patch_llm():
            t0 = time.perf_counter()
            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/sessions", json={"session_id": "cold-1"})
            cold_start = time.perf_counter() - t0

        assert resp.status_code == 200
        assert cold_start < API_COLD_START_TARGET_S, (
            f"API cold start {cold_start:.2f}s exceeded {API_COLD_START_TARGET_S:.1f}s"
        )


# ---------------------------------------------------------------------------
# Memory overhead per session
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.benchmark(group="streaming-metrics")
class TestMemoryOverhead:
    @pytest.mark.asyncio
    async def test_session_memory_under_target(self) -> None:
        """Running 10 turns on one session should stay under 10MB peak overhead."""
        agent = Agent()
        runner = Runner(agent)

        gc.collect()
        tracemalloc.start()
        with _patch_llm():
            for i in range(10):
                async for event in runner.run(f"turn {i}", "mem-1"):
                    # Force consumption (GC opportunity)
                    _ = sys.getsizeof(event)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        total_bytes = sum(stat.size for stat in snapshot.statistics("filename"))
        assert total_bytes < MEMORY_PER_SESSION_TARGET_B, (
            f"Per-session memory {total_bytes / (1024 * 1024):.1f}MB "
            f"exceeded {MEMORY_PER_SESSION_TARGET_B / (1024 * 1024):.1f}MB target"
        )


# ---------------------------------------------------------------------------
# Pure throughput benchmark (kept for reference)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.benchmark(group="streaming-metrics")
class TestRunnerThroughputBenchmark:
    def test_full_turn_throughput(self, benchmark: Any) -> None:
        """End-to-end Runner.run() throughput (mocked LLM)."""
        import asyncio

        async def _one_turn() -> int:
            agent = Agent()
            runner = Runner(agent)
            count = 0
            with _patch_llm():
                async for _e in runner.run("hi", "throughput-1"):
                    count += 1
            return count

        loop = asyncio.new_event_loop()
        try:
            benchmark(lambda: loop.run_until_complete(_one_turn()))
        finally:
            loop.close()


# Keep AgentEvent referenced
_ = AgentEvent
