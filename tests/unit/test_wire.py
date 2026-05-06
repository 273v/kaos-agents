"""Tests for wire format serializers (Phase 3).

Covers:
- events_to_sse produces valid SSE format
- events_to_jsonl produces valid JSONL format
- events_to_ws produces serialized dicts
- SSE reconnection ID matches sequence number
- Multi-event streams maintain order
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from kaos_agents.events import (
    IntentClassified,
    KaosEvent,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
    TurnStart,
)
from kaos_agents.wire import events_to_jsonl, events_to_sse, events_to_ws


async def _make_event_stream() -> list[KaosEvent]:
    """Create a realistic event sequence."""
    return [
        TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r", turn_number=1),
        IntentClassified(
            timestamp=1.1,
            sequence=1,
            session_id="s",
            run_id="r",
            intent="tool_use",
            confidence=0.9,
            reasoning="needs tools",
        ),
        ToolCallStart(
            timestamp=1.2,
            sequence=2,
            session_id="s",
            run_id="r",
            call_id="tc_01",
            tool_name="kaos-source-fr-search",
        ),
        ToolCallResult(
            timestamp=1.3,
            sequence=3,
            session_id="s",
            run_id="r",
            call_id="tc_01",
            tool_name="kaos-source-fr-search",
            result_summary="Found 3",
            is_error=False,
            duration_ms=500,
        ),
        TextDelta(timestamp=1.4, sequence=4, session_id="s", run_id="r", content="I found 3"),
        TurnComplete(
            timestamp=1.5,
            sequence=5,
            session_id="s",
            run_id="r",
            text="I found 3",
            intent="tool_use",
        ),
    ]


async def _async_iter(items: Sequence[KaosEvent]) -> Any:
    """Convert a list to an async iterator."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# SSE format tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSSE:
    @pytest.mark.asyncio
    async def test_sse_format_structure(self) -> None:
        """Each SSE message has event:, data:, id: fields and ends with \\n\\n."""
        events = await _make_event_stream()
        messages: list[str] = []
        async for msg in events_to_sse(_async_iter(events)):
            messages.append(msg)

        assert len(messages) == 6
        for msg in messages:
            assert msg.endswith("\n\n")
            assert "event: " in msg
            assert "data: " in msg
            assert "id: " in msg

    @pytest.mark.asyncio
    async def test_sse_event_type(self) -> None:
        """SSE event: field matches the snake_case type name."""
        events = [TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r", turn_number=1)]
        messages: list[str] = []
        async for msg in events_to_sse(_async_iter(events)):
            messages.append(msg)

        assert "event: turn_start\n" in messages[0]

    @pytest.mark.asyncio
    async def test_sse_data_is_valid_json(self) -> None:
        """SSE data: field contains valid JSON."""
        events = await _make_event_stream()
        async for msg in events_to_sse(_async_iter(events)):
            # Extract data line
            for line in msg.strip().split("\n"):
                if line.startswith("data: "):
                    data_str = line[len("data: ") :]
                    parsed = json.loads(data_str)
                    assert "type" in parsed

    @pytest.mark.asyncio
    async def test_sse_id_matches_sequence(self) -> None:
        """SSE id: field matches the event sequence number."""
        events = await _make_event_stream()
        seq = 0
        async for msg in events_to_sse(_async_iter(events)):
            for line in msg.strip().split("\n"):
                if line.startswith("id: "):
                    assert line == f"id: {seq}"
                    seq += 1


# ---------------------------------------------------------------------------
# JSONL format tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJSONL:
    @pytest.mark.asyncio
    async def test_jsonl_one_line_per_event(self) -> None:
        """Each event produces exactly one line ending with \\n."""
        events = await _make_event_stream()
        lines: list[str] = []
        async for line in events_to_jsonl(_async_iter(events)):
            lines.append(line)

        assert len(lines) == 6
        for line in lines:
            assert line.endswith("\n")
            assert line.count("\n") == 1

    @pytest.mark.asyncio
    async def test_jsonl_valid_json(self) -> None:
        """Each JSONL line is valid JSON with a type field."""
        events = await _make_event_stream()
        async for line in events_to_jsonl(_async_iter(events)):
            parsed = json.loads(line)
            assert "type" in parsed

    @pytest.mark.asyncio
    async def test_jsonl_preserves_order(self) -> None:
        """Events appear in the same order they were emitted."""
        events = await _make_event_stream()
        types: list[str] = []
        async for line in events_to_jsonl(_async_iter(events)):
            parsed = json.loads(line)
            types.append(parsed["type"])

        assert types == [
            "turn_start",
            "intent_classified",
            "tool_call_start",
            "tool_call_result",
            "text_delta",
            "turn_complete",
        ]


# ---------------------------------------------------------------------------
# WebSocket format tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWebSocket:
    @pytest.mark.asyncio
    async def test_ws_produces_dicts(self) -> None:
        """events_to_ws yields dicts with type discriminator."""
        events = await _make_event_stream()
        dicts: list[dict[str, Any]] = []
        async for d in events_to_ws(_async_iter(events)):
            dicts.append(d)

        assert len(dicts) == 6
        for d in dicts:
            assert isinstance(d, dict)
            assert "type" in d

    @pytest.mark.asyncio
    async def test_ws_type_values(self) -> None:
        """WebSocket dicts have correct type values."""
        events = [TextDelta(timestamp=1.0, sequence=0, session_id="s", run_id="r", content="hi")]
        dicts: list[dict[str, Any]] = []
        async for d in events_to_ws(_async_iter(events)):
            dicts.append(d)

        assert dicts[0]["type"] == "text_delta"
        assert dicts[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="wire")
class TestWireBenchmark:
    def test_sse_throughput(self, benchmark: Any) -> None:
        """Benchmark SSE serialization throughput."""
        import asyncio

        events = asyncio.get_event_loop().run_until_complete(_make_event_stream())

        async def run_sse() -> int:
            count = 0
            async for _ in events_to_sse(_async_iter(events)):
                count += 1
            return count

        benchmark(lambda: asyncio.get_event_loop().run_until_complete(run_sse()))
