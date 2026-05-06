"""Tests for the streaming agent loop (Phase 1).

Covers:
- BaseAgent.run() yields correct event sequence (TurnStart → IntentClassified → ... → TurnComplete)
- BaseAgent.turn() backward compatibility (collects events, returns AgentResponse)
- Event ordering: sequence numbers are monotonically increasing
- _events_to_response correctly converts events to AgentResponse
- _dispatch_streaming default wraps _dispatch result as TextDelta
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.agent import BaseAgent, _events_to_response
from kaos_agents.events import (
    IntentClassified,
    KaosEvent,
    RunError,
    TextDelta,
    ToolCallResult,
    ToolCallStart,
    ToolCallSummary,
    TurnComplete,
    TurnStart,
)
from kaos_agents.types import AgentResponse, IntentResult, IntentType

# ---------------------------------------------------------------------------
# Helpers: mock VFS and memory for unit tests
# ---------------------------------------------------------------------------


def _make_mock_vfs() -> Any:
    """Create a minimal VFS mock that satisfies BaseAgent.__init__."""
    from kaos_core.types.enums import StorageBackend
    from kaos_core.vfs.core import VirtualFileSystem
    from kaos_core.vfs.models import VFSConfig

    config = VFSConfig(default_backend=StorageBackend.MEMORY)
    return VirtualFileSystem(config=config)


# ---------------------------------------------------------------------------
# Test: _events_to_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventsToResponse:
    def test_basic_conversion(self) -> None:
        """TurnStart + IntentClassified + TextDelta + TurnComplete → AgentResponse."""
        events: list[KaosEvent] = [
            TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r", turn_number=3),
            IntentClassified(
                timestamp=1.1,
                sequence=1,
                session_id="s",
                run_id="r",
                intent="respond",
                confidence=0.95,
                reasoning="Simple query",
            ),
            TextDelta(timestamp=1.2, sequence=2, session_id="s", run_id="r", content="Hello!"),
            TurnComplete(
                timestamp=1.3,
                sequence=3,
                session_id="s",
                run_id="r",
                text="Hello!",
                intent="respond",
                tokens_used=50,
            ),
        ]
        response = _events_to_response(events, "s")
        assert isinstance(response, AgentResponse)
        assert response.text == "Hello!"
        assert response.intent.intent == IntentType.RESPOND
        assert response.intent.confidence == 0.95
        assert response.turn_number == 3
        assert response.tokens_used == 50

    def test_with_tool_calls(self) -> None:
        """ToolCallResult events are collected into AgentResponse.tool_calls."""
        events: list[KaosEvent] = [
            TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r", turn_number=1),
            IntentClassified(
                timestamp=1.1,
                sequence=1,
                session_id="s",
                run_id="r",
                intent="tool_use",
                confidence=0.9,
                reasoning="Needs tools",
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
            TextDelta(
                timestamp=1.4,
                sequence=4,
                session_id="s",
                run_id="r",
                content="Found 3 results.",
            ),
            TurnComplete(
                timestamp=1.5,
                sequence=5,
                session_id="s",
                run_id="r",
                text="Found 3 results.",
                intent="tool_use",
                tool_calls=(
                    ToolCallSummary(
                        tool_name="kaos-source-fr-search", call_id="tc_01", duration_ms=500
                    ),
                ),
            ),
        ]
        response = _events_to_response(events, "s")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "kaos-source-fr-search"
        assert response.intent.intent == IntentType.TOOL_USE

    def test_empty_events(self) -> None:
        """Empty event list produces a minimal AgentResponse."""
        response = _events_to_response([], "s")
        assert response.text == ""
        assert response.intent.intent == IntentType.RESPOND
        assert response.turn_number == 0

    def test_fallback_to_text_deltas(self) -> None:
        """Without TurnComplete, response is concatenated TextDelta content."""
        events: list[KaosEvent] = [
            TextDelta(timestamp=1.0, sequence=0, session_id="s", run_id="r", content="Part 1. "),
            TextDelta(timestamp=1.1, sequence=1, session_id="s", run_id="r", content="Part 2."),
        ]
        response = _events_to_response(events, "s")
        assert response.text == "Part 1. Part 2."


# ---------------------------------------------------------------------------
# Test: BaseAgent.run() event sequence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseAgentRun:
    @pytest.mark.asyncio
    async def test_run_yields_turn_start_and_complete(self) -> None:
        """run() always yields TurnStart as first event and TurnComplete as last."""
        vfs = _make_mock_vfs()
        agent = BaseAgent(vfs)

        # Patch classify to return a simple RESPOND intent (avoids LLM call)
        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch.object(agent, "_classify", new_callable=AsyncMock, return_value=mock_intent),
            patch.object(
                agent, "_dispatch", new_callable=AsyncMock, return_value=("Test response", [])
            ),
        ):
            events: list[KaosEvent] = []
            async for event in agent.run("Hello", "test-session"):
                events.append(event)

        assert len(events) >= 3  # TurnStart, IntentClassified, TextDelta/TurnComplete
        assert isinstance(events[0], TurnStart)
        assert isinstance(events[-1], TurnComplete)

    @pytest.mark.asyncio
    async def test_run_yields_intent_classified(self) -> None:
        """run() yields IntentClassified after classification."""
        vfs = _make_mock_vfs()
        agent = BaseAgent(vfs)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=0.85, reasoning="test")
        with (
            patch.object(agent, "_classify", new_callable=AsyncMock, return_value=mock_intent),
            patch.object(agent, "_dispatch", new_callable=AsyncMock, return_value=("Response", [])),
        ):
            events: list[KaosEvent] = []
            async for event in agent.run("Hello", "test-session"):
                events.append(event)

        intent_events = [e for e in events if isinstance(e, IntentClassified)]
        assert len(intent_events) == 1
        assert intent_events[0].intent == "respond"
        assert intent_events[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_run_sequence_monotonic(self) -> None:
        """Event sequence numbers are strictly monotonically increasing."""
        vfs = _make_mock_vfs()
        agent = BaseAgent(vfs)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch.object(agent, "_classify", new_callable=AsyncMock, return_value=mock_intent),
            patch.object(agent, "_dispatch", new_callable=AsyncMock, return_value=("Response", [])),
        ):
            events: list[KaosEvent] = []
            async for event in agent.run("Hello", "test-session"):
                events.append(event)

        sequences = [e.sequence for e in events]
        for i in range(1, len(sequences)):
            assert sequences[i] > sequences[i - 1], (
                f"Sequence not monotonic: {sequences[i - 1]} >= {sequences[i]}"
            )

    @pytest.mark.asyncio
    async def test_run_session_id_consistent(self) -> None:
        """All events carry the same session_id."""
        vfs = _make_mock_vfs()
        agent = BaseAgent(vfs)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch.object(agent, "_classify", new_callable=AsyncMock, return_value=mock_intent),
            patch.object(agent, "_dispatch", new_callable=AsyncMock, return_value=("Response", [])),
        ):
            events: list[KaosEvent] = []
            async for event in agent.run("Hello", "my-session"):
                events.append(event)

        for event in events:
            assert event.session_id == "my-session"

    @pytest.mark.asyncio
    async def test_run_error_yields_run_error_event(self) -> None:
        """If dispatch raises, run() yields a RunError event."""
        vfs = _make_mock_vfs()
        agent = BaseAgent(vfs)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch.object(agent, "_classify", new_callable=AsyncMock, return_value=mock_intent),
            patch.object(
                agent, "_dispatch", new_callable=AsyncMock, side_effect=RuntimeError("boom")
            ),
        ):
            events: list[KaosEvent] = []
            async for event in agent.run("Hello", "test-session"):
                events.append(event)

        error_events = [e for e in events if isinstance(e, RunError)]
        assert len(error_events) == 1
        assert "boom" in error_events[0].message


# ---------------------------------------------------------------------------
# Test: BaseAgent.turn() backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseAgentTurnCompat:
    @pytest.mark.asyncio
    async def test_turn_returns_agent_response(self) -> None:
        """turn() returns AgentResponse, same as before the streaming refactor."""
        vfs = _make_mock_vfs()
        agent = BaseAgent(vfs)

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch.object(agent, "_classify", new_callable=AsyncMock, return_value=mock_intent),
            patch.object(
                agent, "_dispatch", new_callable=AsyncMock, return_value=("Hello back!", [])
            ),
        ):
            response = await agent.turn("Hello", "test-session")

        assert isinstance(response, AgentResponse)
        assert response.text == "Hello back!"
        assert response.intent.intent == IntentType.RESPOND
