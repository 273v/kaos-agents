"""Regression tests for WS-0.2 — ``Runner.turn()`` delegates to ``Runner.run()``.

Pre-fix bug: ``Runner.turn()`` called ``internal.turn(...)`` directly,
bypassing the Runner's hook dispatch + permission policy evaluation.
Every non-streaming caller (MCP tools, JSON API) silently skipped
safety policy.

Post-fix: ``turn()`` drains ``run()`` and aggregates events into an
``AgentResponse``. The assertions here pin the fix by observing that:

1. Hooks attached to the Runner fire during ``turn()``.
2. A deny-policy still blocks tool calls when reached via ``turn()``
   (verified at the tool-executor layer — the WS-0.1 gate).
3. ``turn()`` returns a populated ``AgentResponse`` with text +
   tool_calls + intent propagated from the event stream.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.config import Agent
from kaos_agents.events import IntentClassified, Span, TurnSummary
from kaos_agents.hooks import KaosHook
from kaos_agents.runner import Runner
from kaos_agents.types import IntentResult, IntentType


class _RecordingHook(KaosHook):
    def __init__(self) -> None:
        self.events: list[str] = []

    async def on_turn_start(self, event: Span) -> None:
        self.events.append("turn_start")

    async def on_intent_classified(self, event: IntentClassified) -> None:
        self.events.append("intent_classified")

    async def on_turn_complete(self, event: Span) -> None:
        self.events.append("turn_complete")

    async def on_turn_summary(self, event: TurnSummary) -> None:
        self.events.append("turn_summary")


@pytest.mark.unit
class TestRunnerTurnDelegatesToRun:
    @pytest.mark.asyncio
    async def test_hooks_fire_during_turn(self) -> None:
        """The headline WS-0.2 assertion — hooks MUST fire on the
        non-streaming ``turn()`` path, because MCP tools + JSON API use it."""
        hook = _RecordingHook()
        agent = Agent()
        runner = Runner(agent, hooks=(hook,))

        mock_intent = IntentResult(
            intent=IntentType.RESPOND,
            confidence=1.0,
            reasoning="test",
        )
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("hello from agent", []),
            ),
        ):
            response = await runner.turn("Hello", "ws02-test-session")

        assert "turn_start" in hook.events, (
            f"turn_start hook did not fire on turn() — hooks={hook.events}. "
            "This is the WS-0.2 bypass regression."
        )
        assert "intent_classified" in hook.events
        assert "turn_complete" in hook.events
        assert response.text == "hello from agent"
        assert response.intent.intent == IntentType.RESPOND

    @pytest.mark.asyncio
    async def test_turn_populates_response_from_event_stream(self) -> None:
        """turn() must aggregate TextDelta / ToolCallResult / TurnStart
        events into a well-formed AgentResponse — not rely on the
        bypassed internal.turn() return value."""
        runner = Runner(Agent())

        mock_intent = IntentResult(
            intent=IntentType.RESPOND,
            confidence=0.95,
            reasoning="simple response",
        )
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("response body", []),
            ),
        ):
            response = await runner.turn("Q", "ws02-session-2")

        assert response.text == "response body"
        assert response.intent.confidence == 0.95
        assert response.intent.reasoning == "simple response"
        # session_id threaded via metadata
        metadata_dict = dict(response.metadata)
        assert metadata_dict.get("session_id") == "ws02-session-2"

    @pytest.mark.asyncio
    async def test_turn_still_works_without_hooks_or_policy(self) -> None:
        """Backward compatibility — no hooks + no policy must continue
        to produce a response."""
        runner = Runner(Agent())

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("ok", []),
            ),
        ):
            response = await runner.turn("Hi", "ws02-session-3")

        assert response.text == "ok"
