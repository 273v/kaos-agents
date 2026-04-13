"""Live integration tests for the agent — real LLM classify + respond.

Tests the actual LLM-backed classify_intent and _simple_respond paths
without mocking. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import pytest
from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

from kaos_agents.agent import BaseAgent
from kaos_agents.context.classify import classify_intent
from kaos_agents.memory.session import SessionMemory
from kaos_agents.models import IntentType


def _memory_vfs() -> VirtualFileSystem:
    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


MODEL = "anthropic:claude-haiku-4-5"


@pytest.mark.live
class TestClassifyLive:
    """Test LLM-based intent classification with real API calls."""

    async def test_greeting_classified_as_respond(self):
        mem = SessionMemory("test")
        result = await classify_intent("Hello!", mem, model=MODEL)
        assert result.intent == IntentType.RESPOND
        assert result.confidence > 0.5

    async def test_tool_request_classified(self):
        mem = SessionMemory("test")
        result = await classify_intent("Extract the dates from contract.pdf", mem, model=MODEL)
        assert result.intent in (IntentType.TOOL_USE, IntentType.PLAN)
        assert result.confidence > 0.3

    async def test_complex_request_classified_as_plan(self):
        mem = SessionMemory("test")
        result = await classify_intent(
            "First search the eCFR for emission regulations, then search EDGAR "
            "for Tesla's filings, then cross-reference the findings",
            mem,
            model=MODEL,
        )
        assert result.intent in (IntentType.PLAN, IntentType.TOOL_USE)


@pytest.mark.live
class TestAgentTurnLive:
    """Test full agent turn with real LLM — no mocking."""

    async def test_simple_greeting(self):
        vfs = _memory_vfs()
        agent = BaseAgent(vfs, model=MODEL)
        response = await agent.turn("Hello!", session_id="live-test")

        assert response.text  # Non-empty response
        assert len(response.text) > 5  # Not just "Hi"
        assert response.turn_number == 1
        assert response.intent.intent in (IntentType.RESPOND, IntentType.TOOL_USE)

    async def test_two_turn_conversation(self):
        vfs = _memory_vfs()

        # Turn 1
        agent1 = BaseAgent(vfs, model=MODEL)
        r1 = await agent1.turn("What is the capital of France?", session_id="live-2turn")
        assert r1.text
        assert r1.turn_number == 1

        # Turn 2 — should have memory of turn 1
        agent2 = BaseAgent(vfs, model=MODEL)
        r2 = await agent2.turn("And what about Germany?", session_id="live-2turn")
        assert r2.text
        assert r2.turn_number == 2
