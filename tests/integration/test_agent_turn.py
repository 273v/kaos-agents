"""Integration tests for BaseAgent turn lifecycle.

Uses heuristic classification (no LLM) to test the full turn loop
without requiring API keys. The _simple_respond method requires
kaos-llm-core, so we mock it at the agent level.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

from kaos_agents.agent import BaseAgent
from kaos_agents.memory.store import SessionStore
from kaos_agents.types import IntentResult, IntentType
from kaos_agents.types.memory import MemoryType


def _memory_vfs() -> VirtualFileSystem:
    config = VFSConfig(default_backend=StorageBackend.MEMORY, isolation_mode=IsolationMode.GLOBAL)
    return VirtualFileSystem(config=config)


@pytest.mark.integration
class TestBaseAgentTurn:
    async def test_single_turn_persists(self):
        """A single turn adds messages to memory and persists."""
        vfs = _memory_vfs()
        agent = BaseAgent(vfs)

        # Patch _classify to use heuristic (no LLM needed)
        # Patch _simple_respond to avoid LLM call
        with (
            patch.object(
                agent,
                "_classify",
                return_value=IntentResult(
                    intent=IntentType.RESPOND, confidence=0.9, reasoning="test"
                ),
            ),
            patch.object(
                agent, "_simple_respond", new_callable=AsyncMock, return_value="Hello back!"
            ),
        ):
            response = await agent.turn("Hello!", session_id="test-session")

        assert response.text == "Hello back!"
        assert response.intent.intent == IntentType.RESPOND
        assert response.turn_number == 1

        # Verify persistence
        store = SessionStore(vfs)
        loaded = await store.load("test-session")
        assert loaded.turn_count == 1
        assert loaded.section_item_count(MemoryType.MESSAGES) == 2  # user + assistant

    async def test_multi_turn_accumulates(self):
        """Multiple turns accumulate messages in memory."""
        vfs = _memory_vfs()

        for turn_idx in range(3):
            agent = BaseAgent(vfs)
            with (
                patch.object(
                    agent,
                    "_classify",
                    return_value=IntentResult(
                        intent=IntentType.RESPOND, confidence=0.9, reasoning="test"
                    ),
                ),
                patch.object(
                    agent,
                    "_simple_respond",
                    new_callable=AsyncMock,
                    return_value=f"Response {turn_idx}",
                ),
            ):
                response = await agent.turn(f"Message {turn_idx}", session_id="multi-turn")
            assert response.turn_number == turn_idx + 1

        # Verify accumulated state
        store = SessionStore(vfs)
        loaded = await store.load("multi-turn")
        assert loaded.turn_count == 3
        # 3 turns x 2 messages (user + assistant) = 6
        assert loaded.section_item_count(MemoryType.MESSAGES) == 6

    async def test_tool_use_intent_fallback(self):
        """Tool use intent falls back to simple response in BaseAgent (no runtime)."""
        vfs = _memory_vfs()
        agent = BaseAgent(vfs)

        with (
            patch.object(
                agent,
                "_classify",
                return_value=IntentResult(
                    intent=IntentType.TOOL_USE, confidence=0.7, reasoning="action word"
                ),
            ),
            patch.object(
                agent,
                "_simple_respond",
                new_callable=AsyncMock,
                return_value="I'll help with that.",
            ),
        ):
            response = await agent.turn("extract dates from the contract", session_id="tool-test")

        assert response.text == "I'll help with that."
        assert response.intent.intent == IntentType.TOOL_USE

    async def test_disk_persistence_survives_restart(self, tmp_path: Path):
        """Agent state survives across separate VFS instances (simulates restart)."""
        config = VFSConfig(
            default_backend=StorageBackend.DISK,
            disk_base_path=tmp_path,
            isolation_mode=IsolationMode.GLOBAL,
        )

        # Turn 1
        vfs1 = VirtualFileSystem(config=config)
        agent1 = BaseAgent(vfs1)
        with (
            patch.object(
                agent1,
                "_classify",
                return_value=IntentResult(
                    intent=IntentType.RESPOND, confidence=0.9, reasoning="test"
                ),
            ),
            patch.object(
                agent1, "_simple_respond", new_callable=AsyncMock, return_value="First response"
            ),
        ):
            await agent1.turn("First message", session_id="restart-test")

        # Turn 2 — new VFS, new agent (simulates process restart)
        vfs2 = VirtualFileSystem(config=config)
        agent2 = BaseAgent(vfs2)
        with (
            patch.object(
                agent2,
                "_classify",
                return_value=IntentResult(
                    intent=IntentType.RESPOND, confidence=0.9, reasoning="test"
                ),
            ),
            patch.object(
                agent2, "_simple_respond", new_callable=AsyncMock, return_value="Second response"
            ),
        ):
            response = await agent2.turn("Second message", session_id="restart-test")

        assert response.turn_number == 2

        # Verify full state
        store = SessionStore(vfs2)
        loaded = await store.load("restart-test")
        assert loaded.turn_count == 2
        assert loaded.section_item_count(MemoryType.MESSAGES) == 4
