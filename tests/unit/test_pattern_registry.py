"""Tests for kaos_agents.registry.pattern_registry.

Track 2 chunk 4 — confirms the registry contract:

- The 3 built-in patterns auto-register on patterns/__init__.py import.
- AgentPattern enum values match registry keys.
- Manual register / get / unregister / clear all work.
- Conflict detection raises RegistryError without force=True.
- Custom out-of-tree patterns can register and resolve by name.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from kaos_core.exceptions import RegistryError

from kaos_agents.base.agent import KaosAgent
from kaos_agents.base.event import KaosEvent
from kaos_agents.config import AgentPattern
from kaos_agents.patterns import ChatAgent, PlanExecuteAgent, ResearchAgent
from kaos_agents.registry.pattern_registry import (
    PatternRegistry,
    default_pattern_registry,
)


@pytest.mark.unit
class TestDefaultRegistryPopulated:
    """The 3 built-in patterns auto-register on patterns/ import."""

    def test_chat_registered(self) -> None:
        cls = default_pattern_registry.get(AgentPattern.CHAT.value)
        assert cls is ChatAgent

    def test_plan_registered(self) -> None:
        cls = default_pattern_registry.get(AgentPattern.PLAN.value)
        assert cls is PlanExecuteAgent

    def test_research_registered(self) -> None:
        cls = default_pattern_registry.get(AgentPattern.RESEARCH.value)
        assert cls is ResearchAgent

    def test_list_names_includes_all_three(self) -> None:
        names = default_pattern_registry.list_names()
        for pattern in AgentPattern:
            assert pattern.value in names


@pytest.mark.unit
class TestPatternRegistry:
    def test_register_and_get(self) -> None:
        reg = PatternRegistry()

        class _Stub(KaosAgent):
            """Stub agent."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        reg.register("stub", _Stub)
        assert reg.get("stub") is _Stub

    def test_get_unknown_returns_none(self) -> None:
        reg = PatternRegistry()
        assert reg.get("not-a-pattern") is None

    def test_double_register_raises(self) -> None:
        reg = PatternRegistry()

        class _A(KaosAgent):
            """A."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        class _B(KaosAgent):
            """B."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        reg.register("dup", _A)
        # Same class — idempotent.
        reg.register("dup", _A)
        # Different class same key — error.
        with pytest.raises(RegistryError):
            reg.register("dup", _B)

    def test_force_replaces(self) -> None:
        reg = PatternRegistry()

        class _A(KaosAgent):
            """A."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        class _B(KaosAgent):
            """B."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        reg.register("dup", _A)
        reg.register("dup", _B, force=True)
        assert reg.get("dup") is _B

    def test_unregister(self) -> None:
        reg = PatternRegistry()

        class _A(KaosAgent):
            """A."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        reg.register("a", _A)
        removed = reg.unregister("a")
        assert removed is _A
        assert reg.get("a") is None

    def test_clear(self) -> None:
        reg = PatternRegistry()

        class _A(KaosAgent):
            """A."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        reg.register("a", _A)
        reg.register("b", _A)
        reg.clear()
        assert len(reg) == 0

    def test_membership(self) -> None:
        reg = PatternRegistry()

        class _A(KaosAgent):
            """A."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        reg.register("a", _A)
        assert "a" in reg
        assert "not-a" not in reg
        assert 42 not in reg  # non-string keys False, never raise

    def test_iter_yields_keys(self) -> None:
        reg = PatternRegistry()

        class _A(KaosAgent):
            """A."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        reg.register("a", _A)
        reg.register("b", _A)
        assert set(reg) == {"a", "b"}
