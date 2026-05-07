"""Tests for the KaosAgent + KaosPattern ABCs.

Track 2 chunk 1 — confirms the abstract surface:
- KaosAgent.run is abstract; bare instantiation fails
- Default metadata() works for trivially-named subclasses
- Override metadata() for custom identity
- KaosPattern.dispatch is abstract
- The ABCs are pure — no behavior, no state
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from kaos_agents.base import KaosAgent, KaosEvent, KaosPattern
from kaos_agents.types.metadata import AgentMetadata, PatternMetadata


@pytest.mark.unit
class TestKaosAgentABC:
    def test_cannot_instantiate_abstract(self) -> None:
        """KaosAgent without a run() implementation is abstract."""
        with pytest.raises(TypeError, match="abstract"):
            KaosAgent()  # type: ignore[abstract]

    def test_minimal_subclass_works(self) -> None:
        """A subclass that implements run() is concrete and constructable."""

        class _Trivial(KaosAgent):
            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                # Empty event stream.
                if False:
                    yield None  # type: ignore[unreachable]

        agent = _Trivial()
        assert isinstance(agent, KaosAgent)

    def test_default_metadata_from_classname(self) -> None:
        """Default metadata kebab-cases the class name and pulls __doc__."""

        class _ChatAgentLike(KaosAgent):
            """A trivial chat-like agent."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        meta = _ChatAgentLike.metadata()
        assert isinstance(meta, AgentMetadata)
        assert meta.name == "chat-agent-like"
        assert meta.description == "A trivial chat-like agent."

    def test_metadata_override(self) -> None:
        """Subclasses can override metadata() for custom identity."""

        class _Custom(KaosAgent):
            @classmethod
            def metadata(cls) -> AgentMetadata:
                return AgentMetadata(
                    name="custom-agent",
                    description="A custom agent.",
                    pattern="research",
                    tags=("rag", "verified"),
                )

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        meta = _Custom.metadata()
        assert meta.name == "custom-agent"
        assert meta.pattern == "research"
        assert "rag" in meta.tags

    @pytest.mark.asyncio
    async def test_default_turn_collects_events(self) -> None:
        """The default turn() collects from run() and produces a response."""
        from kaos_agents.events import EventEmitter, IntentClassified, TurnSummary
        from kaos_agents.events.spans import SpanSubject

        class _Yields3(KaosAgent):
            """Yields a turn-start span, intent, and a turn summary."""

            async def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
                emitter = EventEmitter(session_id=session_id, run_id="r1")
                yield emitter.span_start(
                    SpanSubject.TURN,
                    name="turn.1",
                    attributes={"turn_number": 1},
                )
                yield emitter.emit(
                    IntentClassified, intent="respond", confidence=0.9, reasoning="test"
                )
                yield emitter.emit(TurnSummary, text="hi back", intent="respond", tokens_used=10)

        agent = _Yields3()
        response = await agent.turn("hi", "session-1")
        assert response.text == "hi back"
        assert response.intent.intent.value == "respond"
        assert response.tokens_used == 10
        assert response.turn_number == 1


@pytest.mark.unit
class TestKaosPatternABC:
    def test_cannot_instantiate_abstract(self) -> None:
        """KaosPattern without dispatch() is abstract."""
        with pytest.raises(TypeError, match="abstract"):
            KaosPattern()  # type: ignore[abstract]

    def test_default_metadata(self) -> None:
        """Default metadata snake_cases the class name."""

        class _ResearchPattern(KaosPattern):
            """A research pattern."""

            async def dispatch(
                self,
                intent: Any,
                message: str,
                memory: Any,
                context_items: dict[Any, list[Any]],
                emitter: Any,
            ) -> AsyncIterator[KaosEvent]:
                if False:
                    yield None  # type: ignore[unreachable]

        meta = _ResearchPattern.metadata()
        assert isinstance(meta, PatternMetadata)
        assert meta.name == "research_pattern"
        assert meta.description == "A research pattern."
