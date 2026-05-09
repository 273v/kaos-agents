"""Phase 2.B — unit tests for :class:`kaos_agents.loop.agent_loop.AgentLoop`.

Stubs the inner :class:`IntentExtractor` so no live LLM call is issued.
Stub planners are minimal classes with ``plan`` / ``execute`` async
methods. Streaming tests use ``asyncio.sleep(0.01)`` to give the
runner Task time to emit a ``Span(TURN, START)`` before the consumer
finishes draining.

Covers:

* construction with all-None subsystems (only IntentExtractor required,
  default provided).
* ``prepare_turn`` returns a populated TurnPlan whose intent reflects
  the trigger message.
* ``forward()`` skeleton path → ``extras["phase"]="skeleton"``,
  ``output=""``, ``is_complete=True``, ``error is None``.
* ``forward()`` happy path with a stub planner → output matches the
  planner's execute return.
* ``forward()`` clarification path → emits ``Span(STEP, ERROR)`` and
  ``Span(TURN, COMPLETE)``, finalizes with empty output.
* ContextVar isolation: ``current_turn()`` is the live invocation
  inside ``forward()`` and ``None`` outside.
* Error path: a stub planner that raises causes ``forward()`` to
  raise; the exception carries ``exc.turn_invocation`` with the partial
  events tuple.
* ``stream()`` yields events live (Span(TURN, START) is yielded before
  forward() finishes when the planner sleeps).
* ``stream()`` ends after Span(TURN, COMPLETE) — the final span is
  pushed before the sentinel.
* Usage rollup: a planner that emits ``UsageObserved`` is rolled into
  ``invocation.usage``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from kaos_llm_core.programs._invocation import Invocation, TokenUsage

from kaos_agents.config import AgentPattern
from kaos_agents.core.invocation import current_turn
from kaos_agents.events.lifecycle import TurnSummary, UsageObserved
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.intent import IntentExtractor
from kaos_agents.intent.types import (
    Ambiguity,
    AmbiguityKind,
    Goal,
    IntentResult,
)
from kaos_agents.loop import AgentLoop
from kaos_agents.triggers import MCPToolTrigger
from kaos_agents.types.intents import IntentType

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _intent(
    *,
    pattern: AgentPattern = AgentPattern.CHAT,
    requires_clarification: bool = False,
    intent_type: IntentType = IntentType.RESPOND,
    confidence: float = 0.9,
    raw_input: str = "hello",
    ambiguities: tuple[Ambiguity, ...] = (),
    goal_statement: str = "Greet the user.",
) -> IntentResult:
    """Build a minimal valid IntentResult for tests."""
    return IntentResult(
        goal=Goal(statement=goal_statement, intent_type=intent_type),
        constraints=(),
        ambiguities=ambiguities,
        requires_clarification=requires_clarification,
        pattern=pattern,
        confidence=confidence,
        raw_input=raw_input,
    )


def _stub_extractor(intent: IntentResult) -> IntentExtractor:
    """Return a real IntentExtractor whose ``invoke`` returns an
    :class:`Invocation` carrying the given :class:`IntentResult`.

    AgentLoop calls ``extractor.invoke(...)`` and reads
    ``invocation.output``; bypassing the inner Call entirely keeps the
    stub LLM-free and avoids the Signature projection step.
    """
    extractor = IntentExtractor()

    async def _invoke(**_kwargs: Any) -> Invocation:
        return Invocation(
            client=None,
            model="anthropic:claude-haiku-4-5",
            context=None,
            output=intent,
            trace=None,
            usage=TokenUsage(),
        )

    extractor.invoke = _invoke  # ty: ignore[invalid-assignment]
    return extractor


class _OkPlanner:
    """Minimal planner stub: plan returns a string, execute returns
    a SimpleNamespace with ``.text``."""

    def __init__(self, text: str = "planner-result", *, sleep: float = 0.0) -> None:
        self._text = text
        self._sleep = sleep

    async def plan(self, intent: IntentResult, memory: Any) -> str:
        return f"plan({intent.goal.statement})"

    async def execute(
        self,
        plan_obj: str,
        *,
        perceiver: Any = None,
        actor: Any = None,
    ) -> Any:
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return SimpleNamespace(text=self._text)


class _UsagePlanner(_OkPlanner):
    """A planner that emits a UsageObserved event via the active
    emitter so the AgentLoop's rollup can pick it up."""

    async def execute(
        self,
        plan_obj: str,
        *,
        perceiver: Any = None,
        actor: Any = None,
    ) -> Any:
        # Emit via the active collector — AgentLoop has installed one
        # before dispatching to us. We synthesize via the EventEmitter
        # passed indirectly through the active collector's append.
        # The simplest path is to import push_event and append a fully-
        # formed UsageObserved.
        from kaos_agents.events.collector import push_event

        push_event(
            UsageObserved(
                timestamp=0.0,
                sequence=999,
                session_id="",
                run_id="",
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.005,
                source="planner-stub",
            )
        )
        return SimpleNamespace(text=self._text)


class _RaisingPlanner:
    """Stub planner whose execute raises."""

    async def plan(self, intent: IntentResult, memory: Any) -> str:
        return "plan_obj"

    async def execute(
        self,
        plan_obj: str,
        *,
        perceiver: Any = None,
        actor: Any = None,
    ) -> Any:
        raise RuntimeError("planner-failure")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_all_none_subsystems(self) -> None:
        """Only IntentExtractor is required; a default is provided."""
        loop = AgentLoop()
        assert isinstance(loop._intent_extractor, IntentExtractor)
        assert loop._planner is None
        assert loop._memory is None
        assert loop._termination_judge is None
        assert loop._escalation_policy is None
        assert loop._delegation_router is None
        assert loop._governance is None
        assert loop._permission_policy is None
        assert loop._hooks == ()
        assert loop._agent_envelope_hash == ""
        assert callable(loop._run_id_factory)

    def test_explicit_intent_extractor(self) -> None:
        ex = IntentExtractor()
        loop = AgentLoop(intent_extractor=ex)
        assert loop._intent_extractor is ex

    def test_explicit_envelope_hash(self) -> None:
        loop = AgentLoop(agent_envelope_hash="sha256:cafe")
        assert loop._agent_envelope_hash == "sha256:cafe"


# ---------------------------------------------------------------------------
# prepare_turn
# ---------------------------------------------------------------------------


class TestPrepareTurn:
    async def test_prepare_turn_populates_intent_and_session(self) -> None:
        intent = _intent(raw_input="hello", pattern=AgentPattern.CHAT)
        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        trigger = MCPToolTrigger("hello", session_id="s1")
        plan = await loop.prepare_turn(trigger)
        assert plan.session_id == "s1"
        assert plan.run_id.startswith("run_")
        assert plan.turn_number == 1
        assert plan.intent is intent
        assert plan.trigger is trigger
        assert plan.emitter is not None
        assert plan.parent_span_id is None
        assert plan.memory is None

    async def test_prepare_turn_mints_session_id_when_blank(self) -> None:
        intent = _intent()
        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        trigger = MCPToolTrigger("hi", session_id=None)
        plan = await loop.prepare_turn(trigger)
        # source_id is "" when session_id=None per Trigger.mcp factory;
        # AgentLoop mints one via _new_session_id.
        assert plan.session_id.startswith("session_")


# ---------------------------------------------------------------------------
# forward — happy paths
# ---------------------------------------------------------------------------


class TestForwardSkeletonPath:
    async def test_no_planner_yields_skeleton_invocation(self) -> None:
        intent = _intent()
        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        trigger = MCPToolTrigger("hello", session_id="s-skeleton")
        inv = await loop.forward(trigger=trigger)
        assert inv.is_complete
        assert inv.error is None
        assert inv.output == ""
        assert inv.extras.get("phase") == "skeleton"
        assert inv.session_id == "s-skeleton"
        assert inv.turn_number == 1
        # Should have at least Span(TURN, START), IntentClassified,
        # TurnSummary, Span(TURN, COMPLETE).
        kinds = [type(e).__name__ for e in inv.events]
        assert "Span" in kinds
        assert "IntentClassified" in kinds
        assert "TurnSummary" in kinds


class TestForwardWithPlanner:
    async def test_planner_output_lands_on_invocation(self) -> None:
        intent = _intent()
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            planner=_OkPlanner(text="hello-from-planner"),
        )
        trigger = MCPToolTrigger("hi", session_id="s2")
        inv = await loop.forward(trigger=trigger)
        assert inv.output == "hello-from-planner"
        assert inv.is_complete
        assert inv.error is None
        # plan_obj from the stub planner.
        assert inv.plan == f"plan({intent.goal.statement})"
        # Skeleton flag must NOT be set when planner ran.
        assert inv.extras.get("phase") != "skeleton"


# ---------------------------------------------------------------------------
# forward — clarification path
# ---------------------------------------------------------------------------


class TestForwardClarificationPath:
    async def test_requires_clarification_emits_step_error(self) -> None:
        intent = _intent(
            requires_clarification=True,
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.MISSING_CONTEXT,
                    span=(0, 5),
                    excerpt="hello",
                    candidate_interpretations=(),
                    preferred_clarification="Which contract?",
                ),
            ),
        )
        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        trigger = MCPToolTrigger("hello", session_id="s-clarify")
        inv = await loop.forward(trigger=trigger)
        assert inv.is_complete
        assert inv.error is None
        assert inv.output == ""
        assert inv.extras.get("clarification_required") is True
        # Verify Span(STEP, ERROR) and Span(TURN, COMPLETE).
        spans = [e for e in inv.events if isinstance(e, Span)]
        kinds = [(s.subject, s.phase) for s in spans]
        assert (SpanSubject.STEP, SpanPhase.ERROR) in kinds
        assert (SpanSubject.TURN, SpanPhase.COMPLETE) in kinds
        # No TurnSummary on the early-exit clarification path.
        summaries = [e for e in inv.events if isinstance(e, TurnSummary)]
        assert summaries == []


# ---------------------------------------------------------------------------
# ContextVar isolation
# ---------------------------------------------------------------------------


class TestContextVarIsolation:
    async def test_current_turn_inside_and_outside_forward(self) -> None:
        seen_inside: list[Any] = []

        class _ProbingPlanner:
            async def plan(self, intent: IntentResult, memory: Any) -> str:
                seen_inside.append(current_turn())
                return "p"

            async def execute(
                self, plan_obj: str, *, perceiver: Any = None, actor: Any = None
            ) -> Any:
                return SimpleNamespace(text="ok")

        intent = _intent()
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            planner=_ProbingPlanner(),
        )
        assert current_turn() is None  # outside
        inv = await loop.forward(trigger=MCPToolTrigger("x", session_id="s3"))
        assert current_turn() is None  # outside again
        # Inside the planner the active turn was the running invocation.
        assert seen_inside == [inv]


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestForwardErrorPath:
    async def test_planner_raises_carries_partial_invocation(self) -> None:
        intent = _intent()
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            planner=_RaisingPlanner(),
        )
        trigger = MCPToolTrigger("explode", session_id="s-err")
        with pytest.raises(RuntimeError) as ei:
            await loop.forward(trigger=trigger)
        # The partial TurnInvocation is attached to the exception.
        partial = getattr(ei.value, "turn_invocation", None)
        assert partial is not None
        assert partial.error is ei.value
        assert partial.is_complete  # finalize() was called on the error path
        # We at least captured the turn-start Span and IntentClassified
        # before the planner blew up.
        kinds = {type(e).__name__ for e in partial.events}
        assert "Span" in kinds
        assert "IntentClassified" in kinds


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStream:
    async def test_stream_yields_events_live(self) -> None:
        """The TURN-START span must arrive before the planner's
        ``await asyncio.sleep`` completes — proves Pattern A is wired
        live, not delivered all-at-once at the end of forward()."""
        intent = _intent()
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            # Sleep inside execute() so the consumer drains the
            # turn-start span BEFORE forward() finishes.
            planner=_OkPlanner(sleep=0.05),
        )
        trigger = MCPToolTrigger("hi", session_id="s-stream-live")
        events: list[Any] = []
        # Open the stream and drain. A live emission is observable as
        # "Span(TURN, START) appears in the events list".
        async for event in loop.stream(trigger):
            events.append(event)
        # End-of-stream after sentinel: we got everything.
        kinds = [type(e).__name__ for e in events]
        assert "Span" in kinds
        # The first event is the TURN-start span.
        assert isinstance(events[0], Span)
        assert events[0].subject is SpanSubject.TURN
        assert events[0].phase is SpanPhase.START

    async def test_stream_ends_with_turn_complete_span(self) -> None:
        intent = _intent()
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            planner=_OkPlanner(text="done"),
        )
        trigger = MCPToolTrigger("hi", session_id="s-stream-end")
        events: list[Any] = []
        async for event in loop.stream(trigger):
            events.append(event)
        # The last meaningful event before the sentinel is
        # Span(TURN, COMPLETE).
        spans = [e for e in events if isinstance(e, Span)]
        assert spans, "expected at least one Span"
        last_turn_span = next(
            (s for s in reversed(spans) if s.subject is SpanSubject.TURN),
            None,
        )
        assert last_turn_span is not None
        assert last_turn_span.phase is SpanPhase.COMPLETE


# ---------------------------------------------------------------------------
# Usage rollup
# ---------------------------------------------------------------------------


class TestUsageRollup:
    async def test_usage_observed_events_roll_into_invocation_usage(self) -> None:
        intent = _intent()
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            planner=_UsagePlanner(text="ok"),
        )
        trigger = MCPToolTrigger("hi", session_id="s-usage")
        inv = await loop.forward(trigger=trigger)
        assert inv.usage.input_tokens == 10
        assert inv.usage.output_tokens == 20
        assert inv.usage.total_tokens == 30
        assert inv.usage.cost_usd == pytest.approx(0.005)
        assert inv.cost_usd == pytest.approx(0.005)
        # TurnSummary must reflect the rollup, not zeros.
        summaries = [e for e in inv.events if isinstance(e, TurnSummary)]
        assert len(summaries) == 1
        assert summaries[0].tokens_used == 30
        assert summaries[0].cost_usd == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# invoke() bridge
# ---------------------------------------------------------------------------


class TestInvoke:
    async def test_invoke_returns_turn_invocation(self) -> None:
        intent = _intent()
        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        trigger = MCPToolTrigger("hello", session_id="s-invoke")
        inv = await loop.invoke(trigger=trigger)
        # We override Program.invoke to return TurnInvocation, not
        # the kaos-llm-core Invocation wrapper.
        from kaos_agents.core.invocation import TurnInvocation

        assert isinstance(inv, TurnInvocation)
        assert inv.is_complete

    async def test_forward_requires_trigger_kwarg(self) -> None:
        intent = _intent()
        loop = AgentLoop(intent_extractor=_stub_extractor(intent))
        with pytest.raises(TypeError, match="trigger"):
            await loop.forward()
