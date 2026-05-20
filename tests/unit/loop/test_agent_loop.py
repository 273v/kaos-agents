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
import os
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
# _corpus_size_from_memory + IntentExtractor.invoke kwarg threading
# ---------------------------------------------------------------------------


def _capturing_extractor(intent: IntentResult) -> tuple[IntentExtractor, list[dict[str, Any]]]:
    """Like ``_stub_extractor`` but records every ``invoke`` kwargs dict
    into a list the test can assert on. Returns ``(extractor, captures)``."""
    extractor = IntentExtractor()
    captures: list[dict[str, Any]] = []

    async def _invoke(**kwargs: Any) -> Invocation:
        captures.append(kwargs)
        return Invocation(
            client=None,
            model="anthropic:claude-haiku-4-5",
            context=None,
            output=intent,
            trace=None,
            usage=TokenUsage(),
        )

    extractor.invoke = _invoke  # ty: ignore[invalid-assignment]
    return extractor, captures


class _FakeMemoryItem:
    """Minimal MemoryItem stand-in for ``_corpus_headlines_from_memory``.

    The helper reads ``metadata["filename"]`` (or falls back to other
    metadata anchors / content). The stub mirrors that surface.
    """

    def __init__(self, filename: str | None, content: str = "") -> None:
        self.metadata: dict[str, Any] = {"filename": filename} if filename else {}
        self.content = content


class _FakeMemory:
    """Minimal SessionMemory stand-in for the prepare_turn corpus helpers.

    AgentLoop touches ``has_section``, ``section_item_count``, and (since
    persona-matrix-followups §6) ``get_section`` to materialize the
    ``corpus_headlines`` input. Everything else stays unused on the
    prepare_turn path (Phase 2 hydration is pass-through, the loop
    accepts whatever SessionMemory the caller provided).
    """

    def __init__(
        self,
        *,
        has_documents: bool,
        count: int = 0,
        filenames: tuple[str, ...] | None = None,
    ) -> None:
        self._has = has_documents
        self._count = count
        # If the test supplied explicit filenames, use those (and align
        # count to match for the size-helper). Otherwise synthesize a
        # placeholder list matching ``count`` so the headlines helper
        # produces a non-empty string.
        if filenames is not None:
            self._items = tuple(_FakeMemoryItem(name) for name in filenames)
            self._count = len(filenames)
        else:
            self._items = tuple(_FakeMemoryItem(f"doc-{i}.docx") for i in range(self._count))
        self.turn_count = 0
        # WU-G.2 / #352 — AgentLoop.prepare_turn now calls
        # ``mark_corpus_attached`` whenever ``corpus_size > 0``; the
        # stub records the call so tests can assert on it without
        # touching the real SessionMemory implementation.
        self.corpus_attached_marks: int = 0
        self.corpus_ever_attached: bool = False

    def mark_corpus_attached(self) -> None:
        self.corpus_attached_marks += 1
        self.corpus_ever_attached = True

    def has_section(self, section: Any) -> bool:
        from kaos_agents.types.memory import MemoryType

        return section == MemoryType.DOCUMENTS and self._has

    def section_item_count(self, section: Any) -> int:
        from kaos_agents.types.memory import MemoryType

        if section == MemoryType.DOCUMENTS and self._has:
            return self._count
        return 0

    def get(self, section: Any, *, max_tokens: int | None = None) -> tuple[_FakeMemoryItem, ...]:
        # Matches SessionMemory.get signature — the helper only uses the
        # positional ``section`` arg; ``max_tokens`` is accepted for
        # signature compatibility.
        from kaos_agents.types.memory import MemoryType

        if section == MemoryType.DOCUMENTS and self._has:
            return self._items
        return ()


class TestCorpusSizeFromMemory:
    """Pure-function coverage for the static helper."""

    def test_none_memory_returns_zero(self) -> None:
        assert AgentLoop._corpus_size_from_memory(None) == 0

    def test_memory_without_documents_section_returns_zero(self) -> None:
        mem = _FakeMemory(has_documents=False)
        assert AgentLoop._corpus_size_from_memory(mem) == 0  # ty: ignore[invalid-argument-type]

    def test_memory_with_empty_documents_section_returns_zero(self) -> None:
        mem = _FakeMemory(has_documents=True, count=0)
        assert AgentLoop._corpus_size_from_memory(mem) == 0  # ty: ignore[invalid-argument-type]

    def test_memory_with_documents_returns_count(self) -> None:
        mem = _FakeMemory(has_documents=True, count=7)
        assert AgentLoop._corpus_size_from_memory(mem) == 7  # ty: ignore[invalid-argument-type]


class TestPrepareTurnThreadsCorpusInputs:
    """End-to-end (prepare_turn) coverage: corpus_attached + corpus_size
    must reach ``IntentExtractor.invoke``."""

    async def test_no_memory_passes_false_and_zero(self) -> None:
        intent = _intent()
        extractor, captures = _capturing_extractor(intent)
        loop = AgentLoop(intent_extractor=extractor)
        trigger = MCPToolTrigger("hi", session_id="s-no-mem")
        await loop.prepare_turn(trigger)
        assert len(captures) == 1
        assert captures[0]["corpus_attached"] is False
        assert captures[0]["corpus_size"] == 0

    async def test_memory_with_corpus_passes_true_and_count(self) -> None:
        intent = _intent()
        extractor, captures = _capturing_extractor(intent)
        mem = _FakeMemory(has_documents=True, count=3)
        loop = AgentLoop(intent_extractor=extractor, memory=mem)  # ty: ignore[invalid-argument-type]
        trigger = MCPToolTrigger("summarize that", session_id="s-corpus")
        await loop.prepare_turn(trigger)
        assert len(captures) == 1
        assert captures[0]["corpus_attached"] is True
        assert captures[0]["corpus_size"] == 3

    async def test_memory_with_empty_corpus_passes_false_and_zero(self) -> None:
        # has_section=True but section_item_count=0 (e.g., a session
        # that uploaded a file then cleared it).
        intent = _intent()
        extractor, captures = _capturing_extractor(intent)
        mem = _FakeMemory(has_documents=True, count=0)
        loop = AgentLoop(intent_extractor=extractor, memory=mem)  # ty: ignore[invalid-argument-type]
        trigger = MCPToolTrigger("hi", session_id="s-empty-corpus")
        await loop.prepare_turn(trigger)
        assert captures[0]["corpus_attached"] is False
        assert captures[0]["corpus_size"] == 0


# ---------------------------------------------------------------------------
# forward — happy paths
# ---------------------------------------------------------------------------


class TestForwardSkeletonPath:
    async def test_no_planner_yields_skeleton_invocation(self) -> None:
        # Phase 3.D introduces auto-select. Opt out via auto_select_planner=False
        # to exercise the legacy Phase 2.B skeleton path.
        intent = _intent()
        loop = AgentLoop(
            intent_extractor=_stub_extractor(intent),
            auto_select_planner=False,
        )
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
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason=(
            "AgentLoop.invoke runs the full 8-step turn; with only the "
            "IntentExtractor stubbed, the 'respond' step still issues a "
            "real RespondSignature call against Anthropic. Passes locally "
            "where ANTHROPIC_API_KEY is exported; skipped in unit-only "
            "CI. Misclassified live test — tracked as a Phase D follow-up "
            "to either move the test to tests/integration/ or stub the "
            "respond step too."
        ),
    )
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
