"""Tests for :class:`~kaos_agents.planning.hierarchical_planner.HierarchicalPlanner`.

Phase 3.C — decomposes a goal into sub-goals and delegates each to
a sub-agent. The "plan" step produces a tree of (envelope, sub-goal)
specs from the intent; the "execute" step runs each sub-agent
through a fresh ``AgentLoop.invoke()`` and aggregates results into a
parent :class:`PlanResult`.

What we cover:

* :class:`SubAgentSpec` and :class:`HierarchicalPlan` value-type
  construction, defaults, and JSON round-trip.
* :class:`HierarchicalPlanner` construction defaults
  (``stream_mode="value-only"``, ``max_concurrent=3``, ``max_depth=3``,
  ``aggregation="concat"``).
* :class:`HierarchicalPlanner` rejects invalid ``stream_mode`` /
  ``aggregation_strategy``.
* :meth:`HierarchicalPlanner.plan` heuristic fallback (no envelopes,
  no decomposer): 1-sub-agent plan.
* :meth:`HierarchicalPlanner.plan` with explicit envelopes: count
  matches.
* :meth:`HierarchicalPlanner.plan` with ``decomposer_call`` stub:
  invokes the stub and returns its output verbatim.
* :meth:`HierarchicalPlanner._should_forward` per stream_mode
  (Resolved Decision #4):

  * ``"value-only"``: forwards UsageObserved / CitationFound /
    IntentClassified / Span / TurnSummary; suppresses
    TextDelta / ThinkingDelta / ToolCallArgsDelta.
  * ``"full"``: forwards every event.
  * ``"summary-only"``: forwards only TurnSummary and
    Span(SUBAGENT, COMPLETE).

* :meth:`HierarchicalPlanner.execute` happy path with stub agent_loop
  factory: aggregates outputs.
* :meth:`HierarchicalPlanner.execute` propagates value events to the
  parent collector when ``stream_mode="value-only"``.
* :meth:`HierarchicalPlanner.execute` suppresses TextDelta /
  ThinkingDelta in value-only mode.
* :meth:`HierarchicalPlanner.execute` aggregation strategies (concat
  / first / json).
* :meth:`HierarchicalPlanner.execute` rolls all sub-usage into
  :attr:`PlanResult.usage`.
* :meth:`HierarchicalPlanner.execute` enforces ``max_delegation_depth``.
* :class:`Planner` Protocol runtime check passes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kaos_agents.config import AgentPattern
from kaos_agents.core.envelope import AgentEnvelope
from kaos_agents.events.collector import collect_events
from kaos_agents.events.lifecycle import (
    IntentClassified,
    TurnSummary,
    UsageObserved,
)
from kaos_agents.events.research import CitationFound
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.events.stream import TextDelta, ThinkingDelta, ToolCallArgsDelta
from kaos_agents.intent.types import Goal, IntentResult
from kaos_agents.planning.hierarchical_planner import (
    HierarchicalPlan,
    HierarchicalPlanner,
    SubAgentSpec,
    _hierarchical_depth,
    current_hierarchical_depth,
)
from kaos_agents.planning.planner import Plan, Planner, PlanResult
from kaos_agents.types.intents import IntentType
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_intent(
    statement: str = "Summarize the contract.",
    intent_type: IntentType = IntentType.PLAN,
) -> IntentResult:
    return IntentResult(
        goal=Goal(statement=statement, intent_type=intent_type),
        pattern=AgentPattern.PLAN,
        confidence=0.9,
        raw_input=statement,
    )


def _make_envelope(
    name: str = "stub_subagent", model: str = "anthropic:claude-haiku-4-5"
) -> AgentEnvelope:
    return AgentEnvelope(
        pattern=AgentPattern.CHAT,
        instructions="You are a stub sub-agent.",
        model=model,
        name=name,
    )


# Default base-event field values shared across every synthetic
# :class:`KaosEvent` in this suite. Real emission goes through
# :class:`EventEmitter`, which fills these in; tests construct events
# directly, so we provide them explicitly. We expand them as named
# kwargs in each helper rather than via ``**_BASE_FIELDS`` so the
# static type checker can resolve each value's type against the
# pydantic field.
_TS = 0.0
_SEQ = 0
_SID = "test-session"
_RID = "test-run"


def _text_delta(content: str = "x") -> TextDelta:
    return TextDelta(timestamp=_TS, sequence=_SEQ, session_id=_SID, run_id=_RID, content=content)


def _thinking_delta(content: str = "x") -> ThinkingDelta:
    return ThinkingDelta(
        timestamp=_TS, sequence=_SEQ, session_id=_SID, run_id=_RID, content=content
    )


def _tool_args_delta(call_id: str = "c1", content: str = '{"q":"x"}') -> ToolCallArgsDelta:
    return ToolCallArgsDelta(
        timestamp=_TS,
        sequence=_SEQ,
        session_id=_SID,
        run_id=_RID,
        call_id=call_id,
        content=content,
    )


def _usage_observed(
    input_tokens: int = 1, output_tokens: int = 2, total_tokens: int = 3
) -> UsageObserved:
    return UsageObserved(
        timestamp=_TS,
        sequence=_SEQ,
        session_id=_SID,
        run_id=_RID,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _citation_found(claim: str = "x", source_uri: str = "y") -> CitationFound:
    return CitationFound(
        timestamp=_TS,
        sequence=_SEQ,
        session_id=_SID,
        run_id=_RID,
        claim=claim,
        source_uri=source_uri,
        confidence=0.9,
        verified=True,
    )


def _intent_classified(intent: str = "research") -> IntentClassified:
    return IntentClassified(
        timestamp=_TS,
        sequence=_SEQ,
        session_id=_SID,
        run_id=_RID,
        intent=intent,
        confidence=0.9,
    )


def _turn_summary(text: str = "done") -> TurnSummary:
    return TurnSummary(timestamp=_TS, sequence=_SEQ, session_id=_SID, run_id=_RID, text=text)


def _make_span(
    subject: SpanSubject = SpanSubject.SUBAGENT,
    phase: SpanPhase = SpanPhase.COMPLETE,
    name: str = "subagent.stub",
) -> Span:
    return Span(
        timestamp=_TS,
        sequence=_SEQ,
        session_id=_SID,
        run_id=_RID,
        subject=subject,
        phase=phase,
        span_id="span-stub",
        name=name,
    )


class _StubAgentLoop:
    """Stub :class:`AgentLoop` returning a canned :class:`TurnInvocation`-like
    SimpleNamespace and pushing pre-canned events into the active
    collector.
    """

    def __init__(
        self,
        output: str = "stub-out",
        usage: InvocationUsage | None = None,
        events: tuple[Any, ...] = (),
    ) -> None:
        self._output = output
        self._usage = usage or InvocationUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.001,
        )
        self._events = events
        self.invoke_calls: list[Any] = []

    async def invoke(self, *, trigger: Any) -> Any:
        # Push pre-canned events into the active collector to simulate
        # a real sub-agent run.
        from kaos_agents.events.collector import push_event

        self.invoke_calls.append(trigger)
        for event in self._events:
            push_event(event)
        return SimpleNamespace(
            output=self._output,
            usage=self._usage,
            tool_executions=(),
            id="stub-id",
        )


def _stub_factory(loops: list[_StubAgentLoop]) -> Any:
    """Build a factory that returns the next stub loop in a list.

    Each call pops one loop from the list. Tests construct the list
    in declaration order matching the SubAgentSpec order in the plan.
    """
    iterator = iter(loops)

    def _factory(_env: AgentEnvelope) -> _StubAgentLoop:
        return next(iterator)

    return _factory


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        planner = HierarchicalPlanner()
        assert planner.stream_mode == "value-only"
        assert planner.max_concurrent_subagents == 3
        assert planner.max_delegation_depth == 3
        assert planner.aggregation_strategy == "concat"

    def test_explicit_kwargs(self) -> None:
        planner = HierarchicalPlanner(
            stream_mode="full",
            max_concurrent_subagents=5,
            max_delegation_depth=10,
            aggregation_strategy="json",
        )
        assert planner.stream_mode == "full"
        assert planner.max_concurrent_subagents == 5
        assert planner.max_delegation_depth == 10
        assert planner.aggregation_strategy == "json"

    def test_invalid_stream_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="stream_mode must be one of"):
            HierarchicalPlanner(stream_mode="invalid")

    def test_invalid_aggregation_raises(self) -> None:
        with pytest.raises(ValueError, match="aggregation_strategy must be one of"):
            HierarchicalPlanner(aggregation_strategy="weighted")

    def test_decomposer_call_stashed(self) -> None:
        sentinel = AsyncMock()
        planner = HierarchicalPlanner(decomposer_call=sentinel)
        assert planner._decomposer_call is sentinel

    def test_envelopes_stashed(self) -> None:
        env = _make_envelope()
        planner = HierarchicalPlanner(sub_agent_envelopes=(env,))
        assert planner._sub_agents == (env,)


# ---------------------------------------------------------------------------
# SubAgentSpec / HierarchicalPlan value types
# ---------------------------------------------------------------------------


class TestSubAgentSpec:
    def test_construction_defaults(self) -> None:
        env = _make_envelope()
        spec = SubAgentSpec(
            sub_agent_envelope_json=env.model_dump_json(),
            sub_goal="Find party names.",
        )
        assert spec.sub_goal == "Find party names."
        assert spec.sub_agent_name == ""
        assert spec.inputs == {}
        assert spec.pattern == "hierarchical_subagent"

    def test_json_round_trip(self) -> None:
        env = _make_envelope()
        spec = SubAgentSpec(
            sub_agent_envelope_json=env.model_dump_json(),
            sub_goal="g",
            sub_agent_name="foo",
            inputs={"k": "v"},
        )
        revived = SubAgentSpec.model_validate_json(spec.model_dump_json())
        assert revived.sub_goal == "g"
        assert revived.sub_agent_name == "foo"
        assert revived.inputs == {"k": "v"}
        # The envelope JSON survives intact and re-validates.
        env_back = AgentEnvelope.model_validate_json(revived.sub_agent_envelope_json)
        assert env_back.name == env.name


class TestHierarchicalPlan:
    def test_construction_defaults(self) -> None:
        p = HierarchicalPlan()
        assert p.pattern == "hierarchical"
        assert p.parent_goal == ""
        assert p.sub_agents == ()
        assert p.aggregation_strategy == "concat"

    def test_full_construction(self) -> None:
        env = _make_envelope()
        spec = SubAgentSpec(
            sub_agent_envelope_json=env.model_dump_json(),
            sub_goal="g",
        )
        p = HierarchicalPlan(
            parent_goal="parent",
            sub_agents=(spec,),
            aggregation_strategy="first",
        )
        assert p.parent_goal == "parent"
        assert len(p.sub_agents) == 1
        assert p.aggregation_strategy == "first"

    def test_json_round_trip(self) -> None:
        env = _make_envelope()
        spec = SubAgentSpec(
            sub_agent_envelope_json=env.model_dump_json(),
            sub_goal="x",
        )
        p = HierarchicalPlan(
            parent_goal="parent",
            sub_agents=(spec,),
            aggregation_strategy="json",
        )
        revived = HierarchicalPlan.model_validate_json(p.model_dump_json())
        assert revived.parent_goal == "parent"
        assert len(revived.sub_agents) == 1
        assert revived.sub_agents[0].sub_goal == "x"
        assert revived.aggregation_strategy == "json"


# ---------------------------------------------------------------------------
# plan() — heuristic + explicit + decomposer paths
# ---------------------------------------------------------------------------


class TestPlanHeuristic:
    @pytest.mark.asyncio
    async def test_heuristic_yields_single_sub_agent(self) -> None:
        planner = HierarchicalPlanner()
        intent = _make_intent("Find facts.", IntentType.RESEARCH)
        plan = await planner.plan(intent)

        assert isinstance(plan, HierarchicalPlan)
        assert plan.pattern == "hierarchical"
        assert plan.parent_goal == "Find facts."
        assert len(plan.sub_agents) == 1
        assert plan.metadata == {
            "source": "heuristic",
            "intent_type": IntentType.RESEARCH.value,
            "sub_agent_count": 1,
        }
        # The heuristic envelope embeds the goal in instructions.
        env = AgentEnvelope.model_validate_json(plan.sub_agents[0].sub_agent_envelope_json)
        assert "Find facts." in env.instructions
        assert env.pattern == AgentPattern.CHAT
        assert env.model == "anthropic:claude-haiku-4-5"


class TestPlanExplicitEnvelopes:
    @pytest.mark.asyncio
    async def test_envelope_count_matches(self) -> None:
        envs = (
            _make_envelope(name="researcher"),
            _make_envelope(name="writer"),
            _make_envelope(name="critic"),
        )
        planner = HierarchicalPlanner(sub_agent_envelopes=envs)
        intent = _make_intent("Compose memo.", IntentType.PLAN)
        plan = await planner.plan(intent)
        assert len(plan.sub_agents) == 3
        assert [s.sub_agent_name for s in plan.sub_agents] == [
            "researcher",
            "writer",
            "critic",
        ]
        # Each sub-goal mirrors the parent goal in Phase 3.C.
        assert all(s.sub_goal == "Compose memo." for s in plan.sub_agents)
        assert plan.metadata == {
            "source": "explicit_envelopes",
            "intent_type": IntentType.PLAN.value,
            "sub_agent_count": 3,
        }

    @pytest.mark.asyncio
    async def test_unnamed_envelope_gets_default_name(self) -> None:
        env = AgentEnvelope(
            pattern=AgentPattern.CHAT,
            instructions="anon",
            model="anthropic:claude-haiku-4-5",
        )
        planner = HierarchicalPlanner(sub_agent_envelopes=(env,))
        plan = await planner.plan(_make_intent())
        # Default name pattern: subagent_<index>.
        assert plan.sub_agents[0].sub_agent_name == "subagent_0"


class TestPlanDecomposerCall:
    @pytest.mark.asyncio
    async def test_decomposer_call_invocation_output_returned(self) -> None:
        env = _make_envelope()
        custom_plan = HierarchicalPlan(
            parent_goal="from-llm",
            sub_agents=(
                SubAgentSpec(
                    sub_agent_envelope_json=env.model_dump_json(),
                    sub_goal="step-1",
                ),
            ),
        )
        invocation = SimpleNamespace(output=custom_plan)
        invoke_mock = AsyncMock(return_value=invocation)
        call_stub = SimpleNamespace(invoke=invoke_mock)

        planner = HierarchicalPlanner(decomposer_call=call_stub)
        intent = _make_intent("Anything.", IntentType.RESEARCH)

        result = await planner.plan(intent)

        invoke_mock.assert_awaited_once()
        await_args = invoke_mock.await_args
        assert await_args is not None
        assert await_args.kwargs["goal"] == "Anything."
        assert await_args.kwargs["constraints"] == ()
        assert result is custom_plan


# ---------------------------------------------------------------------------
# _should_forward — Resolved Decision #4
# ---------------------------------------------------------------------------


class TestShouldForwardValueOnly:
    """Default mode: collapse stream-deltas, propagate value events."""

    def test_forwards_usage_observed(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_usage_observed()) is True

    def test_forwards_citation_found(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_citation_found()) is True

    def test_forwards_intent_classified(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_intent_classified()) is True

    def test_forwards_span(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_make_span()) is True

    def test_forwards_turn_summary(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_turn_summary()) is True

    def test_suppresses_text_delta(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_text_delta("hello")) is False

    def test_suppresses_thinking_delta(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_thinking_delta("thinking")) is False

    def test_suppresses_tool_call_args_delta(self) -> None:
        planner = HierarchicalPlanner()
        assert planner._should_forward(_tool_args_delta()) is False


class TestShouldForwardFull:
    """Full mode: forward every event."""

    def test_forwards_text_delta(self) -> None:
        planner = HierarchicalPlanner(stream_mode="full")
        assert planner._should_forward(_text_delta("hi")) is True

    def test_forwards_thinking_delta(self) -> None:
        planner = HierarchicalPlanner(stream_mode="full")
        assert planner._should_forward(_thinking_delta("x")) is True

    def test_forwards_value_event(self) -> None:
        planner = HierarchicalPlanner(stream_mode="full")
        assert planner._should_forward(_turn_summary("x")) is True

    def test_forwards_arbitrary_event(self) -> None:
        # Even a non-value, non-stream event passes through in full
        # mode. Use UsageObserved to assert positive forwarding without
        # constructing an exotic event subtype.
        planner = HierarchicalPlanner(stream_mode="full")
        assert planner._should_forward(_usage_observed()) is True


class TestShouldForwardSummaryOnly:
    """Summary-only mode: only TurnSummary + Span(SUBAGENT, COMPLETE)."""

    def test_forwards_turn_summary(self) -> None:
        planner = HierarchicalPlanner(stream_mode="summary-only")
        assert planner._should_forward(_turn_summary("done")) is True

    def test_forwards_subagent_complete_span(self) -> None:
        planner = HierarchicalPlanner(stream_mode="summary-only")
        ev = _make_span(subject=SpanSubject.SUBAGENT, phase=SpanPhase.COMPLETE)
        assert planner._should_forward(ev) is True

    def test_suppresses_subagent_start_span(self) -> None:
        planner = HierarchicalPlanner(stream_mode="summary-only")
        ev = _make_span(subject=SpanSubject.SUBAGENT, phase=SpanPhase.START)
        assert planner._should_forward(ev) is False

    def test_suppresses_step_complete_span(self) -> None:
        # Different subject (STEP) — only SUBAGENT spans pass.
        planner = HierarchicalPlanner(stream_mode="summary-only")
        ev = _make_span(subject=SpanSubject.STEP, phase=SpanPhase.COMPLETE)
        assert planner._should_forward(ev) is False

    def test_suppresses_value_events(self) -> None:
        planner = HierarchicalPlanner(stream_mode="summary-only")
        assert planner._should_forward(_usage_observed()) is False

    def test_suppresses_stream_deltas(self) -> None:
        planner = HierarchicalPlanner(stream_mode="summary-only")
        assert planner._should_forward(_text_delta("x")) is False


# ---------------------------------------------------------------------------
# execute() — sub-agent dispatch + aggregation
# ---------------------------------------------------------------------------


class TestExecuteHappyPath:
    @pytest.mark.asyncio
    async def test_execute_aggregates_outputs(self) -> None:
        env = _make_envelope()
        loops = [_StubAgentLoop(output="result-a"), _StubAgentLoop(output="result-b")]

        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env, env),
            agent_loop_factory=_stub_factory(loops),
        )
        intent = _make_intent("Compose memo.", IntentType.PLAN)
        plan = await planner.plan(intent)

        result = await planner.execute(plan)

        assert isinstance(result, PlanResult)
        # Concat aggregation joins on newline.
        assert "result-a" in result.text
        assert "result-b" in result.text
        # Both outputs survive into output as well.
        assert result.output == result.text
        # Both stub loops were invoked.
        assert len(loops[0].invoke_calls) == 1
        assert len(loops[1].invoke_calls) == 1
        # Metadata reflects sub-agent count.
        assert result.metadata["sub_agents_run"] == 2
        assert len(result.metadata["sub_agent_records"]) == 2

    @pytest.mark.asyncio
    async def test_execute_text_and_output_aliased(self) -> None:
        env = _make_envelope()
        loops = [_StubAgentLoop(output="single")]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env,),
            agent_loop_factory=_stub_factory(loops),
        )
        plan = await planner.plan(_make_intent())
        result = await planner.execute(plan)
        # text and output are populated to the same value.
        assert result.text == result.output

    @pytest.mark.asyncio
    async def test_execute_sums_sub_usage(self) -> None:
        env = _make_envelope()
        u1 = InvocationUsage(input_tokens=5, output_tokens=5, total_tokens=10, cost_usd=0.001)
        u2 = InvocationUsage(input_tokens=7, output_tokens=3, total_tokens=10, cost_usd=0.002)
        loops = [
            _StubAgentLoop(output="a", usage=u1),
            _StubAgentLoop(output="b", usage=u2),
        ]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env, env),
            agent_loop_factory=_stub_factory(loops),
        )
        plan = await planner.plan(_make_intent())
        result = await planner.execute(plan)

        # Totals sum across both sub-agents.
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 8
        assert result.usage.total_tokens == 20
        assert result.usage.cost_usd == pytest.approx(0.003)

    @pytest.mark.asyncio
    async def test_execute_empty_plan_short_circuits(self) -> None:
        planner = HierarchicalPlanner()
        empty = HierarchicalPlan(parent_goal="none", sub_agents=())
        result = await planner.execute(empty)
        assert result.text == ""
        assert result.usage == ZERO_USAGE
        assert result.metadata["sub_agents_run"] == 0


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------


class TestAggregationStrategies:
    @pytest.mark.asyncio
    async def test_concat_joins_on_newline(self) -> None:
        env = _make_envelope()
        loops = [_StubAgentLoop(output="alpha"), _StubAgentLoop(output="beta")]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env, env),
            agent_loop_factory=_stub_factory(loops),
            aggregation_strategy="concat",
        )
        plan = await planner.plan(_make_intent())
        result = await planner.execute(plan)
        assert result.text == "alpha\nbeta"

    @pytest.mark.asyncio
    async def test_first_returns_first_non_empty(self) -> None:
        env = _make_envelope()
        loops = [
            _StubAgentLoop(output=""),
            _StubAgentLoop(output="winner"),
            _StubAgentLoop(output="loser"),
        ]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env, env, env),
            agent_loop_factory=_stub_factory(loops),
            aggregation_strategy="first",
        )
        plan = await planner.plan(_make_intent())
        result = await planner.execute(plan)
        assert result.text == "winner"

    @pytest.mark.asyncio
    async def test_json_returns_dict_string(self) -> None:
        env_a = _make_envelope(name="alpha")
        env_b = _make_envelope(name="beta")
        loops = [_StubAgentLoop(output="ans-a"), _StubAgentLoop(output="ans-b")]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env_a, env_b),
            agent_loop_factory=_stub_factory(loops),
            aggregation_strategy="json",
        )
        plan = await planner.plan(_make_intent())
        result = await planner.execute(plan)

        # Round-trip the JSON string to verify shape.
        decoded = json.loads(result.text)
        assert decoded == {"alpha": "ans-a", "beta": "ans-b"}


# ---------------------------------------------------------------------------
# Event propagation — the key Resolved Decision #4 behaviour.
# ---------------------------------------------------------------------------


class TestEventPropagation:
    @pytest.mark.asyncio
    async def test_value_only_propagates_value_events_to_parent(self) -> None:
        """Open a parent collector. Run the planner. Verify only value
        events appear in the parent's events list — stream deltas
        emitted by the sub-agent are suppressed.
        """
        env = _make_envelope()
        sub_events = (
            _text_delta("hello"),  # suppressed
            _usage_observed(),  # forwarded
            _thinking_delta("thinking"),  # suppressed
            _citation_found(),  # forwarded
            _turn_summary("done"),  # forwarded
        )
        loops = [_StubAgentLoop(output="sub-out", events=sub_events)]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env,),
            agent_loop_factory=_stub_factory(loops),
            stream_mode="value-only",
        )
        plan = await planner.plan(_make_intent())

        with collect_events() as parent_collector:
            await planner.execute(plan)

        # Value events present.
        forwarded = parent_collector.events
        assert any(isinstance(e, UsageObserved) for e in forwarded)
        assert any(isinstance(e, CitationFound) for e in forwarded)
        assert any(isinstance(e, TurnSummary) for e in forwarded)
        # Stream deltas absent.
        assert not any(isinstance(e, TextDelta) for e in forwarded)
        assert not any(isinstance(e, ThinkingDelta) for e in forwarded)

    @pytest.mark.asyncio
    async def test_full_mode_forwards_stream_deltas(self) -> None:
        env = _make_envelope()
        sub_events = (
            _text_delta("hello"),
            _usage_observed(),
        )
        loops = [_StubAgentLoop(output="x", events=sub_events)]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env,),
            agent_loop_factory=_stub_factory(loops),
            stream_mode="full",
        )
        plan = await planner.plan(_make_intent())

        with collect_events() as parent_collector:
            await planner.execute(plan)

        forwarded = parent_collector.events
        assert any(isinstance(e, TextDelta) for e in forwarded)
        assert any(isinstance(e, UsageObserved) for e in forwarded)

    @pytest.mark.asyncio
    async def test_summary_only_drops_value_events(self) -> None:
        env = _make_envelope()
        sub_events = (
            _usage_observed(),  # dropped
            _text_delta("x"),  # dropped
            _make_span(SpanSubject.SUBAGENT, SpanPhase.COMPLETE),  # forwarded
            _turn_summary("done"),  # forwarded
        )
        loops = [_StubAgentLoop(output="x", events=sub_events)]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env,),
            agent_loop_factory=_stub_factory(loops),
            stream_mode="summary-only",
        )
        plan = await planner.plan(_make_intent())

        with collect_events() as parent_collector:
            await planner.execute(plan)

        forwarded = parent_collector.events
        # TurnSummary forwarded.
        assert any(isinstance(e, TurnSummary) for e in forwarded)
        # Subagent COMPLETE span forwarded.
        assert any(
            isinstance(e, Span)
            and e.subject == SpanSubject.SUBAGENT
            and e.phase == SpanPhase.COMPLETE
            for e in forwarded
        )
        # UsageObserved + TextDelta dropped.
        assert not any(isinstance(e, UsageObserved) for e in forwarded)
        assert not any(isinstance(e, TextDelta) for e in forwarded)


# ---------------------------------------------------------------------------
# Depth tracking
# ---------------------------------------------------------------------------


class TestMaxDelegationDepth:
    @pytest.mark.asyncio
    async def test_exceeds_depth_raises(self) -> None:
        """Manually pre-seed the depth ContextVar above the cap to
        simulate a recursive hierarchical-planner stack.
        """
        env = _make_envelope()
        loops = [_StubAgentLoop()]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env,),
            agent_loop_factory=_stub_factory(loops),
            max_delegation_depth=2,
        )
        plan = await planner.plan(_make_intent())

        # Pretend we're 2 hops deep already; the next execute() must
        # refuse before even spinning a sub-agent.
        token = _hierarchical_depth.set(2)
        try:
            with pytest.raises(RuntimeError, match="max_depth=2"):
                await planner.execute(plan)
        finally:
            _hierarchical_depth.reset(token)
        # The factory was NOT consumed — we never spun a sub-agent.
        assert len(loops[0].invoke_calls) == 0

    @pytest.mark.asyncio
    async def test_normal_run_increments_then_decrements_depth(self) -> None:
        env = _make_envelope()
        loops = [_StubAgentLoop()]
        planner = HierarchicalPlanner(
            sub_agent_envelopes=(env,),
            agent_loop_factory=_stub_factory(loops),
        )
        plan = await planner.plan(_make_intent())

        assert current_hierarchical_depth() == 0
        await planner.execute(plan)
        # Cleaned up after execute returned.
        assert current_hierarchical_depth() == 0


# ---------------------------------------------------------------------------
# Coercion + Protocol compliance
# ---------------------------------------------------------------------------


class TestCoercion:
    @pytest.mark.asyncio
    async def test_coerce_bare_plan(self) -> None:
        """A loose :class:`Plan` carrying ``sub_agents`` via
        ``extra='allow'`` is coerced into a :class:`HierarchicalPlan`.
        """
        planner = HierarchicalPlanner()
        bare = Plan(
            pattern="hierarchical",
            metadata={
                "parent_goal": "coerced",
                "sub_agents": (),
                "aggregation_strategy": "concat",
            },
        )
        result = await planner.execute(bare)
        assert isinstance(result, PlanResult)
        # No sub-agents → empty result.
        assert result.text == ""


class TestPlannerProtocolCompliance:
    def test_implements_planner_protocol(self) -> None:
        planner = HierarchicalPlanner()
        assert isinstance(planner, Planner)


class TestDefaultFactoryAntiRecursion:
    """DEFECT-4 regression — default sub-AgentLoop pins ReActPlanner.

    Without this fix, RESEARCH-classified sub-intents would trigger
    classifier-driven auto-select that picks HierarchicalPlanner again,
    recursing until ``max_depth=3`` raises RuntimeError. The default
    factory now sets ``auto_select_planner=False`` and pins
    ``planner=ReActPlanner(...)`` so the recursion can't form.
    """

    def test_default_factory_disables_auto_select(self) -> None:
        from kaos_agents.config import AgentPattern
        from kaos_agents.core.envelope import AgentEnvelope
        from kaos_agents.planning.hierarchical_planner import (
            _default_agent_loop_factory,
        )
        from kaos_agents.planning.react_planner import ReActPlanner

        env = AgentEnvelope(
            pattern=AgentPattern.RESEARCH,
            instructions="sub-agent test",
            model="anthropic:claude-haiku-4-5",
        )
        sub_loop = _default_agent_loop_factory(env)

        assert sub_loop._auto_select_planner is False, (
            "DEFECT-4: default sub-AgentLoop must disable auto-select "
            "so RESEARCH sub-intents don't recurse into HierarchicalPlanner."
        )
        assert isinstance(sub_loop._planner, ReActPlanner), (
            "DEFECT-4: default sub-AgentLoop must pin ReActPlanner as the leaf executor."
        )
