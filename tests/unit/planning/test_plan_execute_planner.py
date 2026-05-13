"""Tests for :class:`~kaos_agents.planning.plan_execute_planner.PlanExecutePlanner`.

Phase 3.B — produces a typed :class:`PlanExecutePlan` and executes it
via kaos-llm-core's :class:`LoopRunner`. The Phase 3.B "vanilla"
strategy is a deterministic heuristic over the intent's
:class:`IntentType`.

What we cover:

* Construction defaults: ``strategy="vanilla"``, ``max_steps=8``.
* Non-vanilla strategy (``"rewoo"`` / ``"compiler"``) is silently
  downgraded to ``"vanilla"``; ``requested_strategy`` preserves the
  original.
* :class:`PlanStep` / :class:`PlanExecutePlan` value-type construction
  defaults + JSON round-trip.
* Heuristic plan generation for each :class:`IntentType`:
  ``RESPOND`` → 1-step plan, ``RESEARCH`` → 2-step plan with a
  ``"perceive"`` step, ``PLAN`` → 3-step plan.
* ``planner_call`` override: when supplied, the call's
  ``invocation.output`` is returned verbatim.
* :meth:`execute` happy path with stub perceiver + actor: every step
  routes to the right dispatcher and the final :attr:`PlanResult.text`
  concatenates the outputs.
* :meth:`execute` with ``perceiver=None`` skips ``"perceive"`` steps
  with a metadata note.
* :meth:`execute` with ``actor=None`` skips ``"act"`` steps with a
  metadata note.
* :meth:`execute` populates BOTH :attr:`PlanResult.text` AND
  :attr:`PlanResult.output` (Phase 2.B AgentLoop's ``getattr`` chain).
* :meth:`execute` records ``metadata["steps_executed"]`` /
  ``["steps_skipped"]`` reflecting actual counts.
* :class:`Planner` Protocol runtime check passes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kaos_agents.action.types import ActionPlan, ActionResult
from kaos_agents.config import AgentPattern
from kaos_agents.intent.types import Constraint, ConstraintKind, Goal, IntentResult
from kaos_agents.perception.types import (
    PerceptionItem,
    PerceptionQuery,
    PerceptionResult,
)
from kaos_agents.planning.plan_execute_planner import (
    PlanExecutePlan,
    PlanExecutePlanner,
    PlanStep,
)
from kaos_agents.planning.planner import Plan, Planner, PlanResult
from kaos_agents.types.intents import IntentType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_intent(
    statement: str = "Summarize the contract.",
    intent_type: IntentType = IntentType.RESEARCH,
) -> IntentResult:
    return IntentResult(
        goal=Goal(statement=statement, intent_type=intent_type),
        pattern=AgentPattern.PLAN,
        confidence=0.9,
        raw_input=statement,
    )


class _StubPerceiver:
    """Minimal duck-typed perceiver — returns a single
    :class:`PerceptionItem` with the query text echoed back.
    """

    def __init__(self, content: str = "found-fact") -> None:
        self._content = content
        self.calls: list[PerceptionQuery] = []
        self.tools: tuple[Any, ...] = ()

    async def forward(self, query: PerceptionQuery) -> PerceptionResult:
        self.calls.append(query)
        return PerceptionResult(
            items=(PerceptionItem(content=self._content, source="stub"),),
            confidence=1.0,
            sources_consulted=("stub",),
        )


class _StubActor:
    """Minimal duck-typed actor — returns a successful
    :class:`ActionResult` for every call.
    """

    def __init__(self, output: str = "ok") -> None:
        self._output = output
        self.calls: list[ActionPlan] = []
        self.tools: tuple[Any, ...] = ()

    async def forward(self, *, plan: ActionPlan) -> ActionResult:
        self.calls.append(plan)
        return ActionResult(tool_name=plan.tool_name, success=True, output=self._output)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        planner = PlanExecutePlanner()
        assert planner.strategy == "vanilla"
        assert planner.requested_strategy == "vanilla"
        assert planner._max_steps == 8
        assert planner._step_timeout == 60.0
        assert planner._planner_call is None

    def test_explicit_max_steps_and_timeout(self) -> None:
        planner = PlanExecutePlanner(max_steps=3, step_timeout_seconds=12.5)
        assert planner._max_steps == 3
        assert planner._step_timeout == 12.5

    def test_rewoo_strategy_downgraded_to_vanilla(self) -> None:
        """Phase 3.B: ``"rewoo"`` is accepted but downgraded — the
        active :attr:`strategy` is ``"vanilla"`` while
        :attr:`requested_strategy` preserves the original.
        """
        planner = PlanExecutePlanner(strategy="rewoo")
        assert planner.strategy == "vanilla"
        assert planner.requested_strategy == "rewoo"

    def test_compiler_strategy_downgraded_to_vanilla(self) -> None:
        planner = PlanExecutePlanner(strategy="compiler")
        assert planner.strategy == "vanilla"
        assert planner.requested_strategy == "compiler"

    def test_planner_call_stashed(self) -> None:
        sentinel = MagicMock()
        planner = PlanExecutePlanner(planner_call=sentinel)
        assert planner._planner_call is sentinel


# ---------------------------------------------------------------------------
# PlanStep / PlanExecutePlan value types
# ---------------------------------------------------------------------------


class TestPlanStep:
    def test_construction_defaults(self) -> None:
        step = PlanStep(step_id="s1", kind="respond", description="say hi")
        assert step.step_id == "s1"
        assert step.kind == "respond"
        assert step.description == "say hi"
        assert step.inputs == {}
        assert step.depends_on == ()
        assert step.pattern == "plan_execute_step"

    def test_json_round_trip(self) -> None:
        step = PlanStep(
            step_id="s2",
            kind="perceive",
            description="lookup",
            inputs={"query_text": "the contract"},
            depends_on=("s1",),
        )
        revived = PlanStep.model_validate_json(step.model_dump_json())
        assert revived.step_id == "s2"
        assert revived.kind == "perceive"
        assert revived.inputs == {"query_text": "the contract"}
        assert revived.depends_on == ("s1",)


class TestPlanExecutePlan:
    def test_construction_defaults(self) -> None:
        p = PlanExecutePlan()
        assert p.pattern == "plan_execute"
        assert p.strategy == "vanilla"
        assert p.goal_statement == ""
        assert p.steps == ()

    def test_full_construction(self) -> None:
        p = PlanExecutePlan(
            strategy="vanilla",
            goal_statement="Find party names.",
            steps=(
                PlanStep(step_id="s1", kind="perceive", description="lookup"),
                PlanStep(step_id="s2", kind="respond", description="answer"),
            ),
        )
        assert p.goal_statement == "Find party names."
        assert len(p.steps) == 2
        assert p.steps[0].kind == "perceive"

    def test_json_round_trip(self) -> None:
        p = PlanExecutePlan(
            goal_statement="Goal.",
            steps=(PlanStep(step_id="s1", kind="respond", description="d"),),
        )
        revived = PlanExecutePlan.model_validate_json(p.model_dump_json())
        assert revived.goal_statement == "Goal."
        assert len(revived.steps) == 1
        assert revived.steps[0].step_id == "s1"


# ---------------------------------------------------------------------------
# plan() — heuristic generator
# ---------------------------------------------------------------------------


class TestPlanHeuristic:
    @pytest.mark.asyncio
    async def test_respond_intent_yields_single_step(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Hello!", IntentType.RESPOND)
        plan = await planner.plan(intent)
        assert isinstance(plan, PlanExecutePlan)
        assert plan.strategy == "vanilla"
        assert plan.goal_statement == "Hello!"
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "respond"
        assert plan.metadata == {
            "source": "heuristic",
            "intent_type": IntentType.RESPOND.value,
        }

    @pytest.mark.asyncio
    async def test_research_intent_yields_perceive_then_respond(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Find party names.", IntentType.RESEARCH)
        plan = await planner.plan(intent)
        assert len(plan.steps) == 2
        assert [s.kind for s in plan.steps] == ["perceive", "respond"]
        # Perceive step carries query text in inputs.
        assert plan.steps[0].inputs == {"query_text": "Find party names."}
        # Sequential dep recorded.
        assert plan.steps[1].depends_on == ("s1",)

    @pytest.mark.asyncio
    async def test_tool_use_intent_yields_act_then_respond(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Open the case file.", IntentType.TOOL_USE)
        plan = await planner.plan(intent)
        assert len(plan.steps) == 2
        assert [s.kind for s in plan.steps] == ["act", "respond"]

    @pytest.mark.asyncio
    async def test_plan_intent_yields_three_steps(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Plan a complete review.", IntentType.PLAN)
        plan = await planner.plan(intent)
        assert len(plan.steps) == 3
        assert [s.kind for s in plan.steps] == ["perceive", "act", "respond"]

    @pytest.mark.asyncio
    async def test_clarify_intent_falls_back_to_respond(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Which contract?", IntentType.CLARIFY)
        plan = await planner.plan(intent)
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "respond"

    @pytest.mark.asyncio
    async def test_plan_ignores_memory(self) -> None:
        """``memory=`` is accepted for protocol compliance but not
        consulted by the heuristic generator (Phase 4 wires
        memory-aware planning).
        """
        planner = PlanExecutePlanner()
        intent = _make_intent("Hello.", IntentType.RESPOND)
        plan_a = await planner.plan(intent, memory=None)
        plan_b = await planner.plan(intent, memory=object())
        # Defaults are stable across memory inputs.
        assert plan_a.steps[0].kind == plan_b.steps[0].kind
        assert plan_a.goal_statement == plan_b.goal_statement


class TestPlannerCallOverride:
    @pytest.mark.asyncio
    async def test_planner_call_invocation_output_returned(self) -> None:
        """When ``planner_call`` is provided, ``plan()`` invokes the
        call and returns its ``invocation.output`` directly.
        """
        custom_plan = PlanExecutePlan(
            goal_statement="custom",
            steps=(PlanStep(step_id="x1", kind="respond", description="hand-crafted"),),
            metadata={"source": "llm"},
        )
        invocation = SimpleNamespace(output=custom_plan)
        invoke_mock = AsyncMock(return_value=invocation)
        call_stub = SimpleNamespace(invoke=invoke_mock)

        planner = PlanExecutePlanner(planner_call=call_stub)
        intent = _make_intent("Anything.", IntentType.RESEARCH)

        result = await planner.plan(intent)

        invoke_mock.assert_awaited_once()
        await_args = invoke_mock.await_args
        assert await_args is not None
        assert await_args.kwargs["goal"] == "Anything."
        assert await_args.kwargs["constraints"] == ()
        assert result is custom_plan

    @pytest.mark.asyncio
    async def test_planner_call_receives_constraints(self) -> None:
        """Constraints from the intent are passed through to the
        planner call as a tuple of value strings.
        """
        custom_plan = PlanExecutePlan()
        invocation = SimpleNamespace(output=custom_plan)
        invoke_mock = AsyncMock(return_value=invocation)
        call_stub = SimpleNamespace(invoke=invoke_mock)
        planner = PlanExecutePlanner(planner_call=call_stub)

        intent = IntentResult(
            goal=Goal(statement="Goal.", intent_type=IntentType.RESEARCH),
            constraints=(
                Constraint(kind=ConstraintKind.DEADLINE, value="by Friday"),
                Constraint(kind=ConstraintKind.BUDGET, value="under $5"),
            ),
        )
        await planner.plan(intent)

        await_args = invoke_mock.await_args
        assert await_args is not None
        assert await_args.kwargs["constraints"] == ("by Friday", "under $5")


# ---------------------------------------------------------------------------
# execute() — step dispatch
# ---------------------------------------------------------------------------


class TestExecuteHappyPath:
    @pytest.mark.asyncio
    async def test_execute_dispatches_each_step(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Plan everything.", IntentType.PLAN)
        plan = await planner.plan(intent)

        perceiver = _StubPerceiver(content="contextual-fact")
        actor = _StubActor(output="action-success")

        result = await planner.execute(plan, perceiver=perceiver, actor=actor)

        assert isinstance(result, PlanResult)
        # All 3 steps ran (perceive, act, respond).
        assert result.metadata["steps_executed"] == 3
        assert result.metadata["steps_skipped"] == 0
        assert result.metadata["plan_step_count"] == 3
        assert result.metadata["strategy"] == "vanilla"
        # Perceiver and actor both called once.
        assert len(perceiver.calls) == 1
        assert len(actor.calls) == 1
        # The respond step concatenated prior outputs.
        assert "contextual-fact" in result.text
        assert "action-success" in result.text

    @pytest.mark.asyncio
    async def test_execute_text_and_output_aliased(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Hello!", IntentType.RESPOND)
        plan = await planner.plan(intent)

        result = await planner.execute(plan)
        # text and output carry the same value so AgentLoop's
        # getattr chain works either way.
        assert result.text == result.output


class TestExecuteSkipsMissingDispatcher:
    @pytest.mark.asyncio
    async def test_perceive_step_skipped_when_perceiver_none(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Find facts.", IntentType.RESEARCH)
        plan = await planner.plan(intent)

        actor = _StubActor()
        result = await planner.execute(plan, perceiver=None, actor=actor)

        # perceive skipped, respond ran.
        assert result.metadata["steps_executed"] == 1
        assert result.metadata["steps_skipped"] == 1
        # Per-step records carry the skip reason.
        records = result.metadata["step_records"]
        perceive_record = next(r for r in records if r["kind"] == "perceive")
        assert perceive_record["skipped"] is True
        assert perceive_record["skip_reason"] == "no perceiver"

    @pytest.mark.asyncio
    async def test_act_step_skipped_when_actor_none(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Tool call.", IntentType.TOOL_USE)
        plan = await planner.plan(intent)

        result = await planner.execute(plan, perceiver=None, actor=None)

        # act skipped, respond ran.
        assert result.metadata["steps_executed"] == 1
        assert result.metadata["steps_skipped"] == 1
        records = result.metadata["step_records"]
        act_record = next(r for r in records if r["kind"] == "act")
        assert act_record["skipped"] is True
        assert act_record["skip_reason"] == "no actor"

    @pytest.mark.asyncio
    async def test_all_steps_skipped_when_no_dispatchers(self) -> None:
        """When every non-respond step is skipped, the plan still
        produces a respond-step output (falling back to the goal
        statement).
        """
        planner = PlanExecutePlanner()
        intent = _make_intent("Plan it.", IntentType.PLAN)
        plan = await planner.plan(intent)

        result = await planner.execute(plan, perceiver=None, actor=None)
        # Only the respond step ran.
        assert result.metadata["steps_executed"] == 1
        assert result.metadata["steps_skipped"] == 2
        # respond falls back to the goal statement when no prior outputs.
        assert result.text == "Plan it."


class TestExecuteResultShape:
    @pytest.mark.asyncio
    async def test_metadata_carries_full_step_count(self) -> None:
        planner = PlanExecutePlanner()
        intent = _make_intent("Hi.", IntentType.RESEARCH)
        plan = await planner.plan(intent)

        perceiver = _StubPerceiver()
        result = await planner.execute(plan, perceiver=perceiver)

        # 2 steps total in the plan.
        assert result.metadata["plan_step_count"] == 2
        # 2 records emitted.
        assert len(result.metadata["step_records"]) == 2

    @pytest.mark.asyncio
    async def test_empty_plan_short_circuits(self) -> None:
        """A :class:`PlanExecutePlan` with zero steps short-circuits
        without invoking :class:`LoopRunner` (whose
        ``max_iterations >= 1`` invariant would otherwise reject it).
        """
        planner = PlanExecutePlanner()
        empty = PlanExecutePlan(goal_statement="empty", steps=())
        result = await planner.execute(empty)
        assert result.text == ""
        assert result.metadata["steps_executed"] == 0
        assert result.metadata["plan_step_count"] == 0

    @pytest.mark.asyncio
    async def test_max_steps_caps_plan_length(self) -> None:
        """When the plan has more steps than ``max_steps``, only the
        first ``max_steps`` are executed.
        """
        planner = PlanExecutePlanner(max_steps=2)
        # Hand-construct a 3-step plan to exceed the cap.
        plan = PlanExecutePlan(
            goal_statement="cap test",
            steps=(
                PlanStep(step_id="s1", kind="respond", description="one"),
                PlanStep(step_id="s2", kind="respond", description="two"),
                PlanStep(step_id="s3", kind="respond", description="three"),
            ),
        )
        result = await planner.execute(plan)
        assert result.metadata["steps_executed"] == 2

    @pytest.mark.asyncio
    async def test_coerce_bare_plan_into_plan_execute_plan(self) -> None:
        """A loose :class:`Plan` carrying ``steps`` via
        ``extra='allow'`` is coerced into a :class:`PlanExecutePlan`
        before execution.
        """
        planner = PlanExecutePlanner()
        bare = Plan(
            pattern="plan_execute",
            metadata={
                "strategy": "vanilla",
                "goal_statement": "coerced",
                "steps": (),
            },
        )
        # The coercer should still execute (empty plan path).
        result = await planner.execute(bare)
        assert isinstance(result, PlanResult)


class TestPlannerProtocolCompliance:
    def test_implements_planner_protocol(self) -> None:
        """Runtime ``isinstance`` against the
        :class:`Planner` Protocol passes.
        """
        planner = PlanExecutePlanner()
        assert isinstance(planner, Planner)
