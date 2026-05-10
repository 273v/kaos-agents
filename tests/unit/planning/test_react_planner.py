"""Tests for :class:`~kaos_agents.planning.react_planner.ReActPlanner`.

Phase 3.A — wraps kaos-llm-core :class:`ReAct`. The planner does the
*projection* work (Plan in, PlanResult out, usage rolled up). The
inner ReAct loop is stubbed so these tests stay offline; live
behaviour belongs in the integration tier.

What we cover:

* Construction with default kwargs and with explicit ``model`` /
  ``max_iterations`` / ``instructions`` / ``signature`` /
  ``react_program`` / ``tools``.
* :meth:`ReActPlanner.plan` returns a :class:`ReActPlan` with
  ``pattern == "react"`` and ``goal_statement == intent.goal.statement``.
* :meth:`ReActPlanner.execute` invokes the underlying ReAct and
  projects the result into a :class:`PlanResult` with both ``text`` and
  ``output`` populated.
* :meth:`ReActPlanner.execute` rolls the kaos-llm-core
  :class:`Invocation` usage into :class:`PlanResult.usage` via
  :meth:`InvocationUsage.from_llm_usage`.
* :meth:`ReActPlanner._resolve_tools` precedence: explicit > perceiver
  + actor > empty.
* :meth:`ReActPlanner.execute` reuses a pre-constructed
  :class:`ReAct` when ``react_program=`` is provided.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kaos_agents.config import AgentPattern
from kaos_agents.intent.types import Goal, IntentResult
from kaos_agents.planning.planner import Plan, PlanResult
from kaos_agents.planning.react_planner import ReActPlan, ReActPlanner
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
        pattern=AgentPattern.CHAT,
        confidence=0.9,
        raw_input=statement,
    )


def _stub_invocation(
    answer: str = "42",
    *,
    iterations: int = 2,
    input_tokens: int = 10,
    output_tokens: int = 20,
    total_tokens: int = 30,
    cost_usd: float = 0.001,
) -> SimpleNamespace:
    """Build a synthetic ``Invocation``-like object for stubbing
    :meth:`ReAct.invoke`. The real :class:`Invocation` exposes
    ``output`` (a :class:`ReActResult`) and ``usage`` (a
    :class:`TokenUsage`); :class:`SimpleNamespace` is duck-typed
    enough for both projection helpers
    (:meth:`ReActPlanner._extract_text`,
    :meth:`ReActPlanner._extract_usage`,
    :meth:`ReActPlanner._extract_iterations`).
    """
    react_result = SimpleNamespace(
        outputs={"answer": answer},
        iterations_used=iterations,
        stop_reason="TERMINATED",
    )
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
    return SimpleNamespace(output=react_result, usage=usage)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        planner = ReActPlanner()
        assert planner._model == "anthropic:claude-haiku-4-5"
        assert planner._max_iterations == 10
        assert "helpful assistant" in planner._instructions
        assert planner._tools is None
        assert planner._react is None

    def test_explicit_model_and_iterations(self) -> None:
        planner = ReActPlanner(
            model="openai:gpt-5.4-nano",
            max_iterations=5,
            instructions="Be terse.",
        )
        assert planner._model == "openai:gpt-5.4-nano"
        assert planner._max_iterations == 5
        assert planner._instructions == "Be terse."

    def test_explicit_tools_stashed(self) -> None:
        sentinel_tool = object()
        planner = ReActPlanner(tools=(sentinel_tool,))
        assert planner._tools == (sentinel_tool,)

    def test_pre_constructed_react_program_stashed(self) -> None:
        # Use a sentinel object — _build_react isn't called when
        # react_program is provided, so the type doesn't matter here.
        sentinel = object()
        planner = ReActPlanner(react_program=sentinel)  # ty: ignore[invalid-argument-type]
        assert planner._react is sentinel


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


class TestPlan:
    @pytest.mark.asyncio
    async def test_plan_returns_react_plan(self) -> None:
        planner = ReActPlanner()
        intent = _make_intent("Find party names.")
        plan = await planner.plan(intent)
        assert isinstance(plan, ReActPlan)
        assert plan.pattern == "react"
        assert plan.goal_statement == "Find party names."
        assert plan.tool_hints == ()
        assert plan.metadata == {"intent_type": IntentType.RESEARCH.value}

    @pytest.mark.asyncio
    async def test_plan_ignores_memory_in_phase_3a(self) -> None:
        """Phase 3.A: ``memory=`` is accepted for protocol compliance
        but not consulted (Phase 4 wires memory-aware planning)."""
        planner = ReActPlanner()
        intent = _make_intent("Hello.")
        plan_a = await planner.plan(intent, memory=None)
        plan_b = await planner.plan(intent, memory=object())
        assert plan_a == plan_b


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_invokes_pre_constructed_react(self) -> None:
        """A pre-built :class:`ReAct` passed via ``react_program=`` is
        the same instance used inside :meth:`execute` — verified by
        making :meth:`ReAct.invoke` an :class:`AsyncMock`."""
        invoke_mock = AsyncMock(return_value=_stub_invocation(answer="42"))
        react_stub = SimpleNamespace(invoke=invoke_mock)

        planner = ReActPlanner(react_program=react_stub)  # ty: ignore[invalid-argument-type]
        intent = _make_intent("What is 6 times 7?")
        plan = await planner.plan(intent)

        result = await planner.execute(plan)

        invoke_mock.assert_awaited_once()
        # The goal statement is passed under the ``task`` keyword.
        await_args = invoke_mock.await_args
        assert await_args is not None
        assert await_args.kwargs == {"task": "What is 6 times 7?"}

        assert isinstance(result, PlanResult)
        assert result.text == "42"
        assert result.output == "42"  # alias

    @pytest.mark.asyncio
    async def test_execute_text_and_output_aliased(self) -> None:
        """:class:`PlanResult.text` and :class:`PlanResult.output`
        carry the same value so the AgentLoop's ``getattr`` chain
        succeeds either way."""
        invoke_mock = AsyncMock(return_value=_stub_invocation(answer="hello"))
        react_stub = SimpleNamespace(invoke=invoke_mock)
        planner = ReActPlanner(react_program=react_stub)  # ty: ignore[invalid-argument-type]
        plan = await planner.plan(_make_intent("Greet."))

        result = await planner.execute(plan)
        assert result.text == result.output == "hello"

    @pytest.mark.asyncio
    async def test_execute_rolls_up_usage(self) -> None:
        """Per-invocation usage flows into :class:`PlanResult.usage`
        via :meth:`InvocationUsage.from_llm_usage`."""
        invoke_mock = AsyncMock(
            return_value=_stub_invocation(
                input_tokens=11, output_tokens=22, total_tokens=33, cost_usd=0.05
            )
        )
        react_stub = SimpleNamespace(invoke=invoke_mock)
        planner = ReActPlanner(react_program=react_stub)  # ty: ignore[invalid-argument-type]
        plan = await planner.plan(_make_intent("Compute."))

        result = await planner.execute(plan)
        assert result.usage.input_tokens == 11
        assert result.usage.output_tokens == 22
        assert result.usage.total_tokens == 33
        assert result.usage.cost_usd == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_execute_metadata_carries_iterations(self) -> None:
        invoke_mock = AsyncMock(return_value=_stub_invocation(answer="ok", iterations=4))
        react_stub = SimpleNamespace(invoke=invoke_mock)
        planner = ReActPlanner(react_program=react_stub)  # ty: ignore[invalid-argument-type]
        plan = await planner.plan(_make_intent("Step."))

        result = await planner.execute(plan)
        assert result.metadata == {"react_iterations": 4}

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_metadata_goal_for_bare_plan(self) -> None:
        """When called with a bare :class:`Plan` (not a
        :class:`ReActPlan`), :meth:`execute` reads the goal text
        from ``plan.metadata['goal_statement']`` if present."""
        invoke_mock = AsyncMock(return_value=_stub_invocation(answer="ok"))
        react_stub = SimpleNamespace(invoke=invoke_mock)
        planner = ReActPlanner(react_program=react_stub)  # ty: ignore[invalid-argument-type]

        bare_plan = Plan(
            pattern="react",
            metadata={"goal_statement": "From metadata."},
        )
        await planner.execute(bare_plan)

        await_args = invoke_mock.await_args
        assert await_args is not None
        assert await_args.kwargs == {"task": "From metadata."}

    @pytest.mark.asyncio
    async def test_execute_fallback_default_goal(self) -> None:
        """When neither :class:`ReActPlan` nor metadata carries a goal,
        :meth:`execute` invokes ReAct with a sensible default.
        """
        invoke_mock = AsyncMock(return_value=_stub_invocation(answer="ok"))
        react_stub = SimpleNamespace(invoke=invoke_mock)
        planner = ReActPlanner(react_program=react_stub)  # ty: ignore[invalid-argument-type]

        bare_plan = Plan(pattern="react")  # no metadata, no goal_statement
        await planner.execute(bare_plan)

        await_args = invoke_mock.await_args
        assert await_args is not None
        assert await_args.kwargs == {"task": "Respond to the user's request."}


# ---------------------------------------------------------------------------
# _resolve_tools precedence
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_explicit_tools_win(self) -> None:
        """An explicit ``tools=`` kwarg overrides the perceiver/actor
        fan-out unconditionally."""
        sentinel = object()
        planner = ReActPlanner(tools=(sentinel,))
        perceiver = SimpleNamespace(tools=("p_tool",))
        actor = SimpleNamespace(tools=("a_tool",))

        resolved = planner._resolve_tools(perceiver, actor)
        assert resolved == (sentinel,)

    def test_perceiver_and_actor_union_when_no_explicit_tools(self) -> None:
        planner = ReActPlanner()
        perceiver = SimpleNamespace(tools=("p1", "p2"))
        actor = SimpleNamespace(tools=("a1",))

        resolved = planner._resolve_tools(perceiver, actor)
        # Order: perceiver tools first, then actor tools.
        assert resolved == ("p1", "p2", "a1")

    def test_only_perceiver_supplies_tools(self) -> None:
        planner = ReActPlanner()
        perceiver = SimpleNamespace(tools=("p1",))
        resolved = planner._resolve_tools(perceiver, None)
        assert resolved == ("p1",)

    def test_only_actor_supplies_tools(self) -> None:
        planner = ReActPlanner()
        actor = SimpleNamespace(tools=("a1",))
        resolved = planner._resolve_tools(None, actor)
        assert resolved == ("a1",)

    def test_no_perceiver_no_actor_no_tools(self) -> None:
        planner = ReActPlanner()
        resolved = planner._resolve_tools(None, None)
        assert resolved == ()

    def test_perceiver_without_tools_attribute_is_safe(self) -> None:
        """A perceiver/actor without a ``.tools`` attribute degrades to
        an empty list — Phase 3.A duck-typed contract."""
        planner = ReActPlanner()
        perceiver = SimpleNamespace()  # no .tools
        actor = SimpleNamespace()  # no .tools
        resolved = planner._resolve_tools(perceiver, actor)
        assert resolved == ()


# ---------------------------------------------------------------------------
# Stubbing the global ReAct constructor (path used when react_program=None)
# ---------------------------------------------------------------------------


class TestBuildReact:
    def test_build_react_uses_default_signature(self) -> None:
        """When no Signature is supplied, :class:`ReActPlanner` uses
        :class:`_DefaultReActSignature` as the input/output contract.
        """
        from kaos_agents.planning.react_planner import _DefaultReActSignature

        planner = ReActPlanner()
        assert planner._signature is _DefaultReActSignature

    @pytest.mark.asyncio
    async def test_execute_path_with_no_pre_constructed_react(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``react_program`` is None, :meth:`execute` calls
        :meth:`ReActPlanner._build_react` to construct a fresh
        :class:`ReAct`. We monkey-patch :meth:`_build_react` to return
        a stub so we never actually hit the kaos-llm-core constructor
        (which validates tools-list shape and other things outside
        Phase 3.A's scope).
        """
        invoke_mock = AsyncMock(return_value=_stub_invocation(answer="ok"))
        react_stub = SimpleNamespace(invoke=invoke_mock)

        planner = ReActPlanner()

        def _stub_build(self: ReActPlanner, tools: tuple[Any, ...]) -> Any:
            return react_stub

        monkeypatch.setattr(ReActPlanner, "_build_react", _stub_build)

        plan = await planner.plan(_make_intent("Hi."))
        result = await planner.execute(plan)
        assert result.text == "ok"
        invoke_mock.assert_awaited_once_with(task="Hi.")
