"""Tests for the Planner Protocol value types and runtime checking.

Phase 3.A — :class:`Plan`, :class:`PlanResult`, and the
:class:`Planner` Protocol.

What we cover:

* :class:`Plan` / :class:`PlanResult` construction defaults
* JSON round-trip via ``model_dump_json`` / ``model_validate_json``
* ``extra="allow"`` lets subclasses (and ad-hoc Plan instances) carry
  arbitrary metadata-shaped fields
* ``frozen=True`` rejects post-construction mutation
* :class:`Planner` Protocol via ``runtime_checkable`` —
  classes that implement both methods pass ``isinstance``; those
  missing one or both fail.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from kaos_agents.intent.types import IntentResult
from kaos_agents.planning.planner import Plan, Planner, PlanResult
from kaos_agents.types.tool_call import ToolExecution
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage


class TestPlan:
    def test_construction_defaults(self) -> None:
        p = Plan(pattern="react")
        assert p.pattern == "react"
        assert p.metadata == {}

    def test_metadata_is_carried(self) -> None:
        p = Plan(pattern="plan_execute", metadata={"k": "v", "n": 7})
        assert p.metadata == {"k": "v", "n": 7}

    def test_json_round_trip(self) -> None:
        p = Plan(pattern="react", metadata={"a": 1})
        revived = Plan.model_validate_json(p.model_dump_json())
        assert revived.pattern == p.pattern
        assert revived.metadata == p.metadata

    def test_frozen_mutation_rejected(self) -> None:
        p = Plan(pattern="react")
        with pytest.raises(ValidationError):
            p.pattern = "plan_execute"  # type: ignore[misc]

    def test_extra_allow_accepts_arbitrary_fields(self) -> None:
        """``extra='allow'`` is required so subclasses (e.g.
        :class:`ReActPlan`) can ship typed extras *and* so callers can
        construct a Plan with ad-hoc fields without subclassing first.
        """
        p = Plan.model_validate(
            {
                "pattern": "react",
                "metadata": {},
                "goal_statement": "Summarize the contract.",
                "tool_hints": ["read", "search"],
            }
        )
        # extra="allow" stores unknown fields on the model.
        # ``getattr`` keeps ``ty check`` happy here — extras are dynamic
        # by construction and don't appear on the static class.
        assert getattr(p, "goal_statement") == "Summarize the contract."  # noqa: B009
        assert getattr(p, "tool_hints") == ["read", "search"]  # noqa: B009


class TestPlanResult:
    def test_construction_defaults(self) -> None:
        r = PlanResult()
        assert r.text == ""
        assert r.output == ""
        assert r.tool_executions == ()
        assert r.usage == ZERO_USAGE
        assert r.metadata == {}

    def test_text_and_output_aliased_by_caller(self) -> None:
        """We populate both ``text`` and ``output`` to the same value
        so Phase 2.B AgentLoop's ``getattr(..., "text") or
        getattr(..., "output")`` lookup works either way."""
        r = PlanResult(text="hi", output="hi")
        assert r.text == r.output == "hi"

    def test_full_construction(self) -> None:
        usage = InvocationUsage(input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.01)
        tool_exec = ToolExecution(tool_name="search", call_id="c1")
        r = PlanResult(
            text="answer",
            output="answer",
            tool_executions=(tool_exec,),
            usage=usage,
            metadata={"react_iterations": 3},
        )
        assert r.text == "answer"
        assert r.tool_executions == (tool_exec,)
        assert r.usage == usage
        assert r.metadata == {"react_iterations": 3}

    def test_json_round_trip_minimal(self) -> None:
        """Round-trip the no-arg/default PlanResult — exercises the
        JSON path without involving the (non-pydantic) ToolExecution
        and InvocationUsage value types.
        """
        r = PlanResult(text="hi", output="hi", metadata={"k": "v"})
        revived = PlanResult.model_validate_json(r.model_dump_json())
        assert revived.text == "hi"
        assert revived.output == "hi"
        assert revived.metadata == {"k": "v"}

    def test_frozen_mutation_rejected(self) -> None:
        r = PlanResult(text="hi", output="hi")
        with pytest.raises(ValidationError):
            r.text = "nope"  # type: ignore[misc]


class TestPlannerProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        """A class that implements both ``plan()`` and ``execute()`` is
        an instance of :class:`Planner` at runtime."""

        class Good:
            async def plan(self, intent: IntentResult, memory: Any | None = None) -> Plan:
                return Plan(pattern="react")

            async def execute(
                self,
                plan: Plan,
                *,
                perceiver: Any | None = None,
                actor: Any | None = None,
            ) -> PlanResult:
                return PlanResult(text="hi", output="hi")

        assert isinstance(Good(), Planner)

    def test_missing_both_methods_fails(self) -> None:
        class Bad:
            pass

        assert not isinstance(Bad(), Planner)

    def test_missing_execute_fails(self) -> None:
        class HalfBad:
            async def plan(self, intent: IntentResult, memory: Any | None = None) -> Plan:
                return Plan(pattern="react")

        # ``runtime_checkable`` Protocol checks attribute presence; a
        # class missing ``execute`` is NOT an instance.
        assert not isinstance(HalfBad(), Planner)

    def test_missing_plan_fails(self) -> None:
        class HalfBad2:
            async def execute(
                self,
                plan: Plan,
                *,
                perceiver: Any | None = None,
                actor: Any | None = None,
            ) -> PlanResult:
                return PlanResult()

        assert not isinstance(HalfBad2(), Planner)
