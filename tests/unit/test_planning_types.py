"""Tests for planning type system."""

from __future__ import annotations

import time

import pytest

from kaos_agents.types.plan import (
    ComposeResult,
    Decision,
    EvalMode,
    Judgment,
    PlanBudget,
    PrimitiveTrace,
    RouteResult,
    Step,
    StepType,
    StopReason,
)


class TestStep:
    def test_frozen(self):
        step = Step(id="s1", step_type=StepType.TOOL, description="test")
        with pytest.raises(AttributeError):
            step.id = "s2"  # ty: ignore[invalid-assignment]

    def test_tool_step(self):
        step = Step(
            id="s1",
            step_type=StepType.TOOL,
            description="Search eCFR",
            tool_name="kaos-source-ecfr-search",
            input_spec={"query": "Clean Air Act"},
            expected_output="List of CFR sections",
            depends_on=("s0",),
        )
        assert step.step_type == StepType.TOOL
        assert step.tool_name == "kaos-source-ecfr-search"
        assert step.depends_on == ("s0",)

    def test_llm_step(self):
        step = Step(id="s2", step_type=StepType.LLM, description="Summarize findings")
        assert step.tool_name is None

    def test_subplan_step(self):
        step = Step(id="s3", step_type=StepType.SUBPLAN, description="Research subtopic")
        assert step.step_type == StepType.SUBPLAN

    def test_defaults(self):
        step = Step(id="s1", step_type=StepType.TOOL, description="test")
        assert step.input_spec == {}
        assert step.expected_output == ""
        assert step.depends_on == ()
        assert step.estimated_latency_ms == 0


class TestJudgment:
    def test_structural_judgment(self):
        j = Judgment(
            matched=True, confidence=1.0, reasoning="Tool exists", mode=EvalMode.STRUCTURAL
        )
        assert j.matched
        assert j.mode == EvalMode.STRUCTURAL

    def test_semantic_judgment_with_facts(self):
        j = Judgment(
            matched=True,
            confidence=0.85,
            reasoning="Result addresses the question",
            mode=EvalMode.SEMANTIC,
            new_facts=("CFR 40.60.4 applies",),
            surprises=("Regulation was recently amended",),
        )
        assert len(j.new_facts) == 1
        assert len(j.surprises) == 1

    def test_frozen(self):
        j = Judgment(matched=True, confidence=0.9, reasoning="test", mode=EvalMode.STRUCTURAL)
        with pytest.raises(AttributeError):
            j.confidence = 0.5  # ty: ignore[invalid-assignment]


class TestDecision:
    def test_all_decisions(self):
        for d in Decision:
            assert isinstance(d.value, str)

    def test_route_result(self):
        rr = RouteResult(decision=Decision.CONTINUE, reason="Step succeeded")
        assert rr.decision == Decision.CONTINUE


class TestPlanBudget:
    def test_initial_state(self):
        budget = PlanBudget()
        assert budget.should_stop() is None
        assert budget.steps_executed == 0
        assert budget.cost_usd == 0.0

    def test_max_cost(self):
        budget = PlanBudget(max_cost_usd=0.01)
        budget.record_step(cost_usd=0.005)
        assert budget.should_stop() is None
        budget.record_step(cost_usd=0.006)
        assert budget.should_stop() == StopReason.MAX_COST

    def test_max_tokens(self):
        budget = PlanBudget(max_tokens=100)
        budget.record_step(tokens=90)
        assert budget.should_stop() is None
        budget.record_step(tokens=20)
        assert budget.should_stop() == StopReason.MAX_TOKENS

    def test_max_steps(self):
        budget = PlanBudget(max_steps=3)
        for _ in range(3):
            budget.record_step()
        assert budget.should_stop() == StopReason.MAX_STEPS

    def test_max_replans(self):
        budget = PlanBudget(max_replans=2)
        budget.record_replan()
        assert budget.should_stop() is None
        budget.record_replan()
        assert budget.should_stop() == StopReason.MAX_REPLANS

    def test_max_wall_clock(self):
        budget = PlanBudget(max_wall_clock_seconds=0.01)
        budget.start_time = time.time() - 1.0  # 1 second ago
        assert budget.should_stop() == StopReason.MAX_WALL_CLOCK

    def test_mutable(self):
        """PlanBudget is intentionally mutable."""
        budget = PlanBudget()
        budget.cost_usd = 0.5
        assert budget.cost_usd == 0.5


class TestPlanBudgetTrialIntegration:
    """Phase 5.E — PlanBudget.record_step publishes to active TrialRunner.

    Mirrors the Phase 5.A CostTrackingHook integration: when a plan-execute
    agent runs inside an optimizer trial, each plan step's cost+tokens
    flow into the trial accumulator. Without this, plan steps were
    invisible to the optimizer's budget tracking.
    """

    def test_publishes_cost_and_tokens_to_active_trial(self) -> None:
        from kaos_llm_core.optimization.trial_runner import TrialRunner

        runner = TrialRunner()
        with runner.trial("plan_test") as trial:
            budget = PlanBudget()
            budget.record_step(cost_usd=0.05, tokens=200)
            budget.record_step(cost_usd=0.03, tokens=150)

        # The trial accumulator received both publishes.
        assert trial.cost_usd == pytest.approx(0.08)
        assert trial.total_tokens == 350
        assert trial.n_invocations == 2

        # Local counters are still accurate.
        assert budget.cost_usd == pytest.approx(0.08)
        assert budget.tokens_used == 350
        assert budget.steps_executed == 2

    def test_no_publish_outside_trial_scope(self) -> None:
        """Outside a trial scope, record_step still updates locals
        and silently skips the publish."""
        from kaos_llm_core.optimization.trial_runner import current_trial

        assert current_trial() is None
        budget = PlanBudget()
        budget.record_step(cost_usd=0.10, tokens=500)
        # Local counters intact.
        assert budget.cost_usd == pytest.approx(0.10)
        assert budget.tokens_used == 500
        # Trial scope still empty.
        assert current_trial() is None

    def test_zero_cost_step_skips_publish(self) -> None:
        """Steps with zero cost AND zero tokens skip the import + publish
        path entirely — keeps the no-op fast path zero-overhead."""
        from kaos_llm_core.optimization.trial_runner import TrialRunner

        runner = TrialRunner()
        with runner.trial("plan_zero") as trial:
            budget = PlanBudget()
            # Pure step counter increment — no LLM cost.
            budget.record_step()

        # Trial received nothing (the publish was skipped).
        assert trial.cost_usd == 0.0
        assert trial.total_tokens == 0
        assert trial.n_invocations == 0
        # But local step counter still advanced.
        assert budget.steps_executed == 1


class TestPrimitiveTrace:
    def test_frozen(self):
        trace = PrimitiveTrace(primitive="evaluate", step_id="s1", wall_clock_ms=15.5)
        with pytest.raises(AttributeError):
            trace.wall_clock_ms = 20.0  # ty: ignore[invalid-assignment]

    def test_defaults(self):
        trace = PrimitiveTrace(primitive="route")
        assert trace.step_id is None
        assert trace.wall_clock_ms == 0.0
        assert trace.success is True


class TestComposeResult:
    def test_frozen(self):
        result = ComposeResult(plan_json="{}")
        assert result.stop_reason == StopReason.SUCCESS
        assert result.total_cost_usd == 0.0

    def test_with_traces(self):
        traces = (
            PrimitiveTrace(primitive="act", step_id="s1", wall_clock_ms=500),
            PrimitiveTrace(primitive="evaluate", step_id="s1", wall_clock_ms=10),
        )
        result = ComposeResult(
            plan_json="{}",
            traces=traces,
            steps_executed=1,
            total_cost_usd=0.003,
        )
        assert len(result.traces) == 2
        assert result.steps_executed == 1
