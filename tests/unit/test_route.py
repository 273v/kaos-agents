"""Tests for the Route planning primitive."""

from __future__ import annotations

import time

from kaos_agents.planning.route import route
from kaos_agents.types.plan import (
    Decision,
    EvalMode,
    Judgment,
    PlanBudget,
)


def _judgment(matched: bool = True, confidence: float = 0.8) -> Judgment:
    return Judgment(
        matched=matched,
        confidence=confidence,
        reasoning="test",
        mode=EvalMode.STRUCTURAL,
    )


class TestRouteContinue:
    def test_success_high_confidence(self):
        result = route(_judgment(matched=True, confidence=0.9), PlanBudget())
        assert result.decision == Decision.CONTINUE

    def test_success_at_threshold(self):
        result = route(
            _judgment(matched=True, confidence=0.5),
            PlanBudget(),
            confidence_threshold=0.5,
        )
        assert result.decision == Decision.CONTINUE


class TestRouteReplan:
    def test_failure_triggers_replan(self):
        result = route(_judgment(matched=False, confidence=0.8), PlanBudget())
        assert result.decision == Decision.REPLAN

    def test_low_confidence_triggers_replan(self):
        result = route(
            _judgment(matched=True, confidence=0.4),
            PlanBudget(),
            confidence_threshold=0.5,
        )
        assert result.decision == Decision.REPLAN


class TestRouteDeepen:
    def test_very_low_confidence_triggers_deepen(self):
        result = route(
            _judgment(matched=True, confidence=0.2),
            PlanBudget(),
            confidence_threshold=0.5,
            deepen_threshold=0.3,
        )
        assert result.decision == Decision.DEEPEN


class TestRouteBudget:
    def test_cost_exceeded(self):
        budget = PlanBudget(max_cost_usd=0.01)
        budget.cost_usd = 0.02
        result = route(_judgment(), budget)
        assert result.decision == Decision.STOP_BUDGET

    def test_tokens_exceeded(self):
        budget = PlanBudget(max_tokens=100)
        budget.tokens_used = 200
        result = route(_judgment(), budget)
        assert result.decision == Decision.STOP_BUDGET

    def test_steps_exceeded(self):
        budget = PlanBudget(max_steps=5)
        budget.steps_executed = 5
        result = route(_judgment(), budget)
        assert result.decision == Decision.STOP_BUDGET

    def test_wall_clock_exceeded(self):
        budget = PlanBudget(max_wall_clock_seconds=0.01)
        budget.start_time = time.time() - 1.0
        result = route(_judgment(), budget)
        assert result.decision == Decision.STOP_BUDGET

    def test_budget_checked_before_judgment(self):
        """Budget takes priority over judgment."""
        budget = PlanBudget(max_steps=1)
        budget.steps_executed = 1
        # Even with a great judgment, budget stops us
        result = route(_judgment(matched=True, confidence=1.0), budget)
        assert result.decision == Decision.STOP_BUDGET


class TestRouteCircuitBreaker:
    def test_max_replans_stops(self):
        budget = PlanBudget(max_replans=3)
        result = route(_judgment(matched=False), budget, replan_count=3)
        assert result.decision == Decision.STOP_FAILURE
        assert "Max replans" in result.reason

    def test_under_max_replans_allows_replan(self):
        budget = PlanBudget(max_replans=3)
        result = route(_judgment(matched=False), budget, replan_count=2)
        assert result.decision == Decision.REPLAN


class TestRouteStepId:
    def test_step_id_propagated(self):
        result = route(_judgment(), PlanBudget(), step_id="s1")
        assert result.step_id == "s1"


class TestRouteDecisionPriority:
    def test_budget_before_circuit_breaker(self):
        budget = PlanBudget(max_steps=1, max_replans=3)
        budget.steps_executed = 1
        result = route(_judgment(matched=False), budget, replan_count=3)
        assert result.decision == Decision.STOP_BUDGET  # Not STOP_FAILURE

    def test_circuit_breaker_before_replan(self):
        budget = PlanBudget(max_replans=2)
        result = route(_judgment(matched=False), budget, replan_count=2)
        assert result.decision == Decision.STOP_FAILURE  # Not REPLAN
