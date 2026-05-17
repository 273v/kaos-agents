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
    def test_confident_failure_triggers_replan(self):
        # matched=False AND confidence >= threshold (default 0.5) →
        # judge is confident the tool's output is wrong, REPLAN.
        result = route(_judgment(matched=False, confidence=0.8), PlanBudget())
        assert result.decision == Decision.REPLAN

    def test_low_confidence_triggers_replan(self):
        # matched=True AND confidence < threshold → REPLAN
        # (a different branch — the judge says the result IS satisfactory
        # but only weakly; replan to seek a stronger verdict).
        result = route(
            _judgment(matched=True, confidence=0.4),
            PlanBudget(),
            confidence_threshold=0.5,
        )
        assert result.decision == Decision.REPLAN


class TestRouteSoftMissContinues:
    """Pre-0.1.0a9, ``matched=False`` fired REPLAN unconditionally.

    0.1.0a9 splits the branch by judge confidence: a low-confidence
    rejection ("doesn't match expected but I'm not sure") falls
    through to CONTINUE so the plan keeps going and the
    plan-execute synthesiser can surface the partial result.

    Closes the R1-REAL v2 matrix Tests 3 + 7 regression where 3-5
    successful FR/EDGAR tool calls disappeared because the LLM judge
    wasn't crisply satisfied with the JSON payload.
    """

    def test_unmatched_with_low_confidence_continues(self):
        # matched=False, confidence (0.3) < threshold (0.5) →
        # CONTINUE (judge is uncertain).
        result = route(
            _judgment(matched=False, confidence=0.3),
            PlanBudget(),
            confidence_threshold=0.5,
        )
        assert result.decision == Decision.CONTINUE
        assert "low confidence" in result.reason.lower()

    def test_unmatched_at_threshold_replans(self):
        # Boundary: matched=False, confidence (0.5) == threshold (0.5)
        # → REPLAN (>= is the cutoff).
        result = route(
            _judgment(matched=False, confidence=0.5),
            PlanBudget(),
            confidence_threshold=0.5,
        )
        assert result.decision == Decision.REPLAN

    def test_unmatched_high_confidence_still_replans(self):
        # matched=False, confidence=0.9 → judge is confident, REPLAN.
        # Preserves the pre-0.1.0a9 behavior for the case where the
        # judge actually has evidence the tool output is wrong.
        result = route(
            _judgment(matched=False, confidence=0.9),
            PlanBudget(),
            confidence_threshold=0.5,
        )
        assert result.decision == Decision.REPLAN


class TestRouteLowConfidence:
    """``Decision.DEEPEN`` was removed in kaos-agents 0.1.0a9 — the
    branch collapsed to ``Decision.REPLAN`` inside ``compose`` (see
    pre-0.1.0a9 ``compose.py``: ``if decision in (REPLAN, DEEPEN)``),
    so the very-low-confidence path now flows through the REPLAN
    handler with no behavior change for callers. A future ADaPT-style
    implementation that substep-decomposes the failed step (via
    ``PlanGraph.insert_subplan``) can re-introduce DEEPEN with
    distinct semantics."""

    def test_very_low_confidence_now_triggers_replan(self):
        # Pre-0.1.0a9: ``confidence=0.2`` with ``deepen_threshold=0.3``
        # would return Decision.DEEPEN. The DEEPEN branch is gone; the
        # confidence_threshold check below now catches the same case.
        result = route(
            _judgment(matched=True, confidence=0.2),
            PlanBudget(),
            confidence_threshold=0.5,
        )
        assert result.decision == Decision.REPLAN


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
