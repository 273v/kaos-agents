"""Unit tests for kaos_agents.types.budget + .runtime.escalation."""

from __future__ import annotations

import math

import pytest

from kaos_agents.runtime.escalation import (
    DefaultEscalationPolicy,
    EscalationAction,
    StepOutcome,
)
from kaos_agents.types.budget import UNLIMITED_BUDGET, CostBudget

# ---------------------------------------------------------------------------
# CostBudget invariants
# ---------------------------------------------------------------------------


class TestCostBudgetInvariants:
    def test_zero_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CostBudget(total_usd=0.0)

    def test_negative_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CostBudget(total_usd=-1.0)

    def test_negative_spent_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            CostBudget(total_usd=1.0, spent_usd=-0.1)

    def test_frozen(self) -> None:
        b = CostBudget(total_usd=1.0)
        with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
            b.spent_usd = 0.5  # ty: ignore[invalid-assignment]


class TestCostBudgetMath:
    def test_remaining_clamps_to_zero(self) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=2.0)
        assert b.remaining_usd == 0.0
        assert b.headroom_usd == 0.0  # alias

    def test_fraction_spent_normal(self) -> None:
        b = CostBudget(total_usd=10.0, spent_usd=2.5)
        assert b.fraction_spent == pytest.approx(0.25)

    def test_fraction_spent_overrun(self) -> None:
        b = CostBudget(total_usd=10.0, spent_usd=15.0)
        assert b.fraction_spent == pytest.approx(1.5)

    def test_fraction_spent_inf_budget_zero(self) -> None:
        # Unlimited budget always reads as 0% spent.
        assert UNLIMITED_BUDGET.fraction_spent == 0.0
        spent = UNLIMITED_BUDGET.spend(1000.0)
        assert spent.fraction_spent == 0.0

    def test_exceeded(self) -> None:
        assert not CostBudget(total_usd=1.0, spent_usd=0.5).exceeded
        assert CostBudget(total_usd=1.0, spent_usd=1.0).exceeded  # equality counts
        assert CostBudget(total_usd=1.0, spent_usd=2.0).exceeded

    def test_unlimited_never_exceeded(self) -> None:
        assert not UNLIMITED_BUDGET.exceeded
        spent = UNLIMITED_BUDGET.spend(1e9)
        assert not spent.exceeded


class TestCostBudgetSpend:
    def test_spend_returns_new_instance(self) -> None:
        original = CostBudget(total_usd=1.0, spent_usd=0.2)
        after = original.spend(0.3)
        # Original unchanged
        assert original.spent_usd == pytest.approx(0.2)
        # New has updated spend
        assert after.spent_usd == pytest.approx(0.5)
        assert after.total_usd == pytest.approx(1.0)
        # Identity check — different object
        assert after is not original

    def test_spend_negative_rejected(self) -> None:
        b = CostBudget(total_usd=1.0)
        with pytest.raises(ValueError, match="≥ 0"):
            b.spend(-0.1)

    def test_spend_zero_is_noop(self) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.3)
        assert b.spend(0.0).spent_usd == pytest.approx(0.3)

    def test_with_cap(self) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.5)
        wider = b.with_cap(10.0)
        assert wider.total_usd == 10.0
        assert wider.spent_usd == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# DefaultEscalationPolicy — the decision table
# ---------------------------------------------------------------------------


class TestDefaultEscalationPolicy:
    @pytest.fixture()
    def policy(self) -> DefaultEscalationPolicy:
        return DefaultEscalationPolicy()

    def test_initial_outcome_stays(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0)
        d = policy.choose(budget=b, outcome=StepOutcome.INITIAL, attempt=1)
        assert d.action == EscalationAction.STAY

    def test_success_stays(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.3)
        d = policy.choose(budget=b, outcome=StepOutcome.SUCCESS, attempt=1)
        assert d.action == EscalationAction.STAY

    def test_provider_error_stays(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.3)
        d = policy.choose(budget=b, outcome=StepOutcome.PROVIDER_ERROR, attempt=2)
        assert d.action == EscalationAction.STAY
        assert "transient" in d.reason or "retrying" in d.reason

    def test_refusal_upgrades_when_budget_available(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.2)
        d = policy.choose(budget=b, outcome=StepOutcome.REFUSAL, attempt=1)
        assert d.action == EscalationAction.UPGRADE

    def test_invariant_violation_upgrades(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.2)
        d = policy.choose(budget=b, outcome=StepOutcome.INVARIANT_VIOLATION, attempt=1)
        assert d.action == EscalationAction.UPGRADE

    def test_low_confidence_upgrades_below_50_pct(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.2)
        d = policy.choose(budget=b, outcome=StepOutcome.LOW_CONFIDENCE, attempt=1)
        assert d.action == EscalationAction.UPGRADE

    def test_low_confidence_stays_above_50_pct(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=0.6)
        d = policy.choose(budget=b, outcome=StepOutcome.LOW_CONFIDENCE, attempt=1)
        assert d.action == EscalationAction.STAY

    def test_high_water_mark_blocks_escalation(self, policy: DefaultEscalationPolicy) -> None:
        # At 80% spent, escalation is refused even on refusal.
        b = CostBudget(total_usd=1.0, spent_usd=0.8)
        d = policy.choose(budget=b, outcome=StepOutcome.REFUSAL, attempt=1)
        assert d.action == EscalationAction.STOP
        assert "high-water" in d.reason

    def test_exceeded_budget_stops(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0, spent_usd=1.1)
        d = policy.choose(budget=b, outcome=StepOutcome.SUCCESS, attempt=1)
        assert d.action == EscalationAction.STOP
        assert "budget exceeded" in d.reason

    def test_max_attempts_stops(self, policy: DefaultEscalationPolicy) -> None:
        b = CostBudget(total_usd=1.0)
        d = policy.choose(
            budget=b, outcome=StepOutcome.REFUSAL, attempt=4
        )  # max_attempts_per_step=3
        assert d.action == EscalationAction.STOP
        assert "max_attempts_per_step" in d.reason

    def test_custom_high_water_mark(self) -> None:
        # Constructed with a lower high-water mark
        strict = DefaultEscalationPolicy(high_water_mark=0.4)
        b = CostBudget(total_usd=1.0, spent_usd=0.5)
        d = strict.choose(budget=b, outcome=StepOutcome.REFUSAL, attempt=1)
        assert d.action == EscalationAction.STOP


class TestPolicyFrozen:
    def test_policy_is_frozen_dataclass(self) -> None:
        p = DefaultEscalationPolicy()
        with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
            p.high_water_mark = 0.99  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Sentinel: UNLIMITED_BUDGET
# ---------------------------------------------------------------------------


class TestUnlimitedBudget:
    def test_is_inf(self) -> None:
        assert math.isinf(UNLIMITED_BUDGET.total_usd)

    def test_never_exceeded(self) -> None:
        assert not UNLIMITED_BUDGET.exceeded

    def test_remaining_inf(self) -> None:
        assert math.isinf(UNLIMITED_BUDGET.remaining_usd)

    def test_spend_keeps_inf_cap(self) -> None:
        after = UNLIMITED_BUDGET.spend(1000.0)
        assert math.isinf(after.total_usd)
        assert after.spent_usd == pytest.approx(1000.0)
