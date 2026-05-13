"""Unit tests for kaos_agents.termination.degrade — DegradationPolicy."""

from __future__ import annotations

import pytest

from kaos_agents.termination.degrade import DegradationOutcome, DegradationPolicy


class TestDefaults:
    def test_min_partial_chars_default(self) -> None:
        policy = DegradationPolicy()
        assert policy.min_partial_chars == 32

    def test_below_min_chars_not_accepted(self) -> None:
        policy = DegradationPolicy(min_partial_chars=32)
        outcome = policy.evaluate(kind="budget_exceeded", partial_text="too short")
        assert outcome.accept is False
        assert "too short" in outcome.reason
        assert outcome.partial == ""

    def test_empty_partial_not_accepted(self) -> None:
        policy = DegradationPolicy()
        outcome = policy.evaluate(kind="budget_exceeded", partial_text="")
        assert outcome.accept is False


class TestBudgetExhaustion:
    def test_budget_exceeded_accepted_with_long_text(self) -> None:
        policy = DegradationPolicy(min_partial_chars=10, accept_on_budget_exhaustion=True)
        text = "x" * 50
        outcome = policy.evaluate(kind="budget_exceeded", partial_text=text)
        assert outcome.accept is True
        assert outcome.partial == text
        assert "budget" in outcome.reason

    def test_budget_exceeded_disabled(self) -> None:
        policy = DegradationPolicy(min_partial_chars=10, accept_on_budget_exhaustion=False)
        outcome = policy.evaluate(kind="budget_exceeded", partial_text="x" * 50)
        assert outcome.accept is False


class TestQualityFailure:
    def test_quality_failed_default_not_accepted(self) -> None:
        policy = DegradationPolicy(min_partial_chars=10)
        outcome = policy.evaluate(kind="quality_failed", partial_text="x" * 50)
        assert outcome.accept is False
        assert "not accepted" in outcome.reason

    def test_quality_failed_opt_in_accepted(self) -> None:
        policy = DegradationPolicy(min_partial_chars=10, accept_on_quality_failure=True)
        outcome = policy.evaluate(kind="quality_failed", partial_text="x" * 50)
        assert outcome.accept is True
        assert "quality" in outcome.reason


class TestUnknownKind:
    def test_unknown_kind_not_accepted(self) -> None:
        policy = DegradationPolicy(min_partial_chars=4)
        outcome = policy.evaluate(kind="some_other_kind", partial_text="long enough")
        assert outcome.accept is False
        assert "some_other_kind" in outcome.reason


class TestOutcomeShape:
    def test_outcome_is_dataclass(self) -> None:
        outcome = DegradationOutcome(accept=True, partial="x", reason="y")
        assert outcome.accept is True
        assert outcome.partial == "x"
        assert outcome.reason == "y"

    def test_outcome_defaults(self) -> None:
        outcome = DegradationOutcome(accept=False)
        assert outcome.partial == ""
        assert outcome.reason == ""


class TestCustomThresholds:
    def test_custom_min_chars(self) -> None:
        policy = DegradationPolicy(min_partial_chars=100)
        outcome = policy.evaluate(kind="budget_exceeded", partial_text="x" * 99)
        assert outcome.accept is False
        outcome2 = policy.evaluate(kind="budget_exceeded", partial_text="x" * 101)
        assert outcome2.accept is True

    @pytest.mark.parametrize(
        "kind",
        ["budget_exceeded", "quality_failed", "loop_detected", "failure"],
    )
    def test_min_chars_applies_universally(self, kind: str) -> None:
        policy = DegradationPolicy(
            min_partial_chars=50,
            accept_on_budget_exhaustion=True,
            accept_on_quality_failure=True,
        )
        outcome = policy.evaluate(kind=kind, partial_text="x" * 10)
        assert outcome.accept is False
