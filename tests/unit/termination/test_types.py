"""Unit tests for kaos_agents.termination.types — Decision + SuccessCriteria.

Covers construction defaults, frozen-ness, JSON round-trip, the
DecisionKind enum surface, and SuccessCriteria.from_intent's
duck-typed extraction from an IntentResult-shaped object.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kaos_agents.termination.types import Decision, DecisionKind, SuccessCriteria


class TestDecisionKind:
    """The 8-value enum is the discriminator the AgentLoop pattern-matches on."""

    def test_eight_members_present(self) -> None:
        names = {member.value for member in DecisionKind}
        assert names == {
            "complete",
            "incomplete",
            "budget_exceeded",
            "quality_failed",
            "failure",
            "loop_detected",
            "degraded",
            "escalate",
        }

    def test_string_enum_compares_to_str(self) -> None:
        assert DecisionKind.COMPLETE == "complete"
        assert DecisionKind.LOOP_DETECTED.value == "loop_detected"


class TestDecisionConstruction:
    def test_minimal_construction(self) -> None:
        decision = Decision(kind=DecisionKind.COMPLETE)
        assert decision.kind == DecisionKind.COMPLETE
        assert decision.is_complete is False
        assert decision.allows_replan is False
        assert decision.should_escalate is False
        assert decision.feedback == ""
        assert decision.partial_result is None
        assert decision.confidence == pytest.approx(1.0)
        assert decision.metadata == {}

    def test_full_construction(self) -> None:
        decision = Decision(
            kind=DecisionKind.DEGRADED,
            is_complete=True,
            feedback="budget exhausted",
            partial_result="here is what we have so far ...",
            confidence=0.4,
            metadata={"source": "test"},
        )
        assert decision.kind == DecisionKind.DEGRADED
        assert decision.is_complete is True
        assert decision.partial_result is not None
        assert decision.confidence == pytest.approx(0.4)
        assert decision.metadata["source"] == "test"

    def test_frozen(self) -> None:
        decision = Decision(kind=DecisionKind.COMPLETE)
        with pytest.raises(ValidationError):
            decision.is_complete = True  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        # Use model_validate so a static type checker can't see the
        # unknown kwarg on the Decision constructor signature.
        with pytest.raises(ValidationError):
            Decision.model_validate({"kind": DecisionKind.COMPLETE, "unknown_field": "x"})

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Decision(kind=DecisionKind.COMPLETE, confidence=1.5)
        with pytest.raises(ValidationError):
            Decision(kind=DecisionKind.COMPLETE, confidence=-0.1)

    def test_json_round_trip(self) -> None:
        original = Decision(
            kind=DecisionKind.LOOP_DETECTED,
            is_complete=True,
            should_escalate=True,
            feedback="loop on tool_x",
            confidence=0.9,
            metadata={"step": 3},
        )
        payload = original.model_dump_json()
        restored = Decision.model_validate_json(payload)
        assert restored == original


class TestSuccessCriteria:
    def test_empty_defaults(self) -> None:
        sc = SuccessCriteria()
        assert sc.goal_statement == ""
        assert sc.criteria == ()
        assert sc.must_include == ()
        assert sc.target_format is None
        assert sc.min_confidence == pytest.approx(0.5)

    def test_from_intent_reads_goal(self) -> None:
        intent = SimpleNamespace(
            goal=SimpleNamespace(
                statement="summarize the filing",
                success_criteria=("name parties", "cite governing law"),
                target_format="markdown",
            )
        )
        sc = SuccessCriteria.from_intent(intent)
        assert sc.goal_statement == "summarize the filing"
        assert sc.criteria == ("name parties", "cite governing law")
        assert sc.target_format == "markdown"

    def test_from_intent_no_goal_returns_empty(self) -> None:
        sc = SuccessCriteria.from_intent(SimpleNamespace())
        assert sc == SuccessCriteria()

    def test_from_intent_partial_goal(self) -> None:
        intent = SimpleNamespace(goal=SimpleNamespace(statement="do a thing"))
        sc = SuccessCriteria.from_intent(intent)
        assert sc.goal_statement == "do a thing"
        assert sc.criteria == ()
        assert sc.target_format is None

    def test_frozen(self) -> None:
        sc = SuccessCriteria(goal_statement="x")
        with pytest.raises(ValidationError):
            sc.goal_statement = "y"  # type: ignore[misc]
