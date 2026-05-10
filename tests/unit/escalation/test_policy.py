"""Unit tests for :class:`kaos_agents.escalation.policy.EscalationPolicy`.

Covers the three signal entry points (``evaluate_intent``,
``evaluate_decision``, ``force``), the ``never_escalate`` filter,
and the configured channels accessor.
"""

from __future__ import annotations

from types import SimpleNamespace

from kaos_agents.escalation.kinds import EscalationKind
from kaos_agents.escalation.policy import EscalationDecision, EscalationPolicy


def _intent(
    *,
    requires_clarification: bool = False,
    confidence: float = 1.0,
    ambiguities: tuple[object, ...] = (),
) -> SimpleNamespace:
    """Duck-typed IntentResult-like value for the policy."""
    return SimpleNamespace(
        requires_clarification=requires_clarification,
        confidence=confidence,
        ambiguities=ambiguities,
    )


def _decision(
    *,
    should_escalate: bool = False,
    kind: str = "",
    feedback: str = "",
) -> SimpleNamespace:
    """Duck-typed termination Decision-like value for the policy."""
    return SimpleNamespace(should_escalate=should_escalate, kind=kind, feedback=feedback)


class TestEvaluateIntent:
    def test_no_clarification_returns_no_escalate(self) -> None:
        policy = EscalationPolicy()
        result = policy.evaluate_intent(_intent(requires_clarification=False, confidence=0.1))
        assert result.escalate is False
        assert result.kind is None

    def test_low_confidence_with_clarification_escalates(self) -> None:
        policy = EscalationPolicy(threshold_clarification_confidence=0.5)
        result = policy.evaluate_intent(_intent(requires_clarification=True, confidence=0.3))
        assert result.escalate is True
        assert result.kind is EscalationKind.CLARIFICATION_NEEDED
        assert result.details == {"confidence": 0.3, "ambiguity_count": 0}
        assert result.reason  # non-empty default

    def test_high_confidence_with_clarification_does_not_escalate(self) -> None:
        policy = EscalationPolicy(threshold_clarification_confidence=0.5)
        result = policy.evaluate_intent(_intent(requires_clarification=True, confidence=0.8))
        assert result.escalate is False
        assert result.kind is None

    def test_reads_first_ambiguity_preferred_clarification(self) -> None:
        policy = EscalationPolicy(threshold_clarification_confidence=0.5)
        amb = SimpleNamespace(preferred_clarification="Which contract did you mean?")
        result = policy.evaluate_intent(
            _intent(requires_clarification=True, confidence=0.2, ambiguities=(amb,))
        )
        assert result.escalate is True
        assert result.kind is EscalationKind.CLARIFICATION_NEEDED
        assert result.reason == "Which contract did you mean?"
        assert result.details["ambiguity_count"] == 1


class TestEvaluateDecision:
    def test_no_should_escalate_returns_no_escalate(self) -> None:
        policy = EscalationPolicy()
        result = policy.evaluate_decision(_decision(should_escalate=False, kind="incomplete"))
        assert result.escalate is False
        assert result.kind is None

    def test_maps_decision_kind_to_escalation_kind(self) -> None:
        policy = EscalationPolicy()
        for decision_kind, expected in [
            ("budget_exceeded", EscalationKind.BUDGET_EXCEEDED),
            ("loop_detected", EscalationKind.LOOP_DETECTED),
            ("failure", EscalationKind.EVIDENCE_INSUFFICIENT),
            ("quality_failed", EscalationKind.OUTSIDE_COMPETENCE),
            ("escalate", EscalationKind.DOMAIN_SPECIFIC),  # falls through to default
        ]:
            result = policy.evaluate_decision(
                _decision(should_escalate=True, kind=decision_kind, feedback="why")
            )
            assert result.escalate is True, decision_kind
            assert result.kind is expected, decision_kind
            assert result.reason == "why"
            assert result.details == {"decision_kind": decision_kind}


class TestForce:
    def test_force_returns_escalation(self) -> None:
        policy = EscalationPolicy()
        result = policy.force(
            EscalationKind.APPROVAL_REQUIRED, reason="destructive tool", tool="delete"
        )
        assert result.escalate is True
        assert result.kind is EscalationKind.APPROVAL_REQUIRED
        assert result.reason == "destructive tool"
        assert result.details == {"tool": "delete"}

    def test_never_escalate_filter_suppresses(self) -> None:
        policy = EscalationPolicy(
            never_escalate=(EscalationKind.DOMAIN_SPECIFIC, EscalationKind.LOOP_DETECTED)
        )
        result = policy.force(EscalationKind.DOMAIN_SPECIFIC, reason="x")
        assert result.escalate is False
        assert "suppressed" in result.reason

    def test_never_escalate_filter_in_evaluate_decision(self) -> None:
        # quality_failed → OUTSIDE_COMPETENCE; suppressed
        policy = EscalationPolicy(never_escalate=(EscalationKind.OUTSIDE_COMPETENCE,))
        result = policy.evaluate_decision(
            _decision(should_escalate=True, kind="quality_failed", feedback="bad")
        )
        assert result.escalate is False
        assert "suppressed" in result.reason


class TestPolicyChannels:
    def test_default_channels(self) -> None:
        policy = EscalationPolicy()
        assert policy.channels == ("cli", "http", "mcp-resource")

    def test_custom_channels(self) -> None:
        policy = EscalationPolicy(channels=("mcp-resource",))
        assert policy.channels == ("mcp-resource",)


class TestEscalationDecisionType:
    def test_default_factory_for_details(self) -> None:
        decision = EscalationDecision(escalate=False)
        assert decision.details == {}
        assert decision.kind is None
        assert decision.reason == ""

    def test_frozen(self) -> None:
        import dataclasses

        import pytest

        decision = EscalationDecision(escalate=True, kind=EscalationKind.LOOP_DETECTED)
        # Use setattr() so static type checkers don't flag the
        # intentionally-illegal assignment we're trying to provoke.
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(decision, "escalate", False)  # noqa: B010 — defeats ty static check on frozen dataclass
