"""Unit tests for :class:`kaos_agents.escalation.kinds.EscalationKind`."""

from __future__ import annotations

from kaos_agents.escalation.kinds import EscalationKind


class TestEscalationKind:
    """The 7-value enum is the discriminator the policy/HITL bridge use."""

    def test_seven_members_present(self) -> None:
        names = {member.value for member in EscalationKind}
        assert names == {
            "clarification_needed",
            "approval_required",
            "outside_competence",
            "budget_exceeded",
            "evidence_insufficient",
            "loop_detected",
            "domain_specific",
        }

    def test_members_are_strings(self) -> None:
        for member in EscalationKind:
            assert isinstance(member.value, str)
            assert member.value
            # StrEnum compares to its str value
            assert member == member.value

    def test_members_are_unique(self) -> None:
        # @unique would raise at import time on collision; this is a
        # belt-and-braces check at runtime.
        values = [member.value for member in EscalationKind]
        assert len(values) == len(set(values))
