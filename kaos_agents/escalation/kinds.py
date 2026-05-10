"""EscalationKind enum — paper §Q8."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class EscalationKind(StrEnum):
    CLARIFICATION_NEEDED = "clarification_needed"  # ambiguous intent (Q2)
    APPROVAL_REQUIRED = "approval_required"  # destructive action (Q4)
    OUTSIDE_COMPETENCE = "outside_competence"  # agent recognises its limits
    BUDGET_EXCEEDED = "budget_exceeded"  # cost / time / iter cap
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"  # RAG refused (Q3/Q5)
    LOOP_DETECTED = "loop_detected"  # Q7 detected
    DOMAIN_SPECIFIC = "domain_specific"  # extensible by recipe


__all__ = ["EscalationKind"]
