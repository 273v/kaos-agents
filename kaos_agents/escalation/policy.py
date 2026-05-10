"""EscalationPolicy — when to escalate vs continue.

Phase 4.C baseline: a small rule-engine that maps signals (intent /
termination Decision / events) into an EscalationDecision. Composes
with TerminationJudge: when TerminationJudge sets
``should_escalate=True``, EscalationPolicy reads the kind and
decides the channel + payload.

Constructor kwargs:
  channels: tuple of HITLChannel values to use, in priority order
    (default ("cli", "http", "mcp-resource")). The HITLBridge
    consults this when routing.
  always_escalate: tuple of EscalationKind values that bypass any
    other gate (default — IRREVERSIBLE actions, OUTSIDE_COMPETENCE).
  never_escalate: tuple of EscalationKind values to suppress (default
    empty; use to silence noisy escalations during evaluation runs).
  threshold_clarification_confidence: when intent.confidence is
    below this AND requires_clarification is True, escalate. Default
    0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaos_agents.escalation.kinds import EscalationKind


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    escalate: bool
    kind: EscalationKind | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class EscalationPolicy:
    def __init__(
        self,
        *,
        channels: tuple[str, ...] = ("cli", "http", "mcp-resource"),
        always_escalate: tuple[EscalationKind, ...] = (
            EscalationKind.OUTSIDE_COMPETENCE,
            EscalationKind.LOOP_DETECTED,
        ),
        never_escalate: tuple[EscalationKind, ...] = (),
        threshold_clarification_confidence: float = 0.5,
    ) -> None:
        self._channels = channels
        self._always = always_escalate
        self._never = never_escalate
        self._threshold = threshold_clarification_confidence

    @property
    def channels(self) -> tuple[str, ...]:
        return self._channels

    def evaluate_intent(self, intent: Any) -> EscalationDecision:
        """Decide whether an :class:`IntentResult` warrants escalation."""
        requires = bool(getattr(intent, "requires_clarification", False))
        confidence = float(getattr(intent, "confidence", 1.0))
        if requires and confidence < self._threshold:
            ambiguities = tuple(getattr(intent, "ambiguities", ()) or ())
            return EscalationDecision(
                escalate=True,
                kind=EscalationKind.CLARIFICATION_NEEDED,
                reason=(
                    ambiguities[0].preferred_clarification
                    if ambiguities and getattr(ambiguities[0], "preferred_clarification", None)
                    else "intent ambiguous; clarification required"
                ),
                details={
                    "confidence": confidence,
                    "ambiguity_count": len(ambiguities),
                },
            )
        return EscalationDecision(escalate=False)

    def evaluate_decision(self, decision: Any) -> EscalationDecision:
        """Decide whether a TerminationJudge :class:`Decision` warrants escalation."""
        if not bool(getattr(decision, "should_escalate", False)):
            return EscalationDecision(escalate=False)
        kind_str = str(getattr(decision, "kind", ""))
        kind = self._kind_from_decision(kind_str)
        if kind in self._never:
            return EscalationDecision(escalate=False, reason=f"suppressed: {kind.value}")
        return EscalationDecision(
            escalate=True,
            kind=kind,
            reason=str(getattr(decision, "feedback", "")),
            details={"decision_kind": kind_str},
        )

    def force(
        self, kind: EscalationKind, *, reason: str = "", **details: Any
    ) -> EscalationDecision:
        """Force an escalation regardless of policy gates (e.g. caller-driven)."""
        if kind in self._never:
            return EscalationDecision(escalate=False, reason=f"suppressed: {kind.value}")
        return EscalationDecision(
            escalate=True,
            kind=kind,
            reason=reason,
            details=details,
        )

    @staticmethod
    def _kind_from_decision(decision_kind: str) -> EscalationKind:
        """Map TerminationJudge DecisionKind values to EscalationKind."""
        mapping = {
            "budget_exceeded": EscalationKind.BUDGET_EXCEEDED,
            "loop_detected": EscalationKind.LOOP_DETECTED,
            "failure": EscalationKind.EVIDENCE_INSUFFICIENT,  # RunError → review
            "quality_failed": EscalationKind.OUTSIDE_COMPETENCE,
        }
        return mapping.get(decision_kind, EscalationKind.DOMAIN_SPECIFIC)


__all__ = ["EscalationDecision", "EscalationPolicy"]
