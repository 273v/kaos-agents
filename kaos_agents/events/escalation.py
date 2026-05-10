"""EscalationRequired — generalised escalation event.

Phase 4.C ships this as a NEW event type alongside the existing
:class:`ToolCallApprovalRequired`. They coexist:

  - ``ToolCallApprovalRequired`` keeps its dedicated tool-call
    approval semantics (the only escalation kind kaos-agents had
    pre-Phase-4).
  - ``EscalationRequired`` covers the seven generalised kinds:
    CLARIFICATION_NEEDED, APPROVAL_REQUIRED, OUTSIDE_COMPETENCE,
    BUDGET_EXCEEDED, EVIDENCE_INSUFFICIENT, LOOP_DETECTED,
    DOMAIN_SPECIFIC.

Phase 6 cutover may collapse ToolCallApprovalRequired into
EscalationRequired(kind=APPROVAL_REQUIRED) — until then both fire.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from kaos_agents.events.lifecycle import LifecycleEvent


class EscalationRequired(LifecycleEvent):
    """A turn cannot continue without human / parent input.

    Carried in the event stream like any LifecycleEvent so SSE / JSONL
    consumers see escalations natively. The HITLBridge subscribes to
    these and routes to the appropriate channel.

    Fields:
      kind: one of EscalationKind values (str for wire-safety; enums
        do not survive arbitrary JSON round-trips equally well across
        serialisers).
      reason: human-readable explanation.
      details: free-form payload (e.g. ambiguities for
        CLARIFICATION_NEEDED, tool args for APPROVAL_REQUIRED).
      resume_token: opaque string the HITL caller passes back to
        Runner.resume() to continue the run. Phase 4.C value is the
        TurnInvocation.id (run-scoped uniqueness).
      escalation_id: dedup id; auto-uuid'd if not supplied.
    """

    kind: str = ""  # EscalationKind value
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    resume_token: str = ""
    escalation_id: str = ""


__all__ = ["EscalationRequired"]
