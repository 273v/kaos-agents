"""Escalation subsystem (paper §Q8).

Answers the paper's Q8: how does the agent know when to ask for help?

Phase 4.C ships:

* :class:`EscalationKind` — 7-value StrEnum covering the canonical
  escalation reasons (CLARIFICATION_NEEDED, APPROVAL_REQUIRED,
  OUTSIDE_COMPETENCE, BUDGET_EXCEEDED, EVIDENCE_INSUFFICIENT,
  LOOP_DETECTED, DOMAIN_SPECIFIC).
* :class:`EscalationPolicy` + :class:`EscalationDecision` — a small
  rule-engine that maps signals (intent / termination Decision /
  caller-driven force) into a typed decision.
* :class:`HITLBridge` + :class:`HITLChannel` +
  :func:`escalation_resource_uri` + :class:`EscalationContext` —
  routing of escalations to one of three channels (CLI / HTTP /
  MCP-resource per Resolved Decision #2).
* :class:`EscalationResumePayload` — the data Phase 4.D wires into
  Runner.resume() to continue a paused run.

Phase 4.C is the SUBSYSTEM. Phase 4.D wires it into AgentLoop.
"""

from __future__ import annotations

from kaos_agents.escalation.hitl import (
    EscalationContext,
    HITLBridge,
    HITLChannel,
    escalation_resource_uri,
)
from kaos_agents.escalation.kinds import EscalationKind
from kaos_agents.escalation.policy import EscalationDecision, EscalationPolicy
from kaos_agents.escalation.resume import EscalationResumePayload

__all__ = [
    "EscalationContext",
    "EscalationDecision",
    "EscalationKind",
    "EscalationPolicy",
    "EscalationResumePayload",
    "HITLBridge",
    "HITLChannel",
    "escalation_resource_uri",
]
