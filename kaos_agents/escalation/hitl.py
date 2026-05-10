"""HITLBridge — human-in-the-loop routing for escalations.

Resolved Decision #2: cross-process A2A wire format is the MCP
resource at ``kaos-agents://envelope/<agent_hash>``. The HITLBridge
mirrors that pattern for escalations: each escalation gets a stable
URI ``kaos-agents://escalation/<escalation_id>`` that callers can
fetch / resolve.

Phase 4.C ships:
  - HITLChannel enum with three kinds
  - escalation_resource_uri() helper for stable URIs
  - HITLBridge.surface(escalation, channel=...) — dispatches; Phase
    4.C implementations are best-effort (CLI: print to stderr;
    HTTP: returns the URI for a webhook callback; MCP-resource:
    returns the URI for the MCP server to register).
  - EscalationContext value type bundling escalation + channel +
    URI for in-process resume.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Any

from kaos_agents.escalation.kinds import EscalationKind


@unique
class HITLChannel(StrEnum):
    CLI = "cli"
    HTTP = "http"
    MCP_RESOURCE = "mcp-resource"


def escalation_resource_uri(escalation_id: str) -> str:
    """Return the canonical MCP-resource URI for an escalation.

    Mirrors :func:`kaos_agents.core.envelope.agent_hash` URI form
    for AgentEnvelope. Resolved Decision #2 uses
    ``kaos-agents://envelope/<hash>``; we use
    ``kaos-agents://escalation/<id>`` to match.
    """
    return f"kaos-agents://escalation/{escalation_id}"


@dataclass(slots=True)
class EscalationContext:
    """In-process bundle of an active escalation.

    Caller passes this back to ``Runner.resume_escalation(...)`` (
    Phase 4.D) along with the human/parent's response.
    """

    escalation_id: str
    kind: EscalationKind
    reason: str
    channel: HITLChannel
    uri: str
    resume_token: str
    details: dict[str, Any] = field(default_factory=dict)


class HITLBridge:
    """Routes an escalation to one of three channels.

    Phase 4.C is best-effort: the bridge constructs the
    EscalationContext + URI but does NOT actually wait for input —
    it returns the context to the caller (Runner / AgentLoop) which
    is responsible for the surface-specific wait.
    """

    def __init__(
        self,
        *,
        default_channel: HITLChannel = HITLChannel.MCP_RESOURCE,
    ) -> None:
        self._default = default_channel

    def surface(
        self,
        *,
        kind: EscalationKind,
        reason: str,
        resume_token: str,
        channel: HITLChannel | None = None,
        escalation_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> EscalationContext:
        """Construct an EscalationContext + emit a channel-specific notification.

        For CLI: writes a one-line notice to stderr.
        For HTTP: noop (caller is expected to return the URI from a
          handler).
        For MCP_RESOURCE: noop (the runner registers the resource).
        """
        eid = escalation_id or f"esc_{uuid.uuid4().hex[:12]}"
        ch = channel or self._default
        uri = escalation_resource_uri(eid)
        ctx = EscalationContext(
            escalation_id=eid,
            kind=kind,
            reason=reason,
            channel=ch,
            uri=uri,
            resume_token=resume_token,
            details=dict(details or {}),
        )
        if ch is HITLChannel.CLI:
            self._cli_notify(ctx)
        # HTTP and MCP_RESOURCE are surfaced via return value; Phase
        # 4.D wiring publishes them through the appropriate route.
        return ctx

    @staticmethod
    def _cli_notify(ctx: EscalationContext) -> None:
        """Phase 4.C CLI surface: write to stderr."""
        print(
            f"[escalation] kind={ctx.kind.value} id={ctx.escalation_id}\n"
            f"  reason: {ctx.reason}\n"
            f"  resume: {ctx.uri}",
            file=sys.stderr,
        )


__all__ = [
    "EscalationContext",
    "HITLBridge",
    "HITLChannel",
    "escalation_resource_uri",
]
