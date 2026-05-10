"""Resume helpers for escalation pause/resume.

Phase 4.C ships pure value types and helper functions. Phase 4.D
wires them into ``Runner.pause()`` / ``Runner.resume()``. The
existing :class:`RunState` (in ``runtime/interrupts.py``) is kept
unchanged; Phase 4.C adds an ``EscalationResumePayload`` that
carries the human/parent's response back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_agents.escalation.kinds import EscalationKind


@dataclass(frozen=True, slots=True)
class EscalationResumePayload:
    """The data needed to resume after an escalation.

    Fields:
      escalation_id: matches the EscalationRequired event's id.
      kind: matches the original EscalationKind.
      response: free-form human / parent reply (str, dict, or any).
      decision: "approve" | "deny" | "answer" | "abort". Phase 4.D
        uses this to pick the right Runner.resume path.
      metadata: optional additional fields.
    """

    escalation_id: str
    kind: EscalationKind
    response: Any = None
    decision: str = "answer"  # "approve" | "deny" | "answer" | "abort"
    metadata: dict[str, Any] | None = None


__all__ = ["EscalationResumePayload"]
