"""Escalation-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.escalation` under the stable
``EscalationTrigger`` name so consumers can write::

    from kaos_agents.triggers.escalation import EscalationTrigger
    trigger = EscalationTrigger(
        reason="ambiguous query",
        kind="clarification_needed",
        parent_turn_id="turn-7",
    )

Phase 4+ may replace this alias with a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass driven by
sub-agent escalations (paper §2.4). The import path stays stable
across the transition.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

EscalationTrigger = Trigger.escalation

__all__ = ["EscalationTrigger"]
