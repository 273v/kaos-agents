"""Delegation-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.delegation` under the stable
``DelegationTrigger`` name so consumers can write::

    from kaos_agents.triggers.delegation import DelegationTrigger
    trigger = DelegationTrigger(goal="extract dates", parent_turn_id="turn-7")

Phase 4+ may swap this alias for a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass — though
delegation is naturally a push-based source (parents construct the
trigger directly), so the alias may stay permanent. The import path is
stable regardless.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

DelegationTrigger = Trigger.delegation

__all__ = ["DelegationTrigger"]
