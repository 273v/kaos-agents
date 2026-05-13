"""Scheduled-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.scheduled` under the stable
``ScheduledTrigger`` name so consumers can write::

    from kaos_agents.triggers.schedule import ScheduledTrigger
    trigger = ScheduledTrigger(job_name="nightly-research", schedule="0 2 * * *")

Phase 4+ replaces this alias with a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass that yields
:class:`Trigger` events from a cron/interval scheduler (paper §2.3).
The import path stays stable across the transition.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

ScheduledTrigger = Trigger.scheduled

__all__ = ["ScheduledTrigger"]
