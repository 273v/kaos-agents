"""HTTP-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.http` under the stable
``HTTPMessageTrigger`` name so consumers can write::

    from kaos_agents.triggers.http import HTTPMessageTrigger
    trigger = HTTPMessageTrigger("hello", request_id="req-42")

Phase 4+ may replace this alias with a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass that consumes
inbound FastAPI requests as an async iterator. The import path stays
stable across the transition.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

HTTPMessageTrigger = Trigger.http

__all__ = ["HTTPMessageTrigger"]
