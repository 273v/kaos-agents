"""Webhook-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.webhook` under the stable
``WebhookTrigger`` name so consumers can write::

    from kaos_agents.triggers.webhook import WebhookTrigger
    trigger = WebhookTrigger(event_type="github.push", body=payload)

Phase 4+ may replace this alias with a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass that
consumes inbound webhook deliveries from an external feed (paper §2.1).
The import path stays stable across the transition.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

WebhookTrigger = Trigger.webhook

__all__ = ["WebhookTrigger"]
