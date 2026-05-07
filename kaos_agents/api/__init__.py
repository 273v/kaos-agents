"""kaos-agents HTTP / wire surface — Track 5 chunk T5-3 consolidation.

Subpackage layout:

- :mod:`kaos_agents.api.server` — FastAPI ``create_app`` + REST
  routes for ``/v1/sessions/...`` (was ``kaos_agents.api``)
- :mod:`kaos_agents.api.serve`  — ``kaos-agents-serve`` CLI entry
  point (was ``kaos_agents.serve``)
- :mod:`kaos_agents.api.wire`   — SSE / JSONL / WebSocket event
  serialisers (was ``kaos_agents.wire``)

Top-level entry point (registered via ``[project.scripts]``):

- ``kaos-agents-serve = kaos_agents.api.serve:main``

Public re-exports keep the most-used names accessible at the
subpackage level:
    from kaos_agents.api import create_app, events_to_sse, events_to_jsonl

Pre-T5 leaf modules (``kaos_agents.api`` as a flat module,
``kaos_agents.serve``, ``kaos_agents.wire``) are gone.
"""

from __future__ import annotations

from kaos_agents.api.server import create_app
from kaos_agents.api.wire import events_to_jsonl, events_to_sse, events_to_ws

__all__ = [
    "create_app",
    "events_to_jsonl",
    "events_to_sse",
    "events_to_ws",
]
