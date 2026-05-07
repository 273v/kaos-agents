"""Runtime — concrete agent implementations.

Mirrors :mod:`kaos_core` runtime layout. Holds the canonical
implementations of the ABCs in :mod:`kaos_agents.base`:

- :mod:`.agent` — :class:`BaseAgent`, the canonical :class:`KaosAgent`
  implementation with the 8-step turn loop. (Track 2 chunk 2 moves
  the existing ``kaos_agents/agent.py`` here.)
- :mod:`.runner` — :class:`Runner`, the execution engine that
  consumes an :class:`Agent` config and drives the loop. (Track 2
  chunk 2.)
- :mod:`.delegation` — :class:`DelegatedAgent` + ``agent_as_tool``.
- :mod:`.interrupts` — :class:`PendingToolCall` + :class:`RunState`
  for pause/resume.
- :mod:`.permissions` — :class:`PermissionPolicy` engine. (Value
  types ``PermissionRule`` / ``PermissionDecision`` stay in
  :mod:`kaos_agents.types.permissions`.)
- :mod:`.events_to_response` — events-stream → :class:`AgentResponse`
  conversion (used by the default :meth:`KaosAgent.turn`).

Track 2 chunk 1 only adds :mod:`.events_to_response` for the
:meth:`KaosAgent.turn` default. Subsequent chunks move the rest.
"""

from __future__ import annotations
