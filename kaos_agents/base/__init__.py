"""Abstract base classes (ABCs) for kaos-agents core concepts.

Mirrors the layout of :mod:`kaos_core.base`. Each concept's ABC carries
a frozen pydantic ``KaosModel`` metadata describing its identity, and
the ABC's ``metadata`` accessor returns one of these instances.

The five concepts (added incrementally across chunks 2-5):

- :mod:`kaos_agents.base.event` — :class:`KaosEvent` (chunk 2)
- :mod:`kaos_agents.base.hook` — :class:`KaosHook` (chunk 5)
- :mod:`kaos_agents.base.agent` — :class:`KaosAgent` (Track 2)
- :mod:`kaos_agents.base.pattern` — :class:`KaosPattern` (Track 2)
- :mod:`kaos_agents.base.recipe` — :class:`KaosRecipe` (Track 4)

Pure-strategy contracts (no shared impl) live alongside as
:class:`typing.Protocol` classes — see :mod:`kaos_agents.base.classifier`,
:mod:`kaos_agents.base.context`, :mod:`kaos_agents.base.tool_bridge`.
"""

from __future__ import annotations

from kaos_agents.base.event import KaosEvent

__all__ = ["KaosEvent"]
