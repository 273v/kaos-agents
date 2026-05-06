"""Catalogues for kaos-agents core concepts.

Mirrors :mod:`kaos_core.registry`. Each registry holds instances or
classes keyed by their identity (``metadata.name`` for tools,
``event_type()`` for events) and surfaces lookup, listing, and
filtering helpers.

Subpackages added incrementally across the refactor:

- :mod:`kaos_agents.registry.event_registry` — :class:`EventRegistry`
  (chunk 3)
- :mod:`kaos_agents.registry.hook_registry` — :class:`HookRegistry`
  (chunk 5)
- :mod:`kaos_agents.registry.pattern_registry` — :class:`PatternRegistry`
  (Track 2)
- :mod:`kaos_agents.registry.recipe_registry` — :class:`RecipeRegistry`
  (Track 4)
"""

from __future__ import annotations

from kaos_agents.registry.event_registry import (
    EventRegistry,
    default_event_registry,
)

__all__ = ["EventRegistry", "default_event_registry"]
