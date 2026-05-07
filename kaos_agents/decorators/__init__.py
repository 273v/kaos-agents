"""Tier-1 decorators — the easy on-ramp for kaos-agents.

Mirrors :mod:`kaos_core.decorators` (which provides ``@kaos_tool``).

Currently:

- :func:`hook` — wraps a plain async function as a
  :class:`kaos_agents.hooks.base.KaosHook` and auto-registers into
  :data:`kaos_agents.registry.hook_registry.default_hook_registry`.

Tier-2 (subclass :class:`KaosHook`) and Tier-3 (custom registry +
manual instantiation) remain available for advanced cases — see
:mod:`kaos_agents.hooks` and :mod:`kaos_agents.registry`.

Note on ``@event``: :class:`kaos_agents.base.event.KaosEvent` already
auto-registers every subclass via ``__init_subclass__``, so a separate
``@event`` decorator would be a typed alias for the same behavior.
We don't ship one — declare event subclasses by inheriting from
:class:`KaosEvent` directly.
"""

from __future__ import annotations

from kaos_agents.decorators.hook import FunctionHook, hook

__all__ = ["FunctionHook", "hook"]
