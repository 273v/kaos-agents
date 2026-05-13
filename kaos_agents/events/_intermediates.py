"""Intermediate event categories shared across split event modules.

:class:`LifecycleEvent` is the parent of all "high-level, semantic"
events — tool boundaries, plan steps, research citations, etc. It
lives here (rather than in :mod:`kaos_agents.events.lifecycle`) to
break a circular import: every domain-specific event module needs to
import :class:`LifecycleEvent` to subclass it, and lifecycle.py also
needs it for the turn-level events. Keeping the intermediate in a
separate module gives every domain module a single, dependency-free
import path.

:class:`StreamDelta` lives in :mod:`kaos_agents.events.stream` because
no other module subclasses it (only the three concrete deltas do).
"""

from __future__ import annotations

from kaos_agents.base.event import KaosEvent


class LifecycleEvent(KaosEvent):
    """Base class for high-level, semantic lifecycle events.

    Consumers that drive UI state machines, logging, hooks, audit
    trails — anything that needs "what just happened" rather than
    "what tokens just arrived" — handle these.

    ``isinstance(event, LifecycleEvent)`` is the canonical way to
    route lifecycle events.
    """
