"""Lifecycle hooks for the agent Runner.

Hooks provide deterministic interception points around agent events.
Implement a hook to add logging, metrics, guardrails, approval UI,
cost tracking, or audit trails — without modifying the agent itself.

Layout (chunk 5 — was a single 432-LOC ``hooks.py`` + 259-LOC ``otel.py``):

- :mod:`.base` — :class:`KaosHook` ABC + :class:`HookAction` enum.
- :mod:`.dispatch` — :func:`dispatch_hook` (Runner-side fan-out).
- :mod:`.builtin` — :class:`LoggingHook`, :class:`CostTrackingHook`,
  :class:`AuditHook`.
- :mod:`.otel` — :class:`OTelHook` (optional ``[otel]`` extra).

Public surface: this package re-exports :class:`KaosHook`,
:class:`HookAction`, :func:`dispatch_hook`, and the four built-in
hooks — so callers continue to use ``from kaos_agents.hooks import X``.

Tier-1 (decorator) on-ramp lands in chunk 6 — ``@hook`` will wrap a
function as a FunctionHook and auto-register into
:class:`kaos_agents.registry.hook_registry.default_hook_registry`.
"""

from __future__ import annotations

from kaos_agents.hooks.base import HookAction, KaosHook
from kaos_agents.hooks.builtin import (
    AuditHook,
    CostTrackingHook,
    LoggingHook,
)
from kaos_agents.hooks.dispatch import dispatch_hook
from kaos_agents.hooks.otel import OTelHook

__all__ = [
    "AuditHook",
    "CostTrackingHook",
    "HookAction",
    "KaosHook",
    "LoggingHook",
    "OTelHook",
    "dispatch_hook",
]
