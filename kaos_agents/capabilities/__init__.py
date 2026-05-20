"""Agent-side capability facades.

Plan §1 of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``
calls for unified, mode-parameterised capability primitives that
federate over the per-module tool surfaces (``kaos-web-*``,
``kaos-source-*``, ``kaos-content-*``).

This subpackage holds those agent-side facades. Each facade reads the
:class:`CapabilityRegistry` to find capabilities of the right
:class:`CapabilityKind`, resolves them to concrete
``backing_tool_names`` on the active :class:`KaosRuntime`, invokes the
tools dynamically, and aggregates the results into a uniform value type
(e.g. :class:`RetrievalHit`).

Step 2 ships :func:`retrieve` — the federation facade over the
SEARCH / READ tool surface. Future steps add ``discover`` (plan §5),
``draft``, and so on.

The facades intentionally live in kaos-agents (not kaos-core): the
registry resolution + dynamic ``KaosTool.execute`` invocation is an
agent-side concern. kaos-core stays free of any "compose multiple
tools into one capability" logic.
"""

from __future__ import annotations

from kaos_agents.capabilities.retrieve import (
    RETRIEVE_CAPABILITY_NAME,
    RetrievalHit,
    retrieve,
)

__all__ = [
    "RETRIEVE_CAPABILITY_NAME",
    "RetrievalHit",
    "retrieve",
]
