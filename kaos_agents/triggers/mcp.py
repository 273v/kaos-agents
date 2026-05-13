"""MCP-source trigger factory.

Phase 2.A: re-exports :meth:`Trigger.mcp` under the stable
``MCPToolTrigger`` name so consumers can write::

    from kaos_agents.triggers.mcp import MCPToolTrigger
    trigger = MCPToolTrigger("hello", session_id="s1")

Phase 4+ may replace this alias with a real
:class:`kaos_agents.triggers.base.TriggerSource` subclass that consumes
inbound MCP requests as an async iterator. The import path stays
stable across the transition.
"""

from __future__ import annotations

from kaos_agents.triggers.base import Trigger

MCPToolTrigger = Trigger.mcp

__all__ = ["MCPToolTrigger"]
