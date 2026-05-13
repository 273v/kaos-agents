"""kaos-agents governance — paper §Q10 primitives.

The KaosEvent stream IS the audit log: every governance primitive
emits or consumes KaosEvents so all admin actions are observable and
replayable. The §Q10 surface aggregates:

  Phase 1.C : CircuitBreaker (trip on consecutive errors / cost spike)
              re-exported from kaos_agents.action.circuit
  Phase 0/1 : LeastPrivilege via SessionToolSet (allow/deny rules)
              re-exported from kaos_agents.types.session_tool_set
              + kaos_agents.context.filter_tools
  Phase 5.F : JSONLAuditLogger (durable JSONL audit log → VFS)
  Phase 5.F : StateSnapshot + snapshot/restore helpers
  Phase 5.F : OverrideHook (admin event injection)
"""

from __future__ import annotations

# Phase 1.C — re-export CircuitBreaker for the §Q10 namespace.
from kaos_agents.action.circuit import CircuitBreaker, CircuitState

# Phase 0/1 — re-export LeastPrivilege primitives.
from kaos_agents.context import filter_tools
from kaos_agents.governance.logging import JSONLAuditLogger
from kaos_agents.governance.override import OverrideHook, OverrideKind, OverrideRecord
from kaos_agents.governance.snapshot import (
    StateSnapshot,
    load_snapshot,
    save_snapshot,
)
from kaos_agents.types.session_tool_set import SessionToolSet

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "JSONLAuditLogger",
    "OverrideHook",
    "OverrideKind",
    "OverrideRecord",
    "SessionToolSet",
    "StateSnapshot",
    "filter_tools",
    "load_snapshot",
    "save_snapshot",
]
