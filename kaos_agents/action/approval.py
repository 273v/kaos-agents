"""ApprovalWorkflow — produces a :class:`ToolCallApprovalRequired` event.

The kaos-agents Runner already knows how to suspend on
:class:`ToolCallApprovalRequired` and resume via
:meth:`Runner.resume`. Phase 1.C ships a Program-shaped primitive that
wraps the existing event emission so the Actor can consult it cleanly.

Approval rules by reversibility tier:

- :attr:`Reversibility.REVERSIBLE` — never required (auto-allow).
- :attr:`Reversibility.RECOVERABLE` — never required (auto-allow), but
  logged.
- :attr:`Reversibility.EXTERNALLY_VISIBLE` — required by default;
  override via ``ActionPlan.approval_required=False``.
- :attr:`Reversibility.IRREVERSIBLE` — always required, regardless of
  ``ActionPlan.approval_required``.

A :class:`PermissionPolicy` may override these rules. Phase 1.C
exposes the predicate :meth:`required_for` — wiring it into the Runner
loop is Phase 2.
"""

from __future__ import annotations

from typing import Any

from kaos_agents.action.reversibility import Reversibility
from kaos_agents.action.types import ActionPlan


class ApprovalWorkflow:
    """Decides whether an :class:`ActionPlan` needs approval and emits the event."""

    def required_for(self, plan: ActionPlan) -> bool:
        """Return True iff ``plan`` needs approval before dispatch."""
        if plan.reversibility in (Reversibility.REVERSIBLE, Reversibility.RECOVERABLE):
            return False
        if plan.reversibility == Reversibility.EXTERNALLY_VISIBLE:
            return plan.approval_required
        # IRREVERSIBLE: always require, regardless of plan flag.
        return True

    def build_event(self, plan: ActionPlan, *, reason: str = "") -> tuple[type, dict[str, Any]]:
        """Construct event-class + kwargs for a :class:`ToolCallApprovalRequired`.

        Returns a ``(constructor, kwargs)`` tuple so the caller can use
        their own :class:`EventEmitter` (no global emitter coupling).
        ``arguments`` is serialized as the ``tuple[tuple[str, str], ...]``
        shape the existing event class expects: each tuple is
        ``(arg_name, repr(arg_value))``.
        """
        # Local import keeps this module decoupled from the events
        # package at module load time — useful for environments that
        # exercise the workflow without a full agent runtime.
        from kaos_agents.events.tools import ToolCallApprovalRequired

        arguments: tuple[tuple[str, str], ...] = tuple(
            (str(k), repr(v)) for k, v in plan.args.items()
        )
        kwargs: dict[str, Any] = {
            "tool_name": plan.tool_name,
            "arguments": arguments,
            "reason": reason or f"reversibility={plan.reversibility.value}",
        }
        return ToolCallApprovalRequired, kwargs


__all__ = ["ApprovalWorkflow"]
