"""TurnInvocation — the canonical runtime contract for one agent turn.

Mirrors kaos_llm_core.programs._invocation.Invocation field-for-field at
the agent-turn boundary. One TurnInvocation per turn; per-task isolated
via _active_turn_var ContextVar.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kaos_agents.base.event import KaosEvent
from kaos_agents.types.tool_call import ToolExecution
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage


@dataclass(slots=True)
class TurnInvocation:
    """Complete record of one agent turn execution.

    Constructed at the top of AgentLoop.forward(), mutated through the
    turn (events appended, usage accumulated, tool_executions added),
    finalized at the bottom by setting output / error and finished_at.

    Read by step methods via current_turn() instead of receiving every
    field through arguments. Per-task isolated via _active_turn_var.

    Forward-referenced fields (trigger, intent, plan, escalations) are
    typed Any in Phase 0 because their Trigger / IntentResult /
    PlanResult / EscalationRequired types ship in later phases. They
    will be tightened to the concrete types as those phases land —
    ContextVar isolation and downstream consumers do not care about the
    exact static type today.
    """

    # Identity
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    session_id: str = ""
    run_id: str = ""
    turn_number: int = 0
    agent_envelope_hash: str = ""

    # Phase-1+ value types (typed Any until those phases land)
    trigger: Any = None  # Trigger — Phase 1
    intent: Any = None  # IntentResult — Phase 1
    plan: Any = None  # PlanResult — Phase 3

    # Outputs (mutated during the turn)
    output: str = ""
    tool_executions: tuple[ToolExecution, ...] = ()
    events: tuple[KaosEvent, ...] = ()
    usage: InvocationUsage = ZERO_USAGE
    cost_usd: float = 0.0
    children: tuple[TurnInvocation, ...] = ()
    escalations: tuple[KaosEvent, ...] = ()  # tightens to EscalationRequired in Phase 4
    error: BaseException | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    finished_at: datetime | None = None

    @property
    def is_complete(self) -> bool:
        return self.finished_at is not None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def add_event(self, event: KaosEvent) -> None:
        """Append an event to this turn's stream. Mutates in place."""
        self.events = (*self.events, event)

    def add_child(self, child: TurnInvocation) -> None:
        """Append a nested delegation invocation. Mutates in place.

        Parent usage and cost are accumulated automatically so the
        parent TurnSummary reflects the full subtree.
        """
        self.children = (*self.children, child)
        self.usage = self.usage + child.usage
        self.cost_usd = self.cost_usd + child.cost_usd

    def finalize(
        self,
        *,
        output: str = "",
        error: BaseException | None = None,
    ) -> None:
        """Stamp finished_at and set the terminal output or error."""
        if self.finished_at is not None:
            return  # idempotent
        self.output = output
        self.error = error
        self.finished_at = datetime.now(UTC)


# Per-task active TurnInvocation. Set at the top of AgentLoop.forward(),
# reset in finally. Step methods read it via current_turn() instead of
# threading the invocation through every argument.
_active_turn_var: contextvars.ContextVar[TurnInvocation | None] = contextvars.ContextVar(
    "kaos_agents_active_turn",
    default=None,
)


def current_turn() -> TurnInvocation | None:
    """Return the active TurnInvocation for this task, or None."""
    return _active_turn_var.get()


__all__ = ["TurnInvocation", "_active_turn_var", "current_turn"]
