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
from kaos_agents.types.budget import UNLIMITED_BUDGET, CostBudget
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

    PA12 — the internal cost accumulator is now a :class:`CostBudget`
    (``cost_budget``) instead of a bare ``float``. ``cost_usd`` is
    still exposed as a top-level attribute for backwards compatibility
    with every event emitter and TurnSummary builder, but its value is
    read from ``cost_budget.spent_usd`` on each mutation so the typed
    value is the source of truth. A bare-float overwrite (``inv.cost_usd
    = X``) still works for old call sites, and is mirrored into the
    backing budget by the ``__setattr__`` hook below.
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
    # PA12 — backing CostBudget (frozen value type). Public ``cost_usd``
    # below is a mirror of ``cost_budget.spent_usd`` so existing callers
    # that read the float still work. Defaults to ``UNLIMITED_BUDGET``
    # (cap = inf) when the turn isn't running under a hard cost cap.
    cost_budget: CostBudget = UNLIMITED_BUDGET
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

        PA12 — the cost accumulation goes through the typed
        :class:`CostBudget` (``self.cost_budget = self.cost_budget.spend(...)``)
        rather than a raw ``+=`` on a float. The float mirror
        ``self.cost_usd`` is kept in sync via the ``__setattr__`` hook
        so existing readers keep working. The wire-surface
        :attr:`AgentResponse.cost_usd` and ``TurnSummary.cost_usd``
        contracts (Sprint-3 #10) are unaffected — they continue to
        read the float.
        """
        self.children = (*self.children, child)
        self.usage = self.usage + child.usage
        # PA12 — typed accumulation. The bare-float mirror is updated
        # by the cost_budget setattr-hook so external readers of
        # ``inv.cost_usd`` keep working.
        self.cost_budget = self.cost_budget.spend(child.cost_usd)

    def spend(self, amount_usd: float) -> None:
        """Charge ``amount_usd`` against the turn's cost budget.

        PA12 — typed counterpart to the historical ``inv.cost_usd +=
        amount`` pattern. Use this from emitter / hook code so the
        cost flows through :class:`CostBudget` (and downstream
        budget-aware policies like
        :class:`~kaos_agents.runtime.escalation.DefaultEscalationPolicy`
        see consistent ``fraction_spent`` / ``exceeded`` signals).
        """
        self.cost_budget = self.cost_budget.spend(amount_usd)

    @property
    def budget_exceeded(self) -> bool:
        """True when the turn's :class:`CostBudget` has been
        exhausted. ``UNLIMITED_BUDGET`` (the default) always returns
        False — only a turn constructed with an explicit finite cap
        can report exceeded."""
        return self.cost_budget.exceeded

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

    def __setattr__(self, name: str, value: Any) -> None:
        """PA12 — keep ``cost_usd`` and ``cost_budget.spent_usd`` in sync.

        The two-way mirror exists because:

        1. ``cost_usd: float`` is a long-standing public attribute
           read by emitters, TurnSummary builders, and tests.
        2. ``cost_budget: CostBudget`` is the new typed source of
           truth (PA12).

        When a caller writes ``cost_usd`` directly we mirror into the
        budget; when ``cost_budget`` is replaced we mirror its
        ``spent_usd`` into the float. Either path keeps the two
        attributes coherent so downstream readers never see drift.
        """
        if name == "cost_usd":
            # Bare-float write — mirror into the typed budget. Keep
            # the same cap so a write doesn't accidentally drop the
            # ceiling.
            object.__setattr__(self, "cost_usd", float(value))
            # Don't recurse: read cost_budget via object.__getattribute__
            # to avoid the slot-init bootstrap chicken-and-egg.
            try:
                current = object.__getattribute__(self, "cost_budget")
            except AttributeError:
                current = UNLIMITED_BUDGET
            if current.spent_usd != float(value):
                object.__setattr__(
                    self,
                    "cost_budget",
                    CostBudget(total_usd=current.total_usd, spent_usd=float(value)),
                )
            return
        if name == "cost_budget":
            object.__setattr__(self, "cost_budget", value)
            # Mirror spent into the public float.
            object.__setattr__(self, "cost_usd", float(value.spent_usd))
            return
        object.__setattr__(self, name, value)


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
