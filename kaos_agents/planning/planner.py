"""Planner Protocol — Phase 3 contract for the AgentLoop's
plan-and-execute step.

Phase 2.B :class:`~kaos_agents.loop.agent_loop.AgentLoop.forward` already
calls a duck-typed planner::

    plan_obj = await self._planner.plan(plan.intent, plan.memory)
    exec_result = await self._planner.execute(
        plan_obj, perceiver=plan.perceiver, actor=plan.actor,
    )

Phase 3.A formalises that contract as a :class:`Protocol` so the three
concrete planners (:class:`~kaos_agents.planning.react_planner.ReActPlanner`,
PlanExecutePlanner, HierarchicalPlanner — Phase 3.B / 3.C) are
interchangeable. The AgentLoop reads this Protocol; planner subclasses
implement it.

Resolved Decision §13 #3 — *classifier picks at runtime*: the AgentLoop
selects a planner instance based on
:attr:`~kaos_agents.intent.types.IntentResult.pattern` when no explicit
``planner=`` was passed at construction. That selection logic lives in
Phase 3.D (AgentLoop wiring); Phase 3.A only ships the Protocol + value
types + the first concrete planner (:class:`ReActPlanner`).

The :class:`PlanResult` shape matches the duck-typed contract Phase 2.B
:class:`AgentLoop` reads — both ``text`` and ``output`` are populated
to the same value so
``getattr(exec_result, "text", None) or getattr(exec_result, "output", "")``
works regardless of which attribute the caller probes first.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from kaos_agents.intent.types import IntentResult
from kaos_agents.types.tool_call import ToolExecution
from kaos_agents.types.usage import ZERO_USAGE, InvocationUsage


class Plan(BaseModel):
    """Output of :meth:`Planner.plan`. Subclassed by each planner type.

    Carries the strategy discriminator (``pattern``) and free-form
    metadata. Concrete planners typically subclass :class:`Plan` to add
    typed fields (e.g. ``PlanExecutePlanner`` adds a ``PlanGraph``;
    :class:`~kaos_agents.planning.react_planner.ReActPlan` adds a
    goal statement + tool hints).

    ``extra="allow"`` so subclasses can add typed attributes without
    repeating the model_config dance for every value-type extension.
    ``frozen=True`` so a plan instance survives concurrent
    inspection / persistence without races.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    pattern: str  # "react" | "plan_execute" | "hierarchical"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanResult(BaseModel):
    """Output of :meth:`Planner.execute`.

    Compatible with the duck-typed contract Phase 2.B
    :class:`AgentLoop` reads:
    ``getattr(exec_result, "text", None) or getattr(exec_result, "output", "")``
    — both ``text`` and ``output`` are populated to the same value to
    reduce friction.

    Frozen so callers can safely cache or pass across awaits;
    ``extra="allow"`` so subclasses can attach planner-specific
    extras without reissuing model_config.
    """

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    text: str = ""
    output: str = ""  # alias for text — Phase 2.B reads either
    tool_executions: tuple[ToolExecution, ...] = ()
    usage: InvocationUsage = ZERO_USAGE
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Planner(Protocol):
    """Protocol for a Planner — implements :meth:`plan` and :meth:`execute`.

    The :class:`~kaos_agents.loop.agent_loop.AgentLoop` calls
    :meth:`plan` first to produce a typed :class:`Plan` object, then
    :meth:`execute` to run it. Implementations may use any internal
    strategy (single LLM call, ReAct loop, plan-and-execute over
    LoopRunner, hierarchical delegation, etc.).

    Both methods are async. Both may raise; the AgentLoop catches
    exceptions and tags them onto the partial :class:`TurnInvocation`
    per Phase 0.A's error-path discipline.
    """

    async def plan(
        self,
        intent: IntentResult,
        memory: Any | None = None,
    ) -> Plan:
        """Produce a typed :class:`Plan` from the classified intent."""
        ...

    async def execute(
        self,
        plan: Plan,
        *,
        perceiver: Any | None = None,
        actor: Any | None = None,
    ) -> PlanResult:
        """Run a typed :class:`Plan` and return its result."""
        ...


__all__ = ["Plan", "PlanResult", "Planner"]
