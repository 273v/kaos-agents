"""Plan-related payloads + the one plan event that isn't a span.

Step lifecycle (start / complete / error) is now :class:`Span` events
with ``subject=SpanSubject.STEP``. This module retains:

- :class:`PlanStepSummary` — the per-step row that ships inside
  :class:`PlanProposed.steps`.
- :class:`PlanProposed` — fires once per plan, *before* execution
  begins. It's a fact (the plan content), not a phase marker, so it
  keeps a typed class.
"""

from __future__ import annotations

from kaos_core.types.content import KaosModel
from pydantic import ConfigDict

from kaos_agents.events._intermediates import LifecycleEvent


class PlanStepSummary(KaosModel):
    """Compact summary of a plan step for :class:`PlanProposed`."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    step_id: str
    description: str
    tool_name: str | None = None


class PlanProposed(LifecycleEvent):
    """Emitted when the planner produces a plan before execution.

    Carries the proposed step list and the strategy that produced it.
    Consumers (UI, replay tooling) render the plan from this event;
    individual step phases are :class:`Span` events.
    """

    steps: tuple[PlanStepSummary, ...] = ()
    strategy: str = ""  # "direct", "decompose", "rolling", "adaptive"


__all__ = [
    "PlanProposed",
    "PlanStepSummary",
]
