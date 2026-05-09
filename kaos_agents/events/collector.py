"""Event collectors — contextvar-based event collection for agent turns.

Mirrors kaos_llm_core.observability.collectors but for KaosEvent. The
audit found that today every Span emitted by EventEmitter has
parent_span_id=None (no caller threads it through), and that sub-agent
runs discard the entire child event stream. The collector solves both:

1. Push-based collection: AgentLoop.forward() opens a collect_events()
   scope; every emit() in the inner work pushes the event to the
   collector. Sub-agent runs open child scopes that inherit the parent's
   top span_id, so events from a HierarchicalPlanner's sub-agents
   merge into the parent turn's stream automatically.

2. parent_span_id synthesis: when EventEmitter constructs a Span and
   no explicit parent_span_id is given, it reads the top of the
   collector's span stack — making OTel parenting actually flow without
   threading parent_span_id through every call site.

Collectors nest. The inner collector captures its own events and the
parent collector receives whatever the inner collector chooses to
surface (typically nothing — the inner collector lives on a child
TurnInvocation, and only that invocation's `events` tuple is forwarded
when `add_child` is called on the parent).

Phase 0.B ships the infrastructure. Wiring AgentLoop / Runner to use
it is Phase 2 work; today's BaseAgent.run() and Runner.run() are
unaffected because they never open a collector scope.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from kaos_agents.base.event import KaosEvent
from kaos_agents.events.spans import Span, SpanPhase


@dataclass(slots=True)
class EventCollector:
    """A list-like accumulator with a span stack for parent_span_id resolution."""

    events: list[KaosEvent] = field(default_factory=list)
    # Stack of span_ids for currently-open spans (START seen, COMPLETE/ERROR not yet).
    # The top of the stack is the parent for the next emitted Span.
    _span_stack: list[str] = field(default_factory=list)

    def append(self, event: KaosEvent) -> None:
        self.events.append(event)
        # Maintain the span stack so EventEmitter can read the current parent.
        if not isinstance(event, Span):
            return
        if event.phase is SpanPhase.START:
            self._span_stack.append(event.span_id)
        elif event.phase in (SpanPhase.COMPLETE, SpanPhase.ERROR) and (
            event.span_id in self._span_stack
        ):
            # Pop everything above-and-including this span_id —
            # tolerant of out-of-order completions.
            idx = self._span_stack.index(event.span_id)
            del self._span_stack[idx:]

    def current_parent_span_id(self) -> str | None:
        """Return the span_id at the top of the stack, or None."""
        return self._span_stack[-1] if self._span_stack else None

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[KaosEvent]:
        return iter(self.events)


# Active collector for the current async task. None means no collector is
# active and emit() should not push (preserves backward compat).
_active_collector_var: contextvars.ContextVar[EventCollector | None] = contextvars.ContextVar(
    "kaos_agents_event_collector", default=None
)


@contextmanager
def collect_events(
    *,
    parent_span_id: str | None = None,
) -> Iterator[EventCollector]:
    """Open a fresh event collector for the duration of the with-block.

    Yields an EventCollector. Nested collect_events contexts are
    independent — each captures only events emitted while it is the
    innermost active collector. Pass parent_span_id to seed the span
    stack so the first emitted Span gets the parent (used by sub-agent
    delegation in Phase 4).

    Usage::

        with collect_events() as collector:
            await some_program.forward(...)
        # collector.events now holds every KaosEvent emitted inside, in
        # emission order, with span parenting threaded.
    """
    coll = EventCollector()
    if parent_span_id is not None:
        coll._span_stack.append(parent_span_id)
    token = _active_collector_var.set(coll)
    try:
        yield coll
    finally:
        _active_collector_var.reset(token)


def push_event(event: KaosEvent) -> None:
    """Append an event to the active collector if one is in scope.

    Called by EventEmitter.emit() after construction. No-op when no
    collector is active.
    """
    coll = _active_collector_var.get()
    if coll is None:
        return
    coll.append(event)


def active_collector() -> EventCollector | None:
    """Return the active collector, or None if no collector is in scope.

    Used by EventEmitter to read the current parent_span_id when
    synthesizing a Span without an explicit parent.
    """
    return _active_collector_var.get()


__all__ = [
    "EventCollector",
    "active_collector",
    "collect_events",
    "push_event",
]
