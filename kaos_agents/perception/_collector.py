"""Private event collector for Phase 1.B.

The plan refers to ``kaos_agents.events.collector.{collect_events,
push_event}`` as a Phase 0 deliverable. Phase 0 has not landed in
this branch, so Phase 1.B owns a small private collector here. When
Phase 0 lands, this module will be replaced by re-exports from
:mod:`kaos_agents.events.collector`.

The collector follows the kaos-llm-core ``collect_traces`` pattern:
a ``ContextVar`` holds the stack of active collectors; ``push_event``
appends to the innermost active collector and is a no-op when none is
active.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# ContextVar holding the stack of active collectors (a list, with the
# innermost-active collector at the end). A list-of-lists is required
# so nested ``with collect_events()`` blocks compose without losing
# the outer scope's events.
_active_collectors: contextvars.ContextVar[list[list[Any]] | None] = contextvars.ContextVar(
    "_perception_active_collectors", default=None
)


@contextmanager
def collect_events() -> Iterator[list[Any]]:
    """Open a context that captures events pushed via :func:`push_event`.

    Usage::

        with collect_events() as events:
            await rag.invoke(...)
        # events is now a list of CitationFound / etc. instances.

    Nested ``collect_events`` blocks each get their own list — each
    ``push_event`` appends only to the innermost active list.
    """
    stack = _active_collectors.get()
    if stack is None:
        stack = []
        _active_collectors.set(stack)
    bucket: list[Any] = []
    stack.append(bucket)
    try:
        yield bucket
    finally:
        # Pop only the bucket we pushed — defensive against unexpected
        # mutation of the stack from nested code paths.
        if stack and stack[-1] is bucket:
            stack.pop()


def push_event(event: Any) -> None:
    """Push an event into the innermost active collector.

    No-op when no :func:`collect_events` block is active — preserves
    backward-compat for direct Python callers that don't care about
    the event stream.
    """
    stack = _active_collectors.get()
    if not stack:
        return
    stack[-1].append(event)


__all__ = ["collect_events", "push_event"]
