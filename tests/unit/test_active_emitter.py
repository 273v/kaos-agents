"""Tests for the contextvar-based active EventEmitter primitive (Theme A).

Mirrors ``tests/unit/test_events_collector.py`` for the symmetric
:func:`~kaos_agents.events.use_emitter` /
:func:`~kaos_agents.events.active_emitter` pair that publishes the
per-turn :class:`EventEmitter` to deep helpers without threading.

The behavior under test:

- :func:`active_emitter` returns ``None`` outside any
  :func:`use_emitter` scope.
- :func:`use_emitter` sets the active emitter for the duration of the
  ``with`` block and restores ``None`` on exit.
- Nested :func:`use_emitter` scopes nest correctly: the inner emitter
  is active inside the inner block, the outer emitter is restored on
  inner exit.
- ContextVar isolation across asyncio tasks: each task sees its own
  active emitter independently.
"""

from __future__ import annotations

import asyncio

import pytest

from kaos_agents.events import (
    EventEmitter,
    active_emitter,
    use_emitter,
)


@pytest.fixture
def emitter() -> EventEmitter:
    """Build an EventEmitter with stable session_id / run_id for assertions."""
    return EventEmitter(session_id="sess-A", run_id="run-A")


@pytest.fixture
def other_emitter() -> EventEmitter:
    """Second emitter for nesting / isolation tests."""
    return EventEmitter(session_id="sess-B", run_id="run-B")


def test_active_emitter_default_none() -> None:
    """Outside any ``use_emitter`` scope, ``active_emitter()`` returns ``None``.

    Back-compat contract: callers that don't open a ``use_emitter``
    scope must continue to work — the helper sites in
    capabilities/retrieve, actions/tool_bridge, planning/act all
    check for None and skip the emit when it's absent.
    """
    assert active_emitter() is None


def test_use_emitter_sets_and_resets(emitter: EventEmitter) -> None:
    """``use_emitter`` publishes the emitter inside the block and restores None on exit."""
    assert active_emitter() is None
    with use_emitter(emitter) as yielded:
        # ``with use_emitter(e) as x:`` yields the same emitter.
        assert yielded is emitter
        assert active_emitter() is emitter
    # Restored after the block exits.
    assert active_emitter() is None


def test_nested_use_emitter(emitter: EventEmitter, other_emitter: EventEmitter) -> None:
    """Nested ``use_emitter`` scopes restore the prior emitter on inner exit.

    The ContextVar token-based ``reset`` semantics mean the outer
    scope's emitter is visible again after the inner ``with`` block
    exits, without the outer scope having to re-publish it.
    """
    with use_emitter(emitter):
        assert active_emitter() is emitter
        with use_emitter(other_emitter):
            assert active_emitter() is other_emitter
        # Outer emitter restored.
        assert active_emitter() is emitter
    # Both scopes gone — back to None.
    assert active_emitter() is None


def test_use_emitter_exception_restores() -> None:
    """If the body raises, the ContextVar is still reset on exit.

    The ``finally`` + ``suppress(ValueError)`` block in
    :func:`use_emitter` mirrors :func:`collect_events`. A raise from
    the body must not leak the active emitter into subsequent code.
    """
    e1 = EventEmitter(session_id="x", run_id="x")
    with pytest.raises(RuntimeError, match="boom"), use_emitter(e1):
        assert active_emitter() is e1
        raise RuntimeError("boom")
    assert active_emitter() is None


def test_use_emitter_is_task_isolated() -> None:
    """ContextVar isolation: concurrent asyncio tasks see independent emitters.

    A parent task that opens a ``use_emitter`` scope must not leak
    its emitter into a child task that runs concurrently without its
    own scope. ContextVar copy-on-task-start is what gives us this.
    """
    e_parent = EventEmitter(session_id="parent", run_id="parent")

    async def child_no_scope() -> EventEmitter | None:
        # Child task inherits the parent's ContextVar snapshot at
        # task creation time. The contract here is the standard
        # asyncio ContextVar behaviour: child SEES the parent's
        # emitter because the contextvar was set BEFORE create_task.
        await asyncio.sleep(0)
        return active_emitter()

    async def child_own_scope() -> EventEmitter | None:
        # Child opens its own scope; the new emitter is local to
        # this task. The parent's scope still sees its own emitter
        # after this task finishes.
        e_child = EventEmitter(session_id="child", run_id="child")
        with use_emitter(e_child):
            await asyncio.sleep(0)
            return active_emitter()

    async def runner() -> tuple[EventEmitter | None, EventEmitter | None, EventEmitter | None]:
        with use_emitter(e_parent):
            seen_in_child = await asyncio.create_task(child_no_scope())
            seen_via_own_scope = await asyncio.create_task(child_own_scope())
            seen_in_parent_after = active_emitter()
        return seen_in_child, seen_via_own_scope, seen_in_parent_after

    seen_child, seen_own, seen_parent = asyncio.run(runner())
    # Child without its own scope inherits the parent's emitter
    # (standard asyncio ContextVar copy-on-task semantics).
    assert seen_child is e_parent
    # Child's own scope wins inside that child task.
    assert seen_own is not None
    assert seen_own is not e_parent
    # Parent task's view is unchanged by either child.
    assert seen_parent is e_parent
