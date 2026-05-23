"""Tests for Runner.context_factory per-session KaosContext injection.

The Runner's ``context_factory`` parameter lets hosts that scope their
on-disk VFS layout per session or per tenant (e.g. the single-user-chat
SPA's ``sessions/{tenant}:{sid}/files/`` layout) build a fresh
``KaosContext`` for every ``run`` / ``turn`` / ``delegate`` / ``handoff``
entry. Without this, the SPA had to monkey-patch
``Runner._build_internal_agent`` — that workaround is fragile and the
0.1.13 release removes the need for it.

These tests cover the three branches of ``Runner._resolve_context``:

1. With ``context_factory`` set + a session_id, the factory is called.
2. With ``context_factory`` set but ``session_id=None``, fall through
   to the Runner-level context.
3. Without ``context_factory``, the Runner-level context is always
   returned unchanged regardless of session_id.

The tests use ``KaosRuntime.test_mode()`` per the CLAUDE.md isolation
pattern so no on-disk state leaks across runs.
"""

from __future__ import annotations

from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime

from kaos_agents.config import Agent
from kaos_agents.runtime.runner import Runner


def _build_agent() -> Agent:
    """Construct a minimal Agent suitable for Runner construction."""
    return Agent(instructions="Test agent.")


def test_context_factory_invoked_with_session_id() -> None:
    """When a factory is supplied, _resolve_context builds a per-session context."""
    runtime = KaosRuntime.test_mode()
    agent = _build_agent()

    captured: list[str] = []

    def factory(session_id: str) -> KaosContext:
        captured.append(session_id)
        return KaosContext(
            session_id=session_id,
            default_vfs_namespace=f"sessions/{session_id}/files/",
        )

    runner = Runner(agent, runtime=runtime, context_factory=factory)

    ctx = runner._resolve_context("01ABC")

    assert ctx is not None
    assert ctx.session_id == "01ABC"
    # KaosContext normalizes the namespace; assert the prefix matches.
    assert ctx.default_vfs_namespace.startswith("sessions/01ABC/files")
    # Factory was invoked exactly once with the session_id we passed.
    assert captured == ["01ABC"]


def test_context_factory_fallback_when_session_id_is_none() -> None:
    """A None session_id never calls the factory; the Runner-level context is used."""
    runtime = KaosRuntime.test_mode()
    agent = _build_agent()

    sentinel_ctx = KaosContext(
        session_id="runner-level",
        default_vfs_namespace="runner-level/files/",
    )

    def factory(session_id: str) -> KaosContext:
        raise AssertionError(
            "context_factory must NOT be called when session_id is None; "
            f"got session_id={session_id!r}"
        )

    runner = Runner(
        agent,
        runtime=runtime,
        context=sentinel_ctx,
        context_factory=factory,
    )

    resolved = runner._resolve_context(None)

    assert resolved is sentinel_ctx


def test_resolve_context_without_factory_returns_runner_level_context() -> None:
    """Without a factory, _resolve_context always returns self._context."""
    runtime = KaosRuntime.test_mode()
    agent = _build_agent()

    sentinel_ctx = KaosContext(
        session_id="runner-level",
        default_vfs_namespace="runner-level/files/",
    )

    runner = Runner(agent, runtime=runtime, context=sentinel_ctx)

    # With a session_id, still falls through (no factory installed).
    assert runner._resolve_context("01ABC") is sentinel_ctx
    # With None session_id, same fall-through.
    assert runner._resolve_context(None) is sentinel_ctx


def test_resolve_context_without_factory_or_context_returns_none() -> None:
    """Default Runner (no factory, no context) yields None — preserves legacy behaviour."""
    runtime = KaosRuntime.test_mode()
    agent = _build_agent()

    runner = Runner(agent, runtime=runtime)

    assert runner._resolve_context("01ABC") is None
    assert runner._resolve_context(None) is None
