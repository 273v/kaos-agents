"""Unit tests for the session-isolation architecture pass.

Covers four design errors closed in 0.1.17 — the kaos-agents tooling
historically wrapped kaos-core APIs without preserving the per-call
``KaosContext.session_id`` boundary that kaos-core was designed around.
The tests here pin the four corrections so the regression class
(``#569`` / ``#559`` / ``#582`` / B0.1 family) cannot recur silently.

Each finding has its own ``TestFindingN`` class. Tests use real
``KaosRuntime`` + in-memory VFS — no mocks — so they exercise the
guard at the kaos-core layer that kaos-agents now correctly invokes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.runtime.artifact_hydration import hydrate_artifacts_from_message
from kaos_agents.tools.registry import (
    AgentChatTool,
    AgentMemoryClearTool,
    AgentMemoryQueryTool,
    AgentMemorySearchTool,
    AgentPlanTool,
    _build_invocation_context,
)
from kaos_agents.types.memory import MemoryType

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime() -> KaosRuntime:
    """Real KaosRuntime with isolated in-memory VFS — no mocks."""
    vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
    return KaosRuntime(vfs=vfs)


async def _write_artifact_for_session(
    runtime: KaosRuntime,
    *,
    owner_session_id: str,
    name: str,
    body: bytes,
) -> str:
    """Plant a VFS artifact owned by ``owner_session_id``."""
    path = f"/artifacts/{name}"
    await runtime.vfs.write(path, body, context_id=owner_session_id)
    manifest = await runtime.artifacts.create_from_path(
        path,
        context_id=owner_session_id,
        session_id=owner_session_id,
        name=name,
        mime_type="text/plain",
    )
    return manifest.artifact_id


async def _seed_memory(
    runtime: KaosRuntime,
    *,
    session_id: str,
    messages: Sequence[str] = (),
) -> SessionMemory:
    """Create + persist a SessionMemory for ``session_id`` with optional messages."""
    store = SessionStore(runtime.vfs)
    memory = await store.load_or_create(session_id)
    for m in messages:
        memory.add(MemoryType.MESSAGES, m)
    await store.save(memory)
    return memory


async def _drive_invocation_tool(
    tool: object,
    *,
    inputs: dict[str, object],
    context: KaosContext | None,
) -> dict[str, object]:
    """Drive ``AgentChatTool`` / ``AgentPlanTool`` ``.execute`` with the
    ``Runner`` symbol monkey-patched to a stub that records the kwargs it
    was constructed with (no LLM / runtime cost).

    Returns the captured kwargs dict so a test can assert what the tool
    actually handed to ``Runner(...)``. Used by the Finding 5 tests to
    pin the B0.1-invocation fix without exercising a live model.
    """
    import kaos_agents.runtime.runner as runner_mod

    captured: dict[str, object] = {}

    class _StubRunner:
        def __init__(self, _agent, *, runtime=None, context=None, **_kwargs):
            captured["runner_context"] = context
            captured["runner_runtime"] = runtime

        async def run(self, _message, _session_id, **_kwargs):
            # Empty async generator — the runner_turn helper folds zero
            # events into an empty AgentResponse so we exercise the
            # full tool wrapper without an LLM call.
            return
            yield  # pragma: no cover — unreachable; makes ``run`` a generator

    real_runner = runner_mod.Runner
    runner_mod.Runner = _StubRunner  # ty: ignore[invalid-assignment]
    try:
        result = await tool.execute(inputs, context=context)  # ty: ignore[unresolved-attribute]
    finally:
        runner_mod.Runner = real_runner
    captured["result_is_error"] = bool(getattr(result, "isError", False))
    return captured


# ===========================================================================
# Finding 1 — artifact hydration must enforce caller-session ownership
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestFinding1ArtifactHydrationIsolation:
    """``hydrate_artifacts_from_message`` must NOT return another
    session's artifact body even when the calling session can name its
    artifact_id. Closes finding #1.
    """

    async def test_cross_session_hydration_returns_no_artifacts(self):
        """Owner session A; caller session B. Reference works (the ID
        is in the message) but the hydrated tuple is empty — the
        kaos-core ownership guard refused at resolve."""
        runtime = _make_runtime()
        owner_artifact_id = await _write_artifact_for_session(
            runtime,
            owner_session_id="acme-corp-session",
            name="confidential.txt",
            body=b"PRIVILEGED: client strategy memo",
        )

        # Caller in a DIFFERENT session asks to hydrate the artifact.
        caller_memory = SessionMemory(session_id="beta-corp-session")
        message = f"summarize kaos://artifacts/{owner_artifact_id}"

        hydrated = await hydrate_artifacts_from_message(
            message,
            memory=caller_memory,
            runtime=runtime,
            caller_session_id="beta-corp-session",
        )

        assert hydrated == ()
        # Caller's DOCUMENTS section must be empty — no leak.
        docs = caller_memory.get_recent(MemoryType.DOCUMENTS, 10)
        assert docs == []

    async def test_same_session_hydration_succeeds(self):
        """The owner session can still hydrate its own artifact —
        the new ownership check doesn't break the happy path."""
        runtime = _make_runtime()
        owner_artifact_id = await _write_artifact_for_session(
            runtime,
            owner_session_id="acme-corp-session",
            name="own.txt",
            body=b"OWN content: visible to acme",
        )

        owner_memory = SessionMemory(session_id="acme-corp-session")
        message = f"summarize kaos://artifacts/{owner_artifact_id}"

        hydrated = await hydrate_artifacts_from_message(
            message,
            memory=owner_memory,
            runtime=runtime,
            caller_session_id="acme-corp-session",
        )

        assert len(hydrated) == 1
        assert hydrated[0].artifact_id == owner_artifact_id

    async def test_no_caller_session_id_uses_trusted_path(self):
        """Trusted in-process callers (cleanup, migration) pass
        ``caller_session_id=None`` and intentionally bypass the
        ownership check. This is the escape hatch — preserve it."""
        runtime = _make_runtime()
        owner_artifact_id = await _write_artifact_for_session(
            runtime,
            owner_session_id="acme-corp-session",
            name="trusted.txt",
            body=b"trusted callers may read",
        )

        trusted_memory = SessionMemory(session_id="background-worker")
        message = f"summarize kaos://artifacts/{owner_artifact_id}"

        # caller_session_id=None (default) -> guard bypassed
        hydrated = await hydrate_artifacts_from_message(
            message,
            memory=trusted_memory,
            runtime=runtime,
        )

        assert len(hydrated) == 1
        assert hydrated[0].artifact_id == owner_artifact_id


# ===========================================================================
# Finding 2 — memory tools must scope to the caller's context.session_id
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestFinding2MemoryToolsScopeToContext:
    """The three agent memory MCP tools must never let a caller scoped
    to session A read / clear / search session B's memory by passing
    B's id as a string input. Closes finding #2.
    """

    async def test_memory_query_refuses_cross_session_id(self):
        runtime = _make_runtime()
        await _seed_memory(
            runtime,
            session_id="victim-session",
            messages=("victim secret message",),
        )
        attacker_ctx = KaosContext(session_id="attacker-session", runtime=runtime)

        tool = AgentMemoryQueryTool()
        result = await tool.execute(
            {"session_id": "victim-session", "section": "messages"},
            context=attacker_ctx,
        )

        # Refusal MUST surface as an error result — not silent success
        # on the victim's data.
        assert result.isError is True
        text = getattr(result.content[0], "text", "") if result.content else ""
        assert "does not match the calling session" in text
        assert "attacker-session" in text

    async def test_memory_query_accepts_matching_session_id(self):
        runtime = _make_runtime()
        await _seed_memory(
            runtime,
            session_id="acme-session",
            messages=("acme: filed the brief"),
        )
        own_ctx = KaosContext(session_id="acme-session", runtime=runtime)

        tool = AgentMemoryQueryTool()
        result = await tool.execute(
            {"session_id": "acme-session", "section": "messages"},
            context=own_ctx,
        )
        assert result.isError is False

    async def test_memory_query_uses_context_when_input_omitted(self):
        """Caller need not pass session_id — context is authoritative."""
        runtime = _make_runtime()
        await _seed_memory(
            runtime,
            session_id="acme-session",
            messages=("acme stuff",),
        )
        own_ctx = KaosContext(session_id="acme-session", runtime=runtime)

        tool = AgentMemoryQueryTool()
        result = await tool.execute({"section": "messages"}, context=own_ctx)
        assert result.isError is False

    async def test_memory_clear_refuses_cross_session_id(self):
        """The DESTRUCTIVE tool must also refuse cross-session writes.
        A confused caller asking to clear a sibling session was the
        worst-case path in finding #2 (rm -rf the wrong tenant)."""
        runtime = _make_runtime()
        await _seed_memory(
            runtime,
            session_id="victim-session",
            messages=("important",),
        )
        attacker_ctx = KaosContext(session_id="attacker-session", runtime=runtime)

        tool = AgentMemoryClearTool()
        result = await tool.execute(
            {"session_id": "victim-session"},
            context=attacker_ctx,
        )

        assert result.isError is True
        # And — the proof: victim's data MUST still be there.
        store = SessionStore(runtime.vfs)
        assert await store.exists("victim-session")

    async def test_memory_search_refuses_cross_session_id(self):
        runtime = _make_runtime()
        await _seed_memory(
            runtime,
            session_id="victim-session",
            messages=("search needle",),
        )
        attacker_ctx = KaosContext(session_id="attacker-session", runtime=runtime)

        tool = AgentMemorySearchTool()
        result = await tool.execute(
            {"session_id": "victim-session", "query": "needle"},
            context=attacker_ctx,
        )

        assert result.isError is True

    async def test_memory_query_requires_session_id_without_context(self):
        """CLI / standalone callers (context=None) must still supply
        session_id as input — the helper refuses an empty caller
        scope."""
        runtime = _make_runtime()
        await _seed_memory(runtime, session_id="acme-session")

        tool = AgentMemoryQueryTool()
        # context=None, no inputs.session_id → refusal
        result = await tool.execute({}, context=None)
        assert result.isError is True
        text = getattr(result.content[0], "text", "") if result.content else ""
        assert "no calling-session" in text or "Missing 'session_id'" in text


# ===========================================================================
# Finding 3 — v2 agent loop must build with per-session context
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestFinding3V2LoopUsesPerSessionContext:
    """``Runner._build_agent_loop`` must resolve the per-session
    KaosContext via ``context_factory`` for v2 dispatch, not reuse the
    Runner-level ``self._context``. Closes finding #3.
    """

    async def test_build_agent_loop_uses_context_factory_for_session_id(self):
        """Pass a ``context_factory`` and a ``session_id`` — assert the
        bridged tools receive the factory's per-session context, not
        the Runner-level one."""
        from kaos_agents.config import Agent
        from kaos_agents.runtime.runner import Runner

        runtime = _make_runtime()
        seen_contexts: list[KaosContext | None] = []

        def factory(sid: str) -> KaosContext:
            ctx = KaosContext(session_id=sid, runtime=runtime)
            seen_contexts.append(ctx)
            return ctx

        runner = Runner(
            Agent(),
            runtime=runtime,
            context=KaosContext(session_id="runner-default", runtime=runtime),
            context_factory=factory,
        )

        # _build_agent_loop is the bridge point — call it directly.
        runner._build_agent_loop(session_id="per-call-session")

        # The factory must have been invoked with the per-call id,
        # NOT silently bypassed in favor of the runner-level context.
        assert len(seen_contexts) >= 1
        last = seen_contexts[-1]
        assert last is not None and last.session_id == "per-call-session"

    async def test_build_agent_loop_falls_back_when_no_session_id(self):
        """When no session is in scope (None), the loop builder uses
        the Runner-level context — preserves construction-time and
        trusted-invoke paths."""
        from kaos_agents.config import Agent
        from kaos_agents.runtime.runner import Runner

        runtime = _make_runtime()
        factory_calls: list[str] = []

        def factory(sid: str) -> KaosContext:
            factory_calls.append(sid)
            return KaosContext(session_id=sid, runtime=runtime)

        runner = Runner(
            Agent(),
            runtime=runtime,
            context=KaosContext(session_id="runner-default", runtime=runtime),
            context_factory=factory,
        )

        runner._build_agent_loop(session_id=None)
        # Factory MUST NOT be called for None session_id — that's the
        # trusted / no-session path.
        assert factory_calls == []


# ===========================================================================
# Finding 4 — pause/resume RunState scoped per session
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestFinding4RunStateScopedPerSession:
    """``save_run_state`` / ``load_run_state`` / ``save_event_log`` /
    ``load_event_log`` must scope their VFS reads and writes to the
    session id carried by the RunState. Closes finding #4.

    Under GLOBAL VFS isolation (the kaos-ui SPA default) ``context_id``
    is a no-op at the storage layer — the HTTP approve endpoint's
    post-load tenant check is what enforces the boundary. Under
    NAMESPACE / per-context isolation this scoping is load-bearing.
    These tests use GLOBAL mode (the most permissive case) and
    verify the kwarg plumbing is intact end-to-end — a missing
    kwarg would surface as a ``TypeError`` regardless of mode.
    """

    async def test_save_uses_state_session_id_as_context(self):
        """save_run_state must call vfs.write with
        ``context_id=state.session_id``."""
        from kaos_agents.runtime.interrupts import RunState, save_run_state

        runtime = _make_runtime()
        captured: dict[str, object] = {}
        real_write = runtime.vfs.write

        async def spying_write(path, data, **kwargs):
            captured["path"] = path
            captured["context_id"] = kwargs.get("context_id")
            return await real_write(path, data, **kwargs)

        # Method-assignment is a deliberate test-only monkeypatch: we
        # need to observe the kwargs the production code passes through
        # to ``vfs.write`` (specifically ``context_id=state.session_id``).
        runtime.vfs.write = spying_write  # ty: ignore[invalid-assignment]
        state = RunState(
            run_id="run_isolation_save",
            session_id="acme-session-42",
            original_message="x",
        )

        await save_run_state(state, runtime.vfs)

        assert captured.get("context_id") == "acme-session-42"

    async def test_load_round_trips_under_per_session_scope(self):
        """save -> load round-trips when load is told the right session_id."""
        from kaos_agents.runtime.interrupts import (
            RunState,
            load_run_state,
            save_run_state,
        )

        runtime = _make_runtime()
        state = RunState(
            run_id="run_rt",
            session_id="round-trip-session",
            original_message="hello",
        )
        await save_run_state(state, runtime.vfs)

        loaded = await load_run_state(
            "run_rt",
            runtime.vfs,
            session_id="round-trip-session",
        )
        assert loaded.session_id == "round-trip-session"
        assert loaded.original_message == "hello"

    async def test_event_log_scoped_per_session(self):
        """save_event_log + load_event_log accept session_id kwarg."""
        from kaos_agents.runtime.interrupts import load_event_log, save_event_log

        runtime = _make_runtime()
        # Empty event list still exercises the write path.
        path = await save_event_log(
            (),
            "run_evt",
            runtime.vfs,
            session_id="evt-session",
        )
        assert "run_evt" in path

        loaded = await load_event_log(
            "run_evt",
            runtime.vfs,
            session_id="evt-session",
        )
        assert loaded == []


# ===========================================================================
# Finding 5 — invocation tools (chat/plan) MUST NOT split-brain on
# ``context.session_id`` vs ``inputs["session_id"]``
# ===========================================================================
#
# The security review that opened #B0.1-invocation noted: a caller scoped
# to session "A" (``context.session_id="A"``) calling
# ``AgentChatTool.execute({"session_id": "B", ...})`` previously caused
# memory + artifact hydration to run as B while the Runner's bridged
# runtime tools still saw ``context.session_id == "A"`` — a horizontal
# isolation regression of the same shape that ``_resolve_caller_session_id``
# closed for the memory tools (Finding 2 above).
#
# The fix (``_build_invocation_context``) derives a per-invocation child
# ``KaosContext`` whose ``session_id`` matches the resolver's choice, so
# every downstream boundary (memory writes, hydration's
# ``caller_session_id``, bridged tools' ``context.session_id``, VFS
# namespace lookups, artifact-guard scope) lands on the SAME session id
# for a single turn. We MUST preserve MCP ergonomics (the MCP context's
# ``session_id`` is the caller's ``client_id``, not the agent chat
# session) so a strict refuse is NOT the right answer — the agent-
# invocation surface is explicitly "create / resume the chosen session
# on behalf of the caller". But the chosen session MUST be the only
# scope the agent's machinery sees.


@pytest.mark.unit
class TestFinding5InvocationContextHelper:
    """Sync unit tests for :func:`_build_invocation_context` — the
    helper underpinning the B0.1-invocation fix. Kept separate from
    the async tool-level tests below so the class-level ``asyncio``
    mark doesn't flag the sync methods."""

    def test_build_invocation_context_derives_child_on_mismatch(self):
        """When ``context.session_id != session_id``, the helper returns
        a child context retargeted at the resolved session — preserving
        the runtime and other fields — and stashes the caller's original
        session id under ``metadata["caller_session_id"]`` so audit hooks
        can still log the caller vs target distinction."""
        runtime = _make_runtime()
        attacker_ctx = KaosContext(
            session_id="attacker",
            runtime=runtime,
            metadata={"trace_origin": "mcp-client-X"},
        )

        derived = _build_invocation_context(attacker_ctx, runtime=runtime, session_id="victim")

        assert derived is not None
        assert derived.session_id == "victim"
        # Caller identity preserved for observability — but not as the
        # session_id every downstream resolver consults.
        assert derived.metadata.get("caller_session_id") == "attacker"
        # Parent metadata still propagates.
        assert derived.metadata.get("trace_origin") == "mcp-client-X"
        # Runtime survives the derivation (bridged tools need it).
        assert derived.runtime is runtime

    def test_build_invocation_context_returns_original_when_match(self):
        """When the caller is already scoped to the chosen session, the
        helper hands back the same context object — no allocation and
        no loss of caller-supplied ``default_vfs_namespace`` / trace
        fields."""
        runtime = _make_runtime()
        ctx = KaosContext(session_id="acme", runtime=runtime)
        derived = _build_invocation_context(ctx, runtime=runtime, session_id="acme")
        assert derived is ctx

    def test_build_invocation_context_creates_when_context_none(self):
        """CLI / standalone callers (``context=None``) still get a real
        ``KaosContext`` scoped to the chosen session — bridged tools and
        hydration would otherwise see ``context=None`` and skip
        per-session resolution."""
        runtime = _make_runtime()
        derived = _build_invocation_context(None, runtime=runtime, session_id="acme")
        assert derived is not None
        assert derived.session_id == "acme"
        assert derived.runtime is runtime


@pytest.mark.asyncio
@pytest.mark.unit
class TestFinding5InvocationToolsConsistentSessionScope:
    """Pin the B0.1-invocation fix at the tool boundary: when an
    external caller passes ``inputs["session_id"]`` that differs from
    ``context.session_id``, the Runner the tool builds MUST receive a
    context whose ``session_id`` matches the resolved (input-authoritative)
    chat session. No split-brain.
    """

    async def test_chat_tool_passes_session_scoped_context_to_runner(self):
        """Drive ``AgentChatTool.execute`` with mismatched
        ``context.session_id="attacker"`` and ``inputs.session_id="victim"``.
        Monkey-patch ``Runner`` so we capture what the tool actually
        builds. The captured Runner ``context.session_id`` MUST be
        ``"victim"`` — the same session the resolver picked — so
        bridged runtime tools, hydration's ``caller_session_id``, and
        the Runner-internal ``_resolve_context`` fallback all agree.
        """
        runtime = _make_runtime()
        attacker_ctx = KaosContext(session_id="attacker", runtime=runtime)

        captured = await _drive_invocation_tool(
            AgentChatTool(),
            inputs={"session_id": "victim", "message": "hello"},
            context=attacker_ctx,
        )

        # No infrastructure surfacing error — the stub yields no events
        # so the tool returns an empty-text success envelope.
        assert captured["result_is_error"] is False
        derived_ctx = captured["runner_context"]
        assert isinstance(derived_ctx, KaosContext)
        # THE CORE INVARIANT — the Runner saw the victim session, not
        # the attacker's caller identity.
        assert derived_ctx.session_id == "victim"
        assert derived_ctx.metadata.get("caller_session_id") == "attacker"
        # The runtime threaded through correctly so bridged tools find it.
        assert captured["runner_runtime"] is runtime

    async def test_plan_tool_passes_session_scoped_context_to_runner(self):
        """Same invariant as the chat test, applied to ``AgentPlanTool``
        — sister tool, same call shape, same B0.1-invocation hazard.
        """
        runtime = _make_runtime()
        attacker_ctx = KaosContext(session_id="attacker", runtime=runtime)

        captured = await _drive_invocation_tool(
            AgentPlanTool(),
            inputs={"session_id": "victim", "message": "do the thing"},
            context=attacker_ctx,
        )

        assert captured["result_is_error"] is False
        derived_ctx = captured["runner_context"]
        assert isinstance(derived_ctx, KaosContext)
        assert derived_ctx.session_id == "victim"

    async def test_chat_tool_preserves_context_when_session_matches(self):
        """Positive path: when ``inputs.session_id == context.session_id``,
        the tool hands the original context straight through to Runner —
        the helper's no-op fast path is intact. Verifies the B0.1 fix
        didn't accidentally re-allocate contexts on the legitimate
        happy path."""
        runtime = _make_runtime()
        own_ctx = KaosContext(session_id="acme", runtime=runtime)

        captured = await _drive_invocation_tool(
            AgentChatTool(),
            inputs={"session_id": "acme", "message": "hi"},
            context=own_ctx,
        )

        assert captured["result_is_error"] is False
        # Identity preservation: same object, no child allocation.
        assert captured["runner_context"] is own_ctx
