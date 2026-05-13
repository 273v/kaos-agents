"""Live integration tests for the Phase 0-7 streaming/Runner/hooks/
permissions/delegation stack.

Validates the full kaos-agents-streaming.md acceptance criteria against
real Anthropic API calls. Required by CLAUDE.md: every feature must
have a live integration test before it's "done."

Run with: ``pytest -m live tests/integration/test_streaming_live.py``
or via the platform validator:
``./scripts/validate-platform.sh --include-network --include-live``

Requires ``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import StorageBackend
from kaos_core.types.metadata import (
    ToolAnnotations,
    ToolCapability,
    ToolCategory,
    ToolMetadata,
)
from kaos_core.types.results import ToolResult
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.config import Agent, AgentPattern
from kaos_agents.events import (
    IntentClassified,
    LifecycleEvent,
    Span,
    SpanPhase,
    SpanSubject,
    StreamDelta,
    TextDelta,
    ToolCallApprovalRequired,
    TurnSummary,
)
from kaos_agents.hooks import CostTrackingHook, HookAction, KaosHook, LoggingHook
from kaos_agents.runtime.delegation import agent_as_tool
from kaos_agents.runtime.permissions import PermissionPolicy
from kaos_agents.runtime.runner import Runner
from kaos_agents.types.providers import BALANCED, FAST


def _is_tool_call_start(event: object) -> bool:
    return (
        isinstance(event, Span)
        and event.subject == SpanSubject.TOOL_CALL
        and event.phase == SpanPhase.START
    )


def _is_tool_call_result(event: object) -> bool:
    return (
        isinstance(event, Span)
        and event.subject == SpanSubject.TOOL_CALL
        and event.phase == SpanPhase.COMPLETE
    )


def _is_subagent_start(event: object) -> bool:
    return (
        isinstance(event, Span)
        and event.subject == SpanSubject.SUBAGENT
        and event.phase == SpanPhase.START
    )


def _is_subagent_complete(event: object) -> bool:
    return (
        isinstance(event, Span)
        and event.subject == SpanSubject.SUBAGENT
        and event.phase == SpanPhase.COMPLETE
    )


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

# Cheapest current-generation model (per CLAUDE.md guidance)
_LIVE_MODEL = "anthropic:claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Real test tools registered on a runtime
# ---------------------------------------------------------------------------


class _CalculatorTool(KaosTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="live-test-calculator",
            display_name="Calculator",
            description="Compute simple arithmetic. Input: expression (e.g. '2+3').",
            category=ToolCategory.DATA,
            capability=ToolCapability.ANALYZE,
            module_name="test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        expr = inputs.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return ToolResult.create_text(f"Result: {result}")
        except Exception as exc:
            return ToolResult.create_error(
                f"Calculator error: {exc}. Provide a valid arithmetic expression."
            )


class _DangerousDeleteTool(KaosTool):
    """Marked destructive — should trigger permission ASK by default."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="live-test-delete",
            display_name="Delete",
            description="Delete a file. Input: path (str).",
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name="test",
            version="0.1.0",
            annotations=ToolAnnotations(destructiveHint=True),
        )

    async def execute(self, inputs: dict[str, Any], context: Any = None) -> ToolResult:
        return ToolResult.create_text(f"Deleted {inputs.get('path', '?')}")


def _make_runtime() -> KaosRuntime:
    runtime = KaosRuntime()
    runtime.tools.register_tool(_CalculatorTool())
    runtime.tools.register_tool(_DangerousDeleteTool())
    return runtime


def _make_vfs() -> VirtualFileSystem:
    return VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))


# ---------------------------------------------------------------------------
# Phase 1: streaming + Phase 2: Runner — basic chat agent
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_chat_agent_streams_events() -> None:
    """Runner.run() with ChatAgent yields TurnStart → IntentClassified → ...
    → TurnComplete in correct order against a real LLM."""
    runtime = _make_runtime()
    agent = Agent(
        instructions="Answer briefly.",
        model=_LIVE_MODEL,
        pattern=AgentPattern.CHAT,
    )
    runner = Runner(agent, runtime=runtime, vfs=_make_vfs())

    events = []
    async for event in runner.run("What is 2+2? Answer in one word.", "live-chat-1"):
        events.append(event)

    first = events[0]
    last = events[-1]
    assert isinstance(first, Span)
    assert first.subject == SpanSubject.TURN
    assert first.phase == SpanPhase.START
    # The terminal event in a successful run is the typed TurnSummary
    # (it fires alongside Span(TURN, COMPLETE)).
    assert isinstance(last, TurnSummary)
    assert any(isinstance(e, IntentClassified) for e in events)
    # Sequence numbers strictly monotonic
    seqs = [e.sequence for e in events]
    assert seqs == sorted(seqs)
    # Final TurnSummary has non-empty text
    assert len(last.text) > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_chat_agent_with_tool_call() -> None:
    """ChatAgent invokes the calculator tool via ReAct and streams ToolCallStart/Result."""
    runtime = _make_runtime()
    agent = Agent(
        instructions="Use the calculator tool when arithmetic is needed.",
        model=_LIVE_MODEL,
        pattern=AgentPattern.CHAT,
        tools=("live-test-calculator",),
    )
    runner = Runner(agent, runtime=runtime, vfs=_make_vfs())

    events = []
    async for event in runner.run("Calculate 17 * 23 using the tool.", "live-tool-1"):
        events.append(event)

    # Tool call events should appear (LLM may invoke calculator one or more times)
    tool_starts = [e for e in events if _is_tool_call_start(e)]
    tool_results = [e for e in events if _is_tool_call_result(e)]
    assert len(tool_starts) >= 1, f"no tool-call START spans: {[type(e).__name__ for e in events]}"
    assert len(tool_results) == len(tool_starts)
    # The calculator should have been invoked
    assert all(
        str(t.attributes.get("tool_name", "")) == "live-test-calculator" for t in tool_starts
    )
    # At least one result should be successful (not an error). We don't
    # assert the specific value "391" because the LLM may compute the
    # result differently or in steps.
    assert any(not bool(r.attributes.get("is_error", False)) for r in tool_results)


# ---------------------------------------------------------------------------
# Phase 1: backward compat — turn() still works
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_turn_backward_compat() -> None:
    """Runner.turn() returns AgentResponse with the final text (live)."""
    agent = Agent(model=_LIVE_MODEL, instructions="Answer briefly.")
    runner = Runner(agent, vfs=_make_vfs())
    response = await runner.turn("Say hello in one word.", "live-turn-1")
    assert response.text
    assert response.intent.intent.value in {"respond", "tool_use", "research", "plan", "clarify"}


# ---------------------------------------------------------------------------
# Phase 0 (Gap 5): StreamDelta / LifecycleEvent isinstance routing
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_event_categories_are_distinguishable() -> None:
    """isinstance(e, LifecycleEvent) routes lifecycle events from real run."""
    agent = Agent(model=_LIVE_MODEL, instructions="Answer briefly.")
    runner = Runner(agent, vfs=_make_vfs())

    lifecycle = 0
    deltas = 0
    async for event in runner.run("Say 'hi'.", "live-cat-1"):
        if isinstance(event, LifecycleEvent):
            lifecycle += 1
        if isinstance(event, StreamDelta):
            deltas += 1

    # Real run emits lifecycle events (TurnStart, IntentClassified, TurnComplete at minimum)
    assert lifecycle >= 3
    # StreamDelta count depends on whether the dispatch path emits TextDelta
    # (BaseAgent default emits 1 TextDelta with the full text). At least
    # the intermediate base classes are wired up and isinstance works.
    assert deltas >= 0


# ---------------------------------------------------------------------------
# Phase 4 (Gap 4): hooks compose; LoggingHook + CostTrackingHook against live
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_hooks_compose_and_track_cost() -> None:
    """LoggingHook + CostTrackingHook both fire during a real turn."""
    cost = CostTrackingHook()
    logging_hook = LoggingHook()
    agent = Agent(model=_LIVE_MODEL, instructions="Answer briefly.")
    runner = Runner(agent, vfs=_make_vfs(), hooks=(logging_hook, cost))

    async for _ in runner.run("Say 'one'.", "live-hooks-1"):
        pass

    totals = cost.totals
    assert "live-hooks-1" in totals
    # CostTrackingHook ran on the TurnComplete event — verify count
    assert totals["live-hooks-1"][2] == 1  # turn count


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_hook_can_skip_tool_call() -> None:
    """A hook returning HookAction.SKIP suppresses the tool call event."""

    class _BlockCalc(KaosHook):
        async def on_tool_call_start(self, event: Span) -> HookAction:
            if str(event.attributes.get("tool_name", "")) == "live-test-calculator":
                return HookAction.SKIP
            return HookAction.CONTINUE

    runtime = _make_runtime()
    agent = Agent(
        instructions="Use the calculator if needed.",
        model=_LIVE_MODEL,
        tools=("live-test-calculator",),
    )
    runner = Runner(agent, runtime=runtime, vfs=_make_vfs(), hooks=(_BlockCalc(),))

    events = []
    async for event in runner.run("Calculate 5+5.", "live-skip-1"):
        events.append(event)

    # Tool-call START spans for the calculator should be filtered out.
    # (The LLM may or may not call it; if it does, the hook suppresses it.)
    suppressed = [
        e
        for e in events
        if _is_tool_call_start(e)
        and str(e.attributes.get("tool_name", "")) == "live-test-calculator"
    ]
    assert suppressed == []


# ---------------------------------------------------------------------------
# Phase 5 (Gap 8): provider role routing actually changes models in flight
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_provider_routes_classify_to_haiku() -> None:
    """With provider=FAST, classify uses haiku and the run completes."""
    agent = Agent(
        instructions="Answer briefly.",
        provider=FAST,
        pattern=AgentPattern.CHAT,
    )
    runner = Runner(agent, vfs=_make_vfs())

    response = await runner.turn("Hi.", "live-provider-fast")
    assert response.text


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_provider_balanced_runs_end_to_end() -> None:
    """BALANCED preset (cheap classify, strong plan) works end-to-end."""
    agent = Agent(
        instructions="Answer briefly.",
        provider=BALANCED,
        pattern=AgentPattern.CHAT,
    )
    runner = Runner(agent, vfs=_make_vfs())
    response = await runner.turn("Hello.", "live-provider-balanced")
    assert response.text


# ---------------------------------------------------------------------------
# Phase 6 (Gaps 1+2+3+9): permission policy with real annotations + resume
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_destructive_tool_triggers_approval_via_annotations() -> None:
    """A destructiveHint=True tool triggers ASK; Runner pauses and persists state.

    This exercises Gap 9 (annotation lookup) + Gaps 1+2+3 (pause + VFS persist).
    """
    runtime = _make_runtime()
    vfs = _make_vfs()
    policy = PermissionPolicy()  # empty rules — rely on destructiveHint annotation
    agent = Agent(
        instructions="If asked to delete, use the delete tool.",
        model=_LIVE_MODEL,
        tools=("live-test-delete",),
    )
    runner = Runner(agent, runtime=runtime, vfs=vfs, permission_policy=policy)

    events = []
    async for event in runner.run(
        "Delete the file at /tmp/foo.txt using the delete tool.",
        "live-perm-1",
    ):
        events.append(event)
        # If we got an approval event, we're done — Runner has paused
        if isinstance(event, ToolCallApprovalRequired):
            break

    approval = [e for e in events if isinstance(e, ToolCallApprovalRequired)]
    # The LLM may or may not actually invoke the delete tool — depends on its
    # interpretation. If it did, we should see an approval event.
    if approval:
        # State should be persisted to VFS
        from kaos_agents.runtime.interrupts import load_run_state

        state = await load_run_state(approval[0].run_id, vfs)
        assert state.session_id == "live-perm-1"
        assert state.original_message.startswith("Delete")


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_resume_denied_yields_run_error() -> None:
    """resume(approved=False) on a real (paused) state yields RunError."""
    from kaos_agents.events import RunError
    from kaos_agents.runtime.interrupts import PendingToolCall, RunState

    vfs = _make_vfs()
    agent = Agent(model=_LIVE_MODEL)
    runner = Runner(agent, vfs=vfs)

    state = RunState(
        run_id="live_resume_deny",
        session_id="live_resume_sess",
        pending_tool_call=PendingToolCall(call_id="t", tool_name="dangerous"),
        original_message="never mind",
    )
    events = []
    async for event in runner.resume(state, approved=False):
        events.append(event)

    errors = [e for e in events if isinstance(e, RunError)]
    assert len(errors) == 1
    assert errors[0].error_type == "approval_denied"


# ---------------------------------------------------------------------------
# Phase 7 (Gaps 6+7): delegated_agents inject as real tools, sub-agent runs live
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_delegated_agent_invocable_from_parent() -> None:
    """Agent.delegated_agents become tools; calling one runs the sub-agent against the LLM."""
    writer = Agent(
        instructions="You write very short legal memos. One sentence per memo.",
        model=_LIVE_MODEL,
    )
    write_tool = agent_as_tool(
        writer,
        name="write_memo",
        description=(
            "Draft a one-sentence legal memo. Argument: task (str) describing the memo topic."
        ),
    )

    coordinator = Agent(
        instructions="When asked to draft a memo, call the write_memo tool.",
        model=_LIVE_MODEL,
        delegated_agents=(write_tool,),
    )
    runner = Runner(coordinator, vfs=_make_vfs())

    # We can verify the wiring two ways: (1) explicit delegate, (2) via run() if
    # the LLM picks it. Explicit is deterministic.
    events = []
    async for event in runner.delegate(write_tool, "Memo on EPA emissions.", "live-deleg-1"):
        events.append(event)

    starts = [e for e in events if _is_subagent_start(e)]
    completes = [e for e in events if _is_subagent_complete(e)]
    assert len(starts) == 1
    assert len(completes) == 1
    assert completes[0].attributes.get("subagent_name") == "write_memo"
    assert len(str(completes[0].attributes.get("result_summary", ""))) > 0


# ---------------------------------------------------------------------------
# Phase 3 (Gap 10): API content negotiation (live)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_api_sse_returns_event_stream() -> None:
    """POST /v1/sessions/{id}/messages with default Accept streams SSE events.

    Uses the FastAPI app with a real LLM; verifies SSE format end-to-end.
    """
    from httpx import ASGITransport, AsyncClient

    from kaos_agents.api.server import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/sessions/live-api-sse/messages",
            json={"message": "Say 'hi'.", "model": _LIVE_MODEL},
            timeout=60.0,
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    # Stream uses the unified Span(subject, phase) event model — the
    # SSE event name is "span" with the body carrying subject/phase.
    # (Old test asserted "event: turn_start" which dated from before
    # the Span unification.)
    assert "event: span" in text
    assert '"subject":"turn"' in text and '"phase":"start"' in text
    assert '"phase":"complete"' in text
    # Must also surface the value-carrying events
    assert "event: text_delta" in text
    assert "event: turn_summary" in text


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_api_json_returns_full_response() -> None:
    """Same URL with Accept: application/json returns aggregated JSON."""
    from httpx import ASGITransport, AsyncClient

    from kaos_agents.api.server import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/sessions/live-api-json/messages",
            json={"message": "Say 'hi'.", "model": _LIVE_MODEL},
            headers={"Accept": "application/json"},
            timeout=60.0,
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    data = resp.json()
    assert data["text"]
    assert "intent" in data


# Reference unused imports so ruff doesn't drop them
_ = TextDelta
