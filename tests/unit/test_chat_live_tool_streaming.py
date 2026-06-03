"""Unit tests for live per-tool span streaming in ChatAgent.

The streaming-observability fix wires kaos-llm-core's ReAct per-tool
``ProgramHooks`` (``on_tool_start`` / ``on_tool_end``) through
``ChatAgent._handle_tool_use_streaming`` so each tool call surfaces as a
``Span(TOOL_CALL, start)`` the instant it enters flight and a
``Span(TOOL_CALL, complete)`` the instant it returns — *while* the ReAct
loop is still running, instead of in one burst after the whole loop
finishes.

These tests pin the live behavior with a fake ReAct whose ``invoke``
fires the injected hooks (simulating tool dispatch) before returning:

* Each tool yields exactly one start + one complete span — the
  post-invoke trajectory sweep must NOT re-emit a tool already streamed
  live (no duplicate cards).
* The tool spans are yielded BEFORE the final answer ``TextDelta``
  (i.e. live, mid-run — not appended after).
* A hook that raises never breaks the turn.
"""

from __future__ import annotations

import asyncio
import typing
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.events import EventEmitter, Span, SpanPhase, SpanSubject, TextDelta
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.settings import KaosAgentSettings


def _vfs() -> VirtualFileSystem:
    return VirtualFileSystem(
        config=VFSConfig(
            default_backend=StorageBackend.MEMORY,
            isolation_mode=IsolationMode.GLOBAL,
        )
    )


class _NoopTool(KaosTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-test-noop",
            description="Returns a constant string. For streaming tests.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs, context=None):
        return ToolResult.create_text("noop ok")


def _make_runtime() -> KaosRuntime:
    runtime = KaosRuntime.test_mode()
    runtime.tools.register_tool(_NoopTool())
    return runtime


class _FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 10
        self.output_tokens = 5
        self.total_tokens = 15
        self.cost_usd = 0.0


def _fake_obs(call_id: str, tool_name: str, *, is_error: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        tool_call_id=call_id,
        tool_name=tool_name,
        arguments={"q": "x"},
        result=f"{tool_name} result",
        is_error=is_error,
    )


class _FakeReActResult:
    answer = "the final answer"
    iterations_used = 1
    stop_reason = "TERMINATED"

    def __init__(self, observations: list[SimpleNamespace]) -> None:
        # Single trajectory iteration carrying all observations, so the
        # post-invoke backstop loop sees the same call_ids the hooks
        # streamed (and must skip them).
        self.trajectory = [SimpleNamespace(text="step", tool_results=observations)]
        self.outputs = {"answer": self.answer}


class _FakeInvocation:
    def __init__(self, observations: list[SimpleNamespace]) -> None:
        self.output = _FakeReActResult(observations)
        self.usage = _FakeUsage()
        self.extras: dict[str, typing.Any] = {}


def _make_hook_firing_react(observations: list[SimpleNamespace], *, raise_in_hook: bool = False):
    """Fake ReAct whose ``invoke`` fires the per-tool hooks for each
    observation (start before, end after, with an await between to
    exercise the concurrent drain), then returns a matching invocation."""

    class _FakeReAct:
        def __init__(
            self, sig, *, tools, model, max_iterations, instructions, program_hooks=None
        ) -> None:
            self._hooks = program_hooks

        async def invoke(self, **kwargs: typing.Any) -> _FakeInvocation:
            for obs in observations:
                tc = SimpleNamespace(
                    id=obs.tool_call_id, name=obs.tool_name, arguments=obs.arguments
                )
                if self._hooks is not None and self._hooks.on_tool_start is not None:
                    if raise_in_hook:
                        # observability must never break dispatch
                        self._hooks.on_tool_start(self, None)
                    else:
                        self._hooks.on_tool_start(self, tc)
                await asyncio.sleep(0)  # yield so the drain loop can run
                if self._hooks is not None and self._hooks.on_tool_end is not None:
                    self._hooks.on_tool_end(self, obs)
                await asyncio.sleep(0)
            return _FakeInvocation(observations)

    return _FakeReAct


async def _run(agent: ChatAgent, message: str) -> list:
    from kaos_agents.memory.store import SessionStore

    store = SessionStore(_vfs())
    memory = await store.load_or_create("test-live-stream-session")
    emitter = EventEmitter(session_id="test-live-stream-session", run_id="r1")
    events: list = []
    async for event in agent._handle_tool_use_streaming(message, memory, {}, emitter):
        events.append(event)
    return events


async def _noop_passthrough(*, tools: list, query: str, memory=None) -> list:
    return tools


def _tool_spans(events: list, phase: SpanPhase) -> list[Span]:
    return [
        e
        for e in events
        if isinstance(e, Span) and e.subject == SpanSubject.TOOL_CALL and e.phase == phase
    ]


@pytest.mark.asyncio
async def test_each_tool_streams_one_start_and_one_complete() -> None:
    """Two tool calls → exactly two start spans + two complete spans,
    no duplication from the post-invoke backstop, and the spans precede
    the final answer TextDelta (live, mid-run)."""
    observations = [
        _fake_obs("call-1", "kaos-test-noop"),
        _fake_obs("call-2", "kaos-test-noop"),
    ]
    agent = ChatAgent(_vfs(), runtime=_make_runtime(), settings=KaosAgentSettings())
    fake_react = _make_hook_firing_react(observations)

    with (
        patch("kaos_llm_core.programs.react.ReAct", fake_react),
        patch.object(agent, "_maybe_narrow_tools_via_fitness_ranker", new=_noop_passthrough),
    ):
        events = await _run(agent, message="do two things")

    starts = _tool_spans(events, SpanPhase.START)
    completes = _tool_spans(events, SpanPhase.COMPLETE)
    # Exactly one start + one complete per tool — the backstop loop must
    # skip call_ids already streamed live via the hook bridge.
    assert len(starts) == 2, f"expected 2 start spans, got {len(starts)}"
    assert len(completes) == 2, f"expected 2 complete spans, got {len(completes)}"
    assert {s.attributes["call_id"] for s in starts} == {"call-1", "call-2"}
    assert {c.attributes["call_id"] for c in completes} == {"call-1", "call-2"}

    # complete spans reuse their start's span_id (UI pairing intact).
    start_ids = {s.attributes["call_id"]: s.span_id for s in starts}
    for c in completes:
        assert c.span_id == start_ids[c.attributes["call_id"]]

    # The tool spans are live: they appear BEFORE the final answer text.
    last_complete_idx = max(i for i, e in enumerate(events) if e in completes)
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert text_deltas, "expected a final answer TextDelta"
    final_text_idx = max(i for i, e in enumerate(events) if isinstance(e, TextDelta))
    assert last_complete_idx < final_text_idx, (
        "tool spans must stream BEFORE the final answer (live), "
        f"got last_complete@{last_complete_idx} final_text@{final_text_idx}"
    )
    assert "the final answer" in "".join(td.content for td in text_deltas)


@pytest.mark.asyncio
async def test_complete_span_carries_result_and_error_flag() -> None:
    """The streamed complete span carries the tool result summary and the
    is_error flag from the ToolObservation."""
    observations = [_fake_obs("call-err", "kaos-test-noop", is_error=True)]
    agent = ChatAgent(_vfs(), runtime=_make_runtime(), settings=KaosAgentSettings())
    fake_react = _make_hook_firing_react(observations)

    with (
        patch("kaos_llm_core.programs.react.ReAct", fake_react),
        patch.object(agent, "_maybe_narrow_tools_via_fitness_ranker", new=_noop_passthrough),
    ):
        events = await _run(agent, message="do a failing thing")

    completes = _tool_spans(events, SpanPhase.COMPLETE)
    assert len(completes) == 1
    attrs = completes[0].attributes
    assert attrs["is_error"] is True
    assert attrs["result_summary"] == "kaos-test-noop result"


@pytest.mark.asyncio
async def test_raising_tool_hook_does_not_break_turn() -> None:
    """A hook bridge that is fed bad data (defensive try/except) must not
    abort the turn — the final answer still streams."""
    observations = [_fake_obs("call-1", "kaos-test-noop")]
    agent = ChatAgent(_vfs(), runtime=_make_runtime(), settings=KaosAgentSettings())
    fake_react = _make_hook_firing_react(observations, raise_in_hook=True)

    with (
        patch("kaos_llm_core.programs.react.ReAct", fake_react),
        patch.object(agent, "_maybe_narrow_tools_via_fitness_ranker", new=_noop_passthrough),
    ):
        events = await _run(agent, message="do the thing")

    # The turn completed and the answer reached the stream despite the
    # start hook receiving bad data.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert text_deltas, "expected a final answer TextDelta even when the hook misbehaves"
    assert "the final answer" in "".join(td.content for td in text_deltas)
