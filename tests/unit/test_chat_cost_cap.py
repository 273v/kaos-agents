"""Unit tests for P0.3 — ChatAgent cumulative cost cap.

The 2026-05-23 corpus stress S12 reproduced a $4.97 / 187-tool-call
runaway when the agent walked a 500-doc corpus document-by-document
because the plan-execute ``plan_max_cost_usd`` cap does NOT apply
to the ChatAgent path. P0.3 adds ``chat_max_cost_usd`` to
``KaosAgentSettings`` and wires it into ``ChatAgent._handle_tool_use_streaming``
so the ReAct loop short-circuits with a typed BudgetExceeded event
and an honest refusal when the cap is exceeded.

These tests pin two behaviors:

* When ``chat_max_cost_usd`` is set tight (e.g. $0.001) and the
  ReAct call's reported cost exceeds it, the loop MUST emit a
  ``BudgetExceeded(kind="chat_cost")`` event AND a refusal
  ``TextDelta`` naming the cap, the observed spend, and the
  tool-call count.
* When ``chat_max_cost_usd`` is None (default), the loop runs
  to completion with no BudgetExceeded event.

The tests patch the ReAct invocation at its module-level import
point inside chat.py and synthesize a fake invocation whose
``.usage.cost_usd`` is the load-bearing signal — this keeps the
test deterministic and decouples it from any specific provider's
pricing. The live-tier ``@pytest.mark.live`` marker is retained
per the kaos-agents test discipline (per CLAUDE.md the unit suite
runs without ``-m live`` filter, the live runners gate on it).
"""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest
from kaos_core.base.tool import KaosTool
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.results import ToolResult
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.events import BudgetExceeded, EventEmitter, TextDelta
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
    """Tiny no-op tool so ChatAgent enters the TOOL_USE branch."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-test-noop",
            description="Returns a constant string. For cost-cap tests.",
            category=ToolCategory.TEXT,
            capability=ToolCapability.EXTRACT,
            module_name="kaos-agents-test",
            version="0.1.0",
            annotations=ToolAnnotations(readOnlyHint=True),
        )

    async def execute(self, inputs, context=None):
        return ToolResult.create_text("noop ok")


# ── Synthetic ReAct stubs ─────────────────────────────────────────


class _FakeUsage:
    """Stub matching the duck-type ``InvocationUsage.from_invocation``
    expects on ``invocation.usage``."""

    def __init__(self, cost_usd: float) -> None:
        self.input_tokens = 1_000
        self.output_tokens = 500
        self.total_tokens = 1_500
        self.cost_usd = cost_usd


class _FakeIteration:
    """One trajectory step with one tool result so the cost-cap path
    can count ``n_tool_calls``."""

    text = "step text"

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.tool_results = [
            SimpleNamespace(
                tool_name="kaos-test-noop",
                tool_call_id="call-1",
                arguments={},
                result="noop ok",
                is_error=False,
            )
        ]


class _FakeReActResult:
    """Stub ReAct result with one trajectory entry + an answer."""

    answer = "the answer"
    iterations_used = 1
    stop_reason = "TERMINATED"

    def __init__(self) -> None:
        self.trajectory = [_FakeIteration()]
        self.outputs = {"answer": self.answer}


class _FakeInvocation:
    """Stub Invocation with ``.output`` + ``.usage`` ducks."""

    def __init__(self, cost_usd: float) -> None:
        self.output = _FakeReActResult()
        self.usage = _FakeUsage(cost_usd=cost_usd)
        self.extras: dict[str, typing.Any] = {}


def _make_fake_react(cost_usd: float) -> type:
    """Build a fake ReAct class whose .invoke returns a fixed cost."""

    class _FakeReAct:
        def __init__(self, sig, *, tools, model, max_iterations, instructions) -> None:
            self._sig = sig
            self._cost = cost_usd

        async def invoke(self, **kwargs: typing.Any) -> _FakeInvocation:
            return _FakeInvocation(cost_usd=self._cost)

    return _FakeReAct


# ── Test fixtures + helpers ───────────────────────────────────────


def _make_runtime() -> KaosRuntime:
    runtime = KaosRuntime.test_mode()
    runtime.tools.register_tool(_NoopTool())
    return runtime


async def _run_chat_tool_use(
    agent: ChatAgent,
    *,
    message: str,
) -> list:
    """Drive ``_handle_tool_use_streaming`` and collect emitted events."""
    from kaos_agents.memory.store import SessionStore

    store = SessionStore(_vfs())
    memory = await store.load_or_create("test-cost-cap-session")
    emitter = EventEmitter(session_id="test-cost-cap-session", run_id="r1")

    events: list = []
    async for event in agent._handle_tool_use_streaming(message, memory, {}, emitter):
        events.append(event)
    return events


# ── Case 1: cap set tight + cost exceeded → BudgetExceeded + refusal ──


@pytest.mark.live
@pytest.mark.asyncio
async def test_chat_cost_cap_short_circuits_on_exceed() -> None:
    """When ``chat_max_cost_usd=0.001`` and the ReAct invocation
    reports ``cost_usd=0.05``, the chat loop MUST:

    * emit a ``BudgetExceeded(kind="chat_cost")`` event with
      ``limit=0.001`` and ``actual=0.05``
    * emit a refusal ``TextDelta`` naming the cap, observed spend,
      and tool-call count
    * NOT emit the normal ReAct answer (the function returns early)
    """
    runtime = _make_runtime()
    settings = KaosAgentSettings(chat_max_cost_usd=0.001)
    agent = ChatAgent(_vfs(), runtime=runtime, settings=settings)

    fake_react_cls = _make_fake_react(cost_usd=0.05)

    async def _noop_passthrough(*, tools: list, query: str, memory=None) -> list:
        return tools

    # Disable the M1 fitness ranker — it would issue a real LLM call.
    with (
        patch("kaos_llm_core.programs.react.ReAct", fake_react_cls),
        patch.object(
            agent,
            "_maybe_narrow_tools_via_fitness_ranker",
            new=_noop_passthrough,
        ),
    ):
        events = await _run_chat_tool_use(agent, message="please do the thing")

    # BudgetExceeded must be present with the right fields.
    budget = [e for e in events if isinstance(e, BudgetExceeded)]
    assert len(budget) == 1, (
        f"expected exactly 1 BudgetExceeded, got {len(budget)}: "
        f"events={[type(e).__name__ for e in events]}"
    )
    be = budget[0]
    assert be.kind == "chat_cost", f"kind={be.kind!r}"
    assert be.limit == pytest.approx(0.001), f"limit={be.limit}"
    assert be.actual == pytest.approx(0.05), f"actual={be.actual}"
    assert "exceeded" in be.reason.lower(), f"reason missing 'exceeded': {be.reason!r}"

    # A refusal TextDelta must follow the BudgetExceeded event,
    # naming the cap, observed spend, and tool-call count.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert text_deltas, (
        f"expected a refusal TextDelta after BudgetExceeded; "
        f"got events={[type(e).__name__ for e in events]}"
    )
    refusal_text = text_deltas[-1].content.lower()
    assert "stopped" in refusal_text, refusal_text
    assert "cost cap" in refusal_text, refusal_text
    assert "0.001" in refusal_text, refusal_text
    assert "0.05" in refusal_text, refusal_text
    # The refusal must name how many tool calls were attempted.
    assert "tool call" in refusal_text, refusal_text


# ── Case 2: cap unset (None) → loop runs to completion, no BudgetExceeded ──


@pytest.mark.live
@pytest.mark.asyncio
async def test_chat_cost_cap_disabled_by_default() -> None:
    """When ``chat_max_cost_usd`` is None (the default), the chat
    loop MUST complete normally regardless of cost — no
    BudgetExceeded event is emitted even when the ReAct invocation
    reports an arbitrarily large cost.

    This pins the back-compat guarantee: existing callers see no
    behavior change unless they explicitly opt in by setting the cap.
    """
    runtime = _make_runtime()
    # Default settings — chat_max_cost_usd is None.
    settings = KaosAgentSettings()
    assert settings.chat_max_cost_usd is None
    agent = ChatAgent(_vfs(), runtime=runtime, settings=settings)

    fake_react_cls = _make_fake_react(cost_usd=999.99)

    async def _noop_passthrough(*, tools: list, query: str, memory=None) -> list:
        return tools

    with (
        patch("kaos_llm_core.programs.react.ReAct", fake_react_cls),
        patch.object(
            agent,
            "_maybe_narrow_tools_via_fitness_ranker",
            new=_noop_passthrough,
        ),
    ):
        events = await _run_chat_tool_use(agent, message="please do the thing")

    budget = [e for e in events if isinstance(e, BudgetExceeded)]
    assert not budget, (
        f"BudgetExceeded MUST NOT fire when chat_max_cost_usd is None; "
        f"got {len(budget)} event(s): {budget!r}"
    )
    # The normal answer text must reach the stream — i.e. ReAct
    # didn't short-circuit.
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert text_deltas, "expected at least one TextDelta (the ReAct answer)"
    combined = "".join(td.content for td in text_deltas).lower()
    assert "the answer" in combined, (
        f"expected the fake ReAct answer in the stream, got: {combined!r}"
    )
