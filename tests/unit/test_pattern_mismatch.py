"""Unit tests for the 0.1.0a10 PLAN→tools dispatch fix.

The fix lives at ``BaseAgent._dispatch_streaming`` (kaos_agents/
runtime/agent.py) — when the per-turn intent classifier returns
``IntentType.PLAN`` or ``IntentType.RESEARCH`` but the running agent
class hasn't overridden ``_handle_plan`` / ``_handle_research`` (i.e.,
it's the default ``BaseAgent`` implementation that silently degrades
to ``_handle_respond``), the dispatcher emits a typed
:class:`~kaos_agents.events.lifecycle.PatternMismatch` event and
redirects to ``_handle_tool_use`` so at least ReAct fires.

These tests stub the handlers + intent extractor so no live LLM call
is issued.

See: kaos-modules/docs/plans/kaos-agents-autonomy-improvement-1.md
"""

from __future__ import annotations

from typing import Any

import pytest

from kaos_agents.events import (
    EventEmitter,
    PatternMismatch,
)
from kaos_agents.runtime.agent import BaseAgent
from kaos_agents.types import IntentResult, IntentType, InvocationUsage, ToolExecution
from kaos_agents.types.memory import MemoryItem, MemoryType

# ---------------------------------------------------------------------------
# Stub agents — exercise the dispatcher without a real LLM call.
# ---------------------------------------------------------------------------


class _StubChatAgent(BaseAgent):
    """Subclass of BaseAgent that overrides ``_handle_tool_use``
    (so the redirect path has something to dispatch to) but leaves
    ``_handle_plan`` / ``_handle_research`` at the BaseAgent default
    — i.e. the exact shape that triggers the dispatch bug."""

    def __init__(self) -> None:
        # Skip BaseAgent.__init__ — we don't need vfs / store / model
        # resolution for dispatch-layer tests.
        self._tool_use_call_count = 0
        self._respond_call_count = 0

    async def _handle_tool_use(
        self,
        message: str,
        memory: Any,
        context_items: dict[MemoryType, list[MemoryItem]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        self._tool_use_call_count += 1
        return (
            "stub tool_use ran",
            [ToolExecution(tool_name="stub-tool", arguments=(), result_summary="ok")],
            InvocationUsage(total_tokens=0, cost_usd=0.0),
        )

    async def _handle_respond(
        self,
        message: str,
        memory: Any,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        self._respond_call_count += 1
        return (
            "stub respond ran (this is the silent-degradation path)",
            [],
            InvocationUsage(total_tokens=0, cost_usd=0.0),
        )

    @classmethod
    def metadata(cls):
        from kaos_agents.types.metadata import AgentMetadata

        return AgentMetadata(name="stub-chat", description="stub", pattern="chat")


class _StubPlanAgent(_StubChatAgent):
    """Override ``_handle_plan`` so the pattern-mismatch detector
    should NOT fire for PLAN intent."""

    def __init__(self) -> None:
        super().__init__()
        self._plan_call_count = 0

    async def _handle_plan(
        self,
        message: str,
        memory: Any,
        context_items: dict[MemoryType, list[Any]],
    ) -> tuple[str, list[ToolExecution], InvocationUsage]:
        self._plan_call_count += 1
        return (
            "stub plan ran",
            [ToolExecution(tool_name="plan-tool", arguments=(), result_summary="ok")],
            InvocationUsage(total_tokens=0, cost_usd=0.0),
        )

    @classmethod
    def metadata(cls):
        from kaos_agents.types.metadata import AgentMetadata

        return AgentMetadata(name="stub-plan", description="stub", pattern="plan")


def _intent(itype: IntentType, confidence: float = 0.95) -> IntentResult:
    return IntentResult(intent=itype, confidence=confidence, reasoning="stub")


def _emitter() -> EventEmitter:
    return EventEmitter(session_id="stub-session", run_id="stub-run")


# ---------------------------------------------------------------------------
# _handler_is_default — used by the dispatcher to detect the silent
# fall-through.
# ---------------------------------------------------------------------------


class TestHandlerIsDefault:
    def test_chat_agent_handle_plan_is_default(self) -> None:
        agent = _StubChatAgent()
        # _StubChatAgent does NOT override _handle_plan → detector True.
        assert agent._handler_is_default("_handle_plan") is True
        assert agent._handler_is_default("_handle_research") is True

    def test_chat_agent_handle_tool_use_is_overridden(self) -> None:
        agent = _StubChatAgent()
        # _StubChatAgent DOES override _handle_tool_use → detector False.
        assert agent._handler_is_default("_handle_tool_use") is False

    def test_plan_agent_handle_plan_is_overridden(self) -> None:
        agent = _StubPlanAgent()
        # _StubPlanAgent DOES override _handle_plan → detector False.
        assert agent._handler_is_default("_handle_plan") is False
        # But still falls back to BaseAgent for _handle_research.
        assert agent._handler_is_default("_handle_research") is True


# ---------------------------------------------------------------------------
# _detect_pattern_mismatch — emits PatternMismatch + returns redirect
# handler when the silent-fall-through would otherwise fire.
# ---------------------------------------------------------------------------


class TestDetectPatternMismatch:
    def test_plan_intent_on_chat_agent_emits_mismatch_and_redirects(self) -> None:
        agent = _StubChatAgent()
        redirect, mismatch = agent._detect_pattern_mismatch(_intent(IntentType.PLAN), _emitter())

        # Bound methods don't preserve identity across attribute reads;
        # compare the underlying function instead.
        assert redirect.__func__ is type(agent)._handle_tool_use
        assert isinstance(mismatch, PatternMismatch)
        assert mismatch.classified_intent == "plan"
        assert mismatch.agent_pattern == "chat"
        assert mismatch.recommended_pattern == "plan"
        assert mismatch.fallback_handler == "_handle_tool_use"
        assert "silently degraded to _handle_respond" in mismatch.rationale

    def test_research_intent_no_longer_triggers_mismatch(self) -> None:
        # RESEARCH used to fall through the same silent-degradation path
        # as PLAN. As of the FindingsAgent-backed default
        # (kaos-modules/docs/plans/2026-05-26-retrieval-planner-and-findings-dispatch.md),
        # BaseAgent._handle_research is a real grounding pipeline —
        # no redirect, no PatternMismatch.
        agent = _StubChatAgent()
        redirect, mismatch = agent._detect_pattern_mismatch(
            _intent(IntentType.RESEARCH), _emitter()
        )

        assert redirect is None
        assert mismatch is None

    def test_plan_intent_on_plan_agent_no_mismatch(self) -> None:
        # _StubPlanAgent overrides _handle_plan → no fall-through → no
        # PatternMismatch event, redirect returns None.
        agent = _StubPlanAgent()
        redirect, mismatch = agent._detect_pattern_mismatch(_intent(IntentType.PLAN), _emitter())

        assert redirect is None
        assert mismatch is None

    def test_research_intent_on_plan_agent_no_longer_triggers_mismatch(self) -> None:
        # _StubPlanAgent overrides _handle_plan but not _handle_research.
        # Pre-FindingsAgent default this redirected to _handle_tool_use; the
        # FindingsAgent-backed BaseAgent._handle_research now grounds via the
        # planner + applier, so no PatternMismatch fires on RESEARCH.
        agent = _StubPlanAgent()
        redirect, mismatch = agent._detect_pattern_mismatch(
            _intent(IntentType.RESEARCH), _emitter()
        )

        assert redirect is None
        assert mismatch is None

    def test_respond_intent_no_mismatch(self) -> None:
        # IntentType.RESPOND has a real BaseAgent handler — no silent
        # degradation, no mismatch.
        agent = _StubChatAgent()
        redirect, mismatch = agent._detect_pattern_mismatch(_intent(IntentType.RESPOND), _emitter())

        assert redirect is None
        assert mismatch is None

    def test_tool_use_intent_no_mismatch(self) -> None:
        # IntentType.TOOL_USE always has a path — even BaseAgent's
        # default goes through ReAct via _handle_tool_use.
        agent = _StubChatAgent()
        redirect, mismatch = agent._detect_pattern_mismatch(
            _intent(IntentType.TOOL_USE), _emitter()
        )

        assert redirect is None
        assert mismatch is None


# ---------------------------------------------------------------------------
# End-to-end through _dispatch_streaming — proves the redirect actually
# routes to _handle_tool_use and that _handle_respond is NEVER called.
# ---------------------------------------------------------------------------


class TestDispatchStreamingRedirect:
    @pytest.mark.asyncio
    async def test_plan_intent_on_chat_agent_calls_tool_use_not_respond(self) -> None:
        agent = _StubChatAgent()
        emitter = _emitter()
        events = []
        async for ev in agent._dispatch_streaming(
            _intent(IntentType.PLAN),
            message="search the federal register for ...",
            memory=None,  # ty: ignore[invalid-argument-type] — handler stubs ignore it
            context_items={},
            emitter=emitter,
        ):
            events.append(ev)

        # Redirect fired _handle_tool_use exactly once. _handle_respond
        # was NOT called (that's the pre-0.1.0a10 silent-degradation
        # path we're guarding against).
        assert agent._tool_use_call_count == 1
        assert agent._respond_call_count == 0
        # PatternMismatch event made it into the YIELDED stream — not
        # just an in-process collector. This is the bug 0.1.0a11 fixes:
        # 0.1.0a10 emitted the event via emitter.emit but never yielded
        # it, so production SSE/OTel consumers never saw it.
        mismatch_events = [e for e in events if isinstance(e, PatternMismatch)]
        assert len(mismatch_events) == 1

    @pytest.mark.asyncio
    async def test_plan_intent_on_plan_agent_dispatches_to_plan_handler(self) -> None:
        agent = _StubPlanAgent()
        events = []
        async for ev in agent._dispatch_streaming(
            _intent(IntentType.PLAN),
            message="any",
            memory=None,  # ty: ignore[invalid-argument-type]
            context_items={},
            emitter=_emitter(),
        ):
            events.append(ev)

        # _StubPlanAgent overrides _handle_plan → that's what runs.
        # _handle_tool_use is NOT called and no PatternMismatch fires.
        assert agent._plan_call_count == 1
        assert agent._tool_use_call_count == 0
        assert [e for e in events if isinstance(e, PatternMismatch)] == []
