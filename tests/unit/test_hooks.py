"""Tests for hooks (Phase 4).

Covers:
- BaseHook default no-op behavior
- HookAction enum values
- dispatch_hook calls correct method per event type
- dispatch_hook returns most restrictive action
- Multiple hooks compose in order
- Hook exceptions are swallowed (logged, not raised)
- LoggingHook runs without error
- HookAction.SKIP suppresses events in Runner
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kaos_agents.config import Agent
from kaos_agents.events import (
    EventEmitter,
    IntentClassified,
    RunError,
    Span,
    SpanPhase,
    SpanSubject,
    TextDelta,
    TurnSummary,
)
from kaos_agents.hooks import (
    AuditHook,
    BaseHook,
    CostTrackingHook,
    HookAction,
    LoggingHook,
    dispatch_hook,
)
from kaos_agents.runner import Runner
from kaos_agents.types import IntentResult, IntentType


def _turn_start(turn_number: int = 1, *, session_id: str = "s", run_id: str = "r") -> Span:
    return Span(
        timestamp=1.0,
        sequence=0,
        session_id=session_id,
        run_id=run_id,
        subject=SpanSubject.TURN,
        phase=SpanPhase.START,
        span_id=f"turn-{turn_number}",
        name=f"turn.{turn_number}",
        attributes={"turn_number": turn_number},
    )


def _turn_complete(*, session_id: str = "s", run_id: str = "r", turn_number: int = 1) -> Span:
    return Span(
        timestamp=1.4,
        sequence=4,
        session_id=session_id,
        run_id=run_id,
        subject=SpanSubject.TURN,
        phase=SpanPhase.COMPLETE,
        span_id=f"turn-{turn_number}",
        name=f"turn.{turn_number}",
        duration_ms=10.0,
        attributes={"turn_number": turn_number},
    )


def _tool_call_start(
    *,
    tool_name: str = "test-tool",
    call_id: str = "tc_01",
    session_id: str = "s",
    run_id: str = "r",
) -> Span:
    return Span(
        timestamp=1.2,
        sequence=2,
        session_id=session_id,
        run_id=run_id,
        subject=SpanSubject.TOOL_CALL,
        phase=SpanPhase.START,
        span_id=f"tc-{call_id}",
        name=f"tool.{tool_name}",
        attributes={"tool_name": tool_name, "call_id": call_id, "arguments": ()},
    )


def _tool_call_result(
    *,
    tool_name: str = "test-tool",
    call_id: str = "tc_01",
    is_error: bool = False,
    duration_ms: float = 100.0,
    session_id: str = "s",
    run_id: str = "r",
) -> Span:
    return Span(
        timestamp=1.3,
        sequence=3,
        session_id=session_id,
        run_id=run_id,
        subject=SpanSubject.TOOL_CALL,
        phase=SpanPhase.COMPLETE,
        span_id=f"tc-{call_id}",
        name=f"tool.{tool_name}",
        duration_ms=duration_ms,
        attributes={
            "tool_name": tool_name,
            "call_id": call_id,
            "result_summary": "OK",
            "is_error": is_error,
        },
    )


def _is_tool_call_start(event: object) -> bool:
    return (
        isinstance(event, Span)
        and event.subject == SpanSubject.TOOL_CALL
        and event.phase == SpanPhase.START
    )


def _is_turn_start(event: object) -> bool:
    return (
        isinstance(event, Span)
        and event.subject == SpanSubject.TURN
        and event.phase == SpanPhase.START
    )


def _is_turn_complete(event: object) -> bool:
    return (
        isinstance(event, Span)
        and event.subject == SpanSubject.TURN
        and event.phase == SpanPhase.COMPLETE
    )


@pytest.mark.unit
class TestHookAction:
    def test_values(self) -> None:
        assert HookAction.CONTINUE == "continue"
        assert HookAction.SKIP == "skip"
        assert HookAction.REQUIRE_APPROVAL == "approval"


@pytest.mark.unit
class TestBaseHook:
    @pytest.mark.asyncio
    async def test_default_no_ops(self) -> None:
        """BaseHook methods are no-ops that don't raise."""
        hook = BaseHook()
        await hook.on_turn_start(_turn_start())

    @pytest.mark.asyncio
    async def test_tool_call_start_returns_continue(self) -> None:
        """Default on_tool_call_start returns CONTINUE."""
        hook = BaseHook()
        action = await hook.on_tool_call_start(_tool_call_start())
        assert action == HookAction.CONTINUE


@pytest.mark.unit
class TestDispatchHook:
    @pytest.mark.asyncio
    async def test_dispatch_turn_start(self) -> None:
        """dispatch_hook calls on_turn_start for Span(TURN, START) events."""
        calls: list[Span] = []

        class TrackingHook(BaseHook):
            async def on_turn_start(self, event: Span) -> None:
                calls.append(event)

        event = _turn_start(turn_number=1)
        await dispatch_hook((TrackingHook(),), event)
        assert len(calls) == 1
        assert calls[0] is event

    @pytest.mark.asyncio
    async def test_dispatch_tool_call_start_returns_action(self) -> None:
        """dispatch_hook returns the action from on_tool_call_start."""

        class SkipHook(BaseHook):
            async def on_tool_call_start(self, event: Span) -> HookAction:
                return HookAction.SKIP

        action = await dispatch_hook((SkipHook(),), _tool_call_start(tool_name="dangerous-tool"))
        assert action == HookAction.SKIP

    @pytest.mark.asyncio
    async def test_most_restrictive_wins(self) -> None:
        """When multiple hooks return different actions, most restrictive wins."""

        class ContinueHook(BaseHook):
            async def on_tool_call_start(self, event: Span) -> HookAction:
                return HookAction.CONTINUE

        class ApprovalHook(BaseHook):
            async def on_tool_call_start(self, event: Span) -> HookAction:
                return HookAction.REQUIRE_APPROVAL

        action = await dispatch_hook((ContinueHook(), ApprovalHook()), _tool_call_start())
        assert action == HookAction.REQUIRE_APPROVAL

    @pytest.mark.asyncio
    async def test_hook_exception_swallowed(self) -> None:
        """Hook exceptions are logged and swallowed, not propagated."""

        class BrokenHook(BaseHook):
            async def on_turn_start(self, event: Span) -> None:
                msg = "hook exploded"
                raise RuntimeError(msg)

        # Should not raise
        action = await dispatch_hook((BrokenHook(),), _turn_start())
        assert action == HookAction.CONTINUE

    @pytest.mark.asyncio
    async def test_non_matching_event_returns_continue(self) -> None:
        """Events not matched by any isinstance branch return CONTINUE."""
        from kaos_agents.events import ThinkingDelta

        event = ThinkingDelta(
            timestamp=1.0, sequence=0, session_id="s", run_id="r", content="thinking"
        )
        action = await dispatch_hook((BaseHook(),), event)
        assert action == HookAction.CONTINUE


@pytest.mark.unit
class TestLoggingHook:
    @pytest.mark.asyncio
    async def test_logging_hook_runs(self) -> None:
        """LoggingHook handles events without error."""
        hook = LoggingHook()
        events = [
            _turn_start(),
            IntentClassified(
                timestamp=1.1,
                sequence=1,
                session_id="s",
                run_id="r",
                intent="respond",
                confidence=0.9,
                reasoning="test",
            ),
            _tool_call_start(),
            _tool_call_result(),
            _turn_complete(),
            TurnSummary(
                timestamp=1.4,
                sequence=4,
                session_id="s",
                run_id="r",
                text="Done",
                intent="respond",
            ),
            RunError(
                timestamp=1.5,
                sequence=5,
                session_id="s",
                run_id="r",
                error_type="test",
                message="test error",
                recovery_hint="retry",
            ),
        ]
        for event in events:
            await dispatch_hook((hook,), event)


@pytest.mark.unit
class TestRunnerHookIntegration:
    @pytest.mark.asyncio
    async def test_hooks_called_on_run(self) -> None:
        """Runner.run() dispatches hooks for each event."""
        seen_types: list[str] = []

        class TrackingHook(BaseHook):
            async def on_turn_start(self, event: Span) -> None:
                seen_types.append("turn_start")

            async def on_intent_classified(self, event: IntentClassified) -> None:
                seen_types.append("intent_classified")

            async def on_turn_complete(self, event: Span) -> None:
                seen_types.append("turn_complete")

            async def on_turn_summary(self, event: TurnSummary) -> None:
                seen_types.append("turn_summary")

        agent = Agent()
        runner = Runner(agent, hooks=(TrackingHook(),))

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Response", []),
            ),
        ):
            events = []
            async for event in runner.run("Hello", "test-session"):
                events.append(event)

        assert "turn_start" in seen_types
        assert "turn_complete" in seen_types
        assert "turn_summary" in seen_types
        assert "intent_classified" in seen_types

    @pytest.mark.asyncio
    async def test_skip_suppresses_event(self) -> None:
        """HookAction.SKIP prevents the event from being yielded."""

        class SkipTextHook(BaseHook):
            async def on_text_delta(self, event: TextDelta) -> None:
                pass

            async def on_tool_call_start(self, event: Span) -> HookAction:
                return HookAction.SKIP

        agent = Agent()
        runner = Runner(agent, hooks=(SkipTextHook(),))

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch(
                "kaos_agents.agent.BaseAgent._classify",
                new_callable=AsyncMock,
                return_value=mock_intent,
            ),
            patch(
                "kaos_agents.agent.BaseAgent._dispatch",
                new_callable=AsyncMock,
                return_value=("Response", []),
            ),
        ):
            events = []
            async for event in runner.run("Hello", "test-session"):
                events.append(event)

        # No tool-call START spans should appear (none were generated anyway
        # for a simple RESPOND, but the hook path is exercised)
        tool_starts = [e for e in events if _is_tool_call_start(e)]
        assert len(tool_starts) == 0

    @pytest.mark.asyncio
    async def test_skip_suppresses_real_tool_call_start(self) -> None:
        """HookAction.SKIP on a Span(TOOL_CALL, START) emitted by the dispatch
        path prevents it from being yielded by Runner.run()."""

        class _SkipAllTools(BaseHook):
            async def on_tool_call_start(self, event: Span) -> HookAction:
                return HookAction.SKIP

        agent = Agent()
        runner = Runner(agent, hooks=(_SkipAllTools(),))

        # Emit a tool-call START via a fake dispatch (bypasses real ReAct)
        async def _fake_run(_self, message, session_id):  # type: ignore[no-untyped-def]
            emitter = EventEmitter(session_id=session_id, run_id="r-skip-real")
            yield emitter.span_start(
                SpanSubject.TURN,
                name="turn.1",
                attributes={"turn_number": 1},
            )
            yield emitter.span_start(
                SpanSubject.TOOL_CALL,
                name="tool.some-tool",
                attributes={
                    "tool_name": "some-tool",
                    "call_id": "tc_1",
                    "arguments": (),
                },
            )
            yield emitter.span_complete(
                SpanSubject.TURN,
                span_id="turn-fake",
                name="turn.1",
                duration_ms=1.0,
                attributes={"turn_number": 1},
            )
            yield emitter.emit(
                TurnSummary,
                text="",
                intent="tool_use",
            )

        with patch("kaos_agents.agent.BaseAgent.run", _fake_run):
            events = []
            async for event in runner.run("use the tool", "s-skip-real"):
                events.append(event)

        # SKIP must have suppressed the tool-call START entirely
        tool_starts = [e for e in events if _is_tool_call_start(e)]
        assert len(tool_starts) == 0
        # But other events still flow through
        assert any(_is_turn_start(e) for e in events)
        assert any(isinstance(e, TurnSummary) for e in events)


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------


def _turn_summary(
    *, session_id: str, run_id: str, tokens_used: int, cost_usd: float
) -> TurnSummary:
    return TurnSummary(
        timestamp=1.0,
        sequence=0,
        session_id=session_id,
        run_id=run_id,
        text="",
        intent="respond",
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


@pytest.mark.unit
class TestCostTrackingHook:
    @pytest.mark.asyncio
    async def test_accumulates_tokens_and_cost(self) -> None:
        hook = CostTrackingHook()
        await hook.on_turn_summary(
            _turn_summary(session_id="s1", run_id="r1", tokens_used=100, cost_usd=0.01)
        )
        await hook.on_turn_summary(
            _turn_summary(session_id="s1", run_id="r2", tokens_used=50, cost_usd=0.005)
        )

        totals = hook.totals
        assert totals["s1"] == (150, 0.015, 2)

    @pytest.mark.asyncio
    async def test_per_session_isolation(self) -> None:
        hook = CostTrackingHook()
        await hook.on_turn_summary(
            _turn_summary(session_id="a", run_id="r1", tokens_used=100, cost_usd=0.01)
        )
        await hook.on_turn_summary(
            _turn_summary(session_id="b", run_id="r2", tokens_used=200, cost_usd=0.02)
        )
        totals = hook.totals
        assert totals["a"] == (100, 0.01, 1)
        assert totals["b"] == (200, 0.02, 1)

    @pytest.mark.asyncio
    async def test_reset_all(self) -> None:
        hook = CostTrackingHook()
        await hook.on_turn_summary(
            _turn_summary(session_id="s", run_id="r", tokens_used=100, cost_usd=0.01)
        )
        hook.reset()
        assert hook.totals == {}

    @pytest.mark.asyncio
    async def test_reset_single_session(self) -> None:
        hook = CostTrackingHook()
        await hook.on_turn_summary(
            _turn_summary(session_id="a", run_id="r1", tokens_used=100, cost_usd=0.01)
        )
        await hook.on_turn_summary(
            _turn_summary(session_id="b", run_id="r2", tokens_used=50, cost_usd=0.005)
        )
        hook.reset("a")
        assert "a" not in hook.totals
        assert "b" in hook.totals


@pytest.mark.unit
class TestAuditHook:
    @pytest.mark.asyncio
    async def test_appends_events_to_audit_section(self) -> None:
        """AuditHook writes events to memory AUDIT section via SessionStore."""
        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.memory.store import SessionStore
        from kaos_agents.types.memory import MemoryType

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        store = SessionStore(vfs)

        hook = AuditHook(store)
        session_id = "audit-test"

        event = _turn_start(session_id=session_id, run_id="r", turn_number=1)
        await hook.on_turn_start(event)

        # Reload and verify the audit section
        memory = await store.load(session_id)
        audit_items = memory.get(MemoryType.AUDIT)
        assert len(audit_items) == 1
        # Span events serialize with type "span"
        assert "span" in audit_items[0].content

    @pytest.mark.asyncio
    async def test_audit_failure_is_swallowed(self) -> None:
        """AuditHook errors do not propagate — audit must never break the loop."""

        class BrokenStore:
            async def load_or_create(self, session_id: str):  # type: ignore[no-untyped-def]
                raise RuntimeError("broken")

        hook = AuditHook(BrokenStore())
        # Must not raise
        await hook.on_turn_start(_turn_start())
