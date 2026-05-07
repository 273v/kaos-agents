"""Tests for kaos_agents.decorators.hook — Tier-1 on-ramp.

Covers:
- Bare @hook usage with type-annotation-driven listens_to inference
- @hook(name=..., listens_to=...) explicit kwargs
- Auto-registration into default_hook_registry
- register=False opt-out
- Custom registry= target
- Sync function rejection
- Tier-1 (decorator) and Tier-2 (subclass) coexistence in same Runner
"""

from __future__ import annotations

import pytest

from kaos_agents.decorators import FunctionHook, hook
from kaos_agents.events import (
    EventEmitter,
    IntentClassified,
    Span,
    SpanSubject,
    TurnSummary,
)
from kaos_agents.hooks import HookAction, KaosHook, dispatch_hook
from kaos_agents.registry import HookRegistry, default_hook_registry


@pytest.fixture
def emitter() -> EventEmitter:
    return EventEmitter(session_id="s", run_id="r")


@pytest.mark.unit
class TestBareDecorator:
    """@hook with no kwargs — infers listens_to from the function signature."""

    def setup_method(self) -> None:
        # Snapshot registry to detect leaks
        self._snapshot = set(default_event_registry_safe(default_hook_registry))

    def teardown_method(self) -> None:
        # Drop anything the test added so registry stays clean across tests.
        for name in default_event_registry_safe(default_hook_registry):
            if name not in self._snapshot:
                default_hook_registry.unregister(name)

    @pytest.mark.asyncio
    async def test_intent_classified_inferred_from_annotation(self, emitter: EventEmitter) -> None:
        called_with: list[IntentClassified] = []

        @hook
        async def my_intent_hook(event: IntentClassified) -> None:
            """Captures intents."""
            called_with.append(event)

        # Auto-registered.
        assert isinstance(my_intent_hook, FunctionHook)
        registered = default_hook_registry.get(my_intent_hook.metadata().name)
        assert registered is my_intent_hook
        assert my_intent_hook.metadata().listens_to == ("intent_classified",)

        # Dispatch an IntentClassified event — fires.
        ev = emitter.emit(
            IntentClassified, intent="tool_use", confidence=0.9, reasoning="has tools"
        )
        await dispatch_hook((my_intent_hook,), ev)
        assert len(called_with) == 1
        assert called_with[0].intent == "tool_use"

        # Dispatch a TurnSummary — does NOT fire (different type).
        summary = emitter.emit(TurnSummary, text="ok", intent="respond")
        await dispatch_hook((my_intent_hook,), summary)
        assert len(called_with) == 1, "hook fired on the wrong event type"


@pytest.mark.unit
class TestExplicitKwargs:
    @pytest.mark.asyncio
    async def test_listens_to_explicit(self, emitter: EventEmitter) -> None:
        called = 0

        @hook(name="kaos-agents-test-explicit", listens_to=("turn_summary",), register=False)
        async def watch_summary(event: TurnSummary) -> None:
            nonlocal called
            called += 1

        # register=False so we don't pollute the global default.
        assert default_hook_registry.get("kaos-agents-test-explicit") is None

        # Dispatch matching event — fires.
        summary = emitter.emit(TurnSummary, text="ok")
        await dispatch_hook((watch_summary,), summary)
        assert called == 1

    @pytest.mark.asyncio
    async def test_custom_registry(self) -> None:
        custom = HookRegistry()

        @hook(name="kaos-agents-test-custom-reg", registry=custom)
        async def my_hook(event: IntentClassified) -> None:
            """Lives in a custom registry."""

        assert custom.get("kaos-agents-test-custom-reg") is my_hook
        assert default_hook_registry.get("kaos-agents-test-custom-reg") is None

    @pytest.mark.asyncio
    async def test_listen_to_all_explicit_empty_tuple(self, emitter: EventEmitter) -> None:
        """listens_to=() means 'fire on every event' (catch-all)."""
        events_seen: list[str] = []

        @hook(name="kaos-agents-test-catchall", listens_to=(), register=False)
        async def catchall(event: object) -> None:
            events_seen.append(type(event).__name__)

        # Dispatch a few different event types — all fire.
        await dispatch_hook(
            (catchall,),
            emitter.emit(IntentClassified, intent="respond", confidence=0.5, reasoning=""),
        )
        await dispatch_hook(
            (catchall,),
            emitter.emit(TurnSummary, text="ok"),
        )
        assert events_seen == ["IntentClassified", "TurnSummary"]


@pytest.mark.unit
class TestRejectSyncFunctions:
    def test_sync_def_rejected(self) -> None:
        """Hooks must be async — kaos-agents only awaits."""
        with pytest.raises(TypeError, match="async function"):

            @hook  # ty: ignore[invalid-argument-type]
            def sync_hook(event: IntentClassified) -> None:
                pass


@pytest.mark.unit
class TestCoexistenceOfTiers:
    """Tier-1 (decorator) and Tier-2 (subclass) hooks compose identically."""

    @pytest.mark.asyncio
    async def test_decorator_and_subclass_in_same_pipeline(self, emitter: EventEmitter) -> None:
        decorator_calls: list[str] = []
        subclass_calls: list[str] = []

        @hook(name="kaos-agents-test-tier1", register=False)
        async def my_decorator_hook(event: IntentClassified) -> None:
            decorator_calls.append(event.intent)

        class MySubclassHook(KaosHook):
            """Tier-2 sibling."""

            async def on_intent_classified(self, event: IntentClassified) -> None:
                subclass_calls.append(event.intent)

        sub_hook = MySubclassHook()

        ev = emitter.emit(IntentClassified, intent="research", confidence=0.7, reasoning="docs")
        await dispatch_hook((my_decorator_hook, sub_hook), ev)

        assert decorator_calls == ["research"]
        assert subclass_calls == ["research"]


@pytest.mark.unit
class TestToolCallGate:
    """Decorator-wrapped hooks can return HookAction to gate tool calls."""

    @pytest.mark.asyncio
    async def test_decorator_returns_require_approval(self, emitter: EventEmitter) -> None:
        @hook(name="kaos-agents-test-gate", listens_to=("span",), register=False)
        async def gate(event: Span) -> HookAction:
            attrs = event.attributes
            if attrs.get("tool_name", "").startswith("kaos-web-delete"):
                return HookAction.REQUIRE_APPROVAL
            return HookAction.CONTINUE

        # Approval-required path
        ev = emitter.span_start(
            SpanSubject.TOOL_CALL,
            attributes={"tool_name": "kaos-web-delete-page", "call_id": "tc1", "arguments": ()},
        )
        action = await dispatch_hook((gate,), ev)
        assert action == HookAction.REQUIRE_APPROVAL

        # Continue path
        ev2 = emitter.span_start(
            SpanSubject.TOOL_CALL,
            attributes={"tool_name": "kaos-source-search", "call_id": "tc2", "arguments": ()},
        )
        action2 = await dispatch_hook((gate,), ev2)
        assert action2 == HookAction.CONTINUE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_event_registry_safe(reg: HookRegistry) -> tuple[str, ...]:
    """Snapshot of registry names — defensive copy that's robust to changes."""
    return tuple(reg.list_names())
