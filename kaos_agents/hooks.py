"""Lifecycle hooks for the agent Runner.

Hooks provide deterministic interception points around agent events.
Implement a hook to add logging, metrics, guardrails, approval UI,
cost tracking, or audit trails — without modifying the agent itself.

Two ways to use hooks:

1. **Subclass ``BaseHook``** and override the methods you care about.
   Unoverridden methods are no-ops. This is the recommended pattern.

2. **Implement the structural typing contract** directly — any object
   with the right ``async def on_*`` methods works, no inheritance needed.

Usage::

    class MyHook(BaseHook):
        async def on_tool_call_start(self, event: ToolCallStart) -> HookAction:
            if event.tool_name.startswith("kaos-web-delete"):
                return HookAction.REQUIRE_APPROVAL
            return HookAction.CONTINUE

    runner = Runner(agent, hooks=(MyHook(), LoggingHook()))
    async for event in runner.run("Delete the file", "session-1"):
        ...

Design grounded in competitive research:
- OpenAI Agents SDK: RunHooks with typed per-event callbacks
- Anthropic Claude SDK: 10 hook event types with typed inputs
- CrewAI: ~40 event types via pub/sub event bus

We chose the OpenAI pattern (typed callbacks per event, compose in order)
over the CrewAI pattern (pub/sub) for simplicity and type safety.
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_agents.events import (
        CitationFound,
        EvidenceInsufficient,
        IntentClassified,
        KaosEvent,
        MemoryUpdated,
        PlanProposed,
        RunError,
        StepComplete,
        StepStart,
        TextDelta,
        ToolCallResult,
        ToolCallStart,
        TurnComplete,
        TurnStart,
    )

logger = get_logger(__name__)


@unique
class HookAction(StrEnum):
    """What the hook tells the Runner to do after processing an event.

    Returned by ``on_tool_call_start`` to control tool execution.
    Other hook methods return ``None`` (they observe but don't control).
    """

    CONTINUE = "continue"  # Proceed normally
    SKIP = "skip"  # Skip this tool call (don't execute)
    REQUIRE_APPROVAL = "approval"  # Pause for human approval


class BaseHook:
    """Base hook with no-op defaults for all lifecycle events.

    Subclass and override only the events you care about.
    All methods are async to support I/O-bound hooks (logging to
    remote services, database writes, HTTP callbacks).

    Hook methods receive the event that just occurred. Most return
    ``None`` (observe only). ``on_tool_call_start`` returns a
    ``HookAction`` to control whether the tool executes.

    Hooks are called in registration order. If multiple hooks return
    different ``HookAction`` values, the most restrictive wins:
    ``REQUIRE_APPROVAL > SKIP > CONTINUE``.
    """

    async def on_turn_start(self, event: TurnStart) -> None:
        """Called at the beginning of a turn."""

    async def on_intent_classified(self, event: IntentClassified) -> None:
        """Called after intent classification."""

    async def on_tool_call_start(self, event: ToolCallStart) -> HookAction:
        """Called before a tool is executed. Return action to control execution."""
        return HookAction.CONTINUE

    async def on_tool_call_result(self, event: ToolCallResult) -> None:
        """Called after a tool completes."""

    async def on_text_delta(self, event: TextDelta) -> None:
        """Called for each streaming text chunk."""

    async def on_step_start(self, event: StepStart) -> None:
        """Called when a plan step begins."""

    async def on_step_complete(self, event: StepComplete) -> None:
        """Called when a plan step completes."""

    async def on_plan_proposed(self, event: PlanProposed) -> None:
        """Called when a plan is proposed before execution."""

    async def on_citation_found(self, event: CitationFound) -> None:
        """Called when RAG verifies a citation."""

    async def on_evidence_insufficient(self, event: EvidenceInsufficient) -> None:
        """Called when RAG cannot find sufficient evidence."""

    async def on_memory_updated(self, event: MemoryUpdated) -> None:
        """Called when a memory section is modified."""

    async def on_turn_complete(self, event: TurnComplete) -> None:
        """Called at the end of a turn."""

    async def on_error(self, event: RunError) -> None:
        """Called when an error occurs during execution."""


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------


class LoggingHook(BaseHook):
    """Logs all events via kaos-core structured logging.

    Useful for debugging and operational monitoring.
    All events are logged at DEBUG level except errors (WARNING).
    """

    async def on_turn_start(self, event: TurnStart) -> None:
        logger.debug(
            "hook.turn_start: session=%s turn=%d",
            event.session_id,
            event.turn_number,
        )

    async def on_intent_classified(self, event: IntentClassified) -> None:
        logger.debug(
            "hook.intent: %s (confidence=%.2f)",
            event.intent,
            event.confidence,
        )

    async def on_tool_call_start(self, event: ToolCallStart) -> HookAction:
        logger.debug("hook.tool_start: %s (call_id=%s)", event.tool_name, event.call_id)
        return HookAction.CONTINUE

    async def on_tool_call_result(self, event: ToolCallResult) -> None:
        level = "warning" if event.is_error else "debug"
        getattr(logger, level)(
            "hook.tool_result: %s %s (%.0fms)",
            event.tool_name,
            "ERROR" if event.is_error else "OK",
            event.duration_ms,
        )

    async def on_step_start(self, event: StepStart) -> None:
        logger.debug("hook.step_start: %s — %s", event.step_id, event.description)

    async def on_step_complete(self, event: StepComplete) -> None:
        logger.debug(
            "hook.step_complete: %s %s (%.0fms)",
            event.step_id,
            "ERROR" if event.is_error else "OK",
            event.duration_ms,
        )

    async def on_turn_complete(self, event: TurnComplete) -> None:
        logger.debug(
            "hook.turn_complete: session=%s tokens=%d cost=$%.4f",
            event.session_id,
            event.tokens_used,
            event.cost_usd,
        )

    async def on_error(self, event: RunError) -> None:
        logger.warning(
            "hook.error: %s — %s. Recovery: %s",
            event.error_type,
            event.message,
            event.recovery_hint,
        )


class CostTrackingHook(BaseHook):
    """Tracks token usage and cost per session.

    Accumulates ``TurnComplete.tokens_used`` and ``TurnComplete.cost_usd``
    into a per-session totals dict. Useful for billing, budget enforcement,
    and usage reports.

    Inspect via ``hook.totals[session_id]`` to get
    ``(total_tokens, total_cost_usd, turn_count)``.
    """

    __slots__ = ("_totals",)

    def __init__(self) -> None:
        # Mutable accumulator — this hook is stateful by design.
        self._totals: dict[str, tuple[int, float, int]] = {}

    @property
    def totals(self) -> dict[str, tuple[int, float, int]]:
        """Per-session (tokens, cost_usd, turn_count) totals."""
        return dict(self._totals)

    def reset(self, session_id: str | None = None) -> None:
        """Reset accumulators. Pass a session_id to clear one, or None for all."""
        if session_id is None:
            self._totals.clear()
        else:
            self._totals.pop(session_id, None)

    async def on_turn_complete(self, event: TurnComplete) -> None:
        tokens, cost, turns = self._totals.get(event.session_id, (0, 0.0, 0))
        self._totals[event.session_id] = (
            tokens + event.tokens_used,
            cost + event.cost_usd,
            turns + 1,
        )
        logger.debug(
            "hook.cost: session=%s turn_tokens=%d turn_cost=$%.4f "
            "total_tokens=%d total_cost=$%.4f turns=%d",
            event.session_id,
            event.tokens_used,
            event.cost_usd,
            tokens + event.tokens_used,
            cost + event.cost_usd,
            turns + 1,
        )


class AuditHook(BaseHook):
    """Appends every event to the AUDIT memory section for tamper-evident log.

    Requires a ``SessionStore`` to write to memory. The AUDIT section is
    pre-configured with ``EvictionPolicy.NONE`` and streaming persistence
    (see SessionMemory default sections).

    Serializes each event to JSON and appends as a memory item. Later,
    ``memory.get(MemoryType.AUDIT)`` returns the full event log for the
    session.
    """

    __slots__ = ("_store",)

    def __init__(self, store: Any) -> None:
        """
        Args:
            store: A ``SessionStore`` instance (from ``kaos_agents.memory.store``).
        """
        self._store = store

    async def _append(self, event: KaosEvent) -> None:
        """Append a single event to the AUDIT section."""
        # Lazy imports to keep hooks.py import-light
        from kaos_agents.events import serialize_event_json
        from kaos_agents.types.memory import MemoryType

        try:
            memory = await self._store.load_or_create(event.session_id)
            memory.add(
                MemoryType.AUDIT,
                serialize_event_json(event),
                metadata={"sequence": event.sequence, "run_id": event.run_id},
            )
            await self._store.save(memory)
        except Exception as exc:
            logger.warning("hook.audit: failed to persist event (swallowed): %s", exc)

    async def on_turn_start(self, event: TurnStart) -> None:
        await self._append(event)

    async def on_intent_classified(self, event: IntentClassified) -> None:
        await self._append(event)

    async def on_tool_call_start(self, event: ToolCallStart) -> HookAction:
        await self._append(event)
        return HookAction.CONTINUE

    async def on_tool_call_result(self, event: ToolCallResult) -> None:
        await self._append(event)

    async def on_step_start(self, event: StepStart) -> None:
        await self._append(event)

    async def on_step_complete(self, event: StepComplete) -> None:
        await self._append(event)

    async def on_turn_complete(self, event: TurnComplete) -> None:
        await self._append(event)

    async def on_error(self, event: RunError) -> None:
        await self._append(event)


# ---------------------------------------------------------------------------
# Hook dispatch helper (used by Runner)
# ---------------------------------------------------------------------------


# Priority order for actions (higher index = more restrictive).
# Module-level constant — avoids re-creating on every dispatch_hook call.
_ACTION_PRIORITY: dict[HookAction, int] = {
    HookAction.CONTINUE: 0,
    HookAction.SKIP: 1,
    HookAction.REQUIRE_APPROVAL: 2,
}


async def dispatch_hook(
    hooks: tuple[BaseHook, ...],
    event: KaosEvent,
) -> HookAction:
    """Run all hooks for an event, returning the most restrictive action.

    For events that return HookAction (on_tool_call_start), the most
    restrictive action wins: REQUIRE_APPROVAL > SKIP > CONTINUE.

    For events that return None, this always returns CONTINUE.

    Hook exceptions are logged and swallowed — hooks never affect
    the control flow of the agent loop.
    """
    from kaos_agents.events import (
        CitationFound,
        EvidenceInsufficient,
        IntentClassified,
        MemoryUpdated,
        PlanProposed,
        RunError,
        StepComplete,
        StepStart,
        TextDelta,
        ToolCallResult,
        ToolCallStart,
        TurnComplete,
        TurnStart,
    )

    most_restrictive = HookAction.CONTINUE

    for hook in hooks:
        try:
            if isinstance(event, TurnStart):
                await hook.on_turn_start(event)
            elif isinstance(event, IntentClassified):
                await hook.on_intent_classified(event)
            elif isinstance(event, ToolCallStart):
                action = await hook.on_tool_call_start(event)
                if _ACTION_PRIORITY.get(action, 0) > _ACTION_PRIORITY[most_restrictive]:
                    most_restrictive = action
            elif isinstance(event, ToolCallResult):
                await hook.on_tool_call_result(event)
            elif isinstance(event, TextDelta):
                await hook.on_text_delta(event)
            elif isinstance(event, StepStart):
                await hook.on_step_start(event)
            elif isinstance(event, StepComplete):
                await hook.on_step_complete(event)
            elif isinstance(event, PlanProposed):
                await hook.on_plan_proposed(event)
            elif isinstance(event, CitationFound):
                await hook.on_citation_found(event)
            elif isinstance(event, EvidenceInsufficient):
                await hook.on_evidence_insufficient(event)
            elif isinstance(event, MemoryUpdated):
                await hook.on_memory_updated(event)
            elif isinstance(event, TurnComplete):
                await hook.on_turn_complete(event)
            elif isinstance(event, RunError):
                await hook.on_error(event)
        except Exception as exc:
            logger.warning(
                "Hook %s raised %s on %s (swallowed): %s",
                type(hook).__name__,
                type(exc).__name__,
                type(event).__name__,
                exc,
            )

    return most_restrictive
