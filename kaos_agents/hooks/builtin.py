"""Built-in :class:`KaosHook` subclasses for common observability needs.

Three hooks ship out of the box:

- :class:`LoggingHook` — kaos-core structured logging at every event.
- :class:`CostTrackingHook` — per-session token + cost accumulator.
- :class:`AuditHook` — append every event to the AUDIT memory section
  for tamper-evident replay.

The OpenTelemetry hook (:class:`kaos_agents.hooks.otel.OTelHook`)
lives in its own module because it has an optional dependency
(``opentelemetry-api``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.hooks.base import HookAction, KaosHook
from kaos_agents.types.metadata import HookMetadata

if TYPE_CHECKING:
    from kaos_agents.events import (
        IntentClassified,
        KaosEvent,
        RunError,
        Span,
        TurnSummary,
    )

logger = get_logger(__name__)


class LoggingHook(KaosHook):
    """Logs all events via kaos-core structured logging.

    Useful for debugging and operational monitoring.
    All events are logged at DEBUG level except errors (WARNING).
    """

    @classmethod
    def metadata(cls) -> HookMetadata:
        return HookMetadata(
            name="kaos-agents-logging-hook",
            description="Logs lifecycle events via kaos-core structured logging.",
            listens_to=(
                "span",
                "intent_classified",
                "turn_summary",
                "run_error",
            ),
        )

    async def on_turn_start(self, event: Span) -> None:
        logger.debug(
            "hook.turn_start: session=%s turn=%s",
            event.session_id,
            event.attributes.get("turn_number"),
        )

    async def on_intent_classified(self, event: IntentClassified) -> None:
        logger.debug(
            "hook.intent: %s (confidence=%.2f)",
            event.intent,
            event.confidence,
        )

    async def on_tool_call_start(self, event: Span) -> HookAction:
        attrs = event.attributes
        logger.debug(
            "hook.tool_start: %s (call_id=%s)",
            attrs.get("tool_name", ""),
            attrs.get("call_id", ""),
        )
        return HookAction.CONTINUE

    async def on_tool_call_result(self, event: Span) -> None:
        attrs = event.attributes
        is_error = bool(attrs.get("is_error", False))
        level = "warning" if is_error else "debug"
        getattr(logger, level)(
            "hook.tool_result: %s %s (%.0fms)",
            attrs.get("tool_name", ""),
            "ERROR" if is_error else "OK",
            event.duration_ms or 0.0,
        )

    async def on_step_start(self, event: Span) -> None:
        attrs = event.attributes
        logger.debug(
            "hook.step_start: %s — %s",
            attrs.get("step_id", ""),
            attrs.get("description", ""),
        )

    async def on_step_complete(self, event: Span) -> None:
        attrs = event.attributes
        is_error = bool(attrs.get("is_error", False))
        logger.debug(
            "hook.step_complete: %s %s (%.0fms)",
            attrs.get("step_id", ""),
            "ERROR" if is_error else "OK",
            event.duration_ms or 0.0,
        )

    async def on_turn_complete(self, event: Span) -> None:
        logger.debug(
            "hook.turn_complete: session=%s duration_ms=%.0f",
            event.session_id,
            event.duration_ms or 0.0,
        )

    async def on_turn_summary(self, event: TurnSummary) -> None:
        logger.debug(
            "hook.turn_summary: session=%s tokens=%d cost=$%.4f",
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


class CostTrackingHook(KaosHook):
    """Tracks token usage and cost per session.

    Accumulates ``TurnSummary.tokens_used`` and ``TurnSummary.cost_usd``
    into a per-session totals dict. Useful for billing, budget enforcement,
    and usage reports.

    Inspect via ``hook.totals[session_id]`` to get
    ``(total_tokens, total_cost_usd, turn_count)``.
    """

    __slots__ = ("_totals",)

    def __init__(self) -> None:
        # Mutable accumulator — this hook is stateful by design.
        self._totals: dict[str, tuple[int, float, int]] = {}

    @classmethod
    def metadata(cls) -> HookMetadata:
        return HookMetadata(
            name="kaos-agents-cost-tracking-hook",
            description="Per-session token + cost accumulator from TurnSummary.",
            listens_to=("turn_summary",),
        )

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

    async def on_turn_summary(self, event: TurnSummary) -> None:
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


class AuditHook(KaosHook):
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

    @classmethod
    def metadata(cls) -> HookMetadata:
        return HookMetadata(
            name="kaos-agents-audit-hook",
            description="Appends every event to the session AUDIT memory section.",
            # Empty listens_to = listens to all events.
            listens_to=(),
        )

    async def _append(self, event: KaosEvent) -> None:
        """Append a single event to the AUDIT section."""
        # Lazy imports to keep hooks/builtin.py import-light
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

    async def on_turn_start(self, event: Span) -> None:
        await self._append(event)

    async def on_intent_classified(self, event: IntentClassified) -> None:
        await self._append(event)

    async def on_tool_call_start(self, event: Span) -> HookAction:
        await self._append(event)
        return HookAction.CONTINUE

    async def on_tool_call_result(self, event: Span) -> None:
        await self._append(event)

    async def on_step_start(self, event: Span) -> None:
        await self._append(event)

    async def on_step_complete(self, event: Span) -> None:
        await self._append(event)

    async def on_turn_complete(self, event: Span) -> None:
        await self._append(event)

    async def on_turn_summary(self, event: TurnSummary) -> None:
        await self._append(event)

    async def on_error(self, event: RunError) -> None:
        await self._append(event)


__all__ = [
    "AuditHook",
    "CostTrackingHook",
    "LoggingHook",
]
