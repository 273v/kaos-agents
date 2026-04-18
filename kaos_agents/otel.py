"""OpenTelemetry hook for kaos-agents.

Emits OTel spans for agent lifecycle events so traces appear in
Jaeger, Grafana Tempo, or any OTLP-compatible backend.

Span hierarchy::

    [turn]
      ├─ [intent_classification]
      ├─ [tool_call: tool_name]
      ├─ [tool_call: tool_name]
      ├─ [step: description]  (plan-execute only)
      │    └─ [tool_call: ...]
      └─ [step: ...]

Requires ``opentelemetry-api`` (the ``[otel]`` extra). Construction
raises ``ImportError`` early if the package is missing.

Usage::

    from kaos_agents.otel import OTelHook

    hook = OTelHook(service_name="kaos-agents")
    runner = Runner(agent, hooks=(hook,))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_agents.hooks import BaseHook, HookAction

if TYPE_CHECKING:
    from kaos_agents.events import (
        IntentClassified,
        RunError,
        StepComplete,
        StepStart,
        ToolCallResult,
        ToolCallStart,
        TurnComplete,
        TurnStart,
    )

logger = get_logger(__name__)


class OTelHook(BaseHook):
    """Emits OpenTelemetry spans for agent lifecycle events.

    Each turn becomes a parent span. Tool calls and plan steps become
    child spans within the turn. All spans carry ``kaos.*`` attributes
    for filtering in your tracing backend.

    Active spans are tracked in three dicts keyed by ``(session_id, run_id)``
    for turns, ``call_id`` for tool calls, and ``step_id`` for plan steps.
    Spans are ended on the corresponding ``_complete`` or ``_error`` event.

    Args:
        service_name: OTel service name (unused here but conventional).
        tracer_name: OTel tracer instrument name. Defaults to ``"kaos.agents"``.
    """

    __slots__ = (
        "_step_spans",
        "_tool_spans",
        "_tracer",
        "_turn_spans",
    )

    def __init__(
        self,
        *,
        service_name: str = "kaos-agents",
        tracer_name: str = "kaos.agents",
    ) -> None:
        try:
            from opentelemetry import trace  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "OTelHook requires opentelemetry-api. "
                "Install with: uv pip install kaos-agents[otel]. "
                "Alternative: use LoggingHook for structured logging without OTel."
            ) from exc

        self._tracer = trace.get_tracer(tracer_name)
        self._turn_spans: dict[tuple[str, str], Any] = {}
        self._tool_spans: dict[str, Any] = {}
        self._step_spans: dict[str, Any] = {}

    def _turn_key(self, event: Any) -> tuple[str, str]:
        return (event.session_id, event.run_id)

    async def on_turn_start(self, event: TurnStart) -> None:
        """Start a parent ``agent.turn`` span."""
        span = self._tracer.start_span(
            "agent.turn",
            attributes={
                "kaos.session_id": event.session_id,
                "kaos.run_id": event.run_id,
                "kaos.turn_number": event.turn_number,
            },
        )
        self._turn_spans[self._turn_key(event)] = span
        logger.debug("otel.turn_start: session=%s turn=%d", event.session_id, event.turn_number)

    async def on_intent_classified(self, event: IntentClassified) -> None:
        """Create and immediately end an ``agent.intent_classification`` child span."""
        key = self._turn_key(event)
        parent = self._turn_spans.get(key)
        if parent is None:
            return

        from opentelemetry import trace  # type: ignore[import-not-found]

        ctx = trace.set_span_in_context(parent)
        span = self._tracer.start_span(
            "agent.intent_classification",
            context=ctx,
            attributes={
                "kaos.intent": event.intent,
                "kaos.confidence": event.confidence,
                "kaos.reasoning": event.reasoning[:200] if event.reasoning else "",
            },
        )
        span.end()

    async def on_tool_call_start(self, event: ToolCallStart) -> HookAction:
        """Start an ``agent.tool_call.{name}`` child span under the turn."""
        key = self._turn_key(event)
        parent = self._turn_spans.get(key)

        from opentelemetry import trace  # type: ignore[import-not-found]

        ctx = trace.set_span_in_context(parent) if parent is not None else None

        span = self._tracer.start_span(
            f"agent.tool_call.{event.tool_name}",
            context=ctx,
            attributes={
                "kaos.tool_name": event.tool_name,
                "kaos.call_id": event.call_id,
                "kaos.session_id": event.session_id,
            },
        )
        self._tool_spans[event.call_id] = span
        return HookAction.CONTINUE

    async def on_tool_call_result(self, event: ToolCallResult) -> None:
        """End the tool call span with status and duration."""
        span = self._tool_spans.pop(event.call_id, None)
        if span is None:
            return

        from opentelemetry.trace import StatusCode  # type: ignore[import-not-found]

        span.set_attribute("kaos.duration_ms", event.duration_ms)
        span.set_attribute("kaos.is_error", event.is_error)
        if event.is_error:
            span.set_status(StatusCode.ERROR, event.result_summary[:200])
        else:
            span.set_status(StatusCode.OK)
        span.end()

    async def on_step_start(self, event: StepStart) -> None:
        """Start an ``agent.step.{id}`` child span under the turn."""
        key = self._turn_key(event)
        parent = self._turn_spans.get(key)

        from opentelemetry import trace  # type: ignore[import-not-found]

        ctx = trace.set_span_in_context(parent) if parent is not None else None

        span = self._tracer.start_span(
            f"agent.step.{event.step_id}",
            context=ctx,
            attributes={
                "kaos.step_id": event.step_id,
                "kaos.description": event.description[:200],
                "kaos.session_id": event.session_id,
            },
        )
        self._step_spans[event.step_id] = span

    async def on_step_complete(self, event: StepComplete) -> None:
        """End the step span with status and duration."""
        span = self._step_spans.pop(event.step_id, None)
        if span is None:
            return

        from opentelemetry.trace import StatusCode  # type: ignore[import-not-found]

        span.set_attribute("kaos.duration_ms", event.duration_ms)
        span.set_attribute("kaos.is_error", event.is_error)
        if event.is_error:
            span.set_status(StatusCode.ERROR, event.result_summary[:200])
        else:
            span.set_status(StatusCode.OK)
        span.end()

    async def on_turn_complete(self, event: TurnComplete) -> None:
        """End the turn span with token/cost attributes."""
        key = self._turn_key(event)
        span = self._turn_spans.pop(key, None)
        if span is None:
            return

        from opentelemetry.trace import StatusCode  # type: ignore[import-not-found]

        span.set_attribute("kaos.tokens_used", event.tokens_used)
        span.set_attribute("kaos.cost_usd", event.cost_usd)
        span.set_attribute("kaos.intent", event.intent)
        span.set_attribute("kaos.tool_call_count", len(event.tool_calls))
        span.set_status(StatusCode.OK)
        span.end()
        logger.debug(
            "otel.turn_complete: session=%s tokens=%d cost=$%.4f",
            event.session_id,
            event.tokens_used,
            event.cost_usd,
        )

    async def on_error(self, event: RunError) -> None:
        """End the turn span with ERROR status."""
        key = self._turn_key(event)
        span = self._turn_spans.pop(key, None)
        if span is None:
            return

        from opentelemetry.trace import StatusCode  # type: ignore[import-not-found]

        span.set_status(StatusCode.ERROR, event.message[:200])
        span.set_attribute("kaos.error_type", event.error_type)
        span.set_attribute("kaos.recovery_hint", event.recovery_hint[:200])
        span.end()
        logger.warning("otel.error: %s — %s", event.error_type, event.message)
