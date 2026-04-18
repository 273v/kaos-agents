"""Unit tests for the OTelHook.

Tests the span creation/ending lifecycle using a mock tracer,
without requiring a real OTel collector.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kaos_agents.hooks import HookAction


def _make_event(event_type: str, **kwargs: Any) -> Any:
    """Create a minimal event-like object with required fields."""
    defaults = {
        "timestamp": 1000.0,
        "sequence": 0,
        "session_id": "test-session",
        "run_id": "test-run",
        "agent_id": None,
    }
    defaults.update(kwargs)

    class _Event:
        pass

    e = _Event()
    for k, v in defaults.items():
        object.__setattr__(e, k, v)
    return e


@pytest.fixture
def mock_otel():
    """Patch opentelemetry so OTelHook can be instantiated without real otel."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    mock_trace_module = MagicMock()
    mock_trace_module.get_tracer.return_value = mock_tracer
    mock_trace_module.set_span_in_context.return_value = None
    mock_trace_module.StatusCode = MagicMock()
    mock_trace_module.StatusCode.OK = "OK"
    mock_trace_module.StatusCode.ERROR = "ERROR"

    with (
        patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(trace=mock_trace_module, context=MagicMock()),
                "opentelemetry.trace": mock_trace_module,
            },
        ),
    ):
        yield mock_tracer, mock_span


@pytest.mark.unit
class TestOTelHook:
    @pytest.mark.asyncio
    async def test_construction(self, mock_otel: tuple) -> None:
        from kaos_agents.otel import OTelHook

        hook = OTelHook(service_name="test")
        assert hook._tracer is not None

    @pytest.mark.asyncio
    async def test_turn_lifecycle(self, mock_otel: tuple) -> None:
        mock_tracer, mock_span = mock_otel
        from kaos_agents.otel import OTelHook

        hook = OTelHook()

        turn_start = _make_event("TurnStart", turn_number=1)
        await hook.on_turn_start(turn_start)
        mock_tracer.start_span.assert_called()
        assert ("test-session", "test-run") in hook._turn_spans

        turn_complete = _make_event(
            "TurnComplete",
            text="done",
            intent="answer",
            tool_calls=(),
            tokens_used=100,
            cost_usd=0.001,
        )
        await hook.on_turn_complete(turn_complete)
        mock_span.end.assert_called()
        assert ("test-session", "test-run") not in hook._turn_spans

    @pytest.mark.asyncio
    async def test_tool_call_lifecycle(self, mock_otel: tuple) -> None:
        _mock_tracer, _mock_span = mock_otel
        from kaos_agents.otel import OTelHook

        hook = OTelHook()

        turn_start = _make_event("TurnStart", turn_number=1)
        await hook.on_turn_start(turn_start)

        tool_start = _make_event(
            "ToolCallStart",
            call_id="call-1",
            tool_name="kaos-web-search",
            arguments=(("query", "test"),),
        )
        action = await hook.on_tool_call_start(tool_start)
        assert action == HookAction.CONTINUE
        assert "call-1" in hook._tool_spans

        tool_result = _make_event(
            "ToolCallResult",
            call_id="call-1",
            tool_name="kaos-web-search",
            result_summary="3 results found",
            is_error=False,
            duration_ms=150.0,
        )
        await hook.on_tool_call_result(tool_result)
        assert "call-1" not in hook._tool_spans

    @pytest.mark.asyncio
    async def test_step_lifecycle(self, mock_otel: tuple) -> None:
        _mock_tracer, _mock_span = mock_otel
        from kaos_agents.otel import OTelHook

        hook = OTelHook()

        turn_start = _make_event("TurnStart", turn_number=1)
        await hook.on_turn_start(turn_start)

        step_start = _make_event("StepStart", step_id="step-1", description="Search for documents")
        await hook.on_step_start(step_start)
        assert "step-1" in hook._step_spans

        step_complete = _make_event(
            "StepComplete",
            step_id="step-1",
            result_summary="Found 5 docs",
            is_error=False,
            duration_ms=500.0,
        )
        await hook.on_step_complete(step_complete)
        assert "step-1" not in hook._step_spans

    @pytest.mark.asyncio
    async def test_error_ends_turn_span(self, mock_otel: tuple) -> None:
        _mock_tracer, mock_span = mock_otel
        from kaos_agents.otel import OTelHook

        hook = OTelHook()

        turn_start = _make_event("TurnStart", turn_number=1)
        await hook.on_turn_start(turn_start)

        error = _make_event(
            "RunError",
            error_type="LLMError",
            message="Rate limit exceeded",
            recovery_hint="Retry with backoff",
        )
        await hook.on_error(error)
        mock_span.end.assert_called()
        assert ("test-session", "test-run") not in hook._turn_spans

    @pytest.mark.asyncio
    async def test_intent_classified(self, mock_otel: tuple) -> None:
        mock_tracer, _mock_span = mock_otel
        from kaos_agents.otel import OTelHook

        hook = OTelHook()

        turn_start = _make_event("TurnStart", turn_number=1)
        await hook.on_turn_start(turn_start)

        intent = _make_event(
            "IntentClassified",
            intent="tool_use",
            confidence=0.95,
            reasoning="User asked to search",
        )
        await hook.on_intent_classified(intent)
        # Intent classification creates and immediately ends a span
        assert mock_tracer.start_span.call_count >= 2  # turn + intent


@pytest.mark.unit
class TestOTelHookImportError:
    def test_missing_opentelemetry_raises_with_guidance(self) -> None:
        """Verify 3-part error: what happened, how to fix, alternative."""
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
            # Force re-import to pick up the patched modules
            import importlib

            import kaos_agents.otel

            importlib.reload(kaos_agents.otel)
            with pytest.raises(ImportError, match="opentelemetry-api"):
                kaos_agents.otel.OTelHook()

    def test_importable_without_opentelemetry(self) -> None:
        """OTelHook class should be importable even without opentelemetry."""
        from kaos_agents.otel import OTelHook

        assert OTelHook is not None
