"""Tests for agent response and intent models."""

from __future__ import annotations

import pytest

from kaos_agents.models import AgentResponse, IntentResult, IntentType, ToolCallRecord


class TestIntentType:
    def test_all_types_are_strings(self):
        for it in IntentType:
            assert isinstance(it.value, str)

    def test_known_types(self):
        assert IntentType.RESPOND == "respond"
        assert IntentType.TOOL_USE == "tool_use"
        assert IntentType.RESEARCH == "research"
        assert IntentType.PLAN == "plan"
        assert IntentType.CLARIFY == "clarify"


class TestIntentResult:
    def test_frozen(self):
        result = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="test")
        with pytest.raises(AttributeError):
            result.confidence = 0.5  # ty: ignore[invalid-assignment]

    def test_fields(self):
        result = IntentResult(
            intent=IntentType.TOOL_USE, confidence=0.75, reasoning="Action keyword"
        )
        assert result.intent == IntentType.TOOL_USE
        assert result.confidence == 0.75
        assert result.reasoning == "Action keyword"


class TestToolCallRecord:
    def test_frozen(self):
        record = ToolCallRecord(
            tool_name="kaos-pdf-extract-parse",
            arguments={"path": "/tmp/test.pdf"},
            result_summary="15 pages extracted",
        )
        assert record.tool_name == "kaos-pdf-extract-parse"
        assert not record.is_error

    def test_error_record(self):
        record = ToolCallRecord(
            tool_name="kaos-web-fetch",
            arguments={"url": "https://example.com"},
            result_summary="Connection refused",
            is_error=True,
        )
        assert record.is_error


class TestAgentResponse:
    def test_minimal_response(self):
        intent = IntentResult(intent=IntentType.RESPOND, confidence=0.9, reasoning="greeting")
        response = AgentResponse(text="Hello!", intent=intent)
        assert response.text == "Hello!"
        assert response.tool_calls == ()
        assert response.artifacts == ()
        assert response.turn_number == 0

    def test_response_with_tool_calls(self):
        intent = IntentResult(intent=IntentType.TOOL_USE, confidence=0.8, reasoning="extract")
        tc = ToolCallRecord(
            tool_name="kaos-pdf-extract-parse",
            arguments={"path": "/tmp/test.pdf"},
            result_summary="15 pages",
        )
        response = AgentResponse(
            text="Extracted 15 pages.",
            intent=intent,
            tool_calls=(tc,),
            turn_number=3,
        )
        assert len(response.tool_calls) == 1
        assert response.turn_number == 3
