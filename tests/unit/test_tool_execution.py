"""Tests for the merged :class:`ToolExecution` value type.

Track 3 chunk A2 — confirms:
- ``ToolCallRecord`` is now a back-compat alias for ``ToolExecution``
- New fields (call_id, timing, usage, plan/step) round-trip correctly
- ``begin()`` / ``with_completion()`` lifecycle works
- ``to_summary()`` projects cleanly to the wire-side ToolCallSummary
- ``to_dict()`` produces a JSON-friendly dict (used by Memory.ACTIONS metadata)
- Frozen dataclass — assignment raises
"""

from __future__ import annotations

import time

import pytest

from kaos_agents.events.tools import ToolCallSummary
from kaos_agents.types import ToolCallRecord, ToolExecution


@pytest.mark.unit
class TestToolExecutionAlias:
    def test_tool_call_record_is_tool_execution(self) -> None:
        """The chunk-A2 alias preserves backward compat for callers
        that still import ``ToolCallRecord``."""
        assert ToolCallRecord is ToolExecution


@pytest.mark.unit
class TestToolExecutionFields:
    def test_default_construction(self) -> None:
        te = ToolExecution(tool_name="x")
        assert te.tool_name == "x"
        assert te.call_id == ""
        assert te.arguments == ()
        assert te.result_summary == ""
        assert te.is_error is False
        assert te.started_at is None
        assert te.ended_at is None
        assert te.duration_ms == 0.0
        assert te.cost_usd == 0.0
        assert te.input_tokens == 0
        assert te.output_tokens == 0
        assert te.plan_id is None
        assert te.step_id is None
        assert te.metadata == {}

    def test_from_dict_args_threads_all_fields(self) -> None:
        te = ToolExecution.from_dict_args(
            tool_name="kaos-source-fr-search",
            arguments={"query": "EPA enforcement", "limit": 10},
            call_id="tc-7af2",
            result_summary="3 documents",
            is_error=False,
            started_at=100.0,
            ended_at=100.5,
            duration_ms=500.0,
            cost_usd=0.012,
            input_tokens=120,
            output_tokens=80,
            plan_id="plan-1",
            step_id="s001",
            metadata={"region": "us-east-1"},
        )
        assert te.call_id == "tc-7af2"
        assert te.duration_ms == 500.0
        assert te.cost_usd == 0.012
        assert te.input_tokens == 120
        assert te.output_tokens == 80
        assert te.plan_id == "plan-1"
        assert te.step_id == "s001"
        assert te.metadata == {"region": "us-east-1"}
        # arguments are sorted for determinism
        assert te.arguments == (("limit", 10), ("query", "EPA enforcement"))

    def test_frozen(self) -> None:
        te = ToolExecution(tool_name="x")
        with pytest.raises((AttributeError, Exception)):
            object.__setattr__(te, "tool_name", "y")
            te.tool_name = "y"


@pytest.mark.unit
class TestLifecycleHelpers:
    def test_begin_sets_started_at(self) -> None:
        te = ToolExecution.begin(
            tool_name="kaos-web-fetch",
            call_id="tc-x",
            arguments={"url": "https://example.com"},
        )
        assert te.tool_name == "kaos-web-fetch"
        assert te.call_id == "tc-x"
        assert te.started_at is not None
        assert te.ended_at is None
        assert te.arguments == (("url", "https://example.com"),)

    def test_with_completion_computes_duration(self) -> None:
        started = ToolExecution.begin(tool_name="x", call_id="c1")
        time.sleep(0.005)  # 5ms
        done = started.with_completion(
            result_summary="ok",
            cost_usd=0.001,
            input_tokens=10,
            output_tokens=5,
        )
        assert done.tool_name == "x"
        assert done.call_id == "c1"
        assert done.result_summary == "ok"
        assert done.cost_usd == 0.001
        assert done.input_tokens == 10
        assert done.output_tokens == 5
        # duration_ms is positive and roughly matches sleep
        assert done.duration_ms > 0
        # started_at preserved
        assert done.started_at == started.started_at
        # ended_at is set
        assert done.ended_at is not None

    def test_with_completion_preserves_correlation(self) -> None:
        started = ToolExecution.begin(tool_name="x", call_id="c1", plan_id="plan-9", step_id="s007")
        done = started.with_completion(result_summary="ok")
        assert done.plan_id == "plan-9"
        assert done.step_id == "s007"


@pytest.mark.unit
class TestToSummary:
    def test_summary_keeps_telemetry_drops_payload(self) -> None:
        te = ToolExecution.from_dict_args(
            tool_name="kaos-source-fr-search",
            arguments={"query": "EPA"},
            call_id="tc-1",
            result_summary="3 documents",
            is_error=False,
            duration_ms=120.0,
            cost_usd=0.001,
            input_tokens=50,
            output_tokens=30,
            plan_id="plan-1",
            step_id="s001",
        )
        summary = te.to_summary()
        assert isinstance(summary, ToolCallSummary)
        # Telemetry preserved
        assert summary.tool_name == "kaos-source-fr-search"
        assert summary.call_id == "tc-1"
        assert summary.duration_ms == 120.0
        assert summary.cost_usd == 0.001
        assert summary.input_tokens == 50
        assert summary.output_tokens == 30
        assert summary.plan_id == "plan-1"
        assert summary.step_id == "s001"
        # Payload (arguments + result text) NOT on summary — that's the point
        assert not hasattr(summary, "arguments")
        assert not hasattr(summary, "result_summary")


@pytest.mark.unit
class TestToDict:
    def test_to_dict_round_trip_friendly(self) -> None:
        te = ToolExecution.from_dict_args(
            tool_name="x",
            arguments={"a": 1, "b": "two"},
            call_id="c1",
            result_summary="ok",
            duration_ms=100.0,
            plan_id="p1",
            step_id="s1",
        )
        d = te.to_dict()
        # All fields present
        assert d["tool_name"] == "x"
        assert d["call_id"] == "c1"
        # arguments serialized as list-of-pairs (json-friendly)
        assert d["arguments"] == [["a", 1], ["b", "two"]]
        assert d["result_summary"] == "ok"
        assert d["duration_ms"] == 100.0
        assert d["plan_id"] == "p1"
        assert d["step_id"] == "s1"
        # metadata is a fresh dict, not the same object
        assert d["metadata"] == {}
        assert d["metadata"] is not te.metadata
