"""Tests for plan_id / step_id correlation on tool execution records.

Track 2 chunk 5 — kelvin-port #14. Confirms that:

- :class:`ToolCallRecord` carries optional ``plan_id`` / ``step_id``.
- :class:`ToolCallSummary` carries optional ``plan_id`` / ``step_id``.
- Both default to ``None`` (chat / research / direct-respond turns
  don't fill them).
- Round-trip serialization preserves them.
"""

from __future__ import annotations

import pytest

from kaos_agents.events import (
    ToolCallSummary,
    deserialize_event,
    serialize_event,
)
from kaos_agents.events.lifecycle import TurnSummary
from kaos_agents.types import ToolCallRecord


@pytest.mark.unit
class TestToolCallRecord:
    def test_default_plan_id_is_none(self) -> None:
        rec = ToolCallRecord(tool_name="kaos-source-fr-search", arguments=())
        assert rec.plan_id is None
        assert rec.step_id is None

    def test_plan_id_explicit(self) -> None:
        rec = ToolCallRecord(
            tool_name="kaos-source-fr-search",
            arguments=(),
            plan_id="plan_abc",
            step_id="s001",
        )
        assert rec.plan_id == "plan_abc"
        assert rec.step_id == "s001"

    def test_from_dict_args_threads_plan_id(self) -> None:
        rec = ToolCallRecord.from_dict_args(
            tool_name="kaos-source-edgar-fetch",
            arguments={"cik": "0000320193"},
            plan_id="plan_xyz",
            step_id="s002",
        )
        assert rec.plan_id == "plan_xyz"
        assert rec.step_id == "s002"
        assert rec.arguments == (("cik", "0000320193"),)

    def test_record_remains_frozen(self) -> None:
        """Adding the new optional fields didn't break frozen-ness."""
        rec = ToolCallRecord(tool_name="x", plan_id="p")
        with pytest.raises((AttributeError, Exception)):
            # Frozen dataclass — setting any field raises FrozenInstanceError
            # at runtime. ty knows the property is read-only at static
            # analysis time too; suppress the spurious diagnostic.
            object.__setattr__(rec, "plan_id", "other")
            rec.plan_id = "other"  # ty: ignore[invalid-assignment]


@pytest.mark.unit
class TestToolCallSummary:
    def test_default_plan_id_is_none(self) -> None:
        summary = ToolCallSummary(tool_name="kaos-web-fetch", call_id="tc1")
        assert summary.plan_id is None
        assert summary.step_id is None

    def test_plan_id_explicit(self) -> None:
        summary = ToolCallSummary(
            tool_name="kaos-web-fetch",
            call_id="tc1",
            plan_id="plan_q1",
            step_id="s003",
            duration_ms=42.0,
        )
        assert summary.plan_id == "plan_q1"
        assert summary.step_id == "s003"
        assert summary.duration_ms == 42.0


@pytest.mark.unit
class TestRoundTripSerialization:
    """ToolCallSummary nested in TurnSummary serializes plan_id correctly."""

    def test_turn_summary_with_correlated_tool_calls(self) -> None:
        turn = TurnSummary(
            timestamp=1.0,
            sequence=0,
            session_id="s",
            run_id="r",
            text="ok",
            tool_calls=(
                ToolCallSummary(
                    tool_name="kaos-source-fr-search",
                    call_id="tc1",
                    plan_id="plan_abc",
                    step_id="s001",
                    duration_ms=120.0,
                ),
                ToolCallSummary(
                    tool_name="kaos-content-extract",
                    call_id="tc2",
                    plan_id="plan_abc",
                    step_id="s002",
                    duration_ms=85.0,
                ),
            ),
        )

        serialized = serialize_event(turn)
        assert serialized["tool_calls"][0]["plan_id"] == "plan_abc"
        assert serialized["tool_calls"][0]["step_id"] == "s001"
        assert serialized["tool_calls"][1]["step_id"] == "s002"

        roundtripped = deserialize_event(serialized)
        assert isinstance(roundtripped, TurnSummary)
        assert roundtripped.tool_calls[0].plan_id == "plan_abc"
        assert roundtripped.tool_calls[1].step_id == "s002"
