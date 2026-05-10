"""Tests for kaos_agents.governance.snapshot — StateSnapshot + helpers."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

import pytest

from kaos_agents.core.invocation import TurnInvocation
from kaos_agents.events import IntentClassified, TextDelta
from kaos_agents.governance.snapshot import (
    StateSnapshot,
    _intent_to_dict,
    load_snapshot,
    save_snapshot,
)
from kaos_agents.types.intents import IntentResult, IntentType
from kaos_agents.types.usage import InvocationUsage


class _StubVFS:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def read(self, path: str, context_id: str | None = None) -> bytes:
        if path not in self.store:
            raise FileNotFoundError(path)
        return self.store[path]

    async def write(self, path: str, data: bytes, context_id: str | None = None) -> int:
        self.store[path] = data
        return len(data)


def _make_invocation(*, with_events: bool = True) -> TurnInvocation:
    inv = TurnInvocation(
        session_id="sess-1",
        run_id="run-1",
        turn_number=2,
        agent_envelope_hash="abc123",
        usage=InvocationUsage(input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001),
        cost_usd=0.001,
        intent=IntentResult(intent=IntentType.TOOL_USE, confidence=0.85, reasoning="example"),
    )
    if with_events:
        inv.add_event(
            IntentClassified(
                timestamp=1.0,
                sequence=0,
                session_id="sess-1",
                run_id="run-1",
                intent="tool_use",
                confidence=0.85,
                reasoning="example",
            )
        )
        inv.add_event(
            TextDelta(
                timestamp=1.5,
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                content="hi",
            )
        )
    inv.output = "answer"
    inv.extras["custom"] = "value"
    return inv


def test_snapshot_construction_and_frozen() -> None:
    snap = StateSnapshot(
        snapshot_id="snap_1",
        captured_at=datetime.now(UTC),
        session_id="s",
        run_id="r",
        turn_number=1,
        agent_envelope_hash="h",
        output="o",
        intent=None,
        events_json=(),
        usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
        cost_usd=0.0,
    )
    assert snap.snapshot_id == "snap_1"
    # frozen dataclass: assignment must fail. setattr() so static type
    # checkers don't flag the intentionally-illegal assignment.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(snap, "snapshot_id", "other")  # noqa: B010 — defeats ty static check on frozen dataclass


def test_from_invocation_captures_all_fields() -> None:
    inv = _make_invocation()
    snap = StateSnapshot.from_invocation(inv)
    assert snap.session_id == "sess-1"
    assert snap.run_id == "run-1"
    assert snap.turn_number == 2
    assert snap.agent_envelope_hash == "abc123"
    assert snap.output == "answer"
    assert snap.cost_usd == pytest.approx(0.001)
    assert snap.usage["input_tokens"] == 10
    assert snap.usage["total_tokens"] == 30
    assert snap.intent == {
        "intent": "tool_use",
        "confidence": 0.85,
        "reasoning": "example",
        "usage": None,
    }
    assert len(snap.events_json) == 2
    assert "intent_classified" in snap.events_json[0]
    assert "text_delta" in snap.events_json[1]
    assert snap.extras == {"custom": "value"}


def test_from_invocation_explicit_snapshot_id() -> None:
    snap = StateSnapshot.from_invocation(_make_invocation(), snapshot_id="custom-id")
    assert snap.snapshot_id == "custom-id"


def test_snapshot_id_unique_across_calls() -> None:
    a = StateSnapshot.from_invocation(_make_invocation())
    b = StateSnapshot.from_invocation(_make_invocation())
    assert a.snapshot_id != b.snapshot_id


def test_to_json_from_json_roundtrip() -> None:
    snap = StateSnapshot.from_invocation(_make_invocation())
    payload = snap.to_json()
    parsed = json.loads(payload)
    # JSON shape sanity check.
    assert parsed["session_id"] == "sess-1"
    assert isinstance(parsed["events_json"], list)
    # Round-trip equals.
    again = StateSnapshot.from_json(payload)
    assert again.snapshot_id == snap.snapshot_id
    assert again.session_id == snap.session_id
    assert again.run_id == snap.run_id
    assert again.turn_number == snap.turn_number
    assert again.events_json == snap.events_json
    assert again.usage == snap.usage
    assert again.cost_usd == snap.cost_usd
    assert again.intent == snap.intent
    assert again.extras == snap.extras


def test_deserialized_events_returns_typed_events() -> None:
    snap = StateSnapshot.from_invocation(_make_invocation())
    events = snap.deserialized_events()
    assert len(events) == 2
    assert isinstance(events[0], IntentClassified)
    assert events[0].intent == "tool_use"
    assert isinstance(events[1], TextDelta)
    assert events[1].content == "hi"


async def test_save_snapshot_default_path() -> None:
    vfs = _StubVFS()
    snap = StateSnapshot.from_invocation(_make_invocation())
    path = await save_snapshot(snap, vfs)
    assert path == f"snapshots/{snap.snapshot_id}.json"
    assert path in vfs.store
    # Body is the to_json() payload.
    assert vfs.store[path].decode("utf-8") == snap.to_json()


async def test_save_snapshot_custom_path() -> None:
    vfs = _StubVFS()
    snap = StateSnapshot.from_invocation(_make_invocation())
    path = await save_snapshot(snap, vfs, path="archive/my.json")
    assert path == "archive/my.json"
    assert "archive/my.json" in vfs.store


async def test_load_snapshot_roundtrip_via_vfs() -> None:
    vfs = _StubVFS()
    original = StateSnapshot.from_invocation(_make_invocation())
    path = await save_snapshot(original, vfs)
    loaded = await load_snapshot(vfs, path)
    assert loaded.snapshot_id == original.snapshot_id
    assert loaded.events_json == original.events_json
    assert loaded.intent == original.intent
    assert loaded.usage == original.usage


async def test_load_snapshot_missing_path_raises() -> None:
    vfs = _StubVFS()
    with pytest.raises(FileNotFoundError):
        await load_snapshot(vfs, "snapshots/missing.json")


def test_intent_to_dict_handles_none() -> None:
    assert _intent_to_dict(None) is None


def test_intent_to_dict_handles_dataclass_with_enum() -> None:
    intent = IntentResult(intent=IntentType.PLAN, confidence=0.5, reasoning="r")
    out = _intent_to_dict(intent)
    assert out == {"intent": "plan", "confidence": 0.5, "reasoning": "r", "usage": None}


def test_intent_to_dict_handles_pydantic_like_object() -> None:
    class FakePydantic:
        def model_dump(self, mode: str = "json") -> dict[str, object]:
            return {"x": 1, "mode": mode}

    out = _intent_to_dict(FakePydantic())
    assert out == {"x": 1, "mode": "json"}


def test_intent_to_dict_falls_back_to_raw_input() -> None:
    class WithRawInput:
        raw_input = "hello there"

    out = _intent_to_dict(WithRawInput())
    assert out == {"raw_input": "hello there"}


def test_from_invocation_with_no_intent() -> None:
    inv = _make_invocation()
    inv.intent = None
    snap = StateSnapshot.from_invocation(inv)
    assert snap.intent is None


def test_from_invocation_no_events() -> None:
    inv = TurnInvocation(session_id="s", run_id="r", turn_number=0)
    snap = StateSnapshot.from_invocation(inv)
    assert snap.events_json == ()
    assert snap.deserialized_events() == ()
