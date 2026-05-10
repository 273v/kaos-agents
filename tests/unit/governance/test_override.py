"""Tests for kaos_agents.governance.override — OverrideHook + value types."""

from __future__ import annotations

from datetime import datetime

import pytest

from kaos_agents.events.collector import collect_events
from kaos_agents.events.emitter import EventEmitter
from kaos_agents.events.spans import Span, SpanPhase, SpanSubject
from kaos_agents.governance.override import (
    OverrideHook,
    OverrideKind,
    OverrideRecord,
)


def test_override_kind_has_four_members() -> None:
    members = {member.value for member in OverrideKind}
    assert members == {
        "force_stop",
        "clear_section",
        "replay_snapshot",
        "inject_event",
    }


def test_override_record_construction_defaults() -> None:
    rec = OverrideRecord(
        kind=OverrideKind.FORCE_STOP,
        issued_by="alice",
        reason="manual stop",
    )
    assert rec.kind is OverrideKind.FORCE_STOP
    assert rec.issued_by == "alice"
    assert rec.reason == "manual stop"
    assert isinstance(rec.issued_at, datetime)
    assert rec.payload == {}


def test_override_record_is_frozen_slotted() -> None:
    import dataclasses

    rec = OverrideRecord(kind=OverrideKind.INJECT_EVENT, issued_by="x", reason="y")
    # setattr() so static type checkers don't flag the intentionally-illegal
    # assignment used to verify frozen semantics at runtime.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(rec, "reason", "other")  # noqa: B010 — defeats ty static check on frozen dataclass
    assert not hasattr(rec, "__dict__")


def test_record_appends_to_records() -> None:
    hook = OverrideHook()
    rec = hook.record(
        kind=OverrideKind.FORCE_STOP,
        issued_by="alice",
        reason="stop",
    )
    assert isinstance(rec, OverrideRecord)
    assert hook.records == (rec,)


def test_record_authorized_admins_none_accepts_any() -> None:
    hook = OverrideHook(authorized_admins=None)
    hook.record(kind=OverrideKind.FORCE_STOP, issued_by="random", reason="r")
    assert len(hook.records) == 1


def test_record_authorized_admins_rejects_unauthorized() -> None:
    hook = OverrideHook(authorized_admins=("alice", "bob"))
    with pytest.raises(PermissionError) as excinfo:
        hook.record(kind=OverrideKind.FORCE_STOP, issued_by="mallory", reason="hax")
    assert "mallory" in str(excinfo.value)
    assert hook.records == ()  # nothing recorded on rejection


def test_record_authorized_admins_accepts_listed_issuer() -> None:
    hook = OverrideHook(authorized_admins=("alice",))
    rec = hook.record(kind=OverrideKind.CLEAR_SECTION, issued_by="alice", reason="r")
    assert rec.issued_by == "alice"
    assert hook.records == (rec,)


def test_record_emits_span_event_via_emitter() -> None:
    hook = OverrideHook()
    emitter = EventEmitter(session_id="s", run_id="r")
    # Capture via the standard collector context — every emit() pushes
    # into the active collector if one is in scope.
    with collect_events() as collector:
        hook.record(
            kind=OverrideKind.FORCE_STOP,
            issued_by="admin-1",
            reason="emergency stop",
            emitter=emitter,
            payload={"detail": "memory exhausted"},
        )
    spans = [e for e in collector.events if isinstance(e, Span)]
    assert len(spans) == 1
    span = spans[0]
    assert span.subject is SpanSubject.STEP
    assert span.phase is SpanPhase.ERROR
    assert span.error_type == "AdminOverride"
    assert "admin-1" in (span.error_message or "")
    assert "emergency stop" in (span.error_message or "")
    assert span.attributes["override_kind"] == "force_stop"
    assert span.attributes["issued_by"] == "admin-1"
    assert span.attributes["reason"] == "emergency stop"
    assert span.attributes["detail"] == "memory exhausted"
    assert span.name == "override.force_stop"


def test_record_without_emitter_does_not_raise() -> None:
    hook = OverrideHook()
    rec = hook.record(
        kind=OverrideKind.INJECT_EVENT,
        issued_by="alice",
        reason="r",
    )
    assert rec.kind is OverrideKind.INJECT_EVENT


def test_records_property_returns_immutable_view() -> None:
    hook = OverrideHook()
    hook.record(kind=OverrideKind.FORCE_STOP, issued_by="x", reason="y")
    snap1 = hook.records
    assert isinstance(snap1, tuple)
    # Mutating the returned tuple isn't possible (it's a tuple), but
    # appending to the internal list must NOT affect previously
    # returned snapshots.
    hook.record(kind=OverrideKind.CLEAR_SECTION, issued_by="x", reason="y2")
    snap2 = hook.records
    assert len(snap1) == 1  # snapshot is independent
    assert len(snap2) == 2


def test_record_payload_defaults_to_empty_dict() -> None:
    hook = OverrideHook()
    rec = hook.record(kind=OverrideKind.FORCE_STOP, issued_by="x", reason="y")
    assert rec.payload == {}


def test_record_payload_is_copied_not_aliased() -> None:
    hook = OverrideHook()
    payload = {"a": 1}
    rec = hook.record(kind=OverrideKind.FORCE_STOP, issued_by="x", reason="y", payload=payload)
    payload["a"] = 2
    assert rec.payload == {"a": 1}  # snapshot at record time


def test_override_hook_metadata_well_formed() -> None:
    md = OverrideHook.metadata()
    assert md.name == "kaos-agents-override-hook"
    assert md.listens_to == ()
