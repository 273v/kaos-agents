"""Unit tests for :mod:`kaos_agents.triggers.base`.

Covers the value-type discipline (enum membership, factory shapes,
default isolation, frozen-instance contract, JSON round-trip,
``KaosEvent`` lineage). The eight source-tagged factories are exercised
in turn — each verifies its ``kind``, ``source_id`` derivation, and
the per-source ``payload`` shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kaos_agents.base.event import KaosEvent
from kaos_agents.triggers.base import Trigger, TriggerKind

# ----------------------------------------------------------------------
# TriggerKind enum
# ----------------------------------------------------------------------


class TestTriggerKind:
    """The 8-member discriminator enum."""

    def test_has_eight_members(self) -> None:
        assert len(TriggerKind) == 8

    def test_member_values(self) -> None:
        assert TriggerKind.MCP.value == "mcp"
        assert TriggerKind.HTTP.value == "http"
        assert TriggerKind.CLI.value == "cli"
        assert TriggerKind.SCHEDULED.value == "scheduled"
        assert TriggerKind.ESCALATION.value == "escalation"
        assert TriggerKind.WEBHOOK.value == "webhook"
        assert TriggerKind.FILESYSTEM.value == "filesystem"
        assert TriggerKind.DELEGATION.value == "delegation"

    def test_is_str_enum(self) -> None:
        # StrEnum members compare equal to their string values.
        assert TriggerKind.MCP == "mcp"
        assert TriggerKind.HTTP == "http"


# ----------------------------------------------------------------------
# Source-tagged factories
# ----------------------------------------------------------------------


class TestMCPFactory:
    def test_basic(self) -> None:
        t = Trigger.mcp("hello", session_id="s1")
        assert t.kind is TriggerKind.MCP
        assert t.source_id == "s1"
        assert t.payload["message"] == "hello"
        assert t.payload["tool_name"] is None

    def test_with_tool_name(self) -> None:
        t = Trigger.mcp("hi", session_id="s2", tool_name="kaos-agent-chat")
        assert t.payload["tool_name"] == "kaos-agent-chat"

    def test_no_session_id_means_empty_source(self) -> None:
        t = Trigger.mcp("hi")
        assert t.source_id == ""


class TestHTTPFactory:
    def test_basic(self) -> None:
        t = Trigger.http("hello", request_id="req-1")
        assert t.kind is TriggerKind.HTTP
        assert t.source_id == "req-1"
        assert t.payload["message"] == "hello"
        assert t.payload["request_id"] == "req-1"
        assert t.payload["headers"] == {}

    def test_headers_are_copied_to_dict(self) -> None:
        # Pass an arbitrary Mapping; payload should hold a plain dict.
        headers: Mapping[str, str] = {"x-trace": "abc", "user-agent": "pytest"}
        t = Trigger.http("hi", request_id="r", headers=headers)
        assert t.payload["headers"] == {"x-trace": "abc", "user-agent": "pytest"}


class TestCLIFactory:
    def test_basic(self) -> None:
        t = Trigger.cli("hello", session_id="repl-7")
        assert t.kind is TriggerKind.CLI
        assert t.source_id == "repl-7"
        assert t.payload["message"] == "hello"
        assert t.payload["tty"] is False

    def test_tty_flag(self) -> None:
        t = Trigger.cli("hi", tty=True)
        assert t.payload["tty"] is True


class TestScheduledFactory:
    def test_basic(self) -> None:
        t = Trigger.scheduled(job_name="nightly")
        assert t.kind is TriggerKind.SCHEDULED
        assert t.source_id == "nightly"
        assert t.payload["job_name"] == "nightly"
        assert t.payload["schedule"] is None
        # fired_at default → ISO-8601 UTC string.
        assert isinstance(t.payload["fired_at"], str)
        # Should round-trip through datetime.fromisoformat.
        parsed = datetime.fromisoformat(t.payload["fired_at"])
        assert parsed.tzinfo is not None

    def test_explicit_fired_at(self) -> None:
        ts = datetime(2026, 5, 9, 14, 0, 0, tzinfo=UTC)
        t = Trigger.scheduled(job_name="cron-1", fired_at=ts, schedule="0 14 * * *")
        assert t.payload["fired_at"] == "2026-05-09T14:00:00+00:00"
        assert t.payload["schedule"] == "0 14 * * *"


class TestEscalationFactory:
    def test_basic(self) -> None:
        t = Trigger.escalation(reason="ambiguous query")
        assert t.kind is TriggerKind.ESCALATION
        assert t.payload["reason"] == "ambiguous query"
        assert t.payload["kind"] == "domain_specific"
        assert t.payload["details"] == {}
        assert t.source_id == ""

    def test_with_parent_and_details(self) -> None:
        t = Trigger.escalation(
            reason="needs human",
            kind="clarification_needed",
            parent_turn_id="turn-7",
            details={"missing_fields": ["jurisdiction"]},
        )
        assert t.payload["kind"] == "clarification_needed"
        assert t.payload["parent_turn_id"] == "turn-7"
        assert t.source_id == "turn-7"
        assert t.payload["details"] == {"missing_fields": ["jurisdiction"]}


class TestWebhookFactory:
    def test_basic(self) -> None:
        t = Trigger.webhook(event_type="github.push")
        assert t.kind is TriggerKind.WEBHOOK
        # source_id falls back to event_type when not supplied.
        assert t.source_id == "github.push"
        assert t.payload["event_type"] == "github.push"
        assert t.payload["body"] is None
        assert t.payload["signature"] is None

    def test_with_body_and_signature(self) -> None:
        t = Trigger.webhook(
            event_type="stripe.payment_succeeded",
            body={"amount": 1000},
            signature="t=1620000000,v1=abc",
            source_id="stripe-livemode",
        )
        assert t.source_id == "stripe-livemode"
        assert t.payload["body"] == {"amount": 1000}
        assert t.payload["signature"] == "t=1620000000,v1=abc"


class TestFilesystemFactory:
    def test_basic(self) -> None:
        t = Trigger.filesystem(path="/var/log/app.log")
        assert t.kind is TriggerKind.FILESYSTEM
        assert t.source_id == "/var/log/app.log"
        assert t.payload["path"] == "/var/log/app.log"
        assert t.payload["event"] == "modified"
        assert t.payload["stat"] == {}

    def test_with_event_and_stat(self) -> None:
        t = Trigger.filesystem(
            path="/tmp/foo.txt",
            event="created",
            stat={"size": 42, "mtime": 1234567890},
        )
        assert t.payload["event"] == "created"
        assert t.payload["stat"] == {"size": 42, "mtime": 1234567890}


class TestDelegationFactory:
    def test_basic(self) -> None:
        t = Trigger.delegation(goal="extract dates", parent_turn_id="turn-7")
        assert t.kind is TriggerKind.DELEGATION
        assert t.source_id == "turn-7"
        assert t.payload["goal"] == "extract dates"
        assert t.payload["parent_turn_id"] == "turn-7"
        assert t.payload["sub_agent_hash"] is None

    def test_with_sub_agent_hash(self) -> None:
        t = Trigger.delegation(
            goal="run RAG",
            parent_turn_id="turn-9",
            sub_agent_hash="abc123",
        )
        assert t.payload["sub_agent_hash"] == "abc123"


# ----------------------------------------------------------------------
# Trigger value-type discipline
# ----------------------------------------------------------------------


class TestTriggerValueType:
    """Frozen, slotted-equivalent, default-isolated value type."""

    def test_occurred_at_is_tz_aware_utc(self) -> None:
        t = Trigger.mcp("x")
        assert isinstance(t.occurred_at, datetime)
        assert t.occurred_at.tzinfo is not None
        # datetime.now(UTC) returns the UTC tzinfo; the trigger should
        # share the same offset.
        assert t.occurred_at.utcoffset() == datetime.now(UTC).utcoffset()

    def test_default_payload_is_independent(self) -> None:
        """No aliased mutable defaults — each trigger has its own dict."""
        t1 = Trigger.mcp("a")
        t2 = Trigger.mcp("b")
        assert t1.payload is not t2.payload

    def test_default_metadata_is_independent(self) -> None:
        t1 = Trigger.mcp("a")
        t2 = Trigger.mcp("b")
        assert t1.metadata is not t2.metadata

    def test_metadata_classmethod_still_callable(self) -> None:
        """The instance ``metadata`` field shadows the classmethod, but
        ``KaosEvent.metadata`` continues to resolve via the parent class.

        ty's view of ``Trigger.metadata`` follows the pydantic field
        annotation (``Mapping[str, str]``), so we go through the parent
        class to keep the type checker happy while still exercising
        the same underlying classmethod.
        """
        meta = KaosEvent.metadata.__func__(Trigger)  # type: ignore[attr-defined]
        assert meta.name == "trigger"
        assert meta.category == "lifecycle"

    def test_correlation_id_passes_through(self) -> None:
        t = Trigger.mcp("x", correlation_id="trace-abc")
        assert t.correlation_id == "trace-abc"

    def test_metadata_passes_through(self) -> None:
        t = Trigger.cli("x", metadata={"client": "kaos-agent-cli"})
        assert t.metadata == {"client": "kaos-agent-cli"}

    def test_is_frozen(self) -> None:
        """Pydantic ``frozen=True`` raises ``ValidationError`` on assignment."""
        t = Trigger.mcp("x")
        with pytest.raises(ValidationError):
            t.kind = TriggerKind.HTTP  # type: ignore[misc]

    def test_unknown_field_rejected(self) -> None:
        """``extra="forbid"`` rejects mistyped construction kwargs."""
        # Build via ``model_validate`` rather than the constructor so ty
        # doesn't flag the deliberately-unknown kwarg statically.
        with pytest.raises(ValidationError):
            Trigger.model_validate(
                {
                    "timestamp": 0.0,
                    "sequence": 0,
                    "session_id": "",
                    "run_id": "",
                    "kind": TriggerKind.MCP,
                    "bogus_field": "oops",
                }
            )

    def test_is_kaos_event(self) -> None:
        t = Trigger.mcp("x")
        assert isinstance(t, KaosEvent)

    def test_pre_turn_base_field_defaults(self) -> None:
        """Triggers fire before a session/run is bound — the inherited
        :class:`KaosEvent` base fields use empty pre-turn defaults."""
        t = Trigger.mcp("x")
        assert t.session_id == ""
        assert t.run_id == ""
        assert t.agent_id is None
        assert t.sequence == 0
        assert isinstance(t.timestamp, float)


class TestTriggerJSONRoundTrip:
    """Pydantic JSON serialization round-trip across all 8 kinds."""

    @pytest.mark.parametrize(
        ("trigger", "expected_kind"),
        [
            (Trigger.mcp("hi", session_id="s"), TriggerKind.MCP),
            (Trigger.http("hi", request_id="r"), TriggerKind.HTTP),
            (Trigger.cli("hi", session_id="c", tty=True), TriggerKind.CLI),
            (Trigger.scheduled(job_name="j", schedule="* * * * *"), TriggerKind.SCHEDULED),
            (
                Trigger.escalation(reason="r", kind="clarification_needed", parent_turn_id="t"),
                TriggerKind.ESCALATION,
            ),
            (Trigger.webhook(event_type="evt"), TriggerKind.WEBHOOK),
            (Trigger.filesystem(path="/p", event="modified"), TriggerKind.FILESYSTEM),
            (Trigger.delegation(goal="g", parent_turn_id="p"), TriggerKind.DELEGATION),
        ],
    )
    def test_json_round_trip(self, trigger: Trigger, expected_kind: TriggerKind) -> None:
        raw = trigger.model_dump_json()
        rebuilt = Trigger.model_validate_json(raw)
        assert rebuilt.kind is expected_kind
        assert rebuilt == trigger

    def test_kind_is_serialized_as_string(self) -> None:
        """``StrEnum`` serializes to its plain string value on the wire."""
        import json

        t = Trigger.mcp("hi", session_id="s")
        decoded = json.loads(t.model_dump_json())
        assert decoded["kind"] == "mcp"


class TestTriggerNotAutoRegistered:
    """Phase 2.A intentionally does not register Trigger in the default
    event registry — that wiring is a Phase 2.B/2.C concern."""

    def test_not_in_default_event_registry(self) -> None:
        from kaos_agents.registry.event_registry import default_event_registry

        # ``Trigger`` is suppressed via ``register=False`` on the class
        # declaration. The discriminator name "trigger" must not point
        # to ``Trigger`` (or anything else) in the default registry.
        # If the registry holds a different class under that name from
        # an unrelated event type, that's fine — just assert the
        # specific class is absent.
        existing = default_event_registry.get("trigger")
        assert existing is not Trigger
