"""Unit tests for :class:`kaos_agents.events.escalation.EscalationRequired`.

Verifies construction, defaults, inheritance hierarchy (LifecycleEvent
→ KaosEvent), and JSON wire round-trip via Pydantic.
"""

from __future__ import annotations

from kaos_agents.base.event import KaosEvent
from kaos_agents.escalation.kinds import EscalationKind
from kaos_agents.events import EscalationRequired
from kaos_agents.events._intermediates import LifecycleEvent

_TS = 12345.6
_SID = "test-session"
_RID = "test-run"


class TestEscalationRequiredConstruction:
    """Construction of :class:`EscalationRequired` events."""

    def test_full_construction_with_kind(self) -> None:
        evt = EscalationRequired(
            timestamp=_TS,
            sequence=7,
            session_id=_SID,
            run_id=_RID,
            kind=EscalationKind.CLARIFICATION_NEEDED.value,
            reason="ambiguous reference to 'the contract'",
            details={"ambiguity_count": 1, "confidence": 0.4},
            resume_token="run-xyz",
            escalation_id="esc_abc123",
        )
        assert evt.kind == "clarification_needed"
        assert evt.reason == "ambiguous reference to 'the contract'"
        assert evt.details == {"ambiguity_count": 1, "confidence": 0.4}
        assert evt.resume_token == "run-xyz"
        assert evt.escalation_id == "esc_abc123"
        # Base lifecycle/event fields
        assert evt.timestamp == _TS
        assert evt.sequence == 7
        assert evt.session_id == _SID
        assert evt.run_id == _RID

    def test_defaults(self) -> None:
        """Empty payload defaults: empty details / id / token; default kind = ''."""
        evt = EscalationRequired(
            timestamp=_TS,
            sequence=0,
            session_id=_SID,
            run_id=_RID,
        )
        assert evt.kind == ""
        assert evt.reason == ""
        assert evt.details == {}
        assert evt.resume_token == ""
        assert evt.escalation_id == ""

    def test_subclass_of_lifecycle_event_and_kaos_event(self) -> None:
        """EscalationRequired must be a LifecycleEvent (and so KaosEvent)."""
        assert issubclass(EscalationRequired, LifecycleEvent)
        assert issubclass(EscalationRequired, KaosEvent)
        evt = EscalationRequired(
            timestamp=_TS,
            sequence=0,
            session_id=_SID,
            run_id=_RID,
        )
        assert isinstance(evt, LifecycleEvent)
        assert isinstance(evt, KaosEvent)

    def test_json_round_trip(self) -> None:
        """model_dump_json / model_validate_json must round-trip cleanly."""
        original = EscalationRequired(
            timestamp=_TS,
            sequence=11,
            session_id=_SID,
            run_id=_RID,
            kind=EscalationKind.LOOP_DETECTED.value,
            reason="same step fired 3 times",
            details={"step_id": "s2", "count": 3},
            resume_token="run-xyz",
            escalation_id="esc_aaa111",
        )
        payload = original.model_dump_json()
        restored = EscalationRequired.model_validate_json(payload)
        assert restored == original

    def test_event_type_discriminator(self) -> None:
        """The default snake-case discriminator is ``escalation_required``."""
        assert EscalationRequired.event_type() == "escalation_required"
