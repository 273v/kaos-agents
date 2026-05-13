"""Unit tests for :class:`kaos_agents.escalation.resume.EscalationResumePayload`."""

from __future__ import annotations

import dataclasses

import pytest

from kaos_agents.escalation.kinds import EscalationKind
from kaos_agents.escalation.resume import EscalationResumePayload


class TestEscalationResumePayload:
    def test_construction_full(self) -> None:
        payload = EscalationResumePayload(
            escalation_id="esc_xyz",
            kind=EscalationKind.APPROVAL_REQUIRED,
            response={"answer": "approved"},
            decision="approve",
            metadata={"by": "alice@example.com"},
        )
        assert payload.escalation_id == "esc_xyz"
        assert payload.kind is EscalationKind.APPROVAL_REQUIRED
        assert payload.response == {"answer": "approved"}
        assert payload.decision == "approve"
        assert payload.metadata == {"by": "alice@example.com"}

    def test_defaults(self) -> None:
        payload = EscalationResumePayload(
            escalation_id="esc_xyz",
            kind=EscalationKind.CLARIFICATION_NEEDED,
        )
        assert payload.response is None
        assert payload.decision == "answer"
        assert payload.metadata is None

    def test_frozen(self) -> None:
        payload = EscalationResumePayload(
            escalation_id="esc_xyz",
            kind=EscalationKind.LOOP_DETECTED,
        )
        # Use setattr() so static type checkers don't flag the
        # intentionally-illegal assignment we're trying to provoke.
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(payload, "decision", "abort")  # noqa: B010 — defeats ty static check on frozen dataclass

    def test_decision_accepts_known_values(self) -> None:
        # Phase 4.C documents the four expected values; the type is a
        # bare str so consumers can extend, but a smoke test confirms
        # all four canonical values round-trip.
        for decision in ("approve", "deny", "answer", "abort"):
            payload = EscalationResumePayload(
                escalation_id="esc_x",
                kind=EscalationKind.APPROVAL_REQUIRED,
                decision=decision,
            )
            assert payload.decision == decision

    def test_round_trip_via_asdict(self) -> None:
        payload = EscalationResumePayload(
            escalation_id="esc_xyz",
            kind=EscalationKind.BUDGET_EXCEEDED,
            response="proceed",
            decision="answer",
            metadata={"k": "v"},
        )
        as_dict = dataclasses.asdict(payload)
        assert as_dict["escalation_id"] == "esc_xyz"
        # StrEnum serialises through asdict as the enum (not the str);
        # consumers re-construct by passing the enum back in. We just
        # assert the round-trip yields an equivalent payload.
        restored = EscalationResumePayload(
            escalation_id=as_dict["escalation_id"],
            kind=as_dict["kind"],
            response=as_dict["response"],
            decision=as_dict["decision"],
            metadata=as_dict["metadata"],
        )
        assert restored == payload
