"""Unit tests for FUND-8 grounding policy enforcement.

Exercises ``apply_refusal_policy`` with canned Answer and
InsufficientEvidence objects — no live LLM calls. Verifies:

1. No policy → pass-through.
2. Answer above threshold → pass-through.
3. Answer below threshold → collapse to InsufficientEvidence + event.
4. InsufficientEvidence input → pass-through (already refused).
5. Agent.refusal_policy field propagates correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kaos_agents.config import Agent
from kaos_agents.events import GroundingRefusalTriggered
from kaos_agents.grounding import apply_refusal_policy

# ------------------------------------------------------------------
# Fake GroundedAnswer shapes (matches kaos-llm-core's discriminants)
# ------------------------------------------------------------------


@dataclass
class FakeAnswer:
    kind: str = "answer"
    confidence: float = 0.9
    claims: list[Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.claims is None:
            self.claims = []


@dataclass
class FakeInsufficientEvidence:
    kind: str = "insufficient_evidence"
    reason: str = "not found"


@dataclass
class FakeRefusalPolicy:
    min_confidence: float = 0.7
    require_verification: bool = False
    min_spans_per_claim: int = 1


# ------------------------------------------------------------------
# apply_refusal_policy
# ------------------------------------------------------------------


class TestApplyRefusalPolicy:
    def test_no_policy_passes_through(self) -> None:
        answer = FakeAnswer(confidence=0.3)
        result, event = apply_refusal_policy(answer, None)
        assert result is answer
        assert event is None

    def test_answer_above_threshold_passes(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.7)
        answer = FakeAnswer(confidence=0.85)
        result, event = apply_refusal_policy(answer, policy)
        assert result is answer
        assert event is None

    def test_answer_at_threshold_passes(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.7)
        answer = FakeAnswer(confidence=0.7)
        result, event = apply_refusal_policy(answer, policy)
        assert result is answer
        assert event is None

    def test_answer_below_threshold_collapses(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.7)
        answer = FakeAnswer(confidence=0.5)
        result, event = apply_refusal_policy(
            answer, policy, sequence=42, session_id="s1", run_id="r1"
        )

        assert result.kind == "insufficient_evidence"
        assert "confidence 0.50" in result.reason
        assert "threshold 0.70" in result.reason

        assert event is not None
        assert isinstance(event, GroundingRefusalTriggered)
        assert event.original_confidence == pytest.approx(0.5)
        assert event.min_confidence == pytest.approx(0.7)
        assert event.sequence == 42

    def test_insufficient_evidence_input_passes_through(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.7)
        ie = FakeInsufficientEvidence(reason="already refused")
        result, event = apply_refusal_policy(ie, policy)
        assert result is ie
        assert event is None

    def test_zero_confidence_collapses(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.1)
        answer = FakeAnswer(confidence=0.0)
        result, event = apply_refusal_policy(answer, policy, session_id="s1", run_id="r1")
        assert result.kind == "insufficient_evidence"
        assert event is not None


# ------------------------------------------------------------------
# Agent.refusal_policy field
# ------------------------------------------------------------------


class TestAgentRefusalPolicyField:
    def test_default_is_none(self) -> None:
        agent = Agent()
        assert agent.refusal_policy is None

    def test_create_propagates_policy(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.8)
        agent = Agent.create(refusal_policy=policy)
        assert agent.refusal_policy is not None
        assert agent.refusal_policy.min_confidence == pytest.approx(0.8)

    def test_frozen_immutability(self) -> None:
        agent = Agent()
        with pytest.raises(AttributeError):
            agent.refusal_policy = FakeRefusalPolicy()  # type: ignore[misc]
