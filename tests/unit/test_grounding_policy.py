"""Unit tests for FUND-8 grounding policy enforcement.

Exercises ``apply_refusal_policy`` with real :class:`Answer` /
:class:`InsufficientEvidence` instances from kaos-llm-core — duck-typed
fakes are deliberately rejected by the policy now (the
``isinstance(grounded_answer, Answer)`` check is the safety property
the test must respect). No live LLM calls.

Verifies:

1. No policy → pass-through.
2. Answer above threshold → pass-through.
3. Answer below threshold → collapse to InsufficientEvidence + event.
4. InsufficientEvidence input → pass-through (already refused).
5. Agent.refusal_policy field propagates correctly.
6. Duck-typed objects with a ``kind="answer"`` attribute but no real
   ``Answer`` lineage pass through unchanged (bug-fix regression).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from kaos_llm_core.signatures.grounding import Answer, Claim, InsufficientEvidence, Span

from kaos_agents.config import Agent
from kaos_agents.events import GroundingRefusalTriggered
from kaos_agents.grounding import apply_refusal_policy


def _make_answer(confidence: float, value: str = "x") -> Answer[str]:
    """Construct a real Answer[str] with one Claim. Pydantic-validated.

    ClaimType is a Literal in kaos-llm-core; pass the string
    discriminant directly. ``supporting_spans`` requires at least one
    Span (the safety property "claims must cite something"). Build a
    minimal Span pointing at a synthetic source.
    """
    span = Span(
        source_uri="doc:test",
        quote=value,
        char_span=(0, len(value)),
    )
    claim = Claim(
        statement=value,
        claim_type="factual",
        supporting_spans=[span],
        confidence=confidence,
    )
    return Answer[str](
        value=value,
        claims=[claim],
        confidence=confidence,
    )


@dataclass
class _DuckTypedFakeAnswer:
    """A fake that LOOKS like an Answer but isn't one.

    This is the regression fixture for the duck-typing bug:
    ``getattr(answer, "kind", None) == "answer"`` was True for this
    fake, so the prior implementation applied the confidence policy
    to it. With ``isinstance(answer, Answer)`` it's correctly
    pass-through.
    """

    kind: str = "answer"
    confidence: float = 0.0
    claims: tuple = ()


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
        answer = _make_answer(confidence=0.3)
        result, event = apply_refusal_policy(answer, None)
        assert result is answer
        assert event is None

    def test_answer_above_threshold_passes(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.7)
        answer = _make_answer(confidence=0.85)
        result, event = apply_refusal_policy(answer, policy)
        assert result is answer
        assert event is None

    def test_answer_at_threshold_passes(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.7)
        answer = _make_answer(confidence=0.7)
        result, event = apply_refusal_policy(answer, policy)
        assert result is answer
        assert event is None

    def test_answer_below_threshold_collapses(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.7)
        answer = _make_answer(confidence=0.5)
        result, event = apply_refusal_policy(
            answer, policy, sequence=42, session_id="s1", run_id="r1"
        )

        assert isinstance(result, InsufficientEvidence)
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
        ie = InsufficientEvidence(reason="already refused")
        result, event = apply_refusal_policy(ie, policy)
        assert result is ie
        assert event is None

    def test_zero_confidence_collapses(self) -> None:
        policy = FakeRefusalPolicy(min_confidence=0.1)
        answer = _make_answer(confidence=0.0)
        result, event = apply_refusal_policy(answer, policy, session_id="s1", run_id="r1")
        assert isinstance(result, InsufficientEvidence)
        assert result.kind == "insufficient_evidence"
        assert event is not None

    def test_duck_typed_fake_passes_through(self) -> None:
        """Regression: duck-typed objects with a ``kind`` attribute but
        no real ``Answer`` lineage must pass through unchanged. The
        prior implementation applied the confidence policy to anything
        with ``getattr(obj, "kind") == "answer"``, which silently
        defaulted ``confidence`` to 1.0 on objects missing the field.
        """
        policy = FakeRefusalPolicy(min_confidence=0.7)
        fake = _DuckTypedFakeAnswer(confidence=0.0)  # 0 ≤ threshold 0.7
        result, event = apply_refusal_policy(fake, policy)
        # With the isinstance fix: not an Answer, so pass through.
        assert result is fake
        assert event is None


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
            agent.__setattr__("refusal_policy", FakeRefusalPolicy())
