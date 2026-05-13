"""PromotionPolicy unit tests — Phase 4.A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_agents.memory.institutional import KnowledgeBase
from kaos_agents.memory.promotion import PromotionDecision, PromotionPolicy


@dataclass
class FakeFinding:
    """Duck-typed stand-in for kaos-llm-core ``Cited[T]``.

    Has the attributes :class:`PromotionPolicy.consider` reads:
    ``confidence``, ``statement`` (or ``value``), ``spans``,
    ``is_verified``.
    """

    confidence: float
    statement: str = "the sky is blue"
    spans: tuple[Any, ...] = ()
    is_verified: bool | None = None


class TestPromotionPolicy:
    def test_high_confidence_verified_promotes(self) -> None:
        kb = KnowledgeBase()
        policy = PromotionPolicy()
        finding = FakeFinding(confidence=0.95, is_verified=True)
        decision = policy.consider(
            finding,
            matter_client=("m1", "c1"),
            knowledge_base=kb,
        )
        assert isinstance(decision, PromotionDecision)
        assert decision.promoted is True
        assert decision.entry is not None
        assert decision.entry.statement == "the sky is blue"
        # Verify the entry actually landed in the KB.
        from kaos_agents.memory.institutional import KBQuery

        res = kb.query(KBQuery(query_text="sky", matter_client=("m1", "c1")))
        assert len(res.entries) == 1

    def test_low_confidence_blocks_with_reason(self) -> None:
        kb = KnowledgeBase()
        policy = PromotionPolicy()  # default min 0.85
        finding = FakeFinding(confidence=0.5, is_verified=True)
        decision = policy.consider(
            finding,
            matter_client=("m1", "c1"),
            knowledge_base=kb,
        )
        assert decision.promoted is False
        assert "0.50" in decision.reason
        assert "0.85" in decision.reason
        assert decision.entry is None

    def test_grounding_required_but_not_verified_blocks(self) -> None:
        kb = KnowledgeBase()
        policy = PromotionPolicy(require_grounding=True)
        finding = FakeFinding(confidence=0.95, is_verified=None, spans=())
        decision = policy.consider(
            finding,
            matter_client=("m1", "c1"),
            knowledge_base=kb,
        )
        assert decision.promoted is False
        assert "grounding" in decision.reason

    def test_grounding_relaxed_promotes_without_verification(self) -> None:
        kb = KnowledgeBase()
        policy = PromotionPolicy(require_grounding=False)
        finding = FakeFinding(confidence=0.95, is_verified=None, spans=())
        decision = policy.consider(
            finding,
            matter_client=("m1", "c1"),
            knowledge_base=kb,
        )
        assert decision.promoted is True
        assert decision.entry is not None
        # When relaxed, the entry's grounding_verified flag mirrors the
        # require_grounding setting (False).
        assert decision.entry.grounding_verified is False

    def test_custom_min_confidence_threshold(self) -> None:
        kb = KnowledgeBase()
        policy = PromotionPolicy(min_confidence=0.5)
        finding = FakeFinding(confidence=0.6, is_verified=True)
        decision = policy.consider(
            finding,
            matter_client=("m1", "c1"),
            knowledge_base=kb,
        )
        assert decision.promoted is True
        # Right below the custom threshold blocks.
        finding_low = FakeFinding(confidence=0.4, is_verified=True)
        decision2 = policy.consider(
            finding_low,
            matter_client=("m1", "c1"),
            knowledge_base=kb,
        )
        assert decision2.promoted is False

    def test_promotion_does_not_dedup_in_phase_4a(self) -> None:
        """Phase 4.A KB has no dedup — calling consider twice writes two entries.

        Phase 4+ may add content-hash dedup; documenting the current
        behavior so future regressions are caught.
        """
        kb = KnowledgeBase()
        policy = PromotionPolicy()
        finding = FakeFinding(confidence=0.95, is_verified=True)
        d1 = policy.consider(finding, matter_client=("m1", "c1"), knowledge_base=kb)
        d2 = policy.consider(finding, matter_client=("m1", "c1"), knowledge_base=kb)
        assert d1.promoted is True
        assert d2.promoted is True
        from kaos_agents.memory.institutional import KBQuery

        res = kb.query(KBQuery(query_text="sky", matter_client=("m1", "c1")))
        assert len(res.entries) == 2  # not deduped
