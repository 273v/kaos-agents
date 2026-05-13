"""PromotionPolicy — when does a session finding become institutional?

Resolved Decision #5: auto-promote when confidence >= 0.85 AND the
finding is grounding-verified (its supporting spans verified by the
:class:`Cited[T]` verifier). No human-review queue in Phase 4.A —
this is the "high-volume" path. A future Phase 4+ extension may add
opt-in human review for sensitive matters.

Inputs:
  - A finding (typically a kaos-llm-core ``Cited[T]`` with confidence
    + spans).
  - The ``matter_client`` namespace.
  - The current ``KnowledgeBase`` instance.

Output: True if promoted (and the entry was added), False otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_agents.memory.institutional import KBEntry, KnowledgeBase


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    entry: KBEntry | None = None


class PromotionPolicy:
    """Resolved Decision #5: confidence >= 0.85 + grounding-verified.

    Constructor kwargs:
      min_confidence: default 0.85.
      require_grounding: default True. Set False to disable the
        verifier check (Phase 4+ may relax for short-living
        memos).
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.85,
        require_grounding: bool = True,
    ) -> None:
        self._min_confidence = min_confidence
        self._require_grounding = require_grounding

    def consider(
        self,
        finding: Any,
        *,
        matter_client: tuple[str, str],
        knowledge_base: KnowledgeBase,
        finding_id: str | None = None,
    ) -> PromotionDecision:
        """Evaluate a finding and (if it qualifies) write to the KB.

        ``finding`` is duck-typed: must have ``.confidence`` (float),
        ``.statement`` or ``.value`` (str), and (when grounding is
        required) ``.is_verified`` (bool) or non-empty ``.spans``.
        """
        confidence = float(getattr(finding, "confidence", 0.0))
        if confidence < self._min_confidence:
            return PromotionDecision(
                promoted=False,
                reason=f"confidence {confidence:.2f} < {self._min_confidence:.2f}",
            )

        if self._require_grounding:
            verified = bool(
                getattr(finding, "is_verified", None) or getattr(finding, "spans", None)
            )
            if not verified:
                return PromotionDecision(
                    promoted=False,
                    reason="grounding not verified",
                )

        statement = (
            getattr(finding, "statement", None) or getattr(finding, "value", None) or str(finding)
        )
        entry_id = (
            finding_id
            or getattr(finding, "id", None)
            or f"kb_{int(confidence * 1000):04d}_{abs(hash(statement)) % 100000:05d}"
        )
        spans = tuple(getattr(finding, "spans", ()) or ())
        entry = KBEntry(
            id=entry_id,
            statement=str(statement),
            matter_client=matter_client,
            confidence=confidence,
            grounding_verified=self._require_grounding,
            provenance=spans,
        )
        knowledge_base.add(entry)
        return PromotionDecision(
            promoted=True,
            reason=(f"confidence {confidence:.2f} + grounding_verified={self._require_grounding}"),
            entry=entry,
        )


__all__ = ["PromotionDecision", "PromotionPolicy"]
