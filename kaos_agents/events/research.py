"""Research / RAG lifecycle events.

When the agent answers from a corpus, it emits a stream of
:class:`CitationFound` events for each verified claim. If the
corpus doesn't contain enough evidence,
:class:`EvidenceInsufficient` fires instead — the "refuses when
uncertain" pattern. :class:`GroundingRefusalTriggered` marks the
specific case of the Agent's ``refusal_policy`` collapsing a
low-confidence ``Answer[T]`` into ``InsufficientEvidence``.
"""

from __future__ import annotations

from kaos_agents.events._intermediates import LifecycleEvent


class CitationFound(LifecycleEvent):
    """Emitted when RAG verifies a claim against source documents."""

    claim: str = ""
    source_uri: str = ""
    confidence: float = 0.0
    verified: bool = False


class EvidenceInsufficient(LifecycleEvent):
    """Emitted when RAG cannot find sufficient evidence to answer.

    This is the "refuses when uncertain" pattern (Everlaw Deep Dive).
    """

    reason: str = ""
    what_would_resolve: str = ""


class GroundingRefusalTriggered(LifecycleEvent):
    """Emitted when the Agent's refusal_policy collapses an Answer.

    Fires when a ``GroundedAnswer`` comes back as ``Answer[T]`` but the
    answer's confidence is below ``refusal_policy.min_confidence``, or
    when span verification fails with ``require_verification=True``.
    The Answer is replaced with ``InsufficientEvidence`` downstream.

    Listeners (logging hooks, plan-execute replan logic) use this to
    surface the refusal and decide whether to retry, replan, or stop.
    """

    original_confidence: float = 0.0
    min_confidence: float = 0.0
    reason: str = ""


__all__ = [
    "CitationFound",
    "EvidenceInsufficient",
    "GroundingRefusalTriggered",
]
