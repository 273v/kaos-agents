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


class CitationVerified(LifecycleEvent):
    """B0.8 — post-worker citation verification result.

    Emitted once per case citation found in the worker's draft after
    the AgenticLoop's post-worker gate ran
    :func:`kaos_agents.citations.verify_case_citation` against
    CourtListener.

    Status semantics (see
    :class:`kaos_agents.citations.VerificationStatus`):

    - ``verified`` — CourtListener echoed the cite (canonical pass).
    - ``mismatch`` — case_name or year disagrees with CourtListener.
    - ``not_found`` — CourtListener has no cluster for this cite.
    - ``unreachable`` — network / 5xx / disabled (no verdict).
    - ``skipped`` — missing structural fields (volume / reporter / page).

    The AgenticLoop folds ``mismatch`` and ``not_found`` results into
    a forced replan with the diagnostic surfaced in
    ``thinking_note``; ``unreachable`` and ``skipped`` are recorded
    for the audit trail but do not force a replan.
    """

    raw_cite: str = ""
    """The ``Citation.raw`` text we verified — audit-trail anchor."""

    status: str = ""
    """``VerificationStatus`` literal as a string (event payloads are
    serializable; we keep the Literal type-system constraint at the
    callsite, not on the wire)."""

    courtlistener_url: str = ""
    """Canonical CourtListener URL when the cite matched a cluster.
    Empty string otherwise. Frontend can render this as a click-target
    so reviewers can compare the cited cluster to the agent's claim."""

    observed_case_name: str = ""
    """What CourtListener said the case is called (for ``mismatch``
    audit). Empty when not applicable."""

    observed_year: int = 0
    """What CourtListener said the decision year is (for ``mismatch``
    audit). Zero when not applicable."""

    diagnostic: str = ""
    """One-sentence rationale for non-``verified`` outcomes — the same
    text the AgenticLoop folds into ``thinking_note`` on replan."""


__all__ = [
    "CitationFound",
    "CitationVerified",
    "EvidenceInsufficient",
    "GroundingRefusalTriggered",
]
