"""Grounding subsystem — refusal policy enforcement + citation extraction.

This package collects the post-processing verifiers that run on
assistant output before TurnSummary fires:

- :func:`apply_refusal_policy` — collapses low-confidence ``Answer[T]``
  to ``InsufficientEvidence`` and produces a ``GroundingRefusalTriggered``
  event when the policy fires (FUND-8).
- :func:`make_evidence_insufficient_event` — builds an
  ``EvidenceInsufficient`` event from a collapsed result.
- :func:`emit_citations_for_text` — runs ``kaos_citations.extract_citations``
  over assistant text and emits ``CitationFound`` events into the active
  EventEmitter.

The previous layout had a top-level ``grounding.py`` (refusal policy)
that was later shadowed by a new ``grounding/`` package (citation
verifier). The package re-exports the legacy refusal-policy API
unchanged so ``from kaos_agents.grounding import apply_refusal_policy``
keeps working.
"""

from __future__ import annotations

from kaos_agents.grounding.citation_verifier import emit_citations_for_text
from kaos_agents.grounding.refusal_policy import (
    apply_refusal_policy,
    make_evidence_insufficient_event,
)

__all__ = [
    "apply_refusal_policy",
    "emit_citations_for_text",
    "make_evidence_insufficient_event",
]
