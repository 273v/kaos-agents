"""Citation verification primitives (broad-reliability roadmap §B0.8).

kaos-citations is intentionally extract-only (per its AGENTS.md:
"Do not add citation resolution, URL fetching, source retrieval, or
claim verification"). The orchestration layer — extract → resolve →
verify → emit event — lives here, where it composes cleanly with the
AgenticLoop's post-worker gate.

Public surface:

- :class:`VerificationStatus` — Literal of outcome states.
- :class:`CitationVerificationResult` — frozen dataclass with status +
  observed cluster URL + diagnostic message.
- :func:`verify_case_citation` — verify a single CaseCitation against
  CourtListener.
- :func:`verify_citations_in_text` — extract case citations from text
  and verify each in parallel.

The CourtListener client is self-contained (httpx-based) so this
module has no new optional-dependency footprint. Resolution requires
network and is gated by ``KAOS_AGENT_CITATION_VERIFY_ENABLED=1`` so
unit tests + sandbox runs don't make live HTTP calls.
"""

from __future__ import annotations

from kaos_agents.citations.verifier import (
    CitationVerificationResult,
    VerificationStatus,
    verify_case_citation,
    verify_citations_in_text,
)

__all__ = [
    "CitationVerificationResult",
    "VerificationStatus",
    "verify_case_citation",
    "verify_citations_in_text",
]
