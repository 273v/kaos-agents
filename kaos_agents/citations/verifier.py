"""Citation verification against CourtListener (broad-reliability §B0.8).

Resolves a :class:`kaos_citations.CaseCitation` record against the
CourtListener v4 REST API and returns a structured result an
AgenticLoop post-worker gate can act on.

Pre-B0.8 nothing caught the case where an agent emitted
``Brown v. Board of Education, 347 U.S. 495 (1954)`` (the real page
is 483 — every cite-checker except a memorized human catches that).
Harvey publishes a 0.2% citation-error rate; we didn't measure ours.
For attorneys this is *the* malpractice failure mode.

Architecture decisions (per per-repo AGENTS.md):

1. **kaos-citations stays extract-only.** Its AGENTS.md is explicit:
   "Do not add citation resolution, URL fetching, source retrieval,
   or claim verification." Verification is composition; it belongs
   above the extraction primitive.

2. **CourtListener client is self-contained in kaos-agents.** No new
   optional-dependency footprint, no kaos-source detour. The client
   is a thin httpx wrapper around the public v4 ``citation-lookup``
   endpoint. Authenticated requests use ``Authorization: Token <key>``
   from ``KAOS_AGENT_COURTLISTENER_API_KEY`` (or legacy
   ``COURTLISTENER_API_KEY``).

3. **Live HTTP is gated.** ``KAOS_AGENT_CITATION_VERIFY_ENABLED=1`` is
   the master switch. Unit tests + offline sandbox runs see a
   ``status="unreachable"`` result and the AgenticLoop gate
   short-circuits to "verification unavailable" — never a confident
   pass on a fabricated cite.

4. **Conservative semantics.** ``verified`` requires CourtListener to
   echo back a cluster matching ALL of (case_name, volume, reporter,
   page, year). Any discrepancy → ``mismatch``. Network or 5xx →
   ``unreachable`` (the agent must report the cite as unverified,
   never auto-strip it).
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_citations import CaseCitation

logger = get_logger(__name__)

# ── Public contract ─────────────────────────────────────────────────

VerificationStatus = Literal[
    "verified",  # CourtListener matched all canonical fields.
    "mismatch",  # CourtListener matched a cluster but ≥1 field disagreed.
    "not_found",  # CourtListener returned zero clusters for this cite.
    "unreachable",  # Network / 5xx / disabled — neither pass nor fail.
    "skipped",  # Pre-conditions not met (missing volume / reporter / page).
]


@dataclass(frozen=True, slots=True)
class CitationVerificationResult:
    """Outcome of a single citation verification call.

    Frozen value type — safe to attach to events / persist to memory /
    serialize over the wire.
    """

    status: VerificationStatus
    """Verdict — see :data:`VerificationStatus`."""

    raw_cite: str
    """The ``Citation.raw`` text we verified (audit trail anchor)."""

    courtlistener_url: str | None = None
    """When CourtListener returned a cluster, its canonical URL."""

    observed_case_name: str | None = None
    """Case name CourtListener returned (compare to ``CaseCitation.case_name``)."""

    observed_year: int | None = None
    """Year CourtListener returned (compare to ``CaseCitation.year``)."""

    diagnostic: str | None = None
    """One-sentence rationale on ``mismatch`` / ``unreachable`` / ``not_found``.

    Surfaced to the AgenticLoop's ``thinking_note`` when the gate
    forces a replan — gives the worker an actionable hint without
    leaking the underlying network shape."""

    @property
    def is_problem(self) -> bool:
        """True for outcomes that should block a confident answer.

        ``mismatch`` and ``not_found`` are problems the worker must
        address. ``unreachable`` is a problem only insofar as the
        worker must report the cite as unverified — but the gate
        does NOT force a replan on ``unreachable`` because the cite
        might be correct; we just can't confirm.
        """
        return self.status in ("mismatch", "not_found")


# ── CourtListener client ────────────────────────────────────────────

_COURTLISTENER_BASE: Final[str] = "https://www.courtlistener.com/api/rest/v4"
_VERIFY_TIMEOUT_S: Final[float] = 10.0


def _api_key() -> str | None:
    """Resolve the CourtListener API key from env vars.

    Falls back to the legacy unprefixed name documented in the
    kaos-modules CLAUDE.md. ``None`` (anonymous) is acceptable —
    CourtListener serves the citation-lookup endpoint without
    auth, just with a stricter rate limit.
    """
    return (
        os.environ.get("KAOS_AGENT_COURTLISTENER_API_KEY")
        or os.environ.get("COURTLISTENER_API_KEY")
        or None
    )


def _verify_enabled() -> bool:
    """Master switch for live CourtListener HTTP.

    Default: disabled (sandbox safety). Production / live tests must
    set ``KAOS_AGENT_CITATION_VERIFY_ENABLED=1``.
    """
    return os.environ.get("KAOS_AGENT_CITATION_VERIFY_ENABLED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_cite_text(cite: CaseCitation) -> str | None:
    """Render a CaseCitation as a parser-friendly string for CL's
    citation-lookup endpoint.

    Returns None when the cite lacks the structural fields needed
    for resolution (volume + reporter + page). The gate treats this
    as ``skipped`` — neither pass nor fail.
    """
    if cite.volume is None or not cite.reporter or cite.page is None:
        return None
    base = f"{cite.volume} {cite.reporter} {cite.page}"
    if cite.year is not None:
        base = f"{base} ({cite.year})"
    return base


def _names_match(observed: str | None, expected: str | None) -> bool:
    """Loose case-name comparator.

    Both sides are lowercased + non-alphanumeric stripped before
    substring containment. CourtListener returns canonical names
    (``"Brown v. Board of Education"``) while extracted citations
    often have abbreviated / extra-context forms (``"Brown v.
    Board of Education of Topeka"``). We accept either-direction
    containment so common abbreviation noise doesn't fire a false
    ``mismatch``.
    """
    if observed is None or expected is None:
        return False

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    a, b = _norm(observed), _norm(expected)
    if not a or not b:
        return False
    return a in b or b in a


async def _query_courtlistener(
    cite_text: str,
    *,
    client: Any = None,
) -> dict | None:
    """POST one cite to CourtListener's citation-lookup endpoint.

    Returns the first match dict on success, ``None`` on no-match,
    raises ``httpx.HTTPError`` on transport / 5xx — the caller
    decides whether to map that to ``unreachable``.
    """
    headers = {"User-Agent": "kaos-agents citation-verifier"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Token {key}"

    payload = {"text": cite_text}
    url = f"{_COURTLISTENER_BASE}/citation-lookup/"

    if client is None:
        async with httpx.AsyncClient(timeout=_VERIFY_TIMEOUT_S) as fresh_client:
            response = await fresh_client.post(url, json=payload, headers=headers)
    else:
        response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    # CL returns a list-of-dicts per requested cite; we send one cite,
    # so we expect a single envelope. Pre-flight defensive guards
    # against shape drift.
    if not isinstance(data, list) or not data:
        return None
    envelope = data[0]
    clusters = envelope.get("clusters") or []
    if not clusters:
        return None
    # The endpoint orders matches by relevance — first cluster wins.
    return clusters[0]


# ── Single-cite verification ────────────────────────────────────────


async def verify_case_citation(
    cite: CaseCitation,
    *,
    client: Any = None,
) -> CitationVerificationResult:
    """Verify one :class:`CaseCitation` against CourtListener.

    Returns a :class:`CitationVerificationResult`. The caller is
    expected to:

    1. Emit a ``CitationVerified`` event with this result.
    2. If ``result.is_problem``, fold the diagnostic into the
       agent's ``thinking_note`` for the next iteration.

    The function never raises — all error paths map to a
    ``status="unreachable"`` result with the underlying exception
    summarised in ``diagnostic``.
    """
    if not _verify_enabled():
        return CitationVerificationResult(
            status="unreachable",
            raw_cite=cite.raw,
            diagnostic=(
                "CourtListener verification disabled "
                "(set KAOS_AGENT_CITATION_VERIFY_ENABLED=1 to enable)."
            ),
        )

    cite_text = _build_cite_text(cite)
    if cite_text is None:
        return CitationVerificationResult(
            status="skipped",
            raw_cite=cite.raw,
            diagnostic=(
                "missing structural fields (volume / reporter / page) — "
                "cannot resolve against CourtListener"
            ),
        )

    try:
        match = await _query_courtlistener(cite_text, client=client)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "citation_verifier.unreachable: cite=%r exc=%s",
            cite.raw,
            exc,
        )
        return CitationVerificationResult(
            status="unreachable",
            raw_cite=cite.raw,
            diagnostic=f"CourtListener request failed: {exc.__class__.__name__}",
        )

    if match is None:
        return CitationVerificationResult(
            status="not_found",
            raw_cite=cite.raw,
            diagnostic=(
                f"CourtListener has no cluster for {cite_text!r} — "
                "either the cite is fabricated or the reporter is outside "
                "CL's coverage"
            ),
        )

    observed_name = match.get("case_name") or match.get("case_name_full")
    date_filed = match.get("date_filed") or ""
    try:
        observed_year = int(date_filed[:4]) if date_filed else None
    except ValueError:
        observed_year = None

    cluster_url = match.get("absolute_url")
    if cluster_url and not cluster_url.startswith("http"):
        cluster_url = f"https://www.courtlistener.com{cluster_url}"

    name_ok = cite.case_name is None or _names_match(observed_name, cite.case_name)
    year_ok = (
        cite.year is None
        or observed_year is None
        or abs(cite.year - observed_year) <= 1  # CL filing-date sometimes off-by-one
    )

    if name_ok and year_ok:
        return CitationVerificationResult(
            status="verified",
            raw_cite=cite.raw,
            courtlistener_url=cluster_url,
            observed_case_name=observed_name,
            observed_year=observed_year,
        )

    diagnostic_parts = []
    if not name_ok:
        diagnostic_parts.append(
            f"case_name mismatch (expected={cite.case_name!r} observed={observed_name!r})"
        )
    if not year_ok:
        diagnostic_parts.append(f"year mismatch (expected={cite.year} observed={observed_year})")
    return CitationVerificationResult(
        status="mismatch",
        raw_cite=cite.raw,
        courtlistener_url=cluster_url,
        observed_case_name=observed_name,
        observed_year=observed_year,
        diagnostic="; ".join(diagnostic_parts),
    )


# ── Multi-cite verification ─────────────────────────────────────────


async def verify_citations_in_text(
    text: str,
    *,
    max_concurrent: int = 4,
) -> tuple[CitationVerificationResult, ...]:
    """Extract case citations from ``text`` and verify each.

    Used by the AgenticLoop post-worker gate. Extraction is offline
    (pure kaos-citations call); the verification HTTP calls are
    parallelised under a small concurrency cap so a 5-cite answer
    doesn't take 5x the single-call latency.

    Returns a tuple of results in the same order the citations
    appeared in ``text``. Empty tuple when the text has no case
    citations (the common case for non-legal queries).
    """
    from kaos_citations import CaseCitation as _CaseCitation
    from kaos_citations import extract_citations

    citations = extract_citations(text)
    # ``kind == "case"`` narrows the discriminated union at runtime;
    # the isinstance check below is what ty needs to see to narrow
    # the static type to ``CaseCitation``.
    case_citations: list[CaseCitation] = [c for c in citations if isinstance(c, _CaseCitation)]
    if not case_citations:
        return ()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _verify_one(c: CaseCitation) -> CitationVerificationResult:
        async with semaphore:
            return await verify_case_citation(c)

    results = await asyncio.gather(*[_verify_one(c) for c in case_citations])
    return tuple(results)


__all__ = [
    "CitationVerificationResult",
    "VerificationStatus",
    "verify_case_citation",
    "verify_citations_in_text",
]
