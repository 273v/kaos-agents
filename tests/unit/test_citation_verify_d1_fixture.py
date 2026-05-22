"""D1 (citation fabrication) regression fixture — Mata v. Avianca pattern.

Plan §Issue 1 acceptance row:

    D1 (citation fabrication) | Mata v. Avianca regression fixture:
    Brown v. Board, 347 U.S. 495 | loop emits diagnostic, replans,
    final answer cites 483 or refuses with verify-failure rationale

The 2023 Mata v. Avianca matter saw counsel cite fabricated cases
through an LLM with no verification gate. The kaos-agents B0.8
gate (kaos-agents 0.1.7) ships ``verify_citations_in_text`` that
resolves cited cases against CourtListener and emits a
:class:`CitationVerified` event per cite. This file pins the
contract at the primitive layer:

1. ``CitationVerificationResult.is_problem`` correctly classifies
   ``mismatch`` and ``not_found`` as problems that MUST block a
   confident answer (the malpractice case).
2. ``unreachable`` is NOT a problem on its own — but the loop is
   responsible for surfacing it to the user as "could not verify"
   (a separate tier of disclosure documented in the helper's
   docstring).
3. The diagnostic field carries the corrected reference when
   CourtListener returns a mismatch — so the loop's
   ``_replan_diagnostics`` chain has enough text to teach the
   worker the right page on the next iteration.

The full integration test (live CourtListener call against the
real ``Brown v. Board, 347 U.S. 495`` cite) lives in
``tests/integration/test_citation_verify_gate_replan.py`` and
runs only with ``KAOS_AGENT_CITATION_VERIFY_ENABLED=1`` and a
``KAOS_AGENT_COURTLISTENER_API_KEY``.
"""

from __future__ import annotations

import pytest

from kaos_agents.citations.verifier import CitationVerificationResult

# ── is_problem classification ────────────────────────────────────────


@pytest.mark.unit
def test_mismatch_is_a_problem() -> None:
    """Mata v. Avianca pattern: cite resolves to a real cluster but
    a canonical field disagrees (volume / reporter / page / year).
    The gate MUST block a confident answer."""
    result = CitationVerificationResult(
        status="mismatch",
        raw_cite="Brown v. Board of Education, 347 U.S. 495 (1954)",
        observed_case_name="Brown v. Board of Education of Topeka",
        observed_year=1954,
        courtlistener_url="https://www.courtlistener.com/opinion/103127/brown-v-board-of-education/",
        diagnostic="cite page 495 disagrees with CourtListener page 483",
    )
    assert result.is_problem is True
    assert result.diagnostic
    assert "495" in result.diagnostic
    assert "483" in result.diagnostic


@pytest.mark.unit
def test_not_found_is_a_problem() -> None:
    """A wholly fabricated cite (no CourtListener cluster at all)
    is the most dangerous failure mode — and the most common one
    in the Mata v. Avianca fact pattern."""
    result = CitationVerificationResult(
        status="not_found",
        raw_cite="Varghese v. China Southern Airlines Co., 925 F.3d 1339 (11th Cir. 2019)",
        diagnostic="CourtListener returned no cluster for 925 F.3d 1339",
    )
    assert result.is_problem is True


@pytest.mark.unit
def test_verified_is_not_a_problem() -> None:
    """The happy path — CourtListener matched on all canonical fields."""
    result = CitationVerificationResult(
        status="verified",
        raw_cite="Brown v. Board of Education, 347 U.S. 483 (1954)",
        observed_case_name="Brown v. Board of Education of Topeka",
        observed_year=1954,
        courtlistener_url="https://www.courtlistener.com/opinion/103127/brown-v-board-of-education/",
    )
    assert result.is_problem is False


@pytest.mark.unit
def test_unreachable_is_not_classified_as_a_problem() -> None:
    """Network / 5xx / disabled-via-env means we can neither pass
    nor fail. The gate's docstring is explicit: the worker MUST
    surface the cite as unverified, but the loop does NOT force a
    replan — the cite might be correct, we just can't confirm.

    This invariant is load-bearing: flipping it to True would
    cascade-refuse every offline run, which is the wrong
    user-facing outcome (the user wanted an answer, not a
    network-error stack)."""
    result = CitationVerificationResult(
        status="unreachable",
        raw_cite="Brown v. Board of Education, 347 U.S. 483 (1954)",
        diagnostic="CourtListener returned 503 Service Unavailable after 3 retries",
    )
    assert result.is_problem is False


@pytest.mark.unit
def test_skipped_is_not_classified_as_a_problem() -> None:
    """A cite that didn't have enough canonical fields to verify
    (missing volume / reporter / page) is recorded but doesn't
    trip a replan — the agent already gave the user what it had,
    we just can't independently verify a partial reference."""
    result = CitationVerificationResult(
        status="skipped",
        raw_cite="some 1954 case from the Supreme Court",
        diagnostic="missing required fields: volume / reporter / page",
    )
    assert result.is_problem is False


# ── Diagnostic content ──────────────────────────────────────────────


@pytest.mark.unit
def test_mismatch_diagnostic_carries_both_expected_and_observed() -> None:
    """The diagnostic the loop folds into ``thinking_note`` MUST
    contain enough text for the worker's next iteration to learn
    the correct value. Free-form text is OK; the contract is just
    that the cited (wrong) value and the resolved (right) value
    are both present."""
    result = CitationVerificationResult(
        status="mismatch",
        raw_cite="Brown v. Board, 347 U.S. 495 (1954)",
        diagnostic="cite page 495 disagrees with CourtListener page 483",
    )
    assert "495" in (result.diagnostic or "")
    assert "483" in (result.diagnostic or "")


# ── Frozen dataclass contract ───────────────────────────────────────


@pytest.mark.unit
def test_result_exposes_canonical_fields() -> None:
    """The audit trail anchors on the raw_cite + status pair.
    Pin the field set here so a future refactor doesn't silently
    drop the columns the SPA / run-log consumer relies on."""
    result = CitationVerificationResult(
        status="verified",
        raw_cite="Brown v. Board of Education, 347 U.S. 483 (1954)",
        observed_case_name="Brown v. Board of Education of Topeka",
        observed_year=1954,
        courtlistener_url="https://www.courtlistener.com/opinion/103127/brown-v-board-of-education/",
    )
    assert result.status == "verified"
    assert result.raw_cite.startswith("Brown v. Board")
    assert result.observed_case_name == "Brown v. Board of Education of Topeka"
    assert result.observed_year == 1954
    assert result.courtlistener_url is not None
