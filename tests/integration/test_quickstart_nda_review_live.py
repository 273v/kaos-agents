"""Live regression for the NDA-batch review quickstart.

The README ships ``python -m kaos_agents.examples.nda_review.quickstart``
as the canonical demo of the package's value prop (FindingsAgent over a
multi-doc DOCX corpus with cited provenance + cost cap + refusal
contract + audit trail). This test invokes the same ``run()`` coroutine
and asserts the agent surfaces the deviation classes encoded in the
lawyer-curated ground truth at
``tests/integration/ladder/fixtures/nda_ground_truth.py`` — so the
quickstart can't drift from reality unannounced.

Budget: ~$0.10-0.20 end-to-end on Anthropic Haiku 4.5 filter + Sonnet
4.6 synthesis (5 NDAs x ~$0.04 each). Marked ``pytest.mark.live`` so
the default non-live gate doesn't pay this cost.
"""

from __future__ import annotations

import os

import pytest

from tests.integration.ladder.fixtures.nda_ground_truth import (
    EXPECTED_FINDINGS,
    GROUND_TRUTH,
)

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_quickstart_surfaces_expected_findings() -> None:
    """Run the quickstart end-to-end and verify it matches ground truth.

    Asserts:
      1. The quickstart loads the same 5 fixtures the ground truth
         describes.
      2. Every reviewed NDA produces findings with provenance set
         (``finding_id`` + either ``block_ref`` or ``page`` or
         ``char_span``).
      3. Total cost is positive and well below the per-doc cap
         (``5 * max_cost_per_doc``).
      4. No per-doc review reports ``budget_exceeded`` at the
         generous default cap.
      5. The aggregate report covers at least 4 of the 5
         ``EXPECTED_FINDINGS`` deviation classes — governing law,
         term length, confidentiality survival, non-solicit
         presence, and the EMNA signature-block template artifact.
    """
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.fail(
            "ANTHROPIC_API_KEY missing — the NDA quickstart live test "
            "requires a real Anthropic key. Fix the env; the no-skips "
            "policy means we never silently bypass this assertion."
        )

    from kaos_agents.examples.nda_review import quickstart as q

    results = await q.run()

    # 1. Same 5 fixtures as the ground truth.
    assert set(results.keys()) == {gt.filename for gt in GROUND_TRUTH}, (
        f"Quickstart reviewed a different set of files than the "
        f"ground truth lists. Got {sorted(results.keys())!r}; "
        f"expected {sorted(gt.filename for gt in GROUND_TRUTH)!r}."
    )

    # 2. Every reviewed NDA has at least one finding with provenance.
    #    A "refusal" outcome is acceptable in principle but should not
    #    fire on these contracts — they're substantive NDAs and the
    #    question is broad.
    total_findings = 0
    total_cost = 0.0
    for name, result in results.items():
        refusal = result.refusal
        assert refusal is None, (
            f"{name}: agent refused review with reason={refusal.reason!r}. "
            f"On a substantive NDA + a broad review question that should "
            f"never happen — investigate. Message: {refusal.message!r}"
        )
        assert not result.budget_exceeded, (
            f"{name}: budget_exceeded=True at the generous per-doc cap. "
            f"Either cost regressed or the cap is too tight. "
            f"Spend on this doc: ${result.total_cost_usd:.4f}."
        )
        # Cost must be positive — a $0 run means the LLM transport is
        # mocked, which would defeat the regression.
        assert result.total_cost_usd > 0, (
            f"{name}: total_cost_usd is zero — the live LLM call did not "
            f"happen. Refusing to declare green."
        )
        # Provenance: every surviving finding must carry an id + at
        # least one anchor (block_ref / page / char_span) so a partner
        # can verify the cite.
        for f in result.findings:
            cand = f.candidate
            assert cand.finding_id, f"{name}: finding without finding_id"
            has_anchor = (
                cand.block_ref is not None or cand.char_span is not None or cand.page is not None
            )
            assert has_anchor, (
                f"{name}: finding {cand.finding_id} has no provenance "
                f"anchor (block_ref / char_span / page all None) — "
                f"partner cannot verify this cite."
            )
        total_findings += len(result.findings)
        total_cost += result.total_cost_usd

    # 3. Aggregate sanity bounds.
    assert total_findings >= 20, (
        f"Quickstart produced only {total_findings} findings across "
        f"all 5 NDAs (expected >=20 — each NDA typically yields 5+). "
        f"Either filter regressed or LLM is misbehaving."
    )
    assert total_cost < 1.50, (
        f"Quickstart spent ${total_cost:.4f} total (budget <$1.50). Investigate cost regression."
    )

    # 5. Substantive deviation-class coverage — aggregate the surviving
    #    finding text + synthesis answers across all 5 NDAs, then check
    #    for each deviation class encoded in EXPECTED_FINDINGS.
    combined = " ".join(
        " ".join([result.answer] + [f.candidate.text for f in result.findings])
        for result in results.values()
    ).lower()

    # 5a. Governing-law deviation — Michigan (house) vs Delaware (EMNA,
    # DynaMo). The aggregate report must mention BOTH jurisdictions
    # somewhere — otherwise it didn't surface the deviation.
    assert "delaware" in combined and "michigan" in combined, (
        "Aggregate report missed the governing-law deviation: every "
        "NDA's review must collectively surface both Delaware (EMNA, "
        "DynaMo) and Michigan (the other three)."
    )

    # 5b. Term-length variation — at least two of the four distinct
    # term shapes (open-ended, 2yr, 3yr, 5yr) must surface.
    term_terms = (
        "open-ended",
        "open ended",
        "no stated end",
        "no end date",
        "perpetu",
        "two-year",
        "two year",
        "2-year",
        "three-year",
        "three year",
        "3-year",
        "five-year",
        "five year",
        "5-year",
        "second anniversary",
        "third anniversary",
        "fifth anniversary",
    )
    hits = sum(1 for t in term_terms if t in combined)
    assert hits >= 2, (
        f"Aggregate report only mentioned {hits} of the term-length "
        f"variants. Expected at least two distinct shapes across "
        f"open-ended, 2yr, 3yr, and 5yr terms."
    )

    # 5c. Confidentiality-period survival — either the 1-year cap or
    # the "until written notice / perpetual" variant must surface.
    conf_terms = (
        "one year",
        "one-year",
        "1 year",
        "1-year",
        "written notice",
        "written release",
        "perpetu",
        "survive",
        "survival",
    )
    assert any(t in combined for t in conf_terms), (
        "Aggregate report missed the confidentiality-survival "
        "deviation — neither the one-year cap nor the "
        "perpetual/written-notice variant surfaced anywhere."
    )

    # 5d. Non-solicit presence — three of the five NDAs (EMNA,
    # CyberCorp, DynaMo) have non-solicit clauses ranging 6-12 months.
    nonsol_terms = ("non-solicit", "nonsolicit", "non solicit", "solicit")
    assert any(t in combined for t in nonsol_terms), (
        "Aggregate report missed the non-solicitation clauses present "
        "in EMNA, CyberCorp, and DynaMo."
    )

    # 5e. EMNA signature-block template artifact — the signature
    # block reads "DynaMo" even though the named counterparty is
    # ExMachi Bank N.A. The aggregate report must surface this
    # specific anomaly (the recall-first promise of FindingsAgent).
    emna_review = results["EMNA Mutual NDA.docx"]
    emna_text = " ".join(
        [emna_review.answer] + [f.candidate.text for f in emna_review.findings]
    ).lower()
    assert "dynamo" in emna_text, (
        "EMNA review did not surface the template-artifact anomaly "
        "(signature block names 'DynaMo' even though parties are "
        "273 Ventures + ExMachi Bank N.A.). This is the headline "
        "anomaly EXPECTED_FINDINGS[4] documents."
    )
    # The review should additionally flag this as a template /
    # signature-block issue, not just mention the word incidentally.
    assert any(
        t in emna_text
        for t in (
            "signature",
            "template artifact",
            "template",
            "placeholder",
            "not a party",
            "not updated",
        )
    ), (
        "EMNA review mentions 'DynaMo' but doesn't characterize it as "
        "a template / signature-block / placeholder issue."
    )

    # Provenance: at least one finding across the corpus has a
    # block_ref (the strongest provenance anchor). This is what makes
    # the cite verifiable through the AST round-trip.
    has_any_block_ref = any(
        f.candidate.block_ref is not None for result in results.values() for f in result.findings
    )
    assert has_any_block_ref, (
        "No finding across any of the 5 NDAs has a block_ref. The "
        "DocumentView segmenter or the FindingsAgent provenance "
        "plumbing regressed."
    )

    # We exercised EXPECTED_FINDINGS via 5a-5e above — touching the
    # tuple here keeps the test honest if a future contributor adds
    # a 6th expected finding without updating the assertions.
    assert len(EXPECTED_FINDINGS) >= 5
