"""Tier 7: PDF extraction via kaos-pdf.

Goal: agent reads a 1-page government-letter PDF (city council letter
re: docket OST-2008-0299) and answers questions about it. Verifies
kaos-pdf bridges into the agent runtime through ``--with-pdf`` /
``register_pdf_tools``, and the agent can actually pull facts out
of an extracted ContentDocument.

Cost target: ~$0.05.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    collect_events,
    model_for_tier,
    text_from_summary,
)

pytestmark = pytest.mark.live

TIER = 7
BUDGET_USD = 0.10

PDF_FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.mark.asyncio
async def test_pdf_extraction(ladder_runner_factory, turn_session_id):
    """Extract specific facts from a 1-page PDF in the prompt context."""
    # Parse the fixture text once. We pass the extracted text directly
    # in the prompt rather than registering kaos-pdf tools, because
    # the goal here is to verify the agent can REASON about PDF
    # content, not the tool-bridging path (which is covered by tier 2).
    from kaos_pdf import extract_pdf

    assert PDF_FIXTURE.exists(), f"missing fixture: {PDF_FIXTURE}"
    doc = extract_pdf(str(PDF_FIXTURE))
    from kaos_content import serialize_text

    pdf_text = serialize_text(doc)
    assert len(pdf_text) > 100, "PDF parsing returned too little text"

    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a document-extraction assistant. Read the document "
                "below and answer the user's question precisely."
            ),
            "model": model_for_tier(TIER),
        },
    )
    msg = (
        "Read this document and answer two questions:\n"
        "  1. What is the docket number referenced in the subject line?\n"
        "  2. Who is the city's Mayor?\n"
        "Format: 'Docket: <number>; Mayor: <name>'.\n\n"
        f"=== DOCUMENT ===\n{pdf_text}"
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-7")

    text = text_from_summary(events)
    # Ground truth from the PDF (visually inspected):
    # Subject: Docket OST-2008-0299 — EAS at El Centro/Imperial, California
    # Mayor: Geoff Dale
    assert "OST-2008-0299" in text, (
        f"answer must contain docket number 'OST-2008-0299' from the PDF; got: {text[:300]!r}"
    )
    assert "Geoff Dale" in text or "geoff dale" in text.lower(), (
        f"answer must contain Mayor 'Geoff Dale' from the PDF; got: {text[:300]!r}"
    )
