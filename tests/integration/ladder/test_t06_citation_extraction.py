"""Tier 6: citation auto-extraction from assistant text.

The CitationVerifier helper (kaos_agents.grounding.emit_citations_for_text)
auto-fires after every turn produces text. When the answer contains
recognizable citations, ``CitationFound`` events should land in the
event stream. Tier-6 verifies the pipeline by asking an answer that
naturally references a U.S. Code statute.

Cost target: ~$0.05.
"""

from __future__ import annotations

import pytest

from tests.integration.ladder.conftest import (
    assert_within_budget,
    citation_events,
    collect_events,
    model_for_tier,
    text_from_summary,
)

pytestmark = pytest.mark.live

TIER = 6
BUDGET_USD = 0.10


@pytest.mark.asyncio
async def test_citation_auto_extraction(ladder_runner_factory, turn_session_id):
    """An answer that mentions a U.S.C. statute fires CitationFound events."""
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are a legal-research assistant. When you mention a "
                "Supreme Court case or a federal statute, ALWAYS include "
                "a standard legal citation (e.g. 'Miranda v. Arizona, 384 "
                "U.S. 436 (1966)' or '42 U.S.C. § 1983')."
            ),
            "model": model_for_tier(TIER),
        },
    )
    # IMPORTANT: the prompt does NOT contain a citation. The model must
    # produce one from training knowledge — the verifier then runs on
    # the MODEL'S output, not on echoed-from-prompt text. Earlier T06
    # was theater because '42 U.S.C. § 1983' was in the prompt.
    msg = (
        "Briefly explain the Miranda warning requirement in 2-3 sentences. "
        "Include the standard case citation (case name + reporter + year)."
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-6")

    text = text_from_summary(events)
    # First, sanity-check that the model actually produced a citation.
    # If it didn't, the test is testing the model not the verifier —
    # surface that as a separate failure with a clear message.
    import re

    has_case_cite = bool(
        re.search(r"\b\d+\s+U\.?\s?S\.?\s+\d+", text)
        or re.search(r"\bMiranda\s+v\.\s+Arizona", text, re.IGNORECASE)
    )
    assert has_case_cite, (
        f"the model did not produce a recognizable case citation in its "
        f"response. The verifier has nothing to recognize. Answer was: "
        f"{text[:400]!r}"
    )

    # CitationFound events from the kaos-citations integration on the
    # MODEL'S own output (not the prompt). This is the real verifier
    # regression check.
    cites = citation_events(events)
    assert cites, (
        f"the model produced a citation but the verifier emitted zero "
        f"CitationFound events. The citation_verifier integration "
        f"(kaos_agents.grounding.emit_citations_for_text) regressed "
        f"or kaos-citations stopped recognizing case citations. "
        f"Answer text: {text[:400]!r}"
    )

    # At least one detected citation should be a Case citation type
    # (Miranda v. Arizona). The CitationFound.claim is prefixed with
    # the kind: e.g. "[CaseCitation] 384 U.S. 436".
    has_case = any("Case" in str(getattr(c, "claim", "") or "") for c in cites)
    assert has_case, (
        f"expected at least one CitationFound with kind=CaseCitation "
        f"(Miranda v. Arizona). Detected claims: "
        f"{[getattr(c, 'claim', None) for c in cites]!r}"
    )
