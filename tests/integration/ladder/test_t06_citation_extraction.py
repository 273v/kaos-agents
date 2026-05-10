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
                "You are a legal-research assistant. When discussing federal "
                "law, cite the relevant U.S. Code section in standard form "
                "(e.g. '42 U.S.C. § 1983')."
            ),
            "model": model_for_tier(TIER),
        },
    )
    msg = (
        "What is 42 U.S.C. § 1983 (the federal civil rights statute) and what "
        "does it allow plaintiffs to do? Cite the section in your answer."
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-6")

    # CitationFound events fire from the kaos-citations integration.
    cites = citation_events(events)
    assert cites, (
        "expected >=1 CitationFound event. The answer references "
        "42 U.S.C. § 1983 explicitly; if no events fire, the citation "
        "verifier (kaos_agents.grounding.emit_citations_for_text) "
        "regressed or kaos-citations is not installed."
    )

    # The detected claim should mention the U.S.C. cite.
    claims = " ".join(str(getattr(c, "claim", "") or "") for c in cites)
    assert "1983" in claims or "U.S.C." in claims, (
        f"expected the U.S.C. citation in CitationFound.claim text; got: {claims!r}"
    )

    # Answer text actually references the statute.
    text = text_from_summary(events)
    assert "1983" in text, f"answer must mention 42 U.S.C. § 1983 verbatim; got: {text[:300]!r}"
