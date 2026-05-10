"""Tier 12: NDA risk-deviation memo — compose findings into a memo.

The "deal-room senior associate" use case: review a portfolio of
contracts, identify deviations from a standard mutual NDA template,
write a structured memo that references each contract by name. Tests:

- Multi-document analysis in a single agent turn
- Output composition (intro / findings / recommendations structure)
- Per-contract referencing (the memo must name specific NDAs, not
  speak generically)
- The model's ability to identify legal-pattern deviations (mutual
  vs. one-way obligation, term length, IP carve-out presence)

This is the higher-effort cousin of tier 11. Whereas tier 11 outputs
structured data (a row per contract), tier 12 outputs a NARRATIVE
that synthesizes across contracts — closer to what a partner would
actually receive on a deal.

Cost target: ~$0.20 (one OpenAI call with ~10K input, ~1.5K output).
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
from tests.integration.ladder.test_t11_nda_tabular import (
    _load_nda_corpus,
)

pytestmark = pytest.mark.live

TIER = 12
BUDGET_USD = 0.30

NDA_DIR = Path(__file__).parent / "fixtures" / "nda"


@pytest.mark.asyncio
async def test_nda_risk_memo_composition(ladder_runner_factory, turn_session_id):
    """Compose a structured deviation memo across 5 NDAs."""
    docs = _load_nda_corpus()
    assert len(docs) == 5

    context = "\n\n".join(f"=== CONTRACT: {name} ===\n{text[:3000]}" for name, text in docs.items())
    msg = (
        "Below are 5 mutual NDAs from a deal-room. Review them and write "
        "a 1-page risk-deviation memo identifying any contract that DEVIATES "
        "from a standard mutual NDA on these dimensions:\n"
        "  1. Mutuality — obligations actually reciprocal vs. unilateral\n"
        "  2. Term — confidentiality period; flag if perpetual or > 5 years\n"
        "  3. Governing law — flag any non-Michigan or out-of-state choice\n"
        "  4. IP carve-outs — flag missing intellectual-property protection\n"
        "  5. Remedies — flag any waiver of injunctive relief\n\n"
        "STRUCTURE the memo with these exact section headers:\n"
        "## Summary\n"
        "## Findings\n"
        "## Recommendations\n\n"
        "In Findings, reference each deviating contract by filename. If a "
        "contract is fully standard, say so. Be concrete: cite the specific "
        "term, period, or jurisdiction.\n\n"
        f"{context}"
    )
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are senior counsel reviewing NDAs for a portfolio "
                "company. Be precise and reference contracts by exact "
                "filename. Output the memo only — no preamble."
            ),
            "model": model_for_tier(TIER),
        },
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-12")

    text = text_from_summary(events)

    # Structure check: all three section headers present
    for header in ("## Summary", "## Findings", "## Recommendations"):
        assert header in text, (
            f"memo missing required section header {header!r}. Text "
            f"(first 400 chars): {text[:400]!r}"
        )

    # Each section non-empty: text between '## Summary' and '## Findings'
    # must be >0 chars, etc. (Loose floor — a one-word section passes;
    # we're testing structure, not quality.)
    assert text.find("## Summary") < text.find("## Findings") < text.find("## Recommendations"), (
        "section headers out of order"
    )
    summary_body = text.split("## Summary", 1)[1].split("## Findings", 1)[0].strip()
    findings_body = text.split("## Findings", 1)[1].split("## Recommendations", 1)[0].strip()
    recommendations_body = text.split("## Recommendations", 1)[1].strip()
    assert len(summary_body) >= 30, "Summary section too thin"
    assert len(findings_body) >= 100, "Findings section too thin"
    assert len(recommendations_body) >= 30, "Recommendations section too thin"

    # Per-contract referencing: the memo must mention >=3 of the 5
    # contracts by filename or distinctive substring (Acme, BI, CC,
    # DynaMo, EMNA).
    fixture_tokens = ("Acme", "BI", "CC", "DynaMo", "EMNA")
    referenced = sum(1 for tok in fixture_tokens if tok in text)
    assert referenced >= 3, (
        f"memo must reference >=3 of the 5 contracts by name; "
        f"matched only {referenced}. Tokens: {fixture_tokens}"
    )

    # Sanity: total memo length is in a reasonable range. <500 chars
    # is too short to be a real memo; >8000 means the model dumped
    # the contracts back rather than composing.
    assert 500 <= len(text) <= 8000, f"memo length {len(text)} outside expected 500-8000 char range"
