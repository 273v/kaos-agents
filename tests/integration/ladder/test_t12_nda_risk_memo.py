"""Tier 12: NDA risk-deviation memo — graded by LLM-as-judge with ground truth.

The "deal-room senior associate" use case: review a portfolio of
contracts, identify deviations from a standard mutual NDA, write a
structured memo. T12 grades the agent's memo against a manually-
authored EXPECTED_FINDINGS set via a separate LLM-as-judge call —
the judge sees both the memo AND the correct findings, and scores
the memo on coverage / specificity / accuracy.

Cost target: ~$0.30 (one OpenAI call to write the memo, one judge
call to grade it).
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
from tests.integration.ladder.fixtures.nda_ground_truth import (
    EXPECTED_FINDINGS,
)
from tests.integration.ladder.test_t11_nda_tabular import _load_nda_corpus

pytestmark = pytest.mark.live

TIER = 12
BUDGET_USD = 0.40

NDA_DIR = Path(__file__).parent / "fixtures" / "nda"


# Judge runs on a different model than the memo author. Same provider
# generation (claude-sonnet-4-6) — the judge needs a strong model to
# compare two pieces of legal prose accurately. Cost ~$0.05 per call.
JUDGE_MODEL = "anthropic:claude-sonnet-4-6"


async def _judge_memo(memo: str, expected_findings: tuple[str, ...]) -> dict:
    """Score the agent's memo against EXPECTED_FINDINGS via a judge LLM call.

    Returns a dict with keys:
      - "score": int 0-10 (overall correctness/coverage/specificity)
      - "covered": list[int] of expected_findings indices the memo addresses
      - "missing": list[int] of indices the memo did NOT address
      - "incorrect_claims": list[str] of memo statements that are wrong or
        unsupported by the contracts
      - "reasoning": short justification
    """
    from kaos_llm_core import Call, InputField, OutputField, Signature

    findings_block = "\n".join(f"  [{i}] {finding}" for i, finding in enumerate(expected_findings))

    class _MemoJudgeSignature(Signature):
        memo: str = InputField(description="The agent's risk-deviation memo to grade.")
        expected_findings: str = InputField(
            description=(
                "The numbered list of expected findings a correct memo must "
                "address. The memo doesn't have to use the same wording — "
                "just cover the same substantive points."
            )
        )
        score: int = OutputField(
            description=(
                "Overall score 0-10. 0 = memo missed everything or contradicts "
                "the contracts. 10 = memo covers every expected finding "
                "accurately and adds no incorrect claims. Use the rubric: "
                "coverage (4 pts), specificity (3 pts), accuracy (3 pts)."
            )
        )
        covered: list[int] = OutputField(
            description=(
                "Indices (0-based) of expected_findings the memo addresses "
                "substantively (the memo doesn't have to copy the wording)."
            )
        )
        missing: list[int] = OutputField(description="Indices the memo did NOT address.")
        incorrect_claims: list[str] = OutputField(
            description=(
                "Statements in the memo that are factually wrong or "
                "unsupported. Empty list if none."
            )
        )
        reasoning: str = OutputField(description="One short paragraph justifying the score.")

    call = Call(_MemoJudgeSignature, model=JUDGE_MODEL)
    invocation = await call.invoke(memo=memo, expected_findings=findings_block)
    out = invocation.output
    return {
        "score": int(out.score),
        "covered": list(out.covered),
        "missing": list(out.missing),
        "incorrect_claims": list(out.incorrect_claims),
        "reasoning": str(out.reasoning),
    }


@pytest.mark.asyncio
async def test_nda_risk_memo_judged_against_ground_truth(ladder_runner_factory, turn_session_id):
    """LLM-as-judge scores the memo's coverage of EXPECTED_FINDINGS."""
    docs = _load_nda_corpus()
    assert len(docs) == 5

    context = "\n\n".join(f"=== CONTRACT: {name} ===\n{text}" for name, text in docs.items())
    msg = (
        "Below are 5 mutual NDAs from a deal-room. Review them and write "
        "a 1-page risk-deviation memo. Identify EVERY way these contracts "
        "diverge from a hypothetical 'house standard' mutual NDA on these "
        "dimensions:\n"
        "  1. Governing law and venue\n"
        "  2. Term length\n"
        "  3. Confidentiality survival period\n"
        "  4. Non-solicitation clauses (presence and scope)\n"
        "  5. Any template artifacts, drafting errors, or anomalies that "
        "would catch a careful reviewer's eye\n\n"
        "STRUCTURE the memo with these EXACT section headers:\n"
        "## Summary\n"
        "## Findings\n"
        "## Recommendations\n\n"
        "In Findings, reference each contract by filename when the contract "
        "deviates. Be specific — cite exact terms, periods, and "
        "jurisdictions. The reviewing partner needs to act on this memo "
        "without re-reading the contracts.\n\n"
        f"{context}"
    )
    runner, _ = ladder_runner_factory(
        agent_kwargs={
            "instructions": (
                "You are senior counsel reviewing NDAs for a portfolio "
                "company. Output the memo only — no preamble. Reference "
                "contracts by exact filename when discussing them."
            ),
            "model": model_for_tier(TIER),
        },
    )
    events = await collect_events(runner.run(msg, turn_session_id))

    assert_within_budget(events, budget_usd=BUDGET_USD, label="tier-12")

    memo = text_from_summary(events)

    # Structural checks (cheap, deterministic)
    for header in ("## Summary", "## Findings", "## Recommendations"):
        assert header in memo, f"memo missing required section header {header!r}"
    assert 500 <= len(memo) <= 12000, f"memo length {len(memo)} outside expected range"

    # LLM-as-judge: score the memo against EXPECTED_FINDINGS
    judgment = await _judge_memo(memo, EXPECTED_FINDINGS)

    # Coverage + correctness assertions
    score = judgment["score"]
    covered = judgment["covered"]
    missing = judgment["missing"]
    incorrect = judgment["incorrect_claims"]

    # Score floor: 6 of 10 (the rubric is conservative — 4 pts for full
    # coverage, 3 pts for specificity, 3 pts for accuracy). 6 means the
    # memo is materially correct and useful, even if it missed an edge
    # case or paraphrased a finding loosely.
    assert score >= 6, (
        f"tier-12: judge scored memo {score}/10 (target >=6). "
        f"Reasoning: {judgment['reasoning']!r}\n"
        f"Missing findings: {missing}\n"
        f"Incorrect claims: {incorrect}"
    )

    # The memo must cover at least 3 of the 5 expected findings.
    assert len(covered) >= 3, (
        f"tier-12: memo covered only {len(covered)} of "
        f"{len(EXPECTED_FINDINGS)} expected findings. Missing: {missing}. "
        f"Reasoning: {judgment['reasoning']!r}"
    )

    # No more than 1 incorrect claim — small drafting slip is OK,
    # systematic factual errors are not.
    assert len(incorrect) <= 1, (
        f"tier-12: judge flagged {len(incorrect)} incorrect claims in "
        f"the memo: {incorrect}. Reasoning: {judgment['reasoning']!r}"
    )
