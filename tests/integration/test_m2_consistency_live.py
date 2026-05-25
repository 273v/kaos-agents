"""Live evaluation of the M2 reasoning-action consistency critic.

Mirrors the shape of ``test_judge_signature_live.py``: a small case
table x two frontier models, each running the actual critic against
ground-truth text. The :func:`judge_reasoning_action_consistency`
helper is a thin wrapper over :func:`judge_with_rubric` — this file
exercises the rubric STRING against real LLM provider calls.

Cases are sourced from:

* SPA session ``01KS1K6J9XWKCNQ0NPNKXXXP4P`` (gpt-5.4-mini,
  2026-05-19) for the buggy ``contradicts_reasoning`` case — the
  Fed-funds branching prompt where the headline took the wrong
  branch.
* A clean control case for ``consistent``.
* A synthetic but realistic ``contradicts_tool_results`` case.

Required env vars: ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``.

Run with::

    uv run pytest tests/integration/test_m2_consistency_live.py -v -m live
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from kaos_agents.planning.m2_consistency import (
    M2_ALLOWED_LABELS,
    judge_reasoning_action_consistency,
)
from tests.integration._models import critic_model


def _have_key_for(model: str) -> bool:
    provider = model.split(":", 1)[0]
    env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider)
    return bool(env and os.environ.get(env))


@dataclass(frozen=True, slots=True)
class M2Case:
    case_id: str
    response_text: str
    tool_results_text: str
    expected_label: str
    min_confidence: float = 0.6


_CASES: tuple[M2Case, ...] = (
    # Ground-truth contradiction captured live from SPA session
    # 01KS1K6J9XWKCNQ0NPNKXXXP4P (gpt-5.4-mini, 2026-05-19).
    M2Case(
        case_id="spa-fed-branch-contradicts-reasoning",
        response_text=(
            "Branch taken: **upper bound >= 5.0%**.\n\n"
            "The Federal Reserve's current target range is "
            "**4.25%\u20134.50%**, so the upper bound is **4.50%** and "
            "does **not** reach 5.0% (source: "
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).\n\n"
            "Because the upper bound is below 5.0%, the relevant "
            "branch is the **next FOMC meeting** branch. The Fed's "
            "official 2026 FOMC calendar page shows the 2026 "
            "meetings, and the next listed meeting after the "
            "April 8, 2026 release is the **June 16\u201317, 2026** "
            "FOMC meeting."
        ),
        tool_results_text=(
            "kaos-web-search-page result: 'current target range "
            "for the federal funds rate is 4.25 to 4.50 percent'"
        ),
        expected_label="contradicts_reasoning",
    ),
    # Control: headline + body + tools all line up.
    M2Case(
        case_id="consistent-clean-branch",
        response_text=(
            "Branch taken: upper bound < 5.0%.\n\n"
            "The current Federal Reserve target range is 4.25%-4.50%, "
            "so the upper bound is 4.50% and below 5.0%. Therefore "
            "the next FOMC meeting is June 16-17, 2026."
        ),
        tool_results_text=(
            "kaos-web-search-page result: 'current target range "
            "for the federal funds rate is 4.25 to 4.50 percent'"
        ),
        expected_label="consistent",
    ),
    # Tools said one thing; the response asserts something
    # provably different. Should outrank a body-only contradiction.
    M2Case(
        case_id="contradicts-tool-results-fabricated-rate",
        response_text=(
            "The current US Federal Reserve federal funds rate "
            "upper bound is 5.50% (source: federal reserve)."
        ),
        tool_results_text=(
            "kaos-web-search-page result: 'The target range for "
            "the federal funds rate is 4.25 to 4.50 percent.'\n"
            "kaos-web-get-text result: 'effective rate: 4.33%'"
        ),
        expected_label="contradicts_tool_results",
    ),
    # No-tool follow-up recall from prior-turn MESSAGES. Captured
    # 2026-05-24 from SPA session 01KSDNYE2AMZ7CHCFJ5AEY34JP turn 2:
    # the prior turn fetched the FOMC press release (3 tools); the
    # follow-up question asked the agent to repeat the same fact. The
    # worker correctly recalled the rate + date from context. The M2
    # critic flagged the answer as ungrounded because tool_results was
    # empty THIS turn, which forced a refusal that clobbered the
    # worker draft in the UI. The rubric must NOT call this a
    # contradiction — recall-from-MESSAGES is a legitimate grounded
    # answer shape, and empty tool_results this turn doesn't disprove
    # anything the response says.
    M2Case(
        case_id="no-tools-recall-from-prior-turn-fomc",
        response_text=(
            "The target range was 3-1/2 to 3-3/4 percent, and the "
            "FOMC statement was dated April 29, 2026. Source: "
            "Federal Reserve, “Federal Reserve issues FOMC "
            "statement,” "
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            "monetary20260429a.htm."
        ),
        tool_results_text="",
        expected_label="consistent",
    ),
    # No-tool pure-reasoning answer (date arithmetic + FRCP citation).
    # Captured 2026-05-24 from SPA session 01KSDP8F73RWDFX8CEJM4M4X1W
    # turn 5: the question is "what is 21 days after March 15, 2025
    # under FRCP 12(a)(1)(A)(i)?". The worker correctly computed
    # April 5 (Sat) → April 7 (Mon) per FRCP 6(a)(1)(C). The M2
    # critic demanded the agent re-fetch FRCP 12(a) text from eCFR
    # to "ground" the answer, which is unnecessary: the rule cite is
    # well-formed and the arithmetic is verifiable from the rule
    # alone. Pure-reasoning answers must be allowed; the rubric must
    # not require a tool call for every factual claim.
    M2Case(
        case_id="no-tools-pure-reasoning-frcp",
        response_text=(
            "Under FRCP Rule 12(a)(1)(A)(i), the response is due 21 "
            "days after service of the complaint. Assuming the "
            "complaint was filed on March 15, 2025 and there is no "
            "service waiver, the deadline would be:\n\n"
            "March 15, 2025 + 21 days = April 5, 2025\n\n"
            "Because April 5, 2025 falls on a Saturday, the deadline "
            "moves to the next business day under FRCP time-"
            "computation rules:\n\n"
            "Adjusted deadline: Monday, April 7, 2025\n\n"
            "Source: FRCP Rule 12(a)(1)(A)(i) (21-day response "
            "period); FRCP Rule 6(a)(1)(C) (move deadline that "
            "lands on weekend to next business day)."
        ),
        tool_results_text="",
        expected_label="consistent",
    ),
    # No-tool + fabricated content — the ONE case where the critic
    # SHOULD raise a flag. Without tools, contradicts_tool_results
    # cannot apply (rubric rule 1) — so the only correct flag here
    # is contradicts_reasoning when the body's reasoning is
    # internally inconsistent, OR consistent when the headline+body
    # at least line up internally. Here the response confidently
    # asserts a specific rate without any reasoning to support it,
    # but the rubric's job is to catch INTERNAL contradiction, NOT
    # confident-from-training. That's M3's job (grounding).
    # Expected: consistent. (If the critic emits contradicts_*
    # here, the rubric is over-firing on "looks unverified" rather
    # than catching actual contradictions — that's the failure mode
    # we want to prevent.)
    M2Case(
        case_id="no-tools-confident-unverified",
        response_text=("The current US federal funds rate target range is 4.75% to 5.00%."),
        tool_results_text="",
        expected_label="consistent",
    ),
    # No-tool + explicit "I cannot verify" disclaimer. The rubric's
    # "honest can't-verify" carve-out should kick in. The user
    # invited this shape with "If you can't find it, say so
    # explicitly". The response declining to assert anything must
    # NOT be flagged as a contradiction.
    M2Case(
        case_id="no-tools-explicit-cant-verify",
        response_text=(
            "I could not verify the holding of Zarephath Industries "
            "v. Brookfield Capital Trust (2023) with the available "
            "tools. I searched the web for the case name and related "
            "phrases, but I did not find a court opinion or other "
            "authoritative source that identified the holding. I am "
            "not asserting a holding for this case without source "
            "confirmation."
        ),
        tool_results_text="",
        expected_label="consistent",
    ),
)


# M2 consistency is a critic role. Defaults to Sonnet; rerun against
# the OpenAI provider via ``KAOS_TEST_CRITIC_MODEL=openai:gpt-5.4-mini``.
@pytest.mark.live
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.case_id)
async def test_m2_label_emission(case: M2Case) -> None:
    model = critic_model()
    if not _have_key_for(model):
        pytest.skip(f"missing API key for {model}")

    verdict = await judge_reasoning_action_consistency(
        response_text=case.response_text,
        model=model,
        tool_results_text=case.tool_results_text,
    )

    assert not verdict.fell_back, (
        f"{case.case_id} x {model}: fell back unexpectedly; reasoning={verdict.reasoning!r}"
    )
    assert verdict.label in M2_ALLOWED_LABELS, (
        f"{case.case_id} x {model}: emitted label outside allowlist: {verdict.label!r}"
    )
    assert verdict.label == case.expected_label, (
        f"{case.case_id} x {model}: expected {case.expected_label!r}, "
        f"got {verdict.label!r}. reasoning={verdict.reasoning!r}"
    )
    assert verdict.confidence >= case.min_confidence, (
        f"{case.case_id} x {model}: confidence {verdict.confidence} "
        f"below threshold {case.min_confidence}"
    )
    assert verdict.cost_usd >= 0.0
    assert verdict.latency_ms >= 0.0
