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

MODELS: tuple[str, ...] = (
    "openai:gpt-5.4-mini",
    "anthropic:claude-sonnet-4-6",
)


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
)


@pytest.mark.live
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.replace(":", "_"))
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.case_id)
async def test_m2_label_emission(case: M2Case, model: str) -> None:
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
