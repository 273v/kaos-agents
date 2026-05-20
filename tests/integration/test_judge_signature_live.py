"""Live evals for :func:`kaos_agents.planning.judge.judge_with_rubric`.

Plan §6: validate that one generic rubric-parametric JudgeSignature can
subsume the existing bespoke judges. Each case below corresponds to one
of the 5 existing judge classes — same shape, different rubric.

Required env vars: ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``.

Run with::

    uv run pytest tests/integration/test_judge_signature_live.py -v -m live
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from kaos_agents.planning.judge import JudgeVerdict, judge_with_rubric

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
class JudgeCase:
    case_id: str
    rubric: str
    input_text: str
    expected_label: str
    allowed_labels: tuple[str, ...]
    context: str = ""


_CASES: tuple[JudgeCase, ...] = (
    # 1. Subsumes RubricVerdictSignature — binary PASS / FAIL on a
    # deliverable criterion.
    JudgeCase(
        case_id="rubric-pass-fail-positive",
        rubric=(
            "Output one of: pass / fail. PASS iff the deliverable "
            "contains a numerical answer to '2 + 2'. FAIL otherwise."
        ),
        input_text="The answer is 4.",
        expected_label="pass",
        allowed_labels=("pass", "fail"),
    ),
    JudgeCase(
        case_id="rubric-pass-fail-negative",
        rubric=(
            "Output one of: pass / fail. PASS iff the deliverable "
            "contains a numerical answer to '2 + 2'. FAIL otherwise."
        ),
        input_text="I am not sure what you mean by '2 + 2'.",
        expected_label="fail",
        allowed_labels=("pass", "fail"),
    ),
    # 2. Subsumes EvalSemanticSignature — semantic match.
    JudgeCase(
        case_id="semantic-match-positive",
        rubric=(
            "Output one of: matched / unmatched. MATCHED iff the input "
            "text semantically addresses the expectation. UNMATCHED "
            "otherwise. Be strict — surface-level word overlap is not "
            "enough; the input must actually answer the expectation."
        ),
        input_text=(
            "42 USC 7412 establishes the list of hazardous air "
            "pollutants and requires EPA to set emission standards for "
            "major sources."
        ),
        expected_label="matched",
        allowed_labels=("matched", "unmatched"),
        context="Expectation: information about Clean Air Act emission standards.",
    ),
    # 3. Subsumes GoalCheckSignature — 3-way verdict.
    JudgeCase(
        case_id="goal-check-satisfied",
        rubric=(
            "Output one of: satisfied / needs_more_work / "
            "insufficient_evidence. SATISFIED iff the agent's response "
            "directly answers the user's question with grounded "
            "evidence (a tool call result or direct quote). "
            "NEEDS_MORE_WORK iff the answer is partial — the agent "
            "could continue with one more tool call. "
            "INSUFFICIENT_EVIDENCE iff the answer cannot be improved "
            "with more iteration because the underlying data isn't "
            "available."
        ),
        input_text=(
            "USER: who are Michigan's two US senators?\n"
            "AGENT: Michigan's senators are Gary Peters and Elissa "
            "Slotkin (verified via senate.gov/states/MI/senators.htm)."
        ),
        expected_label="satisfied",
        allowed_labels=("satisfied", "needs_more_work", "insufficient_evidence"),
    ),
    # 4. Subsumes EvaluateConditionSignature — predicate over data.
    JudgeCase(
        case_id="condition-predicate-holds",
        rubric=(
            "Output one of: holds / does_not_hold. HOLDS iff the input "
            "text contains evidence that the predicate '$1 million in "
            "revenue or more was reported' is true. DOES_NOT_HOLD "
            "otherwise."
        ),
        input_text=(
            "For fiscal year 2024 the company reported $4.2 million in "
            "total revenue, up from $3.1 million the prior year."
        ),
        expected_label="holds",
        allowed_labels=("holds", "does_not_hold"),
    ),
)


@pytest.mark.live
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.replace(":", "_"))
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.case_id)
async def test_judge_with_rubric_live(case: JudgeCase, model: str) -> None:
    """JudgeSignature emits the expected categorical label on each case.

    Asserts on:
    - ``label`` matches ``expected_label`` (case-insensitive)
    - ``fell_back`` is False (model produced a valid in-rubric label)
    - ``confidence`` is in [0, 1]
    - ``reasoning`` is non-empty
    """
    if not _have_key_for(model):
        pytest.skip(f"no API key for {model}")

    verdict: JudgeVerdict = await judge_with_rubric(
        rubric=case.rubric,
        input_text=case.input_text,
        context=case.context,
        model=model,
        allowed_labels=case.allowed_labels,
    )

    assert not verdict.fell_back, (
        f"[{case.case_id}/{model}] judge fell back — "
        f"label={verdict.label!r} reasoning={verdict.reasoning!r}"
    )
    assert verdict.label == case.expected_label, (
        f"[{case.case_id}/{model}] expected label={case.expected_label!r}, "
        f"got {verdict.label!r}. Reasoning: {verdict.reasoning!r}"
    )
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.reasoning, (
        f"[{case.case_id}/{model}] reasoning was empty — the judge "
        "must cite the criterion that drove its verdict."
    )
