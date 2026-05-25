"""Live evaluation of the GoalCheck (M4 verify-before-answer) critic.

Captures the failure modes that escaped onto the kaos-ui SPA on
2026-05-24 (sessions ``01KSDNYE2AMZ7CHCFJ5AEY34JP`` turn 2 and
``01KSDP8F73RWDFX8CEJM4M4X1W`` turn 5): GoalCheck called
``needs_more_work`` on legitimate no-tool answers — once on a
follow-up recall whose facts came from the prior turn's tool
results, once on pure-reasoning FRCP date arithmetic. Both
verdicts forced the AgenticLoop to burn iteration budget and
ultimately replace the worker's correct draft with a refusal
footer in the UI.

The existing ``tests/unit/test_goal_check.py`` is purely mocked
(verdicts forced via fake critics) and therefore couldn't catch
either rubric over-fire. This file is the live counterpart, modeled
on ``test_m2_consistency_live.py``: a case table x two frontier
models. Each case sends a real ``check_goal`` invocation against the
real provider and asserts the verdict kind matches expectation.

Required env vars: ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``.

Run with::

    uv run pytest tests/integration/test_goal_check_live.py -v -m live
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

from kaos_agents.planning.goal_check import check_goal
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
class GoalCheckCase:
    """One live GoalCheck regression case.

    ``expected_kind`` is the discriminated-union ``kind`` field —
    ``"satisfied"`` / ``"needs_more_work"`` / ``"insufficient_evidence"``.
    """

    case_id: str
    user_message: str
    agent_response: str
    tool_calls_made: list[dict] = field(default_factory=list)
    elevation_trail: list[str] = field(default_factory=list)
    available_groups: list[str] = field(default_factory=lambda: ["web", "source"])
    iteration: int = 1
    expected_kind: str = "satisfied"


# Each case is a discrete contract on the critic's rubric. Adding /
# removing cases here is the canonical way to pin a new failure mode
# the critic must respect before merging.
_CASES: tuple[GoalCheckCase, ...] = (
    # Baseline: worker answered with tool calls. Critic should
    # recognize the grounded shape and return ``satisfied``.
    GoalCheckCase(
        case_id="satisfied-fresh-with-tools",
        user_message="What is the current US federal funds rate? Cite the FOMC source.",
        agent_response=(
            "As of the FOMC statement issued April 29, 2026, the Committee "
            "said it would maintain the target range for the federal funds "
            "rate at 3-1/2 to 3-3/4 percent. Source: Federal Reserve, "
            "“Federal Reserve issues FOMC statement,” "
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            "monetary20260429a.htm."
        ),
        tool_calls_made=[
            {
                "name": "kaos-web-search",
                "is_error": False,
                "summary_excerpt": "FOMC April 29 2026 press release",
            },
            {
                "name": "kaos-web-fetch-page",
                "is_error": False,
                "summary_excerpt": (
                    "target range for the federal funds rate at 3-1/2 to 3-3/4 percent"
                ),
            },
        ],
        expected_kind="satisfied",
    ),
    # The critic SHOULD catch this: fresh factual claim about
    # current state of the world, zero tool calls, no prior-turn
    # context — confident-hallucination pattern. This is the
    # rubric's primary positive case.
    GoalCheckCase(
        case_id="needs-more-work-fresh-no-tools-hallucination",
        user_message="What is the current US federal funds rate?",
        agent_response=("The current US federal funds rate target range is 4.75% to 5.00%."),
        tool_calls_made=[],
        expected_kind="needs_more_work",
    ),
    # SPA 2026-05-24 session 01KSDP8F73RWDFX8CEJM4M4X1W turn 5.
    # The user asked for pure-reasoning FRCP date arithmetic. The
    # worker correctly applied Rule 12(a)(1)(A)(i) (21 days) + Rule
    # 6(a)(1)(C) (weekend rollover). The rubric's own narrative
    # carve-out (line 335-338) names arithmetic as a NOT-
    # hallucination case. Critic should return ``satisfied``;
    # currently it returns ``needs_more_work`` and forces the loop
    # to spend the iteration budget re-fetching FRCP text that
    # adds nothing.
    GoalCheckCase(
        case_id="satisfied-pure-arithmetic-no-tools",
        user_message=(
            "If a complaint was filed on March 15, 2025, and the FRCP "
            "Rule 12(a)(1)(A)(i) gives 21 days to respond, what is the "
            "response deadline (assuming no service waiver)? Show the "
            "date arithmetic."
        ),
        agent_response=(
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
        tool_calls_made=[],
        expected_kind="satisfied",
    ),
    # Per rubric line 335-338's carve-out, pure-definition
    # requests for a stable concept do not need a tool call. The
    # critic should return ``satisfied``.
    GoalCheckCase(
        case_id="satisfied-definition-no-tools",
        user_message="What is JSON Schema?",
        agent_response=(
            "JSON Schema is a declarative language for annotating and "
            "validating JSON document structure. A schema describes "
            "what shape a piece of JSON should have — required fields, "
            "allowed types, value constraints — and validators check "
            "instances against it. The current draft is 2020-12."
        ),
        tool_calls_made=[],
        expected_kind="satisfied",
    ),
    # SPA 2026-05-24 session 01KSDNYE2AMZ7CHCFJ5AEY34JP turn 2.
    # Follow-up referencing the prior turn's tool result. The
    # user's anaphoric phrasing ("the target range you just cited")
    # makes the question only-grounded-by-prior-turn-context.
    # ``tool_calls_made`` is empty THIS turn because the fact was
    # already fetched last turn and is in MESSAGES context. The
    # critic should recognize this shape and return ``satisfied`` —
    # currently it cannot, because GoalCheck has no input field
    # carrying prior-turn evidence and the rubric has no anaphoric-
    # follow-up carve-out. This case is the load-bearing one for
    # the SPA refusal pattern.
    GoalCheckCase(
        case_id="satisfied-follow-up-recall-no-tools",
        user_message=(
            "In one sentence: what was the target range you just "
            "cited, and what was the date of that FOMC statement?"
        ),
        agent_response=(
            "The target range was 3-1/2 to 3-3/4 percent, and the "
            "FOMC statement was dated April 29, 2026. Source: "
            "Federal Reserve, “Federal Reserve issues FOMC "
            "statement,” "
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            "monetary20260429a.htm."
        ),
        tool_calls_made=[],
        # iteration=2 is honest about the loop position; the prior
        # turn fetched the source.
        iteration=2,
        expected_kind="satisfied",
    ),
    # Deterministic pre-check should fire on this shape without
    # an LLM call: the response is a deliverable heading with no
    # body. Asserting this case exercises both the regex pre-check
    # AND the long-tail rubric. Expected: ``needs_more_work``.
    GoalCheckCase(
        case_id="needs-more-work-deliverable-header-then-stop",
        user_message=(
            "Compare these five NDAs and produce a CSV-ready table of governing-law clauses."
        ),
        agent_response=("## CSV-ready table\n\n"),
        tool_calls_made=[
            {
                "name": "kaos-office-parse-docx",
                "is_error": False,
                "summary_excerpt": "NDA #1 parsed",
            },
        ],
        expected_kind="needs_more_work",
    ),
)


# GoalCheck is the canonical critic role — see
# ``tests/integration/_models.py``. Defaults to Sonnet; operators can
# rerun against the OpenAI provider via
# ``KAOS_TEST_CRITIC_MODEL=openai:gpt-5.4-mini``.
@pytest.mark.live
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.case_id)
async def test_goal_check_label_emission(
    case: GoalCheckCase,
) -> None:
    """Hard-gate: GoalCheck verdict KIND matches the case's expectation.

    Failure mode this protects against: rubric over-fires
    ``needs_more_work`` on a no-tool answer that is legitimately
    grounded (follow-up recall, arithmetic, definition) — which is
    the exact pattern that produced the SPA refusal footers users
    saw on 2026-05-24.
    """
    model = critic_model()
    if not _have_key_for(model):
        pytest.skip(f"missing API key for {model}")

    outcome = await check_goal(
        user_message=case.user_message,
        agent_response=case.agent_response,
        tool_calls_made=case.tool_calls_made,
        elevation_trail=case.elevation_trail,
        available_groups=case.available_groups,
        iteration=case.iteration,
        model=model,
    )

    rationale = getattr(outcome.result, "rationale", "") or ""

    assert outcome.kind == case.expected_kind, (
        f"{case.case_id} x {model}: expected kind={case.expected_kind!r}, "
        f"got {outcome.kind!r}. rationale={rationale!r} "
        f"cost=${outcome.cost_usd:.4f} latency_ms={outcome.latency_ms:.0f}"
    )
    assert outcome.cost_usd >= 0.0
    assert outcome.latency_ms >= 0.0
