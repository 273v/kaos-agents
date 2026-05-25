"""Live LLM evals pinning the planner + critic Signature docstrings.

These tests call real LLM APIs (Anthropic Claude Haiku 4.5 by default)
to verify that the structured decision-point Signatures —
:class:`_TurnToolPolicySignature` (planner) and
:class:`_GoalCheckerSignature` (critic) — make the right calls on the
failure scenarios that motivated the
``kaos-modules/docs/plans/thin-worker-prompt.md`` refactor.

Why these tests exist
---------------------

Between 2026-05-13 and 2026-05-16 the kaos-ui SPA worker prompt grew
to ~1,600 tokens of hardcoded English behavior rules. The rules were
duplicating decisions these two Signatures already encode in their
class docstrings — the regression slipped because there was no eval
catching the gap at the Signature layer. Every failed scenario
landed as a worker-prompt patch instead of a docstring tweak.

Each case below is sourced from a 2026-05-16 failing session. When
a future planner / critic docstring edit accidentally regresses the
behavior, the relevant case fails. When a NEW failure mode appears
in the wild, the right reaction is to add a case here, not to add
English to the worker.

Six cases total (3 planner + 3 critic). The pre-B13 wrong-year
hallucination is intentionally NOT in this suite — the critic
cannot fact-check a year without access to the real date, and the
right fix-site for that failure is the worker-side date preamble
(`kaos_ui.agents._date_preamble`), not a critic verdict.

Cost: ~$0.0012 per full run (6 cases at ~$0.0002 each on Haiku).

Run with::

    uv run pytest tests/integration/test_signature_evals_live.py -v

Requires :envvar:`ANTHROPIC_API_KEY` (or whatever provider the
``KAOS_AGENT_PLANNING_LLM_MODEL`` / ``KAOS_AGENT_DEFAULT_LLM_MODEL``
settings resolve to).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kaos_agents.planning.goal_check import (
    GoalCheckNeedsMoreWork,
    GoalCheckSatisfied,
    check_goal,
)
from kaos_agents.planning.policy import plan_turn_tool_policy
from tests.integration._models import critic_model

PLANNER_MODEL = critic_model()
CRITIC_MODEL = critic_model()


# ─── Planner eval cases ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PlannerCase:
    id: str
    user_message: str
    corpus_headlines: str
    corpus_kinds: list[str]
    session_intent: str | None
    raw_turn_groups: list[str] | None
    # The planner's `wanted_groups` (unconstrained by ceiling) must
    # be a SUPERSET of this set.
    expected_groups_subset: set[str]


PLANNER_CASES: list[PlannerCase] = [
    PlannerCase(
        id="lookup_cfpb_2026_no_corpus",
        user_message=("look up the latest CFPB enforcement action against a bank in 2026"),
        corpus_headlines="",
        corpus_kinds=[],
        session_intent="research",
        raw_turn_groups=None,
        # "look up" + "the latest" + a government source → web is
        # the docstring's group-selection shortcut.
        expected_groups_subset={"web"},
    ),
    PlannerCase(
        id="who_teaches_800_with_pdf",
        user_message="who teaches 800",
        corpus_headlines=("course-descriptions-v3-ug.pdf — 245321 bytes, application/pdf"),
        corpus_kinds=["pdf"],
        session_intent="research",
        raw_turn_groups=None,
        # `corpus_kinds` includes "pdf" and the question is plausibly
        # about a course description in the PDF → docs.
        expected_groups_subset={"documents"},
    ),
    PlannerCase(
        id="lookup_continuation_with_pdf",
        user_message="look up who that is",
        corpus_headlines=("course-descriptions-v3-ug.pdf — 245321 bytes, application/pdf"),
        corpus_kinds=["pdf"],
        session_intent="research",
        raw_turn_groups=["documents"],
        # The M2.1 lookup-beyond-corpus rule: a "look up X" against
        # attached docs should bring BOTH documents and web into scope
        # so the agent doesn't have to replan to escalate to web.
        expected_groups_subset={"documents", "web"},
    ),
]


@pytest.mark.live
@pytest.mark.parametrize("case", PLANNER_CASES, ids=lambda c: c.id)
async def test_planner_signature_picks_expected_groups(case: PlannerCase) -> None:
    """The planner Signature's docstring rules must produce a
    ``wanted_groups`` superset that includes the expected groups.

    Asserts on the unconstrained planner output (before
    intersection with the ceiling) so this test isolates the
    Signature's behavior from the ceiling policy.
    """
    wide_ceiling = [
        "web",
        "documents",
        "citations",
        "vfs",
        "retrieval",
        "authoring",
        "forensics",
        "browser",
        "netinfra",
    ]
    plan = await plan_turn_tool_policy(
        user_message=case.user_message,
        recent_turns="",
        corpus_headlines=case.corpus_headlines,
        corpus_kinds=case.corpus_kinds,
        session_intent=case.session_intent,
        raw_turn_groups=case.raw_turn_groups,
        # Wide ceiling so the assertion isolates Signature behavior
        # from the ceiling filter. `wanted_groups` is reconstructed
        # below as `kept_groups | dropped_groups`.
        ceiling_groups=wide_ceiling,
        available_groups=wide_ceiling,
        model=PLANNER_MODEL,
    )
    # `wanted_groups` is the Signature's unconstrained output;
    # `plan_turn_tool_policy` splits it into `kept_groups` (∩ ceiling)
    # and `dropped_groups` (- ceiling). With a wide ceiling those
    # union back to wanted.
    wanted = set(plan.kept_groups) | set(plan.dropped_groups)
    missing = case.expected_groups_subset - wanted
    assert not missing, (
        f"Planner [{case.id}] picked wanted_groups={sorted(wanted)} "
        f"but expected a superset of {sorted(case.expected_groups_subset)}. "
        f"Missing: {sorted(missing)}. "
        f"Rationale: {plan.rationale!r}. "
        "If the docstring should make this case obvious, edit it in "
        "`kaos_agents/planning/policy.py:_TurnToolPolicySignature` and "
        "re-run; do NOT add English to a worker prompt to compensate."
    )


# ─── Critic eval cases ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CriticCase:
    id: str
    user_message: str
    agent_response: str
    tool_calls_made: list[dict]
    elevation_trail: list[str]
    available_groups: list[str]
    iteration: int
    expected_kind: str  # "satisfied" | "needs_more_work" | "insufficient_evidence"
    # When expected_kind is "needs_more_work", the next_action must
    # contain at least one of these keywords (case-insensitive).
    expected_next_action_keywords: tuple[str, ...] = ()


CRITIC_CASES: list[CriticCase] = [
    CriticCase(
        id="hallucinated_person_no_tools",
        user_message="look up who that is",
        agent_response=(
            "J. Bommarito is the lead instructor for the proposed ACC/ITM 2XX "
            "(Continuing Practice in Accounting) special-topics course."
        ),
        tool_calls_made=[],
        elevation_trail=[],
        available_groups=["web", "documents", "citations", "vfs"],
        iteration=1,
        expected_kind="needs_more_work",
        expected_next_action_keywords=("search", "web"),
    ),
    CriticCase(
        id="clarification_when_docs_attached",
        user_message="who teaches 800",
        agent_response=(
            "I need more specific information to answer your question. "
            "The document I have doesn't contain a course numbered exactly '800'."
        ),
        tool_calls_made=[],
        elevation_trail=[],
        available_groups=["documents", "web"],
        iteration=1,
        expected_kind="needs_more_work",
        expected_next_action_keywords=("search", "document"),
    ),
    CriticCase(
        id="grounded_answer_with_tool_call",
        user_message="search the federal register for the latest dairy rule",
        agent_response=(
            "The most recent FR rule on dairy is 90 FR 12345 (2026-05-10), published by USDA AMS."
        ),
        tool_calls_made=[
            {
                "name": "kaos-source-fr-search",
                "is_error": False,
                "summary_excerpt": "1 hit: 90 FR 12345",
            }
        ],
        elevation_trail=[],
        available_groups=["web"],
        iteration=1,
        expected_kind="satisfied",
    ),
]


@pytest.mark.live
@pytest.mark.parametrize("case", CRITIC_CASES, ids=lambda c: c.id)
async def test_critic_signature_emits_expected_verdict(case: CriticCase) -> None:
    """The critic Signature's docstring rules must produce the right
    three-way verdict for each of these scenarios.

    Each scenario is sourced from a 2026-05-16 failing session — the
    cases where the worker hallucinated facts (no tool call) or
    requested clarification while tools were available. The critic
    catches those at the verdict layer; the AgenticLoop then replans
    with the critic's ``next_action`` threaded as ``thinking_note``
    to the next iteration.
    """
    outcome = await check_goal(
        user_message=case.user_message,
        agent_response=case.agent_response,
        tool_calls_made=list(case.tool_calls_made),
        elevation_trail=list(case.elevation_trail),
        available_groups=list(case.available_groups),
        iteration=case.iteration,
        model=CRITIC_MODEL,
    )
    result = outcome.result
    actual_kind = result.kind
    assert actual_kind == case.expected_kind, (
        f"Critic [{case.id}] returned kind={actual_kind!r} "
        f"but expected {case.expected_kind!r}. "
        f"Rationale: {getattr(result, 'rationale', '?')!r}. "
        "If the docstring should make this case obvious, edit it in "
        "`kaos_agents/planning/goal_check.py:_GoalCheckerSignature` "
        "and re-run; do NOT add English to a worker prompt to "
        "compensate."
    )
    if case.expected_next_action_keywords and isinstance(result, GoalCheckNeedsMoreWork):
        next_action = result.next_action.lower()
        matched = [k for k in case.expected_next_action_keywords if k.lower() in next_action]
        assert matched, (
            f"Critic [{case.id}] next_action={result.next_action!r} "
            f"did not contain any of {case.expected_next_action_keywords!r}. "
            "The next_action is what AgenticLoop threads to the next "
            "iteration as `thinking_note` — it must name the action "
            "the worker should take."
        )
    # `satisfied` verdicts on grounded answers are an architectural
    # signal: the critic trusts the tool call as evidence. Assert the
    # result object is the right shape.
    if case.expected_kind == "satisfied":
        assert isinstance(result, GoalCheckSatisfied), (
            f"Critic [{case.id}] returned kind=satisfied but result "
            f"object is {type(result).__name__}, not GoalCheckSatisfied."
        )
