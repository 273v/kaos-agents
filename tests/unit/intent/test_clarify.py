"""Tests for :class:`kaos_agents.intent.clarify.ClarificationLoop`.

Phase 1.A behavior is deterministic — no LLM round-trip — so these are
pure value-type tests around question synthesis from
:class:`Ambiguity` lists.
"""

from __future__ import annotations

import pytest

from kaos_agents.intent.clarify import ClarificationLoop
from kaos_agents.intent.types import (
    Ambiguity,
    AmbiguityKind,
    Goal,
    IntentResult,
)
from kaos_agents.types.intents import IntentType


def _intent(*, requires: bool = False, ambiguities: tuple[Ambiguity, ...] = ()) -> IntentResult:
    """Build an IntentResult with a stub goal."""
    return IntentResult(
        goal=Goal(statement="x", intent_type=IntentType.RESPOND),
        ambiguities=ambiguities,
        requires_clarification=requires,
    )


class TestClarificationLoopPassThrough:
    @pytest.mark.asyncio
    async def test_no_clarification_returns_empty_question_and_same_intent(self):
        loop = ClarificationLoop()
        intent = _intent(requires=False)
        question, returned = await loop.forward(original_message="hi", intent=intent)
        assert question == ""
        # Phase 1.A returns the input intent unchanged.
        assert returned is intent

    @pytest.mark.asyncio
    async def test_no_clarification_with_ambiguities_still_passes_through(self):
        # has_blocking_ambiguity=True does NOT itself trigger clarification
        # — only the explicit ``requires_clarification`` flag does.
        loop = ClarificationLoop()
        intent = _intent(
            requires=False,
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.MISSING_CONTEXT,
                    span=(0, 0),
                    excerpt="",
                ),
            ),
        )
        question, returned = await loop.forward(original_message="hi", intent=intent)
        assert question == ""
        assert returned is intent


class TestClarificationLoopQuestionSynthesis:
    @pytest.mark.asyncio
    async def test_uses_preferred_clarification_when_present(self):
        loop = ClarificationLoop()
        amb = Ambiguity(
            kind=AmbiguityKind.UNKNOWN_REFERENCE,
            span=(4, 16),
            excerpt="the contract",
            preferred_clarification="Which contract are you referring to?",
        )
        intent = _intent(requires=True, ambiguities=(amb,))
        question, _ = await loop.forward(original_message="summarize the contract", intent=intent)
        assert question == "Which contract are you referring to?"

    @pytest.mark.asyncio
    async def test_unknown_reference_template(self):
        loop = ClarificationLoop()
        amb = Ambiguity(
            kind=AmbiguityKind.UNKNOWN_REFERENCE,
            span=(4, 16),
            excerpt="the contract",
        )
        intent = _intent(requires=True, ambiguities=(amb,))
        question, _ = await loop.forward(original_message="summarize the contract", intent=intent)
        assert "the contract" in question
        assert question.endswith("?")

    @pytest.mark.asyncio
    async def test_ambiguous_pronoun_template(self):
        loop = ClarificationLoop()
        amb = Ambiguity(
            kind=AmbiguityKind.AMBIGUOUS_PRONOUN,
            span=(0, 4),
            excerpt="they",
        )
        intent = _intent(requires=True, ambiguities=(amb,))
        question, _ = await loop.forward(original_message="they signed it", intent=intent)
        assert "they" in question
        assert "refer" in question.lower()

    @pytest.mark.asyncio
    async def test_conflicting_requirements_template(self):
        loop = ClarificationLoop()
        amb = Ambiguity(
            kind=AmbiguityKind.CONFLICTING_REQUIREMENTS,
            span=(0, 18),
            excerpt="cheap and the best",
        )
        intent = _intent(requires=True, ambiguities=(amb,))
        question, _ = await loop.forward(original_message="cheap and the best", intent=intent)
        assert "prioritize" in question.lower()

    @pytest.mark.asyncio
    async def test_missing_context_template(self):
        loop = ClarificationLoop()
        amb = Ambiguity(
            kind=AmbiguityKind.MISSING_CONTEXT,
            span=(0, 0),
            excerpt="",
        )
        intent = _intent(requires=True, ambiguities=(amb,))
        question, _ = await loop.forward(original_message="???", intent=intent)
        assert "context" in question.lower() or "work with" in question.lower()

    @pytest.mark.asyncio
    async def test_out_of_domain_template(self):
        loop = ClarificationLoop()
        amb = Ambiguity(
            kind=AmbiguityKind.OUT_OF_DOMAIN,
            span=(0, 5),
            excerpt="paint",
        )
        intent = _intent(requires=True, ambiguities=(amb,))
        question, _ = await loop.forward(original_message="paint the kitchen", intent=intent)
        assert "paint" in question or "outside" in question.lower() or "reframe" in question.lower()

    @pytest.mark.asyncio
    async def test_blocking_ambiguity_wins_over_non_blocking(self):
        # MISSING_CONTEXT should win over UNKNOWN_REFERENCE in the
        # selection priority, regardless of declaration order.
        loop = ClarificationLoop()
        non_blocking = Ambiguity(
            kind=AmbiguityKind.UNKNOWN_REFERENCE,
            span=(0, 3),
            excerpt="the",
            preferred_clarification="Which one?",
        )
        blocking = Ambiguity(
            kind=AmbiguityKind.MISSING_CONTEXT,
            span=(10, 20),
            excerpt="this matter",
            preferred_clarification="Which matter ID?",
        )
        intent = _intent(requires=True, ambiguities=(non_blocking, blocking))
        question, _ = await loop.forward(original_message="...", intent=intent)
        assert question == "Which matter ID?"

    @pytest.mark.asyncio
    async def test_out_of_domain_picked_after_missing_context_absent(self):
        loop = ClarificationLoop()
        non_blocking = Ambiguity(
            kind=AmbiguityKind.UNKNOWN_REFERENCE,
            span=(0, 3),
            excerpt="the",
            preferred_clarification="Which one?",
        )
        out_of_domain = Ambiguity(
            kind=AmbiguityKind.OUT_OF_DOMAIN,
            span=(10, 20),
            excerpt="paint",
            preferred_clarification="I cannot help with painting.",
        )
        intent = _intent(requires=True, ambiguities=(non_blocking, out_of_domain))
        question, _ = await loop.forward(original_message="...", intent=intent)
        assert question == "I cannot help with painting."

    @pytest.mark.asyncio
    async def test_clarification_required_with_no_ambiguities_falls_back(self):
        # Degenerate case: requires_clarification=True but the LLM emitted
        # an empty ambiguities tuple. We still produce *some* question.
        loop = ClarificationLoop()
        intent = _intent(requires=True, ambiguities=())
        question, _ = await loop.forward(original_message="???", intent=intent)
        assert question != ""
        assert question.endswith("?")

    @pytest.mark.asyncio
    async def test_no_preferred_clarification_with_empty_excerpt(self):
        # When neither preferred_clarification nor excerpt are populated,
        # fall back to a generic kind-specific question.
        loop = ClarificationLoop()
        amb = Ambiguity(
            kind=AmbiguityKind.UNKNOWN_REFERENCE,
            span=(0, 0),
            excerpt="",
        )
        intent = _intent(requires=True, ambiguities=(amb,))
        question, _ = await loop.forward(original_message="???", intent=intent)
        assert question == "Which one do you mean?"
