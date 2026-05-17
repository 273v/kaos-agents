"""Tests for :class:`kaos_agents.intent.extractor.IntentExtractor`.

The inner kaos-llm-core Call is stubbed so no live LLM call is issued.
We construct a synthetic :class:`Invocation` whose ``output`` is an
``IntentSignature`` instance and patch ``IntentExtractor._call.invoke``
to return it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from kaos_llm_core.programs._invocation import Invocation, TokenUsage
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.chain_of_thought import ChainOfThought

from kaos_agents.config import AgentPattern
from kaos_agents.intent.extractor import IntentExtractor, _project_signature_output
from kaos_agents.intent.signature import IntentSignature
from kaos_agents.intent.types import (
    Ambiguity,
    AmbiguityKind,
    Constraint,
    ConstraintKind,
    Goal,
    IntentResult,
)
from kaos_agents.types.intents import IntentType


def _make_signature(**overrides: Any) -> IntentSignature:
    """Build an IntentSignature with sensible defaults; overrides win."""
    fields: dict[str, Any] = {
        "message": overrides.pop("message", "test message"),
        "recent_messages": overrides.pop("recent_messages", ""),
        "domain_examples": overrides.pop("domain_examples", ""),
        "goal": overrides.pop(
            "goal",
            Goal(statement="Do X.", intent_type=IntentType.RESPOND),
        ),
    }
    fields.update(overrides)
    return IntentSignature(**fields)


def _make_invocation(output: Any) -> Invocation:
    return Invocation(
        client=None,
        model="anthropic:claude-haiku-4-5",
        context=None,
        output=output,
        trace=None,
        usage=TokenUsage(),
    )


def _stub_call_with(extractor: IntentExtractor, output: Any) -> AsyncMock:
    """Replace the extractor's inner Call.invoke with an AsyncMock that
    returns an Invocation wrapping ``output``. Returns the mock so tests
    can assert on the kwargs the extractor passed."""
    invocation = _make_invocation(output)
    mock = AsyncMock(return_value=invocation)
    # ``_call`` is private but it's the documented stub seam for tests.
    # We replace the bound method with an AsyncMock whose return value
    # is a real Invocation; ty cannot prove the override is sound, so
    # the inline suppression is intentional.
    extractor._call.invoke = mock  # ty: ignore[invalid-assignment]
    return mock


class TestIntentExtractorConstruction:
    def test_default_construction(self):
        ex = IntentExtractor()
        assert ex.chain_of_thought is False
        assert isinstance(ex._call, Call)
        # Bare Call (not ChainOfThought).
        assert not isinstance(ex._call, ChainOfThought)
        # Inner Call wraps IntentSignature.
        assert ex._call.signature is IntentSignature

    def test_explicit_model(self):
        ex = IntentExtractor(model="anthropic:claude-sonnet-4-6")
        assert ex.model == "anthropic:claude-sonnet-4-6"
        # Inner call's resolved model preference matches.
        assert ex._call._model == "anthropic:claude-sonnet-4-6"

    def test_chain_of_thought_wraps_call(self):
        ex = IntentExtractor(chain_of_thought=True)
        assert ex.chain_of_thought is True
        # ChainOfThought is itself a Call subclass.
        assert isinstance(ex._call, ChainOfThought)

    def test_max_retries_threaded(self):
        ex = IntentExtractor(max_retries=3)
        assert ex._call._max_retries == 3

    def test_instructions_override(self):
        custom = "Custom intent instructions for tuned domain."
        ex = IntentExtractor(instructions_override=custom)
        assert ex._call.instructions == custom

    def test_instructions_default_to_signature_docstring(self):
        ex = IntentExtractor()
        # Without an override the Call lifts the IntentSignature docstring.
        assert "intent classifier" in ex._call.instructions.lower()


class TestIntentExtractorForward:
    @pytest.mark.asyncio
    async def test_forward_happy_path_chat(self):
        ex = IntentExtractor()
        sig_out = _make_signature(
            message="hi there",
            goal=Goal(statement="Greeting.", intent_type=IntentType.RESPOND),
            requires_clarification=False,
            pattern=AgentPattern.CHAT,
            confidence=0.95,
        )
        _stub_call_with(ex, sig_out)
        result = await ex.forward(message="hi there")
        assert isinstance(result, IntentResult)
        assert result.goal.intent_type is IntentType.RESPOND
        assert result.pattern is AgentPattern.CHAT
        assert result.confidence == pytest.approx(0.95)
        assert result.requires_clarification is False
        assert result.raw_input == "hi there"

    @pytest.mark.asyncio
    async def test_forward_passes_inputs_to_call(self):
        ex = IntentExtractor()
        sig_out = _make_signature()
        mock = _stub_call_with(ex, sig_out)
        await ex.forward(
            message="extract dates",
            recent_messages="prior message 1\nprior message 2",
            domain_examples="ex1\nex2",
        )
        mock.assert_awaited_once()
        await_args = mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["message"] == "extract dates"
        assert kwargs["recent_messages"] == "prior message 1\nprior message 2"
        assert kwargs["domain_examples"] == "ex1\nex2"

    @pytest.mark.asyncio
    async def test_forward_pattern_plan(self):
        ex = IntentExtractor()
        sig_out = _make_signature(
            goal=Goal(
                statement="First do A, then do B.",
                intent_type=IntentType.PLAN,
            ),
            pattern=AgentPattern.PLAN,
            confidence=0.7,
        )
        _stub_call_with(ex, sig_out)
        result = await ex.forward(message="first do A, then do B")
        assert result.pattern is AgentPattern.PLAN
        assert result.goal.intent_type is IntentType.PLAN

    @pytest.mark.asyncio
    async def test_forward_pattern_research(self):
        ex = IntentExtractor()
        sig_out = _make_signature(
            goal=Goal(
                statement="What does the corpus say about X?",
                intent_type=IntentType.RESEARCH,
            ),
            pattern=AgentPattern.RESEARCH,
            confidence=0.88,
        )
        _stub_call_with(ex, sig_out)
        result = await ex.forward(message="what does the corpus say about X?")
        assert result.pattern is AgentPattern.RESEARCH
        assert result.goal.intent_type is IntentType.RESEARCH

    @pytest.mark.asyncio
    async def test_forward_propagates_constraints_and_ambiguities(self):
        ex = IntentExtractor()
        constraints = (
            Constraint(kind=ConstraintKind.DEADLINE, value="by Friday"),
            Constraint(kind=ConstraintKind.FORMAT, value="json"),
        )
        ambiguities = (
            Ambiguity(
                kind=AmbiguityKind.UNKNOWN_REFERENCE,
                span=(0, 3),
                excerpt="the",
                preferred_clarification="Which contract?",
            ),
        )
        sig_out = _make_signature(
            goal=Goal(statement="Summarize the contract.", intent_type=IntentType.RESEARCH),
            constraints=constraints,
            ambiguities=ambiguities,
            requires_clarification=False,
            pattern=AgentPattern.RESEARCH,
            confidence=0.6,
        )
        _stub_call_with(ex, sig_out)
        result = await ex.forward(message="summarize the contract by Friday in json")
        assert result.constraints == constraints
        assert result.ambiguities == ambiguities

    @pytest.mark.asyncio
    async def test_forward_requires_clarification(self):
        ex = IntentExtractor()
        sig_out = _make_signature(
            goal=Goal(statement="Unclear.", intent_type=IntentType.CLARIFY),
            ambiguities=(
                Ambiguity(
                    kind=AmbiguityKind.MISSING_CONTEXT,
                    span=(0, 0),
                    excerpt="",
                    preferred_clarification="What do you want me to do?",
                ),
            ),
            requires_clarification=True,
            confidence=0.2,
        )
        _stub_call_with(ex, sig_out)
        result = await ex.forward(message="???")
        assert result.requires_clarification is True
        assert result.has_blocking_ambiguity() is True

    @pytest.mark.asyncio
    async def test_confidence_clamped_above(self):
        # An LLM that emits 1.05 should land at 1.0 — the IntentResult
        # validator would otherwise reject it. We clamp instead.
        ex = IntentExtractor()
        sig = _make_signature()

        # Manually construct an output with out-of-range confidence by
        # bypassing the IntentSignature validator (use a plain object).
        class _LooseOutput:
            def __init__(self) -> None:
                self.goal = sig.goal
                self.constraints: tuple = ()
                self.ambiguities: tuple = ()
                self.requires_clarification = False
                self.pattern = AgentPattern.CHAT
                self.confidence = 1.05

        _stub_call_with(ex, _LooseOutput())
        result = await ex.forward(message="x")
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_clamped_below(self):
        ex = IntentExtractor()
        sig = _make_signature()

        class _LooseOutput:
            def __init__(self) -> None:
                self.goal = sig.goal
                self.constraints: tuple = ()
                self.ambiguities: tuple = ()
                self.requires_clarification = False
                self.pattern = AgentPattern.CHAT
                self.confidence = -0.4

        _stub_call_with(ex, _LooseOutput())
        result = await ex.forward(message="x")
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_invoke_returns_invocation_with_output(self):
        # Program.invoke wraps forward() in a trace collector and returns
        # a kaos-llm-core Invocation — verify the wiring works end-to-end.
        ex = IntentExtractor()
        sig_out = _make_signature(
            goal=Goal(statement="Hi.", intent_type=IntentType.RESPOND),
            confidence=1.0,
        )
        _stub_call_with(ex, sig_out)
        invocation = await ex.invoke(message="hi")
        assert isinstance(invocation.output, IntentResult)
        assert invocation.output.raw_input == "hi"


class TestIntentExtractorCorpusAttached:
    """Verify ``corpus_attached`` / ``corpus_size`` thread cleanly into
    the inner Call.invoke kwargs.

    The projection layer is unaffected — these are pure inputs to the
    IntentSignature (rule 6) and do not appear on the IntentResult. The
    LLM is responsible for honoring rule 6 ("indirect document
    references with corpus attached → RESEARCH"); these tests only
    verify the wiring, not the model behavior. Live tests in
    tests/integration/intent/ cover the model behavior with real LLM
    calls.
    """

    @pytest.mark.asyncio
    async def test_defaults_when_not_passed(self):
        ex = IntentExtractor()
        sig_out = _make_signature()
        mock = _stub_call_with(ex, sig_out)
        await ex.forward(message="hello")
        await_args = mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        # Defaults preserve pre-existing behavior for sessions with no
        # attached corpus.
        assert kwargs["corpus_attached"] is False
        assert kwargs["corpus_size"] == 0

    @pytest.mark.asyncio
    async def test_passes_corpus_attached_true_with_size(self):
        ex = IntentExtractor()
        sig_out = _make_signature()
        mock = _stub_call_with(ex, sig_out)
        await ex.forward(message="summarize that", corpus_attached=True, corpus_size=3)
        await_args = mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["corpus_attached"] is True
        assert kwargs["corpus_size"] == 3

    @pytest.mark.asyncio
    async def test_corpus_inputs_coerced_to_bool_and_int(self):
        # Defensive coercion — callers might hand back numpy ints or
        # truthy/falsy non-bool sentinels; forward() must normalise so
        # the Signature validator never sees an unexpected type.
        ex = IntentExtractor()
        sig_out = _make_signature()
        mock = _stub_call_with(ex, sig_out)
        await ex.forward(message="x", corpus_attached=1, corpus_size="5")
        await_args = mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["corpus_attached"] is True
        assert isinstance(kwargs["corpus_attached"], bool)
        assert kwargs["corpus_size"] == 5
        assert isinstance(kwargs["corpus_size"], int)


class TestIntentSignatureCorpusInputs:
    """The Signature itself must accept the new inputs with defaults."""

    def test_defaults_when_omitted(self):
        sig = _make_signature()
        assert sig.corpus_attached is False
        assert sig.corpus_size == 0

    def test_accepts_corpus_attached_true(self):
        sig = _make_signature(corpus_attached=True, corpus_size=12)
        assert sig.corpus_attached is True
        assert sig.corpus_size == 12

    def test_corpus_size_must_be_non_negative(self):
        # ``ge=0`` on the InputField — a negative count is a programming
        # error, surface it rather than silently coercing.
        with pytest.raises(Exception):  # noqa: B017 (pydantic ValidationError)
            _make_signature(corpus_size=-1)


class TestProjectSignatureOutput:
    """Direct tests for the projection helper used by IntentExtractor.forward."""

    def test_basic_projection(self):
        sig = _make_signature(
            message="x",
            goal=Goal(statement="x", intent_type=IntentType.RESPOND),
            confidence=0.5,
        )
        result = _project_signature_output(sig, raw_input="x")
        assert isinstance(result, IntentResult)
        assert result.raw_input == "x"
        assert result.confidence == 0.5

    def test_pattern_string_coercion(self):
        # Some loose stubs hand back the StrEnum's string form rather
        # than the enum itself; the projector must coerce.
        class _LooseOutput:
            def __init__(self) -> None:
                self.goal = Goal(statement="x", intent_type=IntentType.RESPOND)
                self.constraints: tuple = ()
                self.ambiguities: tuple = ()
                self.requires_clarification = False
                self.pattern = "research"  # string, not the enum
                self.confidence = 0.9

        result = _project_signature_output(_LooseOutput(), raw_input="x")
        assert result.pattern is AgentPattern.RESEARCH
