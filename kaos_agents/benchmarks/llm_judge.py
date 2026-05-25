"""LLM-as-judge scorer for benchmark Q&A correctness.

The fuzzy-hint matcher used by :mod:`tests/benchmarks/multiformat_e2e.py`
produces false negatives whenever the agent answers correctly but uses
different wording from the curated ``expected_answer_hint``. For a real
acceptance gate we want a judge that actually reads the question, the
expected behavior (refuse vs. answer), the optional gold hint, and the
agent's verbatim response, then renders a verdict.

Built on the typed kaos-llm-core stack
----------------------------------------
The judge is a :class:`~kaos_llm_core.programs.call.Call` over a typed
:class:`~kaos_llm_core.signatures.signature.Signature`. Validation,
codec selection, structured output, retry, trace, and cost accounting
all flow through ``Call._run_pipeline`` — same path every other LLM
call in the system uses.

Failure mode contract
---------------------
The judge is best-effort scoring infrastructure, NOT a hard gate.
``CallError`` (provider 5xx, auth), ``ValidationRetryExhaustedError``
(unparseable response), missing API keys, and any other exception all
return a :class:`JudgeVerdict` with ``correct=False, confidence=0.0,
reasoning="judge unavailable: <why>"``. The benchmark harness can fall
back to fuzzy matching without crashing the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature
from kaos_llm_core.programs.call import Call

from kaos_agents.types import InvocationUsage

logger = get_logger("benchmarks.llm_judge")


class QAJudgeSignature(Signature):
    """Score whether an agent's response to a Q&A benchmark question is correct.

    You are an impartial grader for a retrieval-augmented question-answering
    benchmark used by lawyers. Decide whether the agent's response is
    CORRECT given the question, whether the question should be answered
    or refused, and (if answerable) the gold answer hint.

    Decision rules:

    1. If ``expected_answerable`` is FALSE: the question is unanswerable
       from the corpus (the relevant document is present but the
       specific fact is absent, OR the topic is entirely outside the
       corpus). The agent is CORRECT only if it explicitly refuses,
       says it cannot find sufficient evidence, declines to answer, or
       otherwise communicates that the answer is not in the materials.
       The agent is WRONG if it provides a confident factual answer
       (even if that answer is true in the world — the agent must not
       pull from its training data).
    2. If ``expected_answerable`` is TRUE: the agent is CORRECT only if
       its response substantively conveys the same fact as
       ``expected_answer_hint``. Paraphrase, reordering, additional
       correct context, or extra citation language are all fine. A
       refusal is WRONG. A different fact is WRONG.
    3. Hedged answers count as refusals only if the agent does not
       commit to a specific factual claim. "I don't have enough
       evidence to say" is a refusal. "It is approximately X, but you
       should verify" is an answer.
    """

    question: str = InputField(description="The question that was asked.")
    expected_answerable: bool = InputField(
        description=(
            "True if the corpus is expected to contain the answer; "
            "False if the agent should refuse / say it cannot find "
            "sufficient evidence."
        )
    )
    expected_hint: str = InputField(
        description=(
            "For answerable questions, a short string describing the "
            "expected fact. Empty string for unanswerable questions."
        )
    )
    agent_answer: str = InputField(description="The agent's full response text (verbatim).")
    agent_refused: bool = InputField(
        description=(
            "True if the agent emitted an explicit "
            "insufficient-evidence signal (EvidenceInsufficient event)."
        )
    )

    correct: bool = OutputField(
        description=("Whether the agent's response is CORRECT under the decision rules above.")
    )
    confidence: float = OutputField(
        description="Verdict confidence in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = OutputField(
        description=(
            "One or two sentences explaining the verdict, naming the "
            "specific fact that did or did not appear."
        )
    )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Observability wrapper around a typed :class:`QAJudgeSignature` verdict.

    The verdict itself (``correct`` / ``confidence`` / ``reasoning``) is
    the typed Pydantic output produced by the judge :class:`Call`. The
    wrapper carries observability metadata (model, cost, tokens) for
    cost rollups in the benchmark JSON. Convenience properties delegate
    to ``verdict`` so callers can read the verdict flatly without
    walking the wrapper.
    """

    verdict: Any  # Pydantic instance — typed at the call site, dynamic here
    judge_model: str = ""
    judge_cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    @property
    def correct(self) -> bool:
        return bool(self.verdict.correct)

    @property
    def confidence(self) -> float:
        return float(self.verdict.confidence)

    @property
    def reasoning(self) -> str:
        return str(self.verdict.reasoning)

    def to_dict(self) -> dict[str, object]:
        """Flatten to the legacy harness shape."""
        return {
            "correct": self.correct,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "judge_model": self.judge_model,
            "judge_cost_usd": self.judge_cost_usd,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
        }


async def llm_judge(
    question: str,
    expected_hint: str | None,
    expected_answerable: bool,
    agent_answer: str,
    agent_refused: bool,
    *,
    model: str = "anthropic:claude-haiku-4-5",
) -> dict:
    """Score an agent's response with an LLM-as-judge call.

    Constructs a typed :class:`Call` over :class:`QAJudgeSignature` and
    invokes it. Errors map to a benign "judge unavailable" verdict so
    the benchmark fanout never aborts on transient infra failures.

    Parameters
    ----------
    question:
        The question that was asked.
    expected_hint:
        Short string describing the expected fact, or ``None`` for
        unanswerable questions (passed through as the empty string at
        the prompt boundary).
    expected_answerable:
        ``True`` if the corpus is expected to contain the answer.
    agent_answer:
        The agent's full response text (concatenated TextDelta stream).
    agent_refused:
        ``True`` if the agent emitted an explicit
        :class:`EvidenceInsufficient` signal.
    model:
        kaos-llm-client model identifier.

    Returns
    -------
    dict
        :meth:`JudgeVerdict.to_dict` payload. On failure, a benign
        verdict with ``correct=False, confidence=0.0, reasoning="judge
        unavailable: ..."``.
    """
    try:
        from kaos_agents._examples import load_examples

        judge_call = Call(
            QAJudgeSignature, model=model, max_retries=2, examples=load_examples("qa_judge")
        )
    except ImportError as exc:
        return _unavailable_verdict(
            f"judge unavailable: kaos-llm-client not installed ({exc})", model
        ).to_dict()

    try:
        invocation = await judge_call.invoke(
            question=question,
            expected_answerable=expected_answerable,
            expected_hint=expected_hint or "",
            agent_answer=agent_answer.strip() or "(empty response)",
            agent_refused=agent_refused,
        )
    except Exception as exc:  # broad: provider 5xx, auth, validation, etc.
        logger.warning("LLM judge failed (%s): %s", type(exc).__name__, exc)
        return _unavailable_verdict(
            f"judge unavailable: {type(exc).__name__}: {exc}", model
        ).to_dict()

    usage = InvocationUsage.from_invocation(invocation)
    verdict = JudgeVerdict(
        verdict=invocation.output,  # typed QAJudgeSignature output
        judge_model=model,
        judge_cost_usd=usage.cost_usd,
        judge_input_tokens=usage.input_tokens,
        judge_output_tokens=usage.output_tokens,
    )
    return verdict.to_dict()


def _unavailable_verdict(reason: str, judge_model: str) -> JudgeVerdict:
    """Build a 'judge unavailable' :class:`JudgeVerdict` with a synthetic verdict."""
    synthetic_output = QAJudgeSignature.model_construct(
        question="",
        expected_answerable=False,
        expected_hint="",
        agent_answer="",
        agent_refused=False,
        correct=False,
        confidence=0.0,
        reasoning=reason,
    )
    return JudgeVerdict(
        verdict=synthetic_output,
        judge_model=judge_model,
    )
