"""LLM-as-judge scorer for benchmark Q&A correctness.

The fuzzy-hint matcher used by :mod:`tests/benchmarks/multiformat_e2e.py`
produces false negatives whenever the agent answers correctly but uses
different wording from the curated ``expected_answer_hint``. For a real
acceptance gate we want a judge that actually reads the question, the
expected behavior (refuse vs. answer), the optional gold hint, and the
agent's verbatim response, then renders a verdict.

This module wraps a single LLM call into a small, well-typed function.
The judge is a cheap model (``anthropic:claude-haiku-4-5`` by default)
because each judgment is a one-shot classification — there is no
benefit to using a strong model.

Failure mode contract
---------------------
The judge is best-effort scoring infrastructure, NOT a hard gate.
If the API key is missing, the network is down, the provider
returns a malformed response, or any other exception fires, the
function returns a :class:`JudgeVerdict` with ``correct=False``,
``confidence=0.0``, and ``reasoning="judge unavailable: <why>"``.
The benchmark harness can then fall back to fuzzy matching without
crashing the run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from kaos_core.logging import get_logger

logger = get_logger("benchmarks.llm_judge")


class JudgeUnavailable(RuntimeError):
    """Raised internally when the judge cannot be invoked.

    Callers of :func:`llm_judge` will never see this — it is caught
    and mapped to a ``correct=False, confidence=0.0`` verdict so a
    benchmark run does not abort on a transient provider error.
    """


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Result of one judgment call."""

    correct: bool
    confidence: float
    reasoning: str
    judge_model: str = ""
    judge_cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_JUDGE_SYSTEM = (
    "You are an impartial grader for a retrieval-augmented question-answering "
    "benchmark used by lawyers. Your job is to decide whether an agent's "
    "response is CORRECT given the question, whether the question should be "
    "answered or refused, and (if answerable) the gold answer hint.\n"
    "\n"
    "Decision rules:\n"
    "1. If `expected_answerable` is FALSE: the question is unanswerable from "
    "the corpus (the relevant document is present but the specific fact is "
    "absent, OR the topic is entirely outside the corpus). The agent is "
    "CORRECT only if it explicitly refuses, says it cannot find sufficient "
    "evidence, declines to answer, or otherwise communicates that the answer "
    "is not in the materials. The agent is WRONG if it provides a confident "
    "factual answer (even if that answer is true in the world — the agent "
    "must not pull from its training data).\n"
    "2. If `expected_answerable` is TRUE: the agent is CORRECT only if its "
    "response substantively conveys the same fact as `expected_answer_hint`. "
    "Paraphrase, reordering, additional correct context, or extra citation "
    "language are all fine. A refusal is WRONG. A different fact is WRONG.\n"
    "3. Hedged answers count as refusals only if the agent does not commit "
    "to a specific factual claim. \"I don't have enough evidence to say\" is "
    "a refusal. \"It is approximately X, but you should verify\" is an "
    "answer.\n"
    "\n"
    "Return strict JSON with keys `correct` (bool), `confidence` (float in "
    "[0,1]), and `reasoning` (one or two sentences explaining the call). "
    "Do not include any other keys."
)


_JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["correct", "confidence", "reasoning"],
    "additionalProperties": False,
}


def _build_user_prompt(
    *,
    question: str,
    expected_answerable: bool,
    expected_hint: str | None,
    agent_answer: str,
    agent_refused: bool,
) -> str:
    expected_clause = (
        "ANSWERABLE — the agent should give the fact below."
        if expected_answerable
        else "UNANSWERABLE — the agent should refuse / decline / say it cannot find evidence."
    )
    hint_clause = (
        f"Expected answer hint: {expected_hint}"
        if expected_answerable and expected_hint
        else "Expected answer hint: (none — the question should be refused)"
    )
    refused_clause = (
        "Agent emitted an explicit insufficient-evidence signal."
        if agent_refused
        else "Agent did not emit an explicit insufficient-evidence signal."
    )
    answer_block = agent_answer.strip() or "(empty response)"
    return (
        f"Question:\n{question}\n\n"
        f"Expected behavior: {expected_clause}\n"
        f"{hint_clause}\n"
        f"{refused_clause}\n\n"
        f"Agent response (verbatim):\n---\n{answer_block}\n---\n\n"
        f"Render your verdict as JSON."
    )


def _coerce_verdict(raw: object) -> tuple[bool, float, str]:
    """Pull (correct, confidence, reasoning) out of a model response.

    The model is asked for strict JSON, but providers occasionally
    wrap it in code fences or append commentary. Be tolerant: try
    direct dict, then JSON-parse a string, then look for the first
    ``{...}`` block.
    """
    if isinstance(raw, dict):
        data: dict[str, object] = raw
    elif isinstance(raw, str):
        s = raw.strip()
        # Strip code fences if present.
        if s.startswith("```"):
            s = s.strip("`")
            # Remove a leading 'json\n' tag if present.
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            # Last resort: find first '{' ... last '}'.
            start = s.find("{")
            end = s.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise JudgeUnavailable("judge returned non-JSON response") from None
            data = json.loads(s[start : end + 1])
    else:
        raise JudgeUnavailable(f"judge returned unexpected type: {type(raw).__name__}")

    if not isinstance(data, dict):
        raise JudgeUnavailable("judge JSON was not an object")
    correct = bool(data.get("correct", False))
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(data.get("reasoning", "")).strip()
    return correct, confidence, reasoning


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

    Parameters
    ----------
    question:
        The question that was asked.
    expected_hint:
        For answerable questions, a short string describing the expected
        fact. ``None`` for unanswerable questions.
    expected_answerable:
        ``True`` if the corpus is expected to contain the answer.
    agent_answer:
        The agent's full response text (concatenated TextDelta stream).
    agent_refused:
        ``True`` if the agent emitted an explicit
        :class:`EvidenceInsufficient` signal.
    model:
        kaos-llm-client model identifier. Defaults to the cheapest current
        Anthropic model (``claude-haiku-4-5``).

    Returns
    -------
    dict
        ``{correct: bool, confidence: float, reasoning: str,
        judge_model: str, judge_cost_usd: float,
        judge_input_tokens: int, judge_output_tokens: int}``. On
        failure (no API key, network error, malformed response) returns
        ``correct=False, confidence=0.0, reasoning="judge unavailable: ..."``.
    """
    try:
        from kaos_llm_client import create_client  # local import: optional dep
    except ImportError as exc:
        verdict = JudgeVerdict(
            correct=False,
            confidence=0.0,
            reasoning=f"judge unavailable: kaos-llm-client not installed ({exc})",
        )
        return verdict.to_dict()

    user_prompt = _build_user_prompt(
        question=question,
        expected_answerable=expected_answerable,
        expected_hint=expected_hint,
        agent_answer=agent_answer,
        agent_refused=agent_refused,
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    try:
        client = create_client(model)
        # Use json() with the structured schema. Anthropic returns the
        # JSON via tool-call arguments (output_json) since GA April 2026,
        # OpenAI returns via response_format. Both surfaces land in
        # ``r.output_json``; the tool-call fallback handles the older
        # Anthropic path defensively.
        response = await client.json_async(messages, schema=_JUDGE_SCHEMA, max_tokens=400)
        raw = response.output_json
        if raw is None and response.tool_calls:
            raw = response.tool_calls[0].arguments
        if raw is None:
            raw = response.text
        correct, confidence, reasoning = _coerce_verdict(raw)
        usage = response.usage
        in_tok = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cost_usd = 0.0
        try:
            from kaos_llm_client.cost import estimate_call_cost

            est = estimate_call_cost(usage, model)
            cost_usd = float(est) if est is not None else 0.0
        except Exception:  # pragma: no cover — best-effort
            cost_usd = 0.0
        verdict = JudgeVerdict(
            correct=correct,
            confidence=confidence,
            reasoning=reasoning or "(no reasoning supplied)",
            judge_model=model,
            judge_cost_usd=cost_usd,
            judge_input_tokens=in_tok,
            judge_output_tokens=out_tok,
        )
        return verdict.to_dict()
    except JudgeUnavailable as exc:
        logger.warning("LLM judge skipped (parse): %s", exc)
        verdict = JudgeVerdict(
            correct=False,
            confidence=0.0,
            reasoning=f"judge unavailable: {exc}",
            judge_model=model,
        )
        return verdict.to_dict()
    except Exception as exc:  # broad: network, auth, provider 5xx, etc.
        logger.warning("LLM judge failed (%s): %s", type(exc).__name__, exc)
        verdict = JudgeVerdict(
            correct=False,
            confidence=0.0,
            reasoning=f"judge unavailable: {type(exc).__name__}: {exc}",
            judge_model=model,
        )
        return verdict.to_dict()
