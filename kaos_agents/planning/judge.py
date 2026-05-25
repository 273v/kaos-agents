"""Generic :class:`JudgeSignature` — rubric-parametric judge.

Plan §6 (Generic JudgeSignature) of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``.

Replaces 5+ bespoke critic Signatures (GoalChecker, RubricVerdict,
EvalSemantic, EvaluateCondition, QAJudge) with one parametric judge
whose **rubric** is the only varying input. Each existing judge gets a
named rubric string; the rest of the contract is uniform.

Shape:

* Input — ``rubric`` (free-form criteria), ``input_text`` (the thing
  being judged), optional ``context`` (background).
* Output — ``label`` (categorical verdict, chosen from the labels the
  rubric enumerates), ``confidence`` ([0, 1] self-rating), ``reasoning``
  (one-paragraph justification).

Examples of rubrics that fit this shape:

* "Output ``satisfied`` / ``needs_more_work`` / ``insufficient_evidence``.
  Satisfied = the response answers the user's question with grounded
  evidence. Needs_more_work = ..." → subsumes GoalCheckSignature.
* "Output ``pass`` / ``fail``. Pass iff the deliverable contains all of:
  X, Y, Z. Otherwise fail." → subsumes RubricVerdictSignature.
* "Output ``matched`` / ``unmatched``. The result is matched iff it
  semantically addresses the expectation X." → subsumes
  EvalSemanticSignature.

Catalog-agnostic, no hardcoded judge-type enums. New judge = new rubric
string, not a new Signature class.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature

logger = logging.getLogger("kaos.agents.planning.judge")


class JudgeSignature(Signature):
    """Evaluate ``input_text`` against ``rubric`` and emit a structured verdict.

    You are an impartial evaluator. The user has supplied a rubric
    describing what to check for, what counts as a positive verdict, and
    what counts as a negative verdict. They have also supplied input
    text — the thing you are judging — and optional context.

    Decision rules:

    1. Read ``rubric`` carefully. It enumerates the labels you may
       emit. Use ONLY labels the rubric names (verbatim, case-
       insensitive). Do not invent additional labels.
    2. Apply the rubric strictly. If the rubric demands every criterion
       hold, all must hold to emit the positive label. If it allows
       partial credit, the rubric will say so — otherwise default to
       strict.
    3. ``confidence`` is your self-rating, 0.0 to 1.0. Reserve
       ``>= 0.9`` for cases where the rubric's positive / negative
       criteria are clearly met or violated. Reserve ``<= 0.3`` for
       cases where the rubric is genuinely ambiguous on this input.
    4. ``reasoning`` is one short paragraph (3-5 sentences max) citing
       the specific rubric criterion that drove the verdict. Do NOT
       restate the rubric verbatim; cite the criterion by name or
       short paraphrase.
    5. When ``context`` is empty, judge from ``input_text`` alone. When
       ``context`` is provided, treat it as supporting background but
       the verdict must still be grounded in ``input_text`` —
       ``context`` cannot be the ONLY basis for the verdict.
    6. If ``rubric`` describes a probabilistic / soft criterion (e.g.
       "if the response is _likely_ correct"), pick the closest
       categorical label and surface uncertainty via ``confidence``,
       not by inventing a fence-sitting label.

    The judge is intentionally domain-agnostic — the rubric carries the
    domain semantics, so the same Signature serves goal-checking,
    rubric-based PASS/FAIL evaluation, semantic-match evaluation,
    predicate evaluation over prior outputs, and Q&A benchmark
    correctness judging.
    """

    rubric: str = InputField(
        description=(
            "Free-form evaluation criteria. Must enumerate the labels "
            "the judge may emit (e.g. 'pass / fail', "
            "'satisfied / needs_more_work / insufficient_evidence', "
            "'matched / unmatched'). May also describe any positive / "
            "negative criteria the judge applies."
        ),
    )
    input_text: str = InputField(
        description=(
            "The text being judged — agent response, retrieved passage, "
            "deliverable, predicate-applicable data."
        ),
    )
    context: str = InputField(
        default="",
        description=(
            "Optional supporting background — prior conversation, "
            "ground-truth hint, expectation description. Empty string "
            "when not applicable."
        ),
    )

    label: str = OutputField(
        description=(
            "Categorical verdict. MUST be one of the labels the rubric "
            "enumerates. The judge enforces nothing schema-wise; the "
            "rubric is load-bearing."
        ),
    )
    confidence: float = OutputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Self-rating, 0.0 to 1.0. See rule 3.",
    )
    reasoning: str = OutputField(
        default="",
        description=(
            "One short paragraph citing the specific rubric criterion "
            "that drove the verdict (3-5 sentences max). See rule 4."
        ),
    )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """Typed outcome of one :func:`judge_with_rubric` invocation.

    Mirrors the kaos-agents value-type discipline — frozen, slotted,
    pointwise constructible.

    Fields:
        label: Categorical verdict emitted by the judge. Lowercased
            and stripped at parse time so downstream comparisons are
            case-insensitive.
        confidence: Float in [0, 1]. Clamped if the model returns a
            value outside that range.
        reasoning: Free-form justification.
        cost_usd: Spend on the underlying LLM call (0.0 when the
            pricing table doesn't know the model).
        latency_ms: Wall-clock latency in milliseconds.
        fell_back: True when the invocation errored and the caller
            should treat the verdict as untrustworthy. ``label`` will
            be the empty string in that case.
    """

    label: str
    confidence: float
    reasoning: str
    cost_usd: float
    latency_ms: float
    fell_back: bool


def _normalize_label(raw: Any) -> str:
    """Lowercase + strip the raw label. Returns empty string for None."""
    if raw is None:
        return ""
    return str(raw).strip().lower()


def _clamp_confidence(raw: Any) -> float:
    """Coerce + clamp to [0, 1]."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


async def judge_with_rubric(
    *,
    rubric: str,
    input_text: str,
    model: str,
    context: str = "",
    allowed_labels: tuple[str, ...] | None = None,
) -> JudgeVerdict:
    """Run :class:`JudgeSignature` and return a typed :class:`JudgeVerdict`.

    Lazy-imports kaos-llm-core so a kaos-agents install without the
    ``[llm]`` extra keeps loading — matches the ``policy.py`` and
    ``tool_fitness.py`` patterns.

    Args:
        rubric: Free-form evaluation criteria. Must enumerate the
            labels the judge may emit. See ``JudgeSignature`` docstring.
        input_text: The thing being judged.
        model: Provider:model string passed to kaos-llm-core's
            :class:`Call`.
        context: Optional supporting background.
        allowed_labels: Optional whitelist. When set, the returned
            ``label`` is verified against this tuple (case-insensitive,
            trimmed); a mismatch sets ``fell_back=True`` and the
            ``label`` field is preserved verbatim for caller inspection.

    Returns:
        :class:`JudgeVerdict`. On any provider error / schema parse
        failure / disallowed label, ``fell_back=True``. The caller
        decides how to handle that (retry, fail-closed, default to a
        conservative verdict, etc.).
    """
    t_start = time.monotonic()
    try:
        from kaos_llm_core.programs.call import Call
    except ImportError as exc:
        logger.warning(
            "JudgeSignature: kaos-llm-core not installed; falling back. err=%s",
            exc,
        )
        return JudgeVerdict(
            label="",
            confidence=0.0,
            reasoning="kaos-llm-core not installed",
            cost_usd=0.0,
            latency_ms=0.0,
            fell_back=True,
        )

    from kaos_agents._examples import load_examples

    call = Call(JudgeSignature, model=model, examples=load_examples("judge"))
    try:
        invocation = await call.invoke(
            rubric=rubric,
            input_text=input_text,
            context=context,
        )
    except Exception as exc:
        logger.warning(
            "JudgeSignature: invocation failed model=%s err=%s",
            model,
            exc,
        )
        return JudgeVerdict(
            label="",
            confidence=0.0,
            reasoning=f"invocation error: {type(exc).__name__}",
            cost_usd=0.0,
            latency_ms=(time.monotonic() - t_start) * 1000,
            fell_back=True,
        )

    output: Any = invocation.output
    raw_label = _normalize_label(getattr(output, "label", None))
    confidence = _clamp_confidence(getattr(output, "confidence", 0.0))
    reasoning = str(getattr(output, "reasoning", "") or "")

    fell_back = False
    if not raw_label:
        fell_back = True
    elif allowed_labels is not None:
        allowed_lower = {a.strip().lower() for a in allowed_labels}
        if raw_label not in allowed_lower:
            fell_back = True

    usage = getattr(invocation, "usage", None)
    cost_usd = float(getattr(usage, "cost_usd", 0.0) or 0.0) if usage is not None else 0.0
    latency_ms = (time.monotonic() - t_start) * 1000

    return JudgeVerdict(
        label=raw_label,
        confidence=confidence,
        reasoning=reasoning,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        fell_back=fell_back,
    )


__all__ = [
    "JudgeSignature",
    "JudgeVerdict",
    "judge_with_rubric",
]
