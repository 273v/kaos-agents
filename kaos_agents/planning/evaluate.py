"""Evaluate primitive — structural and semantic judgment.

Two modes:
- Structural: deterministic checks (tool exists? result is error? schema match?)
  No LLM call. Nanoseconds. Used for most post-execution checks.
- Semantic: LLM-based quality judgment (is this answer good? is this plan feasible?)
  One LLM call via Call(EvalSig). Used when structural checks are insufficient.

Structural mode runs first. If it resolves the judgment, no LLM call is needed.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types.plan import EvalMode, Judgment

logger = get_logger(__name__)


class EvalSemanticSignature(Signature):
    """Judge whether a step's actual result satisfies the expected output.

    Decision rules:

    * Be generous on framing — partial matches count if the substance
      is there.
    * Be strict on facts — if the expectation calls out a specific
      number, name, or section reference, that must appear (or be
      unambiguously implied) in the result.
    * Extract any KEY facts from the result that future steps might
      need to reference. Empty list if nothing notable.
    """

    result_text: str = InputField(description="The actual result produced.")
    expected_description: str = InputField(description="What was expected.")
    additional_context: str = InputField(description="Additional context for judgment.")
    matched: bool = OutputField(description="Whether the result satisfies the expectation.")
    confidence: float = OutputField(
        description="Confidence in this judgment, in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = OutputField(description="One or two sentences explaining the verdict.")
    new_facts: list[str] = OutputField(
        description="Key facts extracted from the result. Empty list if none."
    )


# ---------------------------------------------------------------------------
# Structural evaluation (no LLM)
# ---------------------------------------------------------------------------


def evaluate_structural(
    result: Any,
    expected: str = "",
    *,
    tool_name: str | None = None,
    available_tools: set[str] | None = None,
) -> Judgment | None:
    """Try to evaluate structurally. Returns None if structural check is inconclusive.

    Checks (in order):
    1. Is the result an error? → matched=False
    2. Is the tool name valid? → matched=False if hallucinated
    3. Is the result empty? → matched=False
    4. Otherwise → None (needs semantic evaluation)
    """
    from kaos_agents.planning.result_check import is_empty_result, is_error_result

    result_str = str(result) if result is not None else ""

    # Check for error results
    if is_error_result(result_str):
        return Judgment(
            matched=False,
            confidence=1.0,
            reasoning=f"Step returned an error: {result_str[:200]}",
            mode=EvalMode.STRUCTURAL,
        )

    # Check tool name validity
    if tool_name and available_tools and tool_name not in available_tools:
        return Judgment(
            matched=False,
            confidence=1.0,
            reasoning=f"Tool '{tool_name}' is not registered. "
            f"Available: {sorted(available_tools)[:10]}",
            mode=EvalMode.STRUCTURAL,
        )

    # Check empty result
    if is_empty_result(result_str):
        return Judgment(
            matched=False,
            confidence=0.9,
            reasoning="Step returned empty result.",
            mode=EvalMode.STRUCTURAL,
        )

    # Structural check passed but can't confirm quality
    if not expected:
        # No expectation set — accept any non-error, non-empty result
        return Judgment(
            matched=True,
            confidence=0.7,
            reasoning="Result is non-empty and non-error (no specific expectation to check).",
            mode=EvalMode.STRUCTURAL,
        )

    # Has expectation but structural check can't verify semantic match
    return None


# ---------------------------------------------------------------------------
# Semantic evaluation (LLM)
# ---------------------------------------------------------------------------


async def evaluate_semantic(
    result: Any,
    expected: str,
    *,
    context: str = "",
    model: str = DEFAULT_MODEL,
    emitter: Any = None,
    step_id: str | None = None,
) -> Judgment:
    """Evaluate result quality via LLM.

    Used when structural evaluation is inconclusive and the step has
    an expected output description that needs semantic comparison.

    When ``emitter`` is provided (caller passes the active
    :class:`~kaos_agents.events.emitter.EventEmitter` for this turn),
    a ``Span(SpanSubject.JUDGE, ...)`` pair is emitted around the
    Call so SSE consumers can see what the LLM judge was asked + how
    it scored. The span carries ``step_id``, the truncated
    ``expected``/``result_preview``, and the resulting
    ``matched``/``confidence``/``reasoning``/``mode`` attributes. When
    ``emitter`` is None, no events fire (preserves backward compat for
    direct unit-test callers).
    """
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core import Call

    call = Call(EvalSemanticSignature, model=model)

    # Open the JUDGE span before we issue the Call so observers see the
    # inputs that led to the judgment. Done as a single pair on
    # COMPLETE/ERROR so duration_ms is filled by the emitter.
    judge_span_id: str | None = None
    if emitter is not None:
        from kaos_agents.events.spans import SpanSubject

        span_attrs: dict[str, Any] = {
            "step_id": step_id or "",
            "expected": str(expected)[:200],
            "result_preview": str(result)[:200],
        }
        span = emitter.span_start(
            SpanSubject.JUDGE,
            name=f"judge.{step_id}" if step_id else "judge",
            attributes=span_attrs,
        )
        judge_span_id = span.span_id

    try:
        # ``invoke`` (not bare ``__call__``) so per-step semantic-eval
        # cost flows through ``Invocation.usage`` into the
        # ``PlanBudget``. Was a hole — semantic-eval cost never
        # counted against the per-plan budget.
        invocation = await call.invoke(
            result_text=str(result)[:2000],
            expected_description=expected,
            additional_context=context[:1000],
        )
        output = invocation.output
        judgment = Judgment(
            matched=output.matched,
            confidence=max(0.0, min(1.0, output.confidence)),
            reasoning=output.reasoning,
            mode=EvalMode.SEMANTIC,
            new_facts=tuple(output.new_facts) if output.new_facts else (),
        )
        if emitter is not None and judge_span_id is not None:
            from kaos_agents.events.spans import SpanSubject

            emitter.span_complete(
                SpanSubject.JUDGE,
                span_id=judge_span_id,
                name=f"judge.{step_id}" if step_id else "judge",
                attributes={
                    "step_id": step_id or "",
                    "matched": bool(judgment.matched),
                    "confidence": float(judgment.confidence),
                    "reasoning": str(judgment.reasoning)[:200],
                    "mode": judgment.mode.value,
                },
            )
        return judgment
    except Exception as exc:
        logger.warning("evaluate_semantic failed: %s — marking as unverified", exc)
        if emitter is not None and judge_span_id is not None:
            from kaos_agents.events.spans import SpanSubject

            emitter.span_error(
                SpanSubject.JUDGE,
                span_id=judge_span_id,
                name=f"judge.{step_id}" if step_id else "judge",
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
                attributes={"step_id": step_id or ""},
            )
        return Judgment(
            matched=False,
            confidence=0.0,
            reasoning=f"Semantic evaluation failed ({exc}). Result is unverified.",
            mode=EvalMode.STRUCTURAL,
        )


# ---------------------------------------------------------------------------
# Combined evaluate
# ---------------------------------------------------------------------------


async def evaluate(
    result: Any,
    expected: str = "",
    *,
    tool_name: str | None = None,
    available_tools: set[str] | None = None,
    context: str = "",
    model: str = DEFAULT_MODEL,
    force_semantic: bool = False,
) -> Judgment:
    """Evaluate a result — structural first, semantic fallback.

    Args:
        result: The output to evaluate.
        expected: Description of what was expected.
        tool_name: Tool that produced this result (for validation).
        available_tools: Set of registered tool names (for validation).
        context: Additional context for semantic evaluation.
        model: LLM model for semantic evaluation.
        force_semantic: Skip structural and go straight to LLM.

    Returns:
        Judgment with matched, confidence, reasoning, and mode.
    """
    if not force_semantic:
        structural = evaluate_structural(
            result,
            expected,
            tool_name=tool_name,
            available_tools=available_tools,
        )
        if structural is not None:
            return structural

    # Structural was inconclusive — use semantic
    return await evaluate_semantic(result, expected, context=context, model=model)
