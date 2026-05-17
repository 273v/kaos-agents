"""Conditional-step predicate evaluation.

Implements the ``Step.abort_if`` runtime check added in kaos-agents
0.1.0a9 — see :class:`~kaos_agents.types.plan.Step` for the type-level
contract.

The R1-REAL Test 3 prompt shape this enables::

    Build me a 5-step research plan on CFPB's 1033 open-banking rule,
    execute it, but if any step reveals the rule was vacated or
    stayed, abandon the remaining steps and pivot to the litigation
    status.

The planner generates a 5-step plan with each step carrying
``abort_if="any prior step output indicates the rule was vacated or
stayed"``. Before Compose runs step N, it gathers step N-1's outputs,
passes them + the predicate to :func:`evaluate_condition`, and skips
step N (plus all dependents) when the condition holds.

The evaluation is one cheap haiku-tier ``Call.invoke`` per condition
check — typically <100ms and <$0.001. Conditions on the FIRST step
have no prior outputs to evaluate against, so they are treated as
trivially False.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents.settings import DEFAULT_MODEL

logger = get_logger(__name__)


class EvaluateConditionSignature(Signature):
    """Decide whether a natural-language predicate holds over prior step outputs.

    You are deciding whether a plan should abort or pivot based on
    what earlier steps revealed. You see (a) the predicate the
    planner wrote when constructing this step (``condition``) and
    (b) the outputs of every step that has already executed
    (``prior_outputs``). Return ``holds=true`` only when the prior
    outputs UNAMBIGUOUSLY satisfy the predicate.

    Decision rules:

    1. **Default to false.** If the prior outputs don't clearly
       address the predicate, return ``holds=false`` with a low
       ``confidence``. False positives here cause the agent to
       abandon useful work; false negatives at worst run an extra
       step. Asymmetric cost.
    2. **Strict literal match.** "The rule was vacated or stayed"
       requires the prior output to actually mention vacatur, stay,
       or equivalent legal status. Don't infer from absence of
       evidence ("the docs didn't mention the rule being in force,
       therefore it must be stayed" is NOT a match).
    3. **Cite evidence.** When ``holds=true``, copy the exact
       sentence or phrase from ``prior_outputs`` that proves it
       into ``evidence`` (truncate at 200 chars). Empty when
       ``holds=false``.
    4. **High confidence requires clear evidence.** Reserve
       ``confidence >= 0.8`` for cases where the evidence is a
       direct, unambiguous statement. Reserve ``confidence >= 0.6``
       for strong implications. Below 0.5 means "probably not".
    """

    condition: str = InputField(
        description=(
            "The predicate written by the planner — a natural-language "
            "statement of what would justify aborting or pivoting the "
            "remaining steps. Example: 'the rule was vacated or stayed'."
        ),
    )
    prior_outputs: str = InputField(
        description=(
            "All previously-executed step outputs concatenated as "
            "``[step_id]: <output_preview>`` lines, separated by blank "
            "lines. May be empty when no prior steps have run — in "
            "that case the condition is trivially False (rule 1)."
        ),
    )

    holds: bool = OutputField(
        description=(
            "Whether the prior outputs unambiguously satisfy the "
            "condition. False by default (rule 1)."
        ),
    )
    confidence: float = OutputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Self-rating of the verdict, 0.0 to 1.0. >=0.8 requires "
            "direct evidence; <0.5 reads as 'probably not' (rule 4)."
        ),
    )
    evidence: str = OutputField(
        default="",
        description=(
            "When ``holds=true``: the exact sentence/phrase from "
            "``prior_outputs`` that proves the condition (<=200 chars). "
            "Empty when ``holds=false``."
        ),
    )


def _format_prior_outputs(
    prior_outputs: dict[str, Any],
    *,
    per_step_char_limit: int = 800,
) -> str:
    """Render a dict of completed step outputs for the Signature input.

    Same shape as
    :func:`kaos_agents.patterns.synthesize._format_step_results_for_prompt`
    — ``[step-id]: <truncated_output>`` lines separated by blank
    lines. Smaller per-step cap (800 vs 1200) because abort_if
    typically only needs the key sentence, not the full payload, and
    we want to keep condition-evaluation cost <$0.001.
    """
    if not prior_outputs:
        return ""
    parts: list[str] = []
    for step_id, raw in prior_outputs.items():
        text = str(raw)
        if len(text) > per_step_char_limit:
            text = text[:per_step_char_limit] + "..."
        parts.append(f"[{step_id}]: {text}")
    return "\n\n".join(parts)


async def evaluate_condition(
    condition: str,
    prior_outputs: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    holds_confidence_threshold: float = 0.6,
) -> tuple[bool, str]:
    """Decide whether ``condition`` holds over ``prior_outputs``.

    Returns ``(holds, evidence)``. The ``holds`` flag is ``True``
    only when the inner Signature returns ``holds=true`` AND
    ``confidence >= holds_confidence_threshold`` — the threshold
    gate catches the LLM hedging case ("kind of, sort of").

    Returns ``(False, "")`` without an LLM call when:

    * ``condition`` is empty (no predicate, no check to do).
    * ``prior_outputs`` is empty (first step in the graph, no prior
      evidence to evaluate against — Signature rule 1 says default
      false in that case anyway).
    * The inner Call raises (degraded LLM environment, schema
      retry exhausted, network error). The caller continues
      executing the step rather than abandoning the plan.

    Errors surface as warnings in the logger; never propagated to
    the caller.
    """
    if not condition:
        return False, ""
    if not prior_outputs:
        return False, ""

    from kaos_agents._llm_imports import require_llm_core

    try:
        require_llm_core()
        from kaos_llm_core import Call

        call = Call(EvaluateConditionSignature, model=model)
        invocation = await call.invoke(
            condition=condition,
            prior_outputs=_format_prior_outputs(prior_outputs),
        )
        out = invocation.output
        holds_bool = bool(out.holds) and float(out.confidence) >= holds_confidence_threshold
        evidence_str = str(out.evidence or "")[:200]
        logger.debug(
            "evaluate_condition: holds=%s confidence=%.2f threshold=%.2f → %s",
            bool(out.holds),
            float(out.confidence),
            holds_confidence_threshold,
            holds_bool,
        )
        return holds_bool, evidence_str if holds_bool else ""
    except Exception as exc:
        logger.warning(
            "evaluate_condition failed (%s); defaulting to holds=False so the step executes anyway",
            exc,
        )
        return False, ""
