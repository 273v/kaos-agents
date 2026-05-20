"""M2 reasoning-action consistency critic.

Authored on top of the already-shipped :class:`JudgeSignature`
(``kaos_agents.planning.judge``). Per the lateral-redesign plan §6 and
the agentic-loop-honesty plan Stage 2, new critic families are rubric
strings — not new Signature classes.

This rubric catches the failure mode captured live in SPA session
``01KS1K6J9XWKCNQ0NPNKXXXP4P`` (gpt-5.4-mini, 2026-05-19): the agent
emitted a ``Branch taken: upper bound >= 5.0%`` headline immediately
followed by a body that computed 4.50% and explicitly said "does not
reach 5.0%", then took the ``< 5.0%`` branch in its conclusion. The
reasoning and final answer were correct; the headline was a stale
first-draft commitment the model never edited.

Labels:

* ``consistent`` — the response's headline/announcement is consistent
  with both its own reasoning/body AND the tool-call results
  (provided as ``context``).
* ``contradicts_reasoning`` — the headline asserts X but the body's
  reasoning concludes ~X. The body is correct; the headline is stale.
* ``contradicts_tool_results`` — the headline (and possibly the body)
  asserts something the tool calls' actual results disprove. The tools
  are correct; the response is fabricated or carries-forward from
  training memory.

Use :func:`judge_reasoning_action_consistency` to invoke. Callers
supply the assistant response and the tool-call result texts; the
judge returns a :class:`JudgeVerdict` whose ``label`` is one of the
three above and whose ``fell_back`` flag signals when the verdict
should not be trusted (model error or label outside the allowlist).
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_agents.planning.judge import (
    JudgeVerdict,
    judge_with_rubric,
)

logger = get_logger(__name__)

M2_REASONING_ACTION_RUBRIC: str = """\
Evaluate whether the assistant's response is internally consistent.
You will receive the assistant's full response text as ``input_text``
and the tool-call result summaries (when any tools were invoked) as
``context``.

Emit exactly ONE of these three labels (lowercase, verbatim):

* ``consistent``
* ``contradicts_reasoning``
* ``contradicts_tool_results``

Decision rules:

1. ``contradicts_tool_results`` — when a load-bearing factual claim
   in the response disagrees with what the tool calls actually
   returned. Examples: the tool returned "rate = 4.50%" but the
   response says "rate is above 5.0%"; the tool returned an error
   but the response describes successful retrieval. The tools are
   the ground truth here; the response is wrong. If ``context`` is
   empty (no tools were invoked), this label cannot apply.

2. ``contradicts_reasoning`` — when the response's headline,
   announcement, or summary line at the START of the answer asserts
   one outcome while the body's own reasoning / computation / final
   conclusion asserts the opposite. This is the "stale first-draft
   headline" pattern: the model committed to a branch / verdict /
   classification up front, then computed something different in
   the body, and never went back to fix the headline. The body is
   internally correct; the headline is stale.

3. ``consistent`` — when neither of the above contradictions holds.
   The headline matches the body's reasoning AND the body's claims
   match the tool results (or no tools were invoked).

Edge cases:

* A response with no headline / announcement / summary line —
  treat as ``consistent`` unless ``contradicts_tool_results``
  applies.
* A response that explicitly flags its own uncertainty ("I'm not
  sure but X seems likely") — treat as ``consistent`` if the
  hedging is internally coherent.
* A response that takes a stronger position in the headline than
  the body supports but where the positions don't actively
  contradict (e.g. headline says "definitely X", body says
  "probably X") — treat as ``consistent``. This rubric catches
  contradictions, not over-confidence.
* When both ``contradicts_tool_results`` AND ``contradicts_reasoning``
  apply, emit ``contradicts_tool_results`` (the more fundamental
  failure — the response is ungrounded, not just self-contradicting).

``confidence`` should be high (>= 0.85) when the contradiction is
explicit (the response literally says "X" in one place and "not X"
in another). Reserve lower confidence for cases where the
contradiction requires interpretation."""


M2_ALLOWED_LABELS: tuple[str, ...] = (
    "consistent",
    "contradicts_reasoning",
    "contradicts_tool_results",
)


async def judge_reasoning_action_consistency(
    *,
    response_text: str,
    model: str,
    tool_results_text: str = "",
) -> JudgeVerdict:
    """Run the M2 rubric against an assistant response.

    Args:
        response_text: The full assistant response (headline + body)
            to judge. Typically the text of the final iteration in
            an AgenticLoop turn.
        model: Provider:model string passed to kaos-llm-core.
        tool_results_text: Concatenated tool-call result summaries
            from the same turn, joined with newlines. Pass empty
            string when no tools were invoked — the judge will not
            return ``contradicts_tool_results`` in that case.

    Returns:
        :class:`JudgeVerdict`. ``label`` is one of M2_ALLOWED_LABELS
        on success; ``fell_back=True`` when the model emitted a
        disallowed label or the invocation errored. The caller
        decides how to respond — typically:

        * ``consistent`` → no action; pass through.
        * ``contradicts_reasoning`` → force a re-write iteration
          with a "your headline contradicted your body" directive.
        * ``contradicts_tool_results`` → force a re-grounding
          iteration with a "your answer contradicts what the tools
          returned" directive.
        * ``fell_back=True`` → conservative path; treat as
          ``consistent`` to avoid loop on bad verdicts.
    """
    verdict = await judge_with_rubric(
        rubric=M2_REASONING_ACTION_RUBRIC,
        input_text=response_text,
        context=tool_results_text,
        model=model,
        allowed_labels=M2_ALLOWED_LABELS,
    )
    # Observability — log every M2 invocation so operators don't have
    # to grep memory.json or guess from cost deltas. INFO for trusted
    # verdicts, WARNING when fell_back. The AgenticLoop emits a
    # ConsistencyChecked event in parallel; this log is the
    # CLI / no-event consumer's surface.
    if verdict.fell_back:
        logger.warning(
            "M2 verdict fell back (label=%r confidence=%.2f reason=%r model=%s)",
            verdict.label,
            verdict.confidence,
            verdict.reasoning,
            model,
        )
    else:
        logger.info(
            "M2 verdict label=%s confidence=%.2f cost=$%.4f latency_ms=%.0f "
            "response_chars=%d tool_results_chars=%d model=%s",
            verdict.label,
            verdict.confidence,
            verdict.cost_usd,
            verdict.latency_ms,
            len(response_text),
            len(tool_results_text),
            model,
        )
    return verdict


__all__ = [
    "M2_ALLOWED_LABELS",
    "M2_REASONING_ACTION_RUBRIC",
    "judge_reasoning_action_consistency",
]
