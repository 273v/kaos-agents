"""LLM-based synthesis of plan-execute step results into a narrative answer.

Replaces the f-string blob dump that ``patterns/plan_execute.py:
_synthesize_results`` used through 0.1.0a8. The pre-0.1.0a9 user-facing
output looked like::

    Plan completed with the following results:

    **step-1-2fcdae**: Found 9 Federal Register document(s), showing 9
    {"results": [{"document_number": "2024-30494", "title": "EDGAR
    Filer Access and Account Management", "type": "Rule", ...}]}...

    **step-2-5d7ab4**: EDGAR Filer Access and Account Management (Rule, ...

— raw step IDs, raw JSON, no answer to the actual user question. This
module produces fluent text grounded in the same content.

Surface:

* :class:`SynthesizeFindingsSignature` — kaos-llm-core Signature with the
  inputs the model needs (goal, formatted step results, stop reason)
  and a single ``narrative`` output.
* :func:`synthesize_findings` — async helper that ``Call.invoke``\\s
  the Signature and returns ``(narrative, InvocationUsage)`` so the
  caller can emit ``UsageObserved`` for cost accounting.

The caller (``patterns/plan_execute.py``) is expected to fall back to
the pure f-string formatter (kept as ``_format_plan_response`` in that
module) when synthesis raises — degraded environments without an LLM
configured, network failures, or schema-validation retries that
exhausted all attempts. The fallback path preserves the partial-
findings UX recovery shipped in 0.1.0a7.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types.plan import ComposeResult
from kaos_agents.types.usage import InvocationUsage

logger = get_logger(__name__)


class SynthesizeFindingsSignature(Signature):
    """Write a clear, fluent answer to the user's goal using the
    structured outputs produced by a multi-step plan.

    You are summarising the work the agent already did — every fact
    you write must be traceable to the ``step_results`` block. Do not
    invent details, do not extrapolate beyond the data, and do not
    apologise for what the plan didn't reach when partial work
    succeeded.

    Composition rules:

    1. **Lead with the answer.** First sentence (or two) must address
       the user's original ``goal`` directly using the most relevant
       finding(s). Do not start with "I searched..." / "The plan
       produced..." / etc. — the user asked a question; answer it.
    2. **Cite step IDs inline.** When you reference a specific finding,
       include the step id in brackets, e.g. ``[step-1]``. The SPA
       renders those as links to the run inspector. This is how the
       reader audits the claim against the underlying tool output.
    3. **Preserve uncertainty when the plan stopped early.** If
       ``stop_reason != "success"`` and the findings only partially
       address the goal, state that explicitly in one sentence at the
       end (e.g. "The plan stopped before completing the litigation
       lookup; the document above is the most recent rule but the
       court history wasn't checked."). Do not silently paper over
       gaps — confident-wrong is worse than confident-partial.
    4. **No JSON dumps, no raw step output.** The findings the user
       sees should read as paragraphs, not as ``{"results": [...]}``
       blobs. The structured payload is for the SPA's run inspector;
       the narrative is for the chat surface.
    5. **Be concise.** Two-to-five short paragraphs. If the goal asked
       for a list, return a list; otherwise prose.

    Failure modes to refuse:

    * If every entry in ``step_results`` is an error / empty / off-
      topic, set ``narrative`` to a one-sentence acknowledgement
      ("The plan ran but no usable findings were produced; the steps
      that ran returned errors or off-topic results.") rather than
      fabricating an answer.
    """

    goal: str = InputField(
        description=(
            "The user's original natural-language request, exactly as "
            "the planner saw it. Anchors the narrative — every fact in "
            "the output should address this goal."
        ),
    )
    step_results: str = InputField(
        description=(
            "A flattened render of the executed step outputs. Each step "
            "is shown on its own line as ``[step_id] description: "
            "output_preview`` so the model can reference step ids "
            "inline (rule 2). Long outputs are pre-truncated by the "
            "caller — do not assume the full tool payload is here."
        ),
    )
    stop_reason: str = InputField(
        description=(
            "The plan-execute terminal reason: ``success``, "
            "``needs_replan``, ``max_replans``, ``max_cost``, "
            "``max_steps``, ``max_wall_clock``, or ``failure``. Drives "
            "rule 3 — partial-stop reasons need an explicit gap note."
        ),
    )

    narrative: str = OutputField(
        description=(
            "The user-facing answer, composed per the rules above. "
            "Markdown formatting is fine (bold, lists, links). Step-id "
            "citations look like ``[step-1-2fcdae]`` inline. No raw "
            "JSON or unbroken tool-payload paragraphs."
        ),
    )


def _format_step_results_for_prompt(
    step_results: dict[str, Any],
    *,
    step_descriptions: dict[str, str] | None = None,
    per_step_char_limit: int = 1200,
) -> str:
    """Render ``step_results`` as a multi-line block for the model.

    Format::

        [step-1-2fcdae] description: <truncated_output>
        [step-2-5d7ab4] description: <truncated_output>

    ``per_step_char_limit`` defaults to 1200 because the typical
    Federal Register / EDGAR JSON response is ~3-5KB; 1200 chars is
    enough to carry the document title + key fields without blowing
    out the prompt context budget on a 10-step plan.

    Empty / error-only results are still included so the model can
    truthfully report on what was attempted (per Signature rule 5's
    "every-step-failed → acknowledge, don't invent" fallback).
    """
    descriptions = step_descriptions or {}
    parts: list[str] = []
    for step_id, raw in step_results.items():
        text = str(raw)
        if len(text) > per_step_char_limit:
            text = text[:per_step_char_limit] + "..."
        desc = descriptions.get(step_id, "").strip()
        if desc:
            parts.append(f"[{step_id}] {desc}: {text}")
        else:
            parts.append(f"[{step_id}]: {text}")
    return "\n\n".join(parts)


async def synthesize_findings(
    goal: str,
    result: ComposeResult,
    *,
    model: str = DEFAULT_MODEL,
    step_descriptions: dict[str, str] | None = None,
) -> tuple[str, InvocationUsage]:
    """Compose a narrative answer from a finished ``ComposeResult``.

    Returns ``(narrative, usage)``. The caller is responsible for
    emitting :class:`~kaos_agents.events.lifecycle.UsageObserved` with
    the returned usage so synthesis cost lands on the
    :class:`~kaos_agents.events.lifecycle.TurnSummary` aggregate (and
    on the SPA's per-turn cost line — see kaos-agents 0.1.0a6 "real
    cost accounting" wiring).

    Raises if the inner ``Call.invoke`` fails after retries — callers
    in ``patterns/plan_execute.py`` catch the exception and fall back
    to the pure ``_format_plan_response`` formatter. We do not swallow
    the error here because the caller has more context (which response
    branch we were on, whether to log a warning, whether to record a
    REFLECTION-section memory entry).
    """
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core import Call

    formatted_results = _format_step_results_for_prompt(
        dict(result.step_results),
        step_descriptions=step_descriptions,
    )

    call = Call(SynthesizeFindingsSignature, model=model)
    # ``invoke`` (not bare ``__call__``) so ``Invocation.usage`` is
    # populated for cost accounting.
    invocation = await call.invoke(
        goal=goal,
        step_results=formatted_results,
        stop_reason=result.stop_reason.value,
    )
    narrative = str(invocation.output.narrative).strip()
    usage = InvocationUsage.from_invocation(invocation)

    logger.debug(
        "synthesize_findings: stop=%s, %d step(s), %d narrative chars, cost=$%.4f",
        result.stop_reason.value,
        len(result.step_results),
        len(narrative),
        usage.cost_usd,
    )

    return narrative, usage


def should_attempt_llm_synthesis(result: ComposeResult) -> bool:
    """Gate the LLM synthesis call.

    Returns ``True`` only when there's something worth synthesising —
    at least one non-error step result. The pure-formatter fallback
    handles the all-errors-or-empty cases without spending tokens.

    Centralising this here means the caller in
    ``_handle_plan_streaming`` doesn't have to replicate the
    "is there usable content?" check; the same import is also a
    natural place to evolve the gating policy (e.g., skip synthesis
    on very-short single-result plans) without re-touching the
    streaming loop.
    """
    from kaos_agents.planning.result_check import is_error_result

    if not result.step_results:
        return False
    for raw in result.step_results.values():
        text = str(raw)
        if text and not is_error_result(text):
            return True
    return False


_PUBLIC_SURFACE = (
    "SynthesizeFindingsSignature",
    "synthesize_findings",
    "should_attempt_llm_synthesis",
)
__all__ = list(_PUBLIC_SURFACE)
