"""M4 silent-incompleteness critic.

Authored on top of the already-shipped :class:`JudgeSignature`
(``kaos_agents.planning.judge``). Mirrors :mod:`m2_consistency` and
:mod:`m3_grounding` exactly in shape — *new critic = new rubric, not
a new Signature class*. This keeps the lateral-redesign promise that
the Generic JudgeSignature substrate collapses M2/M3/M4 work into
rubric authoring.

This rubric catches the **confident-partial-answer** failure mode
surfaced live in corpus stress S12 (500-doc deal room, 5 planted
needles, 2026-05-23): agent found 4 of 5 needles, declared partial
victory in a fluent response, did NOT drill for the 5th, and did NOT
acknowledge that one needle was missing. The user asked for "all
five." Same family as M2 (reasoning-action consistency) and M3
(document-grounding fabrication), but on the *completeness* axis —
M2 detects headline-vs-body contradiction, M3 detects
claim-vs-evidence ungroundedness, M4 detects requested-N vs
delivered-N gap with no exhaustion proof.

Labels:

* ``complete`` — the response satisfies the user's quantitative or
  enumerated requirement.
* ``partial`` — the response returned FEWER items than the user
  requested, OR omitted parts of a multi-part question, WITHOUT
  explicitly stating which were not found / not addressed. The
  failing label.
* ``exhausted`` — the response returned fewer items than requested
  AND explicitly states the search/corpus was exhausted. Honest
  fallback; not a failure.
* ``not_quantified`` — the user's request was open-ended. No
  completeness expectation to enforce.

Use :func:`judge_completeness` to invoke. Same wire shape as
:func:`judge_reasoning_action_consistency` / :func:`judge_grounding_fabrication`
so the AgenticLoop can call all three in sequence.
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_agents.planning.judge import (
    JudgeVerdict,
    judge_with_rubric,
    log_verdict,
)

logger = get_logger(__name__)


M4_COMPLETENESS_RUBRIC: str = """\
Evaluate whether the response is COMPLETE relative to what the user
requested. You will receive the user's prompt as ``input_text`` and
the assistant's response as ``context``.

Emit exactly ONE of these four labels (lowercase, verbatim):

* ``complete``
* ``partial``
* ``exhausted``
* ``not_quantified``

Label definitions:

* ``complete``: the response satisfies the user's quantitative or
  enumerated requirement. Examples: user asked for "all five
  needles" and response returned 5; user asked for "three examples"
  and response provided 3 distinct examples; user asked for "the
  list of subsidiaries" and response provided a list with no caveats
  that more exist.
* ``partial``: the response returned FEWER items than the user
  requested, OR omitted parts of a multi-part question, WITHOUT
  explicitly stating which were not found / not addressed. This is
  the silent-incompleteness failure mode — user can't tell that the
  agent gave up early.
* ``exhausted``: the response returned fewer items than requested
  AND explicitly states the search/corpus was exhausted (e.g. "I
  reviewed all N documents in the corpus and found only M
  matching..."). This is the honest fallback; it's NOT a failure.
* ``not_quantified``: the user's request was open-ended (no count,
  no "all/every/each" quantifier, no enumerated multi-part). The
  judge has no completeness expectation to enforce. This is the
  default for conversational and qualitative questions.

Decision rules:

1. Look at the user's request first. If it contains a definite
   quantifier ("all", "every", "each of", "list of all", "N
   ___"), the agent owes either ``complete`` (returned what was
   asked) or ``exhausted`` (honest acknowledgment of fewer).
2. If the user asked a multi-part question (numbered list,
   "and also X, Y, Z"), check each part is addressed. Missing
   parts = ``partial``.
3. ``not_quantified`` is correct for open-ended Q&A. Do not
   over-fire — most conversational turns have no completeness
   expectation.
4. ``partial`` is the only failing label. ``complete``,
   ``exhausted``, and ``not_quantified`` all pass.

Confidence:

* >= 0.9 when the user's quantifier is unambiguous and the response
  count is plain (e.g. user asked for 5, response gives 4 with no
  exhaustion clause = ``partial`` at high confidence).
* <= 0.3 when the response's structure makes counting hard
  (long prose with implicit lists, paraphrased enumerations)."""


M4_ALLOWED_LABELS: tuple[str, ...] = (
    "complete",
    "partial",
    "exhausted",
    "not_quantified",
)


async def judge_completeness(
    *,
    user_prompt: str,
    response_text: str,
    model: str,
) -> JudgeVerdict:
    """Run the M4 rubric against an assistant response.

    Composes the generic JudgeSignature from kaos_agents.planning.judge
    with the COMPLETENESS rubric. The judge reads the user's prompt
    (as ``input_text``) and the agent's response (as ``context``) and
    emits one of {complete, partial, exhausted, not_quantified}.

    Args:
        user_prompt: The user's full message — drives the
            completeness expectation (definite quantifier?
            multi-part?). Passed as ``input_text`` because the
            rubric's primary decision is "what did the user ask
            for?", with the response as supporting context.
        response_text: The full assistant response to evaluate.
            Passed as ``context``.
        model: Provider:model string passed to kaos-llm-core's
            :class:`Call`.

    Returns:
        :class:`JudgeVerdict`. ``label`` is one of
        :data:`M4_ALLOWED_LABELS` on success; ``fell_back=True``
        when the model emitted a disallowed label or the invocation
        errored. The caller decides how to respond — typically:

        * ``complete`` / ``exhausted`` / ``not_quantified`` → no
          action; pass through.
        * ``partial`` → force a re-write iteration with a "you
          returned fewer items than requested without acknowledging
          exhaustion" directive.
        * ``fell_back=True`` → conservative path; treat as
          ``not_quantified`` to avoid loop on bad verdicts.
    """
    verdict = await judge_with_rubric(
        rubric=M4_COMPLETENESS_RUBRIC,
        input_text=user_prompt,
        context=response_text,
        model=model,
        allowed_labels=M4_ALLOWED_LABELS,
    )
    # Same observability discipline as M2/M3 — INFO on trusted verdict,
    # WARNING on fell_back, so operators see every M4 firing without
    # grepping memory.json.
    log_verdict(
        logger,
        "M4",
        verdict,
        model=model,
        char_counts={
            "user_prompt_chars": len(user_prompt),
            "response_chars": len(response_text),
        },
    )
    return verdict


__all__ = [
    "M4_ALLOWED_LABELS",
    "M4_COMPLETENESS_RUBRIC",
    "judge_completeness",
]
