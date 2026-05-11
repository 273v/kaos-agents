"""Reflexion / self-critique loop wrapper.

Wraps any inner pattern with a critique-and-retry loop. After each
iteration the critic looks at the inner pattern's output and either:

- Approves it (score ≥ threshold) → return immediately.
- Critiques it → the inner pattern retries with the critique
  appended to its prompt as ``extra_instruction``.

Generic across chat / plan / research. Drops the hardcoded
"one-shot rag-query → ReAct on insufficient evidence" branch in
``ResearchAgent`` — that's a special case of this pattern with a
fixed critic.

Distinct from :class:`~kaos_agents.runtime.invariants.StepInvariant`:

- ``StepInvariant`` is deterministic (predicate fires Y/N based on
  output text alone). Cheap, but limited expressiveness.
- ``ReflexionLoop`` uses an LLM critic with a scoring rubric. Costs
  one LLM call per iteration but can reason about content quality
  ("does this answer adequately address every part of the question?").

Use the cheap one when you can; reach for this when the quality
signal isn't expressible as a Python predicate.

Architectural shape: ``ReflexionLoop`` is a typed wrapper, not a
pattern enum value. It composes with any BaseAgent. Construct it
around an existing inner agent and call ``run()`` / ``turn()`` as
usual.

Example::

    inner = ChatAgent(model="anthropic:claude-haiku-4-5", ...)
    critic = ReflexionCritic(
        model="anthropic:claude-sonnet-4-6",
        rubric="The answer must cite at least one source and "
               "address every clause in the user's question.",
        threshold=0.7,
    )
    reflexive = ReflexionLoop(inner, critic, max_iterations=3)
    response = await reflexive.turn(message, session_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_agents.base.agent import KaosAgent
    from kaos_agents.types.response import AgentResponse

logger = get_logger(__name__)


_DEFAULT_CRITIC_MODEL: str = "anthropic:claude-sonnet-4-6"
_DEFAULT_MAX_ITERATIONS: int = 3
_DEFAULT_THRESHOLD: float = 0.7


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    """Outcome of a single critic evaluation.

    Attributes:
        score: 0..1 quality score the critic assigned.
        approved: True when ``score >= threshold``.
        feedback: Human-readable critique to inject into the next
            iteration's prompt. Empty when ``approved`` is True.
        reasoning: Short justification — surfaces in audit logs and
            replay diffs.
        cost_usd: USD cost of the critic LLM call. Defaults to 0.0
            for non-LLM critics; LLM critics route through
            ``Call.invoke()`` so ``Invocation.usage.cost_usd`` is
            available and aggregable into the wrapping ReflexionLoop's
            cost trace.
    """

    score: float
    approved: bool
    feedback: str
    reasoning: str
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class ReflexionTrace:
    """Trace of every iteration in a Reflexion run.

    The :class:`ReflexionLoop` returns the final ``AgentResponse``
    augmented with this trace in its ``metadata["reflexion_trace"]``
    so callers can audit how many iterations ran and what the critic
    said at each step.
    """

    iterations: tuple[CritiqueResult, ...] = field(default_factory=tuple)
    final_iteration: int = 0
    """1-based iteration that produced the accepted output. 0 if
    the loop hit max_iterations without approval."""
    accepted: bool = False
    """True when the critic ultimately approved an iteration. False
    when ``max_iterations`` was reached without approval — caller
    gets the best-scored attempt as the final response."""


class ReflexionCritic:
    """The critic side of a Reflexion loop.

    Given a question + the inner pattern's output, returns a
    :class:`CritiqueResult`. The reference implementation uses one
    LLM call via a typed Signature; callers can subclass to implement
    deterministic or composite critics.

    Args:
        model: LLM model for the critique call. Defaults to
            ``anthropic:claude-sonnet-4-6`` — the critic should be
            at least as strong as the inner pattern's model.
        rubric: Free-form description of what the output must
            satisfy. Becomes the system instruction for the critic
            LLM. Be specific: "must cite at least one source AND
            address every clause" beats "must be good."
        threshold: Score in ``[0, 1]`` at or above which the critic
            approves. Default 0.7.
    """

    __slots__ = ("model", "rubric", "threshold")

    def __init__(
        self,
        *,
        rubric: str,
        model: str = _DEFAULT_CRITIC_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        if not rubric or not rubric.strip():
            raise ValueError(
                "ReflexionCritic.rubric must be non-empty. Describe what "
                "the output must satisfy in 1-3 sentences."
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"ReflexionCritic.threshold must be in [0, 1], got {threshold!r}.")
        self.model = model
        self.rubric = rubric
        self.threshold = threshold

    async def critique(
        self,
        question: str,
        output: str,
    ) -> CritiqueResult:
        """Run the critique LLM call.

        Lazy-imports kaos-llm-core so the agent module stays
        importable without the ``[llm]`` extra.
        """
        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.signatures.fields import InputField, OutputField
        from kaos_llm_core.signatures.signature import Signature

        class _ReflexionCritiqueSignature(Signature):
            """Score an agent's output against a rubric."""

            question: str = InputField(description="The original user question.")
            rubric: str = InputField(description="The quality criteria.")
            candidate_output: str = InputField(
                description="The agent's output to evaluate.",
            )
            score: float = OutputField(
                description="Score 0.0..1.0. Avoid 0.5 unless ambiguous.",
            )
            feedback: str = OutputField(
                description=(
                    "Specific, actionable critique. Empty string when the "
                    "output is fully satisfactory. Otherwise: 1-3 sentences "
                    "of what to fix, phrased as instructions for the next "
                    "iteration."
                ),
            )
            reasoning: str = OutputField(
                description="One-sentence justification for the score.",
            )

        call = Call(_ReflexionCritiqueSignature, model=self.model)
        # KC9: use .invoke() instead of bare __call__ so Invocation.usage
        # is available — otherwise the critic's tokens + cost are dropped.
        invocation = await call.invoke(
            question=question,
            rubric=self.rubric,
            candidate_output=output,
        )
        result = invocation.output
        cost = float(getattr(invocation.usage, "cost_usd", 0.0) or 0.0)
        score = float(result.score)
        # Clamp defensively — a poorly-behaved model could emit > 1 or < 0.
        score = max(0.0, min(1.0, score))
        approved = score >= self.threshold
        return CritiqueResult(
            score=score,
            approved=approved,
            feedback=str(result.feedback) if not approved else "",
            reasoning=str(result.reasoning),
            cost_usd=cost,
        )


class ReflexionLoop:
    """Wraps an inner ``KaosAgent`` with critic-and-retry semantics.

    Each iteration:
      1. Inner agent produces an :class:`AgentResponse` for the
         message (+ accumulated feedback in ``extra_instruction``).
      2. Critic scores the response's text.
      3. If approved (score ≥ threshold) or we've hit
         ``max_iterations``, return the response (with trace).
      4. Otherwise append the critique to ``extra_instruction`` and
         loop.

    Args:
        inner: The wrapped agent (any ``KaosAgent`` instance).
        critic: A :class:`ReflexionCritic` to score each iteration.
        max_iterations: Cap on iterations. Default 3 — past this the
            cost typically exceeds the value of further refinement.

    The accepted response is the iteration the critic approved.
    If no iteration is approved by ``max_iterations``, the best-
    scored iteration is returned with ``trace.accepted=False`` so
    callers can detect partial success.
    """

    __slots__ = ("critic", "inner", "max_iterations")

    def __init__(
        self,
        inner: KaosAgent,
        critic: ReflexionCritic,
        *,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                f"ReflexionLoop.max_iterations must be >= 1, got {max_iterations}. "
                "Use 1 to disable the loop (single inner.turn() call with no critique)."
            )
        self.inner = inner
        self.critic = critic
        self.max_iterations = max_iterations

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        """Run the reflexion loop until approved or max iterations.

        Returns the inner agent's final ``AgentResponse``. The
        ``metadata`` dict carries a ``reflexion_trace`` entry with
        the full iteration history.
        """
        best_response: AgentResponse | None = None
        best_score: float = -1.0
        iterations: list[CritiqueResult] = []
        accumulated_feedback: list[str] = []

        for attempt in range(1, self.max_iterations + 1):
            # Compose any prior critique into the prompt for this iteration.
            # The first iteration sees no feedback; subsequent ones see
            # all previous critique messages concatenated.
            extra = _format_feedback(accumulated_feedback)
            response = await _call_inner(self.inner, message, session_id, extra)

            answer_text = response.text or ""
            critique = await self.critic.critique(message, answer_text)
            iterations.append(critique)

            logger.debug(
                "reflexion: iter=%d score=%.3f approved=%s",
                attempt,
                critique.score,
                critique.approved,
            )

            if critique.score > best_score:
                best_score = critique.score
                best_response = response

            if critique.approved:
                trace = ReflexionTrace(
                    iterations=tuple(iterations),
                    final_iteration=attempt,
                    accepted=True,
                )
                return _attach_trace(response, trace)

            accumulated_feedback.append(critique.feedback)

        # All iterations exhausted without approval — return best.
        trace = ReflexionTrace(
            iterations=tuple(iterations),
            final_iteration=self.max_iterations,
            accepted=False,
        )
        # best_response is guaranteed non-None when max_iterations >= 1
        assert best_response is not None
        return _attach_trace(best_response, trace)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_feedback(feedback_history: list[str]) -> str:
    """Build the ``extra_instruction`` text from prior critique messages."""
    nonempty = [f for f in feedback_history if f and f.strip()]
    if not nonempty:
        return ""
    parts = [
        "The previous iteration(s) of this response received the following "
        "critique. Address every point in your next response."
    ]
    for i, feedback in enumerate(nonempty, start=1):
        parts.append(f"  Iteration {i} critique: {feedback}")
    return "\n".join(parts)


async def _call_inner(
    inner: KaosAgent,
    message: str,
    session_id: str,
    extra_instruction: str,
) -> AgentResponse:
    """Call the inner agent's ``turn``, prepending the critic feedback
    to the user message itself.

    The :class:`KaosAgent` ABC has ``turn(message, session_id)`` —
    no extra-instruction kwarg, and silently swallowing the feedback
    via a TypeError fallback (the pre-fix behaviour) defeats the
    whole point of the reflexion loop: the captured telemetry on
    commit ``25b7d6f`` showed identical inner responses across all
    three retry iterations because every iteration received the
    *same* message with the critique nowhere in scope.

    Prepending is the universal-fit choice — it works against any
    :class:`KaosAgent` implementation without requiring the inner to
    opt in. The user's original message stays verbatim after a
    clearly-delimited ``=== CRITIQUE FROM PRIOR ITERATION ===``
    block; well-behaved instruction-tuned models honour the critique
    and patch their next response.

    Trade-off: the prepended critique enters the inner's
    conversation history too. For BaseAgent this means the message
    section accumulates "CRITIQUE / ORIGINAL" wrappers across
    iterations. That's acceptable for the reflexion use case — the
    loop owns the inner's session for the duration of its work and
    the bloat is bounded by ``max_iterations``.
    """
    if not extra_instruction:
        return await inner.turn(message, session_id)

    wrapped = (
        "=== CRITIQUE FROM PRIOR ITERATION ===\n"
        f"{extra_instruction}\n"
        "=== END CRITIQUE ===\n\n"
        "=== ORIGINAL REQUEST ===\n"
        f"{message}\n"
        "=== END REQUEST ===\n\n"
        "Address every point in the critique. Quote the source where the "
        "rubric requires it. Then answer the original request."
    )
    return await inner.turn(wrapped, session_id)


def _attach_trace(response: AgentResponse, trace: ReflexionTrace) -> AgentResponse:
    """Return a new response with the trace attached to metadata.

    ``AgentResponse`` is a frozen dataclass with
    ``metadata: tuple[tuple[str, Any], ...]``. We use
    ``dataclasses.replace`` to build a new instance with the trace
    payload appended as a new ``("reflexion_trace", ...)`` entry.
    """
    import dataclasses

    trace_payload = {
        "final_iteration": trace.final_iteration,
        "accepted": trace.accepted,
        "iterations": [
            {
                "score": it.score,
                "approved": it.approved,
                "reasoning": it.reasoning,
            }
            for it in trace.iterations
        ],
    }
    existing = response.metadata or ()
    # Strip any prior reflexion_trace entry so re-runs don't accumulate.
    filtered = tuple(kv for kv in existing if kv[0] != "reflexion_trace")
    new_metadata = (*filtered, ("reflexion_trace", trace_payload))
    try:
        return dataclasses.replace(response, metadata=new_metadata)
    except (TypeError, ValueError):
        # Defensive — surface the trace in logs if replace fails for
        # some non-standard subclass.
        logger.warning(
            "reflexion: cannot attach trace to %s; trace=%r",
            type(response).__name__,
            trace,
        )
        return response


__all__ = [
    "CritiqueResult",
    "ReflexionCritic",
    "ReflexionLoop",
    "ReflexionTrace",
]
