"""GoalChecker — "did the agent satisfy the user's question?"

Per `kaos-modules/docs/internal/agentic-loop-auto-elevation-plan.md`
§3.2. The Critic node of the Plan + ReAct + Critic pattern. Drives
:func:`AgenticLoop`'s replan-or-return decision after each ReAct
iteration.

Modeled on:

- Everlaw Deep Dive's "insufficient_evidence" gold-standard output
  (`competitive/capabilities/18-refuses-when-uncertain.md`).
- Anthropic's "constitutional AI" critique pattern.
- LangChain's `EvaluatorChain` + Pydantic AI's `result_type` discriminated
  union.

Output is a **three-way discriminated union** instead of a binary
``satisfied: bool``:

- ``satisfied`` — the response answers the user's question. Loop
  returns to user.
- ``needs_more_work`` — the response is incomplete but the agent
  could continue (specific `next_action` provided). Loop replans
  with `next_action` threaded as an agent-internal thinking block.
- ``insufficient_evidence`` — the response cannot be improved with
  more iterations (the corpus / web / etc. genuinely lacks the
  information). Loop returns to user with an explicit "I looked
  here, didn't find it" framing.

Three discrete outcomes → three discrete UX paths (the SPA's
``GoalCheckBadge`` has three states matching what users in the
legal-AI market already expect).

The checker is a cheap Haiku-class call (~$0.0001 per check); the
loop pays for one of these per iteration. Loop budget is bounded by
``SessionPolicy.max_loop_cost_usd`` so worst-case spend on N
iterations is capped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


_BASELINE_GOAL_CHECK_MODEL = "anthropic:claude-haiku-4-5"


def _resolve_goal_check_model() -> str:
    """Read the goal-check model from settings at call time.

    Honors ``KAOS_AGENT_GOAL_CHECK_MODEL`` env override; falls back to
    ``KAOS_AGENT_PLANNING_LLM_MODEL`` (the per-turn planner model);
    falls back to Haiku as a last resort.
    """
    from kaos_agents.settings import KaosAgentSettings

    settings = KaosAgentSettings()
    return (
        getattr(settings, "goal_check_model", None)
        or getattr(settings, "planning_llm_model", None)
        or _BASELINE_GOAL_CHECK_MODEL
    )


# ─── Discriminated union output (three-way) ──────────────────────────


class GoalCheckSatisfied(BaseModel):
    """The agent's response answers the user's question.

    Loop returns to user; SPA renders a green GoalCheckBadge.
    """

    kind: Literal["satisfied"] = "satisfied"
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Critic's self-rated confidence in this verdict.",
    )
    rationale: str = Field(description="One short sentence the user will see in the badge.")


class GoalCheckNeedsMoreWork(BaseModel):
    """The response is incomplete but the agent could continue.

    Loop replans; SPA renders an amber GoalCheckBadge + the
    next_action as an inline "next: ..." chip. The next_action gets
    threaded into the next iteration's agent context as a thinking
    block — NOT as a fake user message (preserves transcript hygiene
    per design doc §7).
    """

    kind: Literal["needs_more_work"] = "needs_more_work"
    next_action: str = Field(
        description=(
            "Imperative one-liner of what the agent should try next "
            "('search SCOTUS directly for the case', 'parse the uploaded "
            "PDF for the answer', 'request the web tool group')."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One sentence the user will see.")


class GoalCheckInsufficientEvidence(BaseModel):
    """The corpus genuinely lacks the information needed to answer.

    Loop returns to user with an "I looked, didn't find it" framing;
    SPA renders a gray GoalCheckBadge (NOT red — refusal is a feature,
    not a failure). Matches Everlaw Deep Dive's gold-standard refusal
    UX from the competitive doc §18.
    """

    kind: Literal["insufficient_evidence"] = "insufficient_evidence"
    missing: str = Field(
        description=(
            "What's missing from the available corpus / web / tools. "
            "Example: 'no Delaware case law on this exact fact "
            "pattern in the available connectors'."
        )
    )
    rationale: str = Field(description="One sentence the user will see in the gray badge.")


# Pydantic discriminated union — the LLM emits exactly one shape via
# the ``kind`` discriminator. The Critic Signature's output field uses
# this union type, so codec validation is automatic.
GoalCheckResult = Annotated[
    GoalCheckSatisfied | GoalCheckNeedsMoreWork | GoalCheckInsufficientEvidence,
    Field(discriminator="kind"),
]


# ─── Lightweight per-turn record (for the AgenticLoop's event stream) ─


@dataclass(frozen=True, slots=True)
class GoalCheckOutcome:
    """The Critic's verdict + cost + latency observability fields.

    The AgenticLoop wraps the LLM's :data:`GoalCheckResult` in this
    outer record so the chat router can emit a ``GoalChecked`` SSE
    event with the cost + latency fields the SPA's CostStrip wants.
    """

    result: GoalCheckResult
    cost_usd: float
    latency_ms: float
    iteration: int

    @property
    def kind(self) -> str:
        return self.result.kind

    @property
    def satisfied(self) -> bool:
        return self.result.kind == "satisfied"

    @property
    def needs_more_work(self) -> bool:
        return self.result.kind == "needs_more_work"

    @property
    def insufficient_evidence(self) -> bool:
        return self.result.kind == "insufficient_evidence"

    @property
    def is_terminal(self) -> bool:
        """True when the loop should stop (satisfied OR insufficient)."""
        return self.result.kind in ("satisfied", "insufficient_evidence")


# ─── Signature (lazy under [llm] extra) ──────────────────────────────


def _build_signature_class() -> type:
    """Lazy-build the Critic Signature under the [llm] extra.

    See ``kaos_agents.planning.policy._build_signature_class`` for the
    rationale — kaos_llm_core is optional, so we don't import it at
    module load.
    """
    from kaos_llm_core import InputField, OutputField, Signature

    class _GoalCheckerSignature(Signature):
        """Decide whether the agent's last response satisfies the user.

        Rules:
          - Output exactly one of three shapes via the ``kind`` field:
            ``satisfied`` / ``needs_more_work`` / ``insufficient_evidence``.
          - Be honest: the loop trusts you. False "satisfied" silently
            ships a bad answer; false "needs_more_work" wastes a turn.
            Asymmetric — prefer ``needs_more_work`` over ``satisfied``
            on close calls.
          - Refusal is a feature, not a failure: ``insufficient_evidence``
            is the right answer when the corpus genuinely lacks the
            information. Do NOT keep iterating on a question the
            available tools can't answer.

        Concrete shortcuts:
          - Agent's response says "I can't / I don't have / sorry" →
            almost certainly NOT satisfied. If the missing tool is in
            ``available_groups`` but not ``elevation_trail``, the
            next action is "request capability X" → ``needs_more_work``.
            If the missing tool is outside the soft ceiling →
            ``insufficient_evidence``.
          - Agent's response answered the question with concrete facts +
            at least one successful tool call → ``satisfied`` (unless
            the question demanded a multi-step output and the agent
            only did step 1).
          - Agent's response is short + generic + no tool calls →
            ``needs_more_work`` (the agent has more it could do).
          - Agent's response cites tool results (block_refs, URLs,
            source spans) → strong signal for ``satisfied``.
          - **Confident-hallucination shortcut (highest-impact case).**
            Agent's response asserts a specific person's identity,
            current role/title, recent date, price, legal status, or
            any other public-record fact, AND ``tool_calls_made`` is
            empty (no successful tool call produced evidence) →
            ``needs_more_work`` with ``next_action`` = "search the
            web for [the asserted fact] before answering". Confident
            hallucination of look-up-able facts is the single highest-
            impact failure this critic catches; trust the absence of
            tool calls more than the model's confidence in the prose.
            (Counter-cases that are NOT hallucination: pure definition
            requests like "what is JSON Schema", arithmetic, language
            tasks, summarization of an already-quoted text — none of
            those need a tool call.)

        Cross-iteration signals (when present):
          - If ``elevation_trail`` shows groups were auto-enabled this
            turn but the agent STILL didn't answer, that's a strong
            ``needs_more_work`` signal — the agent under-used the new
            capabilities.
          - If ``iteration`` is already 2+ and the agent is still in
            the same mode (apologizing / repeating), prefer
            ``insufficient_evidence`` over another ``needs_more_work``
            — diminishing returns.
        """

        user_message: str = InputField(description="The user's original question or instruction.")
        agent_response: str = InputField(
            description="The agent's reply this iteration (concatenated text)."
        )
        tool_calls_made: list[dict] = InputField(
            description=(
                "Every tool call this iteration as a list of "
                "{name, is_error, summary_excerpt}. Empty list means "
                "the agent answered without tools."
            )
        )
        elevation_trail: list[str] = InputField(
            description=(
                "Groups auto-elevated this turn (e.g., ['web']). "
                "Empty when no elevation. Helps the critic recognize "
                "'agent had to elevate to answer — that's fine, not "
                "a refusal'."
            )
        )
        available_groups: list[str] = InputField(
            description=(
                "Every group registered in the runtime, for context. "
                "If the agent gave up but a relevant group is "
                "registered, ``needs_more_work``."
            )
        )
        iteration: int = InputField(description="Current iteration number (1-indexed).")
        result: GoalCheckResult = OutputField(
            description=(
                "Three-way verdict: satisfied / needs_more_work / "
                "insufficient_evidence. See class docstring for the "
                "decision rules."
            )
        )

    return _GoalCheckerSignature


_SIGNATURE_CACHE: type | None = None


def _get_signature() -> type:
    global _SIGNATURE_CACHE
    if _SIGNATURE_CACHE is None:
        _SIGNATURE_CACHE = _build_signature_class()
    return _SIGNATURE_CACHE


# ─── Public entrypoint ───────────────────────────────────────────────


async def check_goal(
    *,
    user_message: str,
    agent_response: str,
    tool_calls_made: list[dict] | None = None,
    elevation_trail: list[str] | None = None,
    available_groups: list[str] | None = None,
    iteration: int = 1,
    model: str | None = None,
) -> GoalCheckOutcome:
    """Run the Critic and return a :class:`GoalCheckOutcome`.

    Best-effort. On any exception (provider error, missing ``[llm]``
    extra, parser failure) returns ``needs_more_work`` with a
    diagnostic rationale so the loop has a chance to recover — NEVER
    returns ``satisfied`` on error (false satisfaction silently ships a
    bad answer).
    """
    used_model = model or _resolve_goal_check_model()
    t_start = time.monotonic()

    try:
        from kaos_llm_core import Call

        signature = _get_signature()
    except ImportError as exc:
        latency_ms = (time.monotonic() - t_start) * 1000
        logger.warning(
            "GoalChecker: kaos-llm-core not installed; defaulting to needs_more_work. err=%s",
            exc,
        )
        return GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="(GoalChecker unavailable — review the answer manually)",
                confidence=0.0,
                rationale="Critic unavailable: kaos-llm-core not installed.",
            ),
            cost_usd=0.0,
            latency_ms=latency_ms,
            iteration=iteration,
        )

    call = Call(signature, model=used_model)  # ty: ignore[invalid-argument-type]

    try:
        invocation = await call.invoke(
            user_message=user_message,
            agent_response=agent_response,
            tool_calls_made=list(tool_calls_made or []),
            elevation_trail=list(elevation_trail or []),
            available_groups=list(available_groups or []),
            iteration=iteration,
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t_start) * 1000
        logger.warning("GoalChecker call failed; defaulting to needs_more_work. err=%s", exc)
        return GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action="(GoalChecker failed — review the answer manually)",
                confidence=0.0,
                rationale=f"Critic call failed: {exc}",
            ),
            cost_usd=0.0,
            latency_ms=latency_ms,
            iteration=iteration,
        )

    latency_ms = (time.monotonic() - t_start) * 1000
    cost_usd = float(getattr(invocation.usage, "cost_usd", 0.0) or 0.0)
    return GoalCheckOutcome(
        result=invocation.output.result,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        iteration=iteration,
    )


__all__ = [
    "GoalCheckInsufficientEvidence",
    "GoalCheckNeedsMoreWork",
    "GoalCheckOutcome",
    "GoalCheckResult",
    "GoalCheckSatisfied",
    "check_goal",
]
