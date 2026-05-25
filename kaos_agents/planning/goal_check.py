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
import re
import time
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


_BASELINE_GOAL_CHECK_MODEL = "anthropic:claude-haiku-4-5"


# ─── Deterministic pre-check: header-then-stop ───────────────────────
#
# 2026-05-19 P0-4 #436: NDA matrix Persona #4 emitted a markdown
# heading naming the deliverable AND ended the turn before any body.
# This is the same failure family as preamble-and-quit (P0-1 #437)
# but with the header alone in place of the preamble alone. The LLM-
# based critic catches the long tail via prompt; this regex catches
# the easy cases deterministically (no extra cost, no extra latency).

_DELIVERABLE_KEYWORDS = (
    "table",
    "list",
    "summary",
    "comparison",
    "csv",
    "memo",
    "scorecard",
    "review",
    "breakdown",
    "matrix",
    "checklist",
    "rubric",
)

# A markdown heading at the start of a line:
#   ATX:   #, ##, ###, ####  (up to 6 hashes)
#   Setext: text followed by === or --- on the next line (rare in LLM output, skip)
# Capture text after the hashes.
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")

# Heuristic for "the heading promised a body": any of our deliverable
# keywords appearing anywhere in the heading text (case-insensitive).
_DELIVERABLE_HEADING_RE = re.compile(
    r"\b(" + "|".join(_DELIVERABLE_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)


def _detect_header_then_stop(agent_response: str) -> str | None:
    """Return a remediation hint if the response ends near a deliverable heading.

    Returns ``None`` when the response looks fine (no trailing
    deliverable heading, OR a heading with substantive body content
    after it). Returns the human-readable issue summary when the
    response ended right after a heading whose text named a
    deliverable — i.e. the "header-then-stop" failure shape.

    "Substantive body" is defined defensively: at least 80 characters
    of non-whitespace, non-heading text after the offending heading.
    A trailing line like ``(see above)`` or a single-line note does
    NOT count — the heading should be followed by the rows / bullets /
    paragraph it names.

    This is a pre-LLM check; the loose threshold is intentional. The
    Critic's LLM-based docstring rule catches the long tail.
    """
    if not agent_response or not agent_response.strip():
        return None

    # Walk lines bottom-up; find the LAST heading line, then assess
    # what (if anything) follows it.
    lines = agent_response.splitlines()
    last_heading_idx: int | None = None
    last_heading_text: str = ""
    for idx in range(len(lines) - 1, -1, -1):
        m = _HEADING_LINE_RE.match(lines[idx])
        if m:
            last_heading_idx = idx
            last_heading_text = m.group(1).strip()
            break

    if last_heading_idx is None:
        return None

    if not _DELIVERABLE_HEADING_RE.search(last_heading_text):
        return None

    # How much real content follows that heading? Strip whitespace +
    # ignore lines that are themselves headings (a heading following a
    # heading is still header-only territory).
    tail = lines[last_heading_idx + 1 :]
    body_chars = 0
    for line in tail:
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_LINE_RE.match(line):
            # Sub-heading without body — keep accumulating; the body
            # should be UNDER the deepest heading.
            continue
        body_chars += len(stripped)

    if body_chars >= 80:
        return None

    return (
        f"You wrote the heading '{last_heading_text}' but did not "
        "emit the body the heading promised. Re-run the turn and "
        "produce the rows / bullets / paragraph that belong under "
        "that heading. Do not stop on a header alone."
    )


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
          - **Structural identifiers must come from tool data.** When
            the answer cites a position in a document — "Section 7",
            "page 12", "paragraph (b)", "Article III", "slide 4",
            "row 3" — that identifier MUST appear in a successful
            tool result this turn (``tool_calls_made`` entry with
            ``is_error=false``, or a structured-content ``path``
            field carrying the identifier). If no tool result
            supplies the identifier, the agent invented it; return
            ``needs_more_work`` with ``next_action`` = "cite by the
            heading text or quoted clause; do not invent section
            numbers". This is the closure for the 2026-05-18 NDA
            persona run where every "Section N" citation was
            confidently wrong despite verbatim clause text being
            right. Quoted clause text without a section identifier
            is acceptable; invented identifiers are not.
          - **Speculative comparative qualifiers.** If the answer
            characterizes a finding with a comparative qualifier
            (``standard``, ``unusual``, ``typical``, ``weird``,
            ``common``, ``rare``, ``normal``, ``atypical``) AND no
            tool result in this turn supplies the baseline that
            qualifier is comparing against, the assertion is
            speculation. Return ``needs_more_work`` with
            ``next_action`` = "remove the comparative qualifier or
            cite the baseline measurement". The 2026-05-18 persona
            run had an agent assert ``Michigan governing law is
            unusual for 273 Ventures`` when 273V is itself a
            Michigan LLC and no tool was called to measure norms.
          - **Deliverable-header-then-stop.** If the user asked for a
            named deliverable (table / list / summary / comparison /
            CSV / memo / scorecard / review / breakdown) AND the
            agent's response ends near a markdown header whose text
            matches that deliverable keyword (``## Table``, ``# CSV-
            ready table``, ``### Summary``, ``## Governing-Law
            Review``, etc.) without the actual body that the heading
            promised, return ``needs_more_work`` with ``next_action``
            = "emit the deliverable body under the heading you wrote;
            do not stop on a header alone". The 2026-05-19 NDA
            persona run had an agent emit
            ``<h2>Governing-Law Review — 5 NDAs</h2>`` plus a one-
            sentence preamble plus ``<h3>CSV-ready table</h3>`` and
            then stop — zero rows. Empty headers are the same failure
            shape as the preamble-and-quit pattern. A deterministic
            pre-check in :func:`check_goal` will also catch the
            common cases; this rule covers the long-tail wording.
          - **Announce-and-quit (future-tense promise without
            execution).** If the agent's response contains a
            future-tense first-person promise to do research —
            phrases like ``I'll now research``, ``I'll search``,
            ``I'll look that up``, ``I'll dispatch tools``, ``let
            me investigate``, ``I'll report back``, ``I'll
            continue and pull``, or a "next steps" list framed as
            things the agent will do — AND ``tool_calls_made``
            for THIS iteration contains zero ``is_error=false``
            entries that actually executed that research, the
            agent has announced work instead of doing it. Return
            ``needs_more_work`` with ``next_action`` = "execute
            the research you promised in the same turn — call FR
            / eCFR / EDGAR / GovInfo / web-search now and produce
            the answer with citations". Same failure family as
            deliverable-header-then-stop: a promise to act is not
            an act. The 2026-05-19 diesel-reproduction shipped
            ``I'll now research the latest applicable federal
            diesel emissions rule and report back with
            citations.`` with zero tool calls — that is the
            canonical case.
          - **Claimed-fetch fabrication.** If the agent's response
            contains any of the phrases ``I fetched``, ``I retrieved``,
            ``I reviewed``, ``I was able to extract``, ``I pulled``,
            ``I read``, ``I downloaded``, ``I opened`` (or
            morphological variants — past-tense first-person verbs
            that claim direct retrieval of a named external resource),
            AND no entry in ``tool_calls_made`` shows a successful
            fetch / retrieval of THAT specific resource this turn
            (``is_error=false``, and the args clearly target the
            named URL / document / filing), the claim is fabricated.
            Return ``needs_more_work`` with ``next_action`` = "drop
            the claim that you fetched / reviewed sources you did
            not actually fetch this turn; either call the fetch tool
            now or rephrase to say you have NOT read the source".
            This is the 2026-05-19 SEC-Climate session where the
            agent claimed it ``fetched and reviewed two practitioner
            commentary pages`` (Cadwalader memo, Ropes & Gray
            viewpoints) when the only fetch in the trace targeted a
            different URL and errored. Asserting first-person
            retrieval of a source the trace doesn't contain is a
            P0 honesty failure on attorney-facing surfaces.
          - **Factual-external-entity question with zero successful
            tool calls.** If the user asked about a *factual external
            entity* — a regulation, statute, case, agency rule,
            public filing, public-company fact, current date / event
            / market / policy state — AND
            ``tool_calls_made`` contains zero successful entries
            (no ``is_error=false`` row), the agent is answering
            from training memory. Return ``needs_more_work`` with
            ``next_action`` = "call the appropriate research tool
            (FR / eCFR / EDGAR / GovInfo / web-search / corpus
            search) and re-answer with citations; do not state
            specific facts about [topic] from training memory".
            Generalizes the confident-hallucination shortcut: even
            "soft" factual claims (current status, latest version,
            present tense regulatory regime) must be tool-grounded.
            The 2026-05-19 ``what is the latest diesel emission
            reg`` session is the canonical case: 9 turns of
            clarification followed by a memory-only answer with
            zero tool calls. Counter-cases (NOT this rule):
            definitions of stable concepts, arithmetic, language
            tasks, summarization of already-quoted text.
            **This rule does NOT apply when the
            ``Anaphoric follow-up recall IS grounded`` OR
            ``Pure deterministic computation IS grounded``
            carve-outs above apply.** Examples that fall under
            the carve-outs (return ``satisfied`` instead): the
            user's question is "in one sentence, what was the X
            you just cited?" (follow-up recall); the user's
            question supplies BOTH the rule citation AND the
            inputs and asks for the deterministic computation
            (e.g., "if a complaint was filed March 15 and FRCP
            12(a) gives 21 days, what is the deadline?" — the
            user supplied ``FRCP 12(a) gives 21 days`` AND
            ``March 15``; the agent's job is calendar math, not
            re-fetching FRCP). Calling FR / eCFR to re-verify a
            rule the user already named in the prompt is wasted
            cost — confidence the agent should have in
            user-supplied premises is the same as confidence in
            user-supplied facts.
          - **Clarification-loop ceiling.** If ``iteration`` >= 2
            AND the agent's response is *another* clarification
            question (request for more information, choice between
            options, "do you mean X or Y?"), the agent is stuck.
            Return ``needs_more_work`` with ``next_action`` =
            "stop asking for clarification; pick the strongest
            reading of the user's request and call the relevant
            research tool now". Two rounds of clarification is the
            ceiling on any factual question.
          - **Domain-conventional shorthand is not ambiguity.** If
            the user's message used a domain-conventional
            abbreviation in context (``GL`` in contracts → governing
            law; ``DD`` in deals → due diligence; ``RFI`` in
            procurement → request for information; ``10-K`` in
            finance → annual filing) and the agent's response is a
            clarification request instead of an answer, return
            ``needs_more_work`` with ``next_action`` = "interpret
            the domain shorthand and answer; only ask for
            clarification when candidate readings are genuinely
            incompatible". Over-refusal to infer conventional
            domain language wastes the user's turn.

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


# ─── Few-shot demonstrations (ground the Signature) ──────────────────
#
# The labelled :class:`~kaos_llm_core.types.Example` pool lives in
# ``kaos_agents/_assets/examples/goal_check.toml`` — see
# :mod:`kaos_agents._examples` for the loader contract. Adding or
# removing examples = edit the TOML; optimizers may rewrite it without
# touching this module. The rubric docstring stays short + abstract;
# the per-incident evidence is data, not code.


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

    # 2026-05-19 P0-4 #436: deterministic pre-check.
    # If the agent's response ends near a heading naming a deliverable
    # with no body, short-circuit to needs_more_work without spending
    # an LLM call. The Critic Signature still covers the long-tail
    # wording the regex can't recognize.
    pre_check_hint = _detect_header_then_stop(agent_response)
    if pre_check_hint is not None:
        latency_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "GoalChecker: header-then-stop deterministic pre-check fired; "
            "skipping LLM call. hint=%s",
            pre_check_hint,
        )
        return GoalCheckOutcome(
            result=GoalCheckNeedsMoreWork(
                next_action=pre_check_hint,
                confidence=1.0,
                rationale=(
                    "Deliverable heading was emitted without the body "
                    "the heading names. Re-run to produce the body."
                ),
            ),
            cost_usd=0.0,
            latency_ms=latency_ms,
            iteration=iteration,
        )

    try:
        from kaos_llm_core import Call

        from kaos_agents._examples import load_examples

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

    call = Call(  # ty: ignore[invalid-argument-type]
        signature,
        model=used_model,
        examples=load_examples("goal_check"),
    )

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
