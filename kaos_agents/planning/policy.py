"""TurnToolPolicy — dynamic per-turn tool-group narrowing.

Promoted from the kaos-ui single-user-chat example
(``examples/single-user-chat/backend/app/services/turn_tool_policy.py``)
into kaos-agents proper per PRD §4 (PR 2) so every consumer can share
the planner without duplicating the Signature.

**What the planner does.** Before each tool-able ReAct turn, the chat
router runs ``plan_turn_tool_policy`` to narrow the
:class:`SessionToolSet` ceiling to just the groups *this specific
message* needs. The chat router intersects the planner's output with
the user's ceiling and hands the result to the proxy filter.

**Why this exists.**

- Users set a generous ceiling once ("documents, citations, vfs, web,
  forensics, retrieval, browser") and want the agent to pick the
  right subset per turn — web for research, documents for upload
  questions, forensics for parsing tasks.
- Sending a full 100+ tool catalog to the LLM on every turn is
  wasteful; a narrowed list improves tool-selection accuracy and
  saves prompt-token spend.
- The planner is a Haiku-class LLM call. Target budget: ≤ $0.0002
  per turn, ≤ 300ms p95.

**Safety semantics.** When the planner is uncertain
(``confidence < threshold``), or the provider call fails, or the
planner emits a set that doesn't intersect the ceiling, we
**abdicate** and use the full ceiling. Refusing to narrow is always
safe; over-narrowing costs the user a wasted turn.

**v2 Signature inputs** (PRD round-2 decision #7 + completion-plan §3
Stage A):

- ``user_message``, ``recent_turns``, ``corpus_headlines`` — existing
- ``corpus_kinds: list[str]`` — Magika-style content classification
  for uploaded files (``["pdf", "spreadsheet", "html"]``)
- ``session_intent: str | None`` — preset chip selection
  (``"research"``, ``"drafting"``, ``"forensics"``) or None
- ``raw_turn_groups: list[str] | None`` — last turn's wanted-groups
  set (pre-intersection), for cross-turn coherence
- ``ceiling_groups``, ``available_groups`` — existing

**v2 output.** The planner produces ``wanted_groups`` unconstrained.
The entrypoint computes ``kept_groups = wanted ∩ ceiling`` and
``dropped_groups = wanted - ceiling`` so the SPA UI can show
"wanted but blocked" badges for the latter.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


_BASELINE_PLANNER_MODEL = "anthropic:claude-haiku-4-5"
_DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def _resolve_planner_model() -> str:
    """Read the planner model from KaosAgentSettings at call time.

    Honors the ``KAOS_AGENT_PLANNING_LLM_MODEL`` env override.
    """
    from kaos_agents.settings import KaosAgentSettings

    return KaosAgentSettings().planning_llm_model


def _resolve_confidence_threshold() -> float:
    """Below this confidence the planner abdicates and the caller uses
    the full ceiling. Defaults to 0.6 — tuned in the prompt examples so
    high-recall scenarios (uploaded files + ambiguous question) trip
    back to the ceiling rather than narrow."""
    from kaos_agents.settings import KaosAgentSettings

    return getattr(
        KaosAgentSettings(),
        "planning_confidence_threshold",
        _DEFAULT_CONFIDENCE_THRESHOLD,
    )


# ─── result types ────────────────────────────────────────────────────


class TurnToolPolicyResult(BaseModel):
    """Structured planner output. JSON-serializable for SSE wire."""

    wanted_groups: list[str] = Field(
        description=(
            "Tool group ids the planner thinks are needed for this turn. "
            "May include groups outside the ceiling — the caller filters."
        )
    )
    rationale: str = Field(
        description=(
            "One short sentence justifying the choice. Surfaced to the user "
            "as a transparency chip ('Picked web for the SEC search')."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Planner's self-rated [0, 1]. Below the caller-specified "
            "threshold (default 0.6) → fall back to the full ceiling."
        ),
    )


@dataclass(frozen=True, slots=True)
class TurnToolPolicy:
    """Per-turn narrowing decision returned to the chat router.

    Frozen + slotted to match kaos-llm-core's value-type conventions.

    Attributes:
        kept_groups: Planner's ``wanted_groups`` intersected with the
            session ceiling. This is the set the proxy filter uses
            for ReAct's tool catalogue.
        dropped_groups: ``wanted_groups - ceiling`` — what the
            planner wanted but the session denied. Surfaced in the
            SPA's CostStrip as "wanted but blocked" badges so users
            see why a turn underperformed.
        rationale: User-facing one-liner. Shown in the
            ToolPolicyBadge chip + the planner row of the CostStrip.
        confidence: Planner's self-rated confidence [0, 1].
        fell_back_to_ceiling: True when ``kept_groups == ceiling`` for
            either of two reasons: (a) ``confidence < threshold``, or
            (b) the planner's wanted set didn't intersect the ceiling.
        cost_usd: LLM cost of the planner call (Haiku-class).
        latency_ms: Wall-clock time of the planner call.
    """

    kept_groups: frozenset[str]
    dropped_groups: frozenset[str]
    rationale: str
    confidence: float
    fell_back_to_ceiling: bool
    cost_usd: float
    latency_ms: float

    @property
    def turn_groups(self) -> frozenset[str]:
        """Back-compat alias for the pre-promotion field name.

        The pre-promotion ``app.services.turn_tool_policy.TurnToolPolicy``
        used ``turn_groups``; callers migrating to the promoted module
        can keep the old attribute name pointed at ``kept_groups``
        without source changes. New code should prefer ``kept_groups``.
        """
        return self.kept_groups


# ─── signature ───────────────────────────────────────────────────────


def _build_signature_class() -> type:
    """Lazy-build the `Signature` subclass under the [llm] extra.

    `kaos_llm_core` is an optional dependency (see kaos-agents AGENTS.md).
    Building the Signature class touches `kaos_llm_core.Signature`,
    `InputField`, `OutputField` — all under the optional `[llm]` extra.
    We defer the import so consumers that don't use the planner don't
    pay the import cost (and don't crash on missing kaos-llm-core).
    """
    from kaos_llm_core import InputField, OutputField, Signature

    class _TurnToolPolicySignature(Signature):
        """Pick the tool groups that can answer the user's message.

        Rules:
          - Output ``wanted_groups`` is what you think would help, even if
            some are outside the ceiling. The caller filters to the
            allowed subset and reports the rest as "wanted but blocked".
          - When in doubt, prefer broader — false narrowing (dropping a
            group the agent actually needed) costs the user a wasted
            turn; false broadening costs at most a few cents of prompt
            tokens. Asymmetric.
          - ``confidence`` is your own self-rated [0, 1]. Output <0.6
            when the message is ambiguous, terse ("hi", "yes"), or
            could plausibly need any group.
          - ``rationale`` is one short sentence the user will see as a
            transparency chip ("Web for the SEC search").

        Group-selection shortcuts:
          - User mentions a website / government source / "search" /
            "live" / "lookup" / a recent event → likely needs ``web``.
          - User refers to an uploaded file / PDF / contract / "the
            document" → likely needs ``documents``.
          - User asks to draft / write / fill a template → likely
            needs ``authoring``.
          - User asks for citations / Bluebook / case law links →
            ``citations``.
          - User asks to inspect / hash / parse raw bytes (emails,
            archives, images, PACER) → ``forensics``.
          - User asks "find X in our docs" / "search the corpus" →
            ``retrieval``.
          - Short greeting / pure language question → empty
            ``wanted_groups`` + low confidence (caller falls back).

        Corpus-kinds hints (when present):
          - ``corpus_kinds`` includes ``"pdf"`` and the question references
            "the document" → favor ``documents``.
          - ``corpus_kinds`` includes ``"spreadsheet"`` → favor
            ``documents`` for parse-and-summarize, plus ``retrieval``
            if the question is "find X in the sheet".
          - ``corpus_kinds`` includes ``"email"`` / ``"archive"`` →
            favor ``forensics``.
          - **Lookup-beyond-corpus rule.** When ``corpus_headlines``
            is non-empty AND the question asks about facts that
            likely go beyond the attached files — names, current
            roles or titles, public records, recent events, prices,
            "who is X", "look up Y", "find the source for Z" —
            include BOTH ``documents`` AND ``web`` in
            ``wanted_groups``. The agent searches the docs first
            (cheaper, deterministic); if the answer isn't there, it
            already has web in scope to escalate without a replan.
            Symmetric: a doc-only ``wanted_groups`` on a public-fact
            lookup is the failure mode that produces "I need more
            information" or confident hallucination.

        Session-intent hints (when present):
          - ``"research"`` (default) → prefer ``web`` + ``documents`` +
            ``retrieval``; avoid ``authoring``.
          - ``"drafting"`` → ``authoring`` is in play; prefer it over
            pure read paths for "draft / write / redline" verbs.
          - ``"forensics"`` → tight ceiling; prefer ``forensics`` + ``vfs``
            and avoid broader research surfaces.

        Cross-turn coherence:
          - If ``raw_turn_groups`` from the prior turn is present and
            the user's message is a follow-up ("continue", "what
            about X", "now do Y"), bias toward the same groups for
            continuity unless the new message clearly signals a
            different direction.

        BM25 surface disambiguation (round-2 decision #6 — three
        BM25 tools live in the ``retrieval`` group; the planner only
        narrows to the *group*, but the agent then selects within the
        group at ReAct time using these distinctions):

          - ``kaos-source-bm25-search`` searches Free Law Project's
            corpus of case law + statutes (external knowledge).
          - ``kaos-nlp-core-bm25-search`` searches the *session
            memory* — past MESSAGES / ACTIONS / FINDINGS / DOCUMENTS
            (what the agent already learned this session).
          - ``kaos-retrieval-bm25`` delegates to the RetrievalAgent
            sub-agent (broader recall when the other two are too
            narrow).
        """

        user_message: str = InputField(description="The user message starting this turn.")
        recent_turns: str = InputField(
            description=(
                "Last 3-5 conversation turns compressed to one line each. "
                "Empty string when no history."
            )
        )
        corpus_headlines: str = InputField(
            description=(
                "One line per uploaded file: 'filename — size, content_type'. "
                "Empty string when nothing uploaded."
            )
        )
        corpus_kinds: list[str] = InputField(
            description=(
                "Coarse content-type labels for uploaded files (e.g. "
                '["pdf", "spreadsheet", "html"]). Sourced from '
                "kaos-nlp-core's content_type classifier. Empty list "
                "when nothing uploaded or all files are unclassified."
            )
        )
        session_intent: str | None = InputField(
            default=None,
            description=(
                "User-selected persona preset at session creation: "
                '"research", "drafting", or "forensics". None when the '
                "user hasn't picked a preset."
            ),
        )
        raw_turn_groups: list[str] | None = InputField(
            default=None,
            description=(
                "Last turn's `wanted_groups` (pre-intersection with "
                "ceiling). Provides cross-turn coherence for follow-up "
                "questions. None on the first turn."
            ),
        )
        ceiling_groups: list[str] = InputField(
            description=(
                "The session's allowed tool groups. Your `wanted_groups` "
                "output may include groups outside this set — the caller "
                "will filter and report the unmet wants to the UI."
            ),
        )
        available_groups: list[str] = InputField(
            description="All tool groups known to the runtime, for context."
        )
        policy: TurnToolPolicyResult = OutputField(
            description=(
                "Wanted tool-group selection for this turn, self-rated "
                "confidence, and a one-sentence rationale."
            )
        )

    return _TurnToolPolicySignature


_SIGNATURE_CACHE: type | None = None


def _get_signature() -> type:
    global _SIGNATURE_CACHE
    if _SIGNATURE_CACHE is None:
        _SIGNATURE_CACHE = _build_signature_class()
    return _SIGNATURE_CACHE


# ─── public entrypoint ───────────────────────────────────────────────


async def plan_turn_tool_policy(
    *,
    user_message: str,
    recent_turns: str = "",
    corpus_headlines: str = "",
    corpus_kinds: list[str] | None = None,
    session_intent: str | None = None,
    raw_turn_groups: list[str] | None = None,
    ceiling_groups: list[str],
    available_groups: list[str],
    model: str | None = None,
    confidence_threshold: float | None = None,
) -> TurnToolPolicy:
    """Run the planner and return a :class:`TurnToolPolicy`.

    Best-effort. On any exception (provider error, timeout, parser
    failure, missing ``[llm]`` extra) returns
    ``fell_back_to_ceiling=True`` with ``kept_groups = frozenset(ceiling_groups)``
    so the agent still gets a useful tool catalog.

    Empty ceiling short-circuits — no planner call. The chat layer
    treats an empty ceiling as "tools disabled for this session" and
    the planner has nothing meaningful to narrow within.
    """
    if not ceiling_groups:
        return TurnToolPolicy(
            kept_groups=frozenset(),
            dropped_groups=frozenset(),
            rationale="Tools disabled for this session.",
            confidence=1.0,
            fell_back_to_ceiling=False,
            cost_usd=0.0,
            latency_ms=0.0,
        )

    used_model = model or _resolve_planner_model()
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else _resolve_confidence_threshold()
    )

    try:
        from kaos_llm_core import Call

        signature = _get_signature()
    except ImportError as exc:
        # `[llm]` extra not installed — abdicate to the ceiling. This
        # is the same shape as a provider error.
        logger.warning(
            "TurnToolPolicy: kaos-llm-core not installed; falling back to ceiling. err=%s",
            exc,
        )
        return TurnToolPolicy(
            kept_groups=frozenset(ceiling_groups),
            dropped_groups=frozenset(),
            rationale=("Planner unavailable (kaos-llm-core not installed). Using full ceiling."),
            confidence=0.0,
            fell_back_to_ceiling=True,
            cost_usd=0.0,
            latency_ms=0.0,
        )

    # signature is a Signature subclass built lazily under [llm]; ty can't
    # see that without an eager kaos_llm_core import (which we deliberately
    # defer to keep the module loadable without the [llm] extra).
    from kaos_agents._examples import load_examples

    call = Call(
        signature,  # ty: ignore[invalid-argument-type]
        model=used_model,
        examples=load_examples("turn_tool_policy"),
    )
    t_start = time.monotonic()
    try:
        invocation = await call.invoke(
            user_message=user_message,
            recent_turns=recent_turns,
            corpus_headlines=corpus_headlines,
            corpus_kinds=list(corpus_kinds or []),
            session_intent=session_intent,
            raw_turn_groups=list(raw_turn_groups or []) if raw_turn_groups else None,
            ceiling_groups=list(ceiling_groups),
            available_groups=list(available_groups),
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t_start) * 1000
        logger.warning(
            "TurnToolPolicy planner failed; falling back to ceiling. err=%s",
            exc,
        )
        return TurnToolPolicy(
            kept_groups=frozenset(ceiling_groups),
            dropped_groups=frozenset(),
            rationale=f"Planner unavailable: {exc}. Using full ceiling.",
            confidence=0.0,
            fell_back_to_ceiling=True,
            cost_usd=0.0,
            latency_ms=latency_ms,
        )

    latency_ms = (time.monotonic() - t_start) * 1000
    raw = invocation.output.policy
    cost_usd = float(getattr(invocation.usage, "cost_usd", 0.0) or 0.0)

    ceiling_set = frozenset(ceiling_groups)
    wanted_set = frozenset(raw.wanted_groups)
    kept = wanted_set & ceiling_set
    dropped = wanted_set - ceiling_set

    if raw.confidence < threshold or not kept:
        # Low confidence OR planner wanted nothing reachable → use the
        # full ceiling. Refusing to narrow is always safe.
        return TurnToolPolicy(
            kept_groups=ceiling_set,
            dropped_groups=dropped,
            rationale=(
                raw.rationale + f" (planner confidence {raw.confidence:.2f} "
                f"< {threshold:.2f}; expanded to ceiling)"
            ),
            confidence=raw.confidence,
            fell_back_to_ceiling=True,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    return TurnToolPolicy(
        kept_groups=kept,
        dropped_groups=dropped,
        rationale=raw.rationale,
        confidence=raw.confidence,
        fell_back_to_ceiling=False,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


__all__ = [
    "TurnToolPolicy",
    "TurnToolPolicyResult",
    "plan_turn_tool_policy",
]
