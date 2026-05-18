"""Typed value types for the Intent subsystem.

These mirror paper §3.2-3.5 of the kaos-agents ground-up rewrite plan
(see ``docs/design/rewrite-plan-ten-questions.md``):

* :class:`Goal` — the primary objective extracted from a trigger.
* :class:`Constraint` — deadlines, budgets, jurisdictions, format/style/scope rules.
* :class:`Ambiguity` — span-level pointers to unclear text + candidate
  interpretations, used by :class:`~kaos_agents.intent.clarify.ClarificationLoop`.
* :class:`IntentResult` — the typed output of
  :class:`~kaos_agents.intent.extractor.IntentExtractor`.

The :class:`IntentResult` here is **distinct from** the legacy
:class:`kaos_agents.types.intents.IntentResult`. The new shape adds
``Goal`` / ``Constraint`` / ``Ambiguity`` / ``pattern`` / ``confidence``
and is consumed by Phase 2's AgentLoop. The legacy type stays untouched
in ``types/intents.py`` so existing tests and ``BaseAgent.run`` keep
working until the cutover.

All types are frozen pydantic models with ``extra="forbid"`` so JSON
round-trips through ``model_dump_json`` / ``model_validate_json`` are
bit-stable and any unknown field is rejected.
"""

from __future__ import annotations

from enum import StrEnum, unique

from pydantic import BaseModel, ConfigDict, Field

from kaos_agents.config import AgentPattern
from kaos_agents.types.intents import IntentType


@unique
class ConstraintKind(StrEnum):
    """The flavor of a single constraint on the goal.

    Mirrors paper §3.4. ``OTHER`` is the catch-all for domain-specific
    constraints that don't fit the canonical taxonomy.
    """

    DEADLINE = "deadline"  # "by Friday", "before close of business"
    BUDGET = "budget"  # "spend less than $50", "max 10 calls"
    JURISDICTION = "jurisdiction"  # "Delaware law", "EU only"
    FORMAT = "format"  # "JSON output", "markdown table"
    STYLE = "style"  # "formal tone", "bullet points only"
    SCOPE = "scope"  # "from 2024 onwards", "EDGAR filings only"
    OTHER = "other"


@unique
class AmbiguityKind(StrEnum):
    """The flavor of unclear text the extractor identified.

    Mirrors paper §3.3. ``MISSING_CONTEXT`` and ``OUT_OF_DOMAIN`` are the
    two kinds that should halt execution by default — see
    :meth:`IntentResult.has_blocking_ambiguity`.
    """

    UNKNOWN_REFERENCE = "unknown_reference"  # "the contract" — which one?
    AMBIGUOUS_PRONOUN = "ambiguous_pronoun"  # "they" with multiple antecedents
    CONFLICTING_REQUIREMENTS = "conflicting_req"  # cheap AND best
    MISSING_CONTEXT = "missing_context"  # no matter_id given
    OUT_OF_DOMAIN = "out_of_domain"  # outside agent's competence


class Goal(BaseModel):
    """Primary objective extracted from a trigger.

    The agent loop consumes ``statement`` as the natural-language goal
    and ``intent_type`` as the dispatch key (CHAT / PLAN / RESEARCH /
    TOOL_USE / CLARIFY). ``success_criteria`` are derived by the
    extractor from constraint analysis and feed
    :class:`~kaos_agents.termination.judge.TerminationJudge` (Phase 4).

    ``matter_client`` is a tuple of ``(matter_id, client_id)`` used by
    the institutional-memory layer (paper §6.4) for namespace isolation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str  # the natural-language goal
    intent_type: IntentType  # CHAT/PLAN/RESEARCH/etc
    success_criteria: tuple[str, ...] = ()  # what "done" looks like
    target_format: str | None = None  # "json" / "markdown" / None
    domain: str | None = None  # "legal" / "financial" / None
    matter_client: tuple[str, str] | None = None  # (matter_id, client_id)


class Constraint(BaseModel):
    """A single constraint on the goal.

    ``mandatory=True`` is the default; ``mandatory=False`` marks
    "preferred but not required" constraints. Constraints flow into
    Plan-Execute's plan emission (so a deadline becomes a step budget)
    and into the TerminationJudge's success-criteria check.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ConstraintKind
    value: str  # constraint payload
    mandatory: bool = True


class Ambiguity(BaseModel):
    """A pointer to unclear text + candidate interpretations.

    ``span`` is character offsets ``(start, end)`` in the original
    trigger text — half-open, like Python slices. ``excerpt`` is the
    actual ambiguous substring; redundant with ``span`` but lets
    consumers display the ambiguity without holding the original text.

    ``preferred_clarification`` is what to ask the user, used by
    :class:`~kaos_agents.intent.clarify.ClarificationLoop` to compose a
    deterministic clarifying question without an extra LLM round-trip.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: AmbiguityKind
    span: tuple[int, int]  # (start, end) char offsets
    excerpt: str  # the actual ambiguous text
    candidate_interpretations: tuple[str, ...] = ()
    preferred_clarification: str = ""


class IntentResult(BaseModel):
    """Output of :class:`~kaos_agents.intent.extractor.IntentExtractor`.

    Distinct from :class:`kaos_agents.types.intents.IntentResult` (legacy).
    The new shape adds Goal / Constraint / Ambiguity / pattern /
    confidence and is consumed by Phase 2's AgentLoop. The legacy type
    is kept for ``BaseAgent.run`` compatibility until cutover.

    ``raw_input`` retains the original trigger payload so downstream
    consumers (clarifier, telemetry) can re-anchor :class:`Ambiguity`
    spans without holding a separate handle to the trigger.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: Goal
    constraints: tuple[Constraint, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    requires_clarification: bool = False
    pattern: AgentPattern = AgentPattern.CHAT
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    targets: tuple[str, ...] = ()
    """Corpus item filenames the intent references, drawn verbatim from
    the session's ``corpus_headlines``. Empty when the intent is
    non-corpus or has no specific corpus reference. The downstream
    planner uses this to scope work and the critic uses it to verify
    coverage. Validated against ``corpus_headlines`` by the extractor
    — unknown filenames are dropped (and logged) rather than passed
    through. See
    ``kaos-modules/docs/plans/persona-matrix-followups.md`` §6."""

    raw_input: str = ""  # the original trigger payload

    def has_blocking_ambiguity(self) -> bool:
        """True when any ambiguity should halt execution by default.

        ``MISSING_CONTEXT`` and ``OUT_OF_DOMAIN`` are the two kinds the
        agent loop cannot reasonably guess past. Other kinds are
        surfaced to the agent for handling (e.g. resolving a pronoun
        from MEMORY) but are not blocking.
        """
        return any(
            a.kind in (AmbiguityKind.MISSING_CONTEXT, AmbiguityKind.OUT_OF_DOMAIN)
            for a in self.ambiguities
        )
