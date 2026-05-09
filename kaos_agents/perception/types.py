"""Typed value types for the Perception subsystem.

Mirrors paper Q3 (perception is read-only fact-finding distinct from
mutation). All wire-facing types are pydantic ``BaseModel`` with
``frozen=True`` and ``extra="forbid"`` so the perception query/result
shape is immutable and strictly schema-checked at the boundary.

Phase 1.B is purely additive — these types are not yet consumed by
any existing pattern in :mod:`kaos_agents.patterns`. Phase 2 wires
them into the agent loop.
"""

from __future__ import annotations

from enum import StrEnum, unique

from kaos_llm_core.signatures.grounding import Span
from pydantic import BaseModel, ConfigDict, Field


@unique
class PerceptionQueryKind(StrEnum):
    """The kind of fact-finding the perceiver should perform.

    The kind determines which subsystems are consulted and in what
    order. ``DOCUMENT_QA`` routes through RAG; ``FACT_LOOKUP`` and
    ``GENERAL_RECALL`` favor session/institutional memory; and
    ``TOOL_QUERY`` targets a specific read-only tool.
    """

    DOCUMENT_QA = "document_qa"  # answer a question against a corpus
    FACT_LOOKUP = "fact_lookup"  # discrete fact retrieval
    GENERAL_RECALL = "general_recall"  # session/institutional memory recall
    TOOL_QUERY = "tool_query"  # call a specific read-only tool


class PerceptionQuery(BaseModel):
    """Input to :meth:`Perceiver.forward` — a typed perception request.

    Attributes:
        query_text: The free-text query.
        kind: Which subsystem(s) to consult. Default is
            :attr:`PerceptionQueryKind.GENERAL_RECALL`.
        required_capabilities: Tool-name patterns. Only read-only
            tools whose name matches at least one pattern (glob
            match) are invoked.
        max_results: Per-source cap on returned items. The total
            output size may be smaller (after dedup / filtering)
            but never larger.
        matter_client: Optional ``(matter_id, client_id)`` pair used
            by Phase 4's institutional KnowledgeBase to scope
            recall. Ignored in Phase 1.B (no KB attached).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_text: str
    kind: PerceptionQueryKind = PerceptionQueryKind.GENERAL_RECALL
    required_capabilities: tuple[str, ...] = ()  # tool name patterns
    max_results: int = Field(default=10, ge=1, le=200)
    matter_client: tuple[str, str] | None = None  # for institutional KB scoping


class PerceptionItem(BaseModel):
    """One returned fact / passage / tool result.

    Attributes:
        content: The textual content of this item.
        source: Provenance string. Conventional values:
            ``"session_memory"``, ``"tool:<tool_name>"``,
            ``"kb:<matter>"``, ``"rag"``.
        citations: Verified spans from kaos-llm-core. Empty when
            the source has no inherent citation surface (e.g., a
            session-memory match).
        score: Optional ranking score. ``None`` when the source
            does not produce a comparable score.
        metadata: Free-form string-keyed metadata. Pydantic
            ``frozen=True`` does not deep-freeze a dict, so callers
            should treat this field as immutable by convention.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    source: str  # "session_memory", "tool:foo", "kb:..."
    citations: tuple[Span, ...] = ()  # verified spans via kaos-llm-core
    score: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class PerceptionRefusal(BaseModel):
    """Returned when the perceiver cannot fulfill the query.

    Maps to an :class:`~kaos_agents.events.research.EvidenceInsufficient`
    or :class:`~kaos_agents.events.research.GroundingRefusalTriggered`
    event in the agent's stream when emitted by Phase 2 wiring. The
    typed refusal is carried on :class:`PerceptionResult.refusal`.

    Attributes:
        reason: Human-readable explanation of why the perceiver gave up.
        kind: Discriminator. Defaults to ``"evidence_insufficient"``;
            wiring may set ``"grounding_refusal"`` when the failure
            is specifically a grounding-policy collapse.
        suggested_alternatives: Names of tools / sub-agents / next
            steps the caller could try. Surfaced into the
            agent-friendly error message in Phase 2.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str
    kind: str = "evidence_insufficient"
    suggested_alternatives: tuple[str, ...] = ()


class PerceptionResult(BaseModel):
    """Output of :meth:`Perceiver.forward` — items + meta + refusal.

    The result is always a single value (never an exception). When
    nothing is found, :attr:`items` is empty and :attr:`refusal` is
    populated. Callers branch on ``result.refusal is None``.

    Attributes:
        items: The retrieved passages / facts / tool outputs in
            consultation order. Sources are merged before return.
        confidence: Aggregate confidence across ``items``. ``1.0``
            for exact-match recall, lower for fuzzy / RAG-grounded
            results. Always ``0.0`` when ``refusal`` is set.
        refusal: ``None`` on success; populated when the perceiver
            could not satisfy the query.
        sources_consulted: Names of the subsystems consulted, in
            order. Useful for telemetry and the perceiver's own
            self-explanation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[PerceptionItem, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    refusal: PerceptionRefusal | None = None
    sources_consulted: tuple[str, ...] = ()  # e.g. ("session", "kb", "tool:foo")


__all__ = [
    "PerceptionItem",
    "PerceptionQuery",
    "PerceptionQueryKind",
    "PerceptionRefusal",
    "PerceptionResult",
]
