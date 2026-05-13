"""Deliverable composition — turn agent state into structured output artifacts.

The agent runs a multi-step plan; intermediate artifacts (FINDINGS Claims,
ExtractionCells, retrieved passages, plan history) accumulate in
:class:`SessionMemory`. This subpackage provides the missing layer: pure
composers + a typed :class:`Deliverable` Protocol that turn that state
into a structured output (markdown, DOCX, JSON, ContentDocument, table).

Three deliverable shapes:

* :class:`TabularDeliverable` — extraction-shaped (rows x columns x cells
  with per-cell ``Cited[T]`` provenance). Backed by
  :class:`kaos_content.TabularDocument`.
* :class:`NarrativeDeliverable` — report-shaped (sections x paragraphs x
  citations). Backed by :class:`kaos_content.ContentDocument`.
* :class:`HybridDeliverable` — narrative sections with embedded tables.
  The Harvey-shape M&A report.

Layered architecture:

* ``types.py`` — Protocol + dataclasses (zero behavior).
* ``walks.py`` — pure iterators over ``SessionMemory``.
* ``citations.py`` — pure citation aggregation + inline rendering.
* ``signatures.py`` — typed Signatures for LLM-driven composition.
* ``composers/`` — orchestrators that wire walks + signatures into
  deliverables.
* ``critic.py`` — :class:`DeliverableCritic` Protocol +
  :class:`RubricDeliverableCritic` implementation.
* ``feedback.py`` — pure feedback formatters for Refine.
* ``refine.py`` — :class:`RefineDeliverable` Program (LoopRunner-backed).

All composers are pure orchestration over kaos-content + kaos-llm-core
primitives. No new transport, no new storage, no parallel abstractions.
"""

from __future__ import annotations

from kaos_agents.output.citations import aggregate_citations, inline_citations_md
from kaos_agents.output.composers import (
    HybridComposeResult,
    NarrativeComposeResult,
    compose_hybrid,
    compose_narrative,
    compose_tabular,
)
from kaos_agents.output.critic import (
    CriterionResult,
    DeliverableCritic,
    DeliverableVerdict,
    RubricCriterion,
    RubricDeliverableCritic,
)
from kaos_agents.output.deliverables import (
    HybridDeliverable,
    NarrativeDeliverable,
    TabularDeliverable,
    empty_narrative,
    narrative_from_blocks,
    paragraph_block,
)
from kaos_agents.output.feedback import (
    FeedbackStrategy,
    format_gap_feedback,
    format_gap_list,
    format_gap_narrative,
)
from kaos_agents.output.refine import (
    IterationRecord,
    RefineDeliverable,
    RefineDeliverableResult,
)
from kaos_agents.output.signatures import (
    DeliverableStructureSignature,
    SectionWriterSignature,
)
from kaos_agents.output.types import (
    Citation,
    CitationStyle,
    Deliverable,
    DeliverableKind,
    SectionSpec,
)
from kaos_agents.output.walks import (
    SpanDedupKey,
    dedupe_spans,
    walk_cells,
    walk_findings,
)

__all__ = [
    "Citation",
    "CitationStyle",
    "CriterionResult",
    "Deliverable",
    "DeliverableCritic",
    "DeliverableKind",
    "DeliverableStructureSignature",
    "DeliverableVerdict",
    "FeedbackStrategy",
    "HybridComposeResult",
    "HybridDeliverable",
    "IterationRecord",
    "NarrativeComposeResult",
    "NarrativeDeliverable",
    "RefineDeliverable",
    "RefineDeliverableResult",
    "RubricCriterion",
    "RubricDeliverableCritic",
    "SectionSpec",
    "SectionWriterSignature",
    "SpanDedupKey",
    "TabularDeliverable",
    "aggregate_citations",
    "compose_hybrid",
    "compose_narrative",
    "compose_tabular",
    "dedupe_spans",
    "empty_narrative",
    "format_gap_feedback",
    "format_gap_list",
    "format_gap_narrative",
    "inline_citations_md",
    "narrative_from_blocks",
    "paragraph_block",
    "walk_cells",
    "walk_findings",
]
