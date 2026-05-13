"""Deliverable types — Protocol + supporting frozen dataclasses.

Pure types, zero behavior. The Protocol fixes the contract every
composer + serializer + critic relies on; concrete deliverable
shapes live in :mod:`kaos_agents.output.deliverables`.

Three top-level types:

* :class:`Deliverable` — Protocol every deliverable shape implements.
* :class:`SectionSpec` — declarative description of one report
  section (id, title, depth, instruction). Used by
  ``compose_narrative`` to drive per-section LLM calls.
* :class:`Citation` — a ``Span`` with display-side metadata
  (which presentation style the rendering should use, where it
  appears in the deliverable). Distinct from ``Span`` because a
  ``Span`` is evidence; a ``Citation`` is evidence rendered into a
  specific output.

The Protocol uses ``TYPE_CHECKING`` imports for ``ContentDocument``
and ``Span`` so this module stays cheap to import. Concrete
implementations resolve those at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument
    from kaos_llm_core.signatures.grounding import Span


# Discriminator for Deliverable.kind. Concrete classes set this as a
# class attribute so callers can branch on `deliverable.kind` without
# isinstance imports across module boundaries.
DeliverableKind = Literal["tabular", "narrative", "hybrid"]


# Citation rendering style. Determines how `inline_citations_md`
# inserts the marker into the body text.
CitationStyle = Literal["inline", "footnote", "bibliography"]


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """Declarative spec for one section of a narrative deliverable.

    Drives :func:`compose_narrative` — for each spec, the composer
    recalls relevant memory, runs ``Call(SectionWriterSignature)``, and
    appends the produced section to the document.

    Attributes:
        id: Stable identifier (e.g., ``"executive-summary"``). Used
            for cross-references and deduplication when refining.
        title: Human-readable heading text.
        depth: Markdown heading depth (1 = top-level, 2 = subsection,
            etc.). Mirrors :class:`kaos_content.Heading.depth`.
        instruction: Optional per-section guidance for the LLM. Empty
            string falls back to the SectionWriterSignature's
            docstring.
        ordering: Where this section sits relative to siblings. Lower
            values render earlier. Composer is stable on ties.
    """

    id: str
    title: str
    depth: int = 2
    instruction: str = ""
    ordering: int = 0


@dataclass(frozen=True, slots=True)
class Citation:
    """A ``Span`` rendered into a specific deliverable.

    A :class:`~kaos_llm_core.signatures.grounding.Span` is evidence
    (this substring appears at this position in this source). A
    Citation is evidence *rendered* — it carries the presentation
    metadata the deliverable needs to inline / footnote / bibliography
    the span correctly.

    Attributes:
        span: The underlying typed evidence.
        style: How this citation is rendered in the deliverable.
        marker: Optional pre-rendered marker text (e.g., ``"[1]"`` or
            ``"^Smith2024"``). Pure-function renderers may set this
            via :func:`assign_markers`. Empty until rendering.
        section_id: Optional ID of the section that introduced this
            citation; lets the renderer scope footnote numbering per
            section.
    """

    span: Span
    style: CitationStyle = "inline"
    marker: str = ""
    section_id: str = ""


@runtime_checkable
class Deliverable(Protocol):
    """A typed output artifact composed from agent plan-step outputs.

    Every Deliverable shape (:class:`TabularDeliverable`,
    :class:`NarrativeDeliverable`, :class:`HybridDeliverable`)
    implements this Protocol. Composers return one of the concrete
    types; consumers (renderers, critics, refiners) work against the
    Protocol.

    The contract is intentionally minimal — three reads, no writes.
    Mutation happens by constructing a new Deliverable, never by
    mutating in place. (``compose_*`` functions are pure with respect
    to the input Deliverable.)
    """

    kind: DeliverableKind

    def to_content_document(self) -> ContentDocument:
        """Render this deliverable as a unified :class:`ContentDocument`.

        Tabular shapes emit a ContentDocument containing a single
        :class:`kaos_content.Table` block; narrative shapes emit one
        with heading/paragraph blocks; hybrid shapes interleave.

        The returned document is fresh — callers may mutate (or pass
        to a builder) without affecting the source.
        """

    def to_markdown(self) -> str:
        """Render this deliverable as markdown.

        Convenience over ``serialize_markdown(self.to_content_document())``
        for the narrative + hybrid shapes; tabular shapes delegate to
        ``serialize_tabular_markdown``.
        """

    def citations(self) -> tuple[Span, ...]:
        """Return the deduplicated, ordered citation list for this deliverable.

        Order is composer-defined (typically: order of first appearance
        in the body). Deduplication is by
        ``(source_uri, char_span)`` per
        :func:`kaos_agents.output.walks.dedupe_spans`.
        """
