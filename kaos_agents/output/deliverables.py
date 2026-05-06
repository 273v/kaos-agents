"""Deliverable dataclasses — concrete shapes implementing the Deliverable Protocol.

Three shapes:

* :class:`TabularDeliverable` — wraps a :class:`kaos_content.TabularDocument`.
* :class:`NarrativeDeliverable` — wraps a :class:`kaos_content.ContentDocument`.
* :class:`HybridDeliverable` — ordered tuple of narrative or tabular
  sections, plus an aggregate citation list.

All three are frozen + slotted. Composers return one of these; consumers
call ``to_content_document()`` / ``to_markdown()`` / ``citations()``
against the Protocol.

Implementation notes
--------------------

* ``to_content_document`` returns a fresh document — callers may pass
  to a builder without affecting the source.
* ``to_markdown`` delegates to existing kaos-content serializers
  (``serialize_markdown`` for narrative, ``serialize_tabular_markdown``
  for tabular). No reinvention.
* Hybrid serialization concatenates section markdown in declared
  order. The unified ``ContentDocument`` round-trip composes a single
  body with heading + (paragraph | table) blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from kaos_content import (
    ContentDocument,
    DocumentBuilder,
    Heading,
    Paragraph,
    TabularDocument,
    Text,
    serialize_markdown,
)
from kaos_content.model.blocks import RawBlock
from kaos_content.serializers.tabular import serialize_tabular_markdown

from kaos_agents.output.types import SectionSpec

if TYPE_CHECKING:
    from kaos_content.model.blocks import Block
    from kaos_llm_core.signatures.grounding import Span


@dataclass(frozen=True, slots=True)
class TabularDeliverable:
    """Extraction-shaped deliverable (rows x columns x cells).

    Wraps a :class:`kaos_content.TabularDocument` produced by
    ``compose_tabular``. Citations are the deduplicated supporting
    spans from the cells' ``Cited[T]`` provenance.
    """

    kind: Literal["tabular"] = field(default="tabular", init=False)
    document: TabularDocument = field(default_factory=TabularDocument)
    spans: tuple[Span, ...] = ()

    def to_content_document(self) -> ContentDocument:
        """Wrap the tabular content in a :class:`ContentDocument`.

        ``kaos_content.model.tabular.Table`` and
        ``kaos_content.model.blocks.Table`` are distinct types with
        no shared base — the tabular model is a Pydantic-typed
        extraction shape, the block model is the AST table block.
        Round-trip via ``serialize_tabular_markdown`` + a markdown
        ``RawBlock`` is the lossless path; downstream serializers
        emit the markdown table verbatim.
        """
        markdown_body = serialize_tabular_markdown(self.document)
        return ContentDocument(
            body=(RawBlock(format="markdown", value=markdown_body),),
        )

    def to_markdown(self) -> str:
        """Serialize via ``serialize_tabular_markdown``."""
        return serialize_tabular_markdown(self.document)

    def citations(self) -> tuple[Span, ...]:
        return self.spans


@dataclass(frozen=True, slots=True)
class NarrativeDeliverable:
    """Report-shaped deliverable (sections x paragraphs x citations).

    Wraps a :class:`kaos_content.ContentDocument` produced by
    ``compose_narrative``. ``structure`` is the ordered tuple of
    :class:`SectionSpec`s the composer used; ``spans`` is the
    deduplicated citation list aggregated across sections.
    """

    kind: Literal["narrative"] = field(default="narrative", init=False)
    document: ContentDocument = field(default_factory=ContentDocument)
    structure: tuple[SectionSpec, ...] = ()
    spans: tuple[Span, ...] = ()

    def to_content_document(self) -> ContentDocument:
        """Return the underlying document — already a ContentDocument."""
        return self.document

    def to_markdown(self) -> str:
        """Serialize via ``serialize_markdown``."""
        return serialize_markdown(self.document)

    def citations(self) -> tuple[Span, ...]:
        return self.spans


@dataclass(frozen=True, slots=True)
class HybridDeliverable:
    """Narrative sections with embedded tables — Harvey-shape M&A reports.

    ``sections`` is an ordered tuple where each entry is either a
    :class:`NarrativeDeliverable` (a textual section) or a
    :class:`TabularDeliverable` (an embedded table). The ``title``
    of each section comes from ``NarrativeDeliverable.structure[0]``
    or ``TabularDeliverable.document.metadata.title``; rendering
    flattens them in declared order.

    Citations are the aggregate across all sections.
    """

    kind: Literal["hybrid"] = field(default="hybrid", init=False)
    sections: tuple[NarrativeDeliverable | TabularDeliverable, ...] = ()
    spans: tuple[Span, ...] = ()
    title: str = ""

    def to_content_document(self) -> ContentDocument:
        """Concatenate every section's content into a unified document.

        Heading hierarchy is preserved (``SectionSpec.depth`` for
        narrative sections; tabular sections get a depth-2 heading
        inserted from the TabularDocument's metadata.title).
        Tabular sections render via a markdown ``RawBlock`` so the
        downstream serializer treats them as pre-rendered markdown
        tables.
        """
        body_blocks: list[Block] = []
        for section in self.sections:
            if isinstance(section, NarrativeDeliverable):
                # Append narrative blocks verbatim.
                body_blocks.extend(section.document.body)
            else:
                # Tabular section — heading + raw markdown table block.
                table_title = section.document.metadata.title
                if table_title:
                    body_blocks.append(Heading(depth=2, children=(Text(value=table_title),)))
                body_blocks.append(
                    RawBlock(
                        format="markdown",
                        value=serialize_tabular_markdown(section.document),
                    )
                )
        return ContentDocument(body=tuple(body_blocks))

    def to_markdown(self) -> str:
        """Serialize via the unified ContentDocument path.

        Tabular sections in a hybrid deliverable are stored as
        markdown ``RawBlock``s (the lossless adapter from
        ``serialize_tabular_markdown``). The serializer default
        scrubs raw blocks for XSS safety; we control the AST end-to-
        end here, so opt in with ``allow_raw_html=True`` to preserve
        the embedded tables.
        """
        return serialize_markdown(self.to_content_document(), allow_raw_html=True)

    def citations(self) -> tuple[Span, ...]:
        return self.spans


def empty_narrative(title: str = "") -> NarrativeDeliverable:
    """Construct an empty :class:`NarrativeDeliverable` with a title.

    Convenience factory for callers / tests that need a Deliverable
    placeholder without going through a composer. The document carries
    just the title in metadata; body is empty.
    """
    builder = DocumentBuilder(title=title)
    return NarrativeDeliverable(document=builder.build())


def narrative_from_blocks(
    blocks: tuple[Block, ...],
    *,
    structure: tuple[SectionSpec, ...] = (),
    spans: tuple[Span, ...] = (),
    title: str = "",
) -> NarrativeDeliverable:
    """Build a :class:`NarrativeDeliverable` from a typed block tuple.

    Used by ``compose_narrative`` after it walks each section's spec +
    recall + Call output and assembles the per-section blocks. Keeps
    the composer's orchestration logic separate from the dataclass
    construction.
    """
    builder = DocumentBuilder(title=title)
    for block in blocks:
        builder.add_block(block)
    return NarrativeDeliverable(
        document=builder.build(),
        structure=structure,
        spans=spans,
    )


def paragraph_block(text: str) -> Paragraph:
    """Construct a :class:`Paragraph` from plain text.

    Uses :class:`Text` inline. Single-purpose helper kept here so
    composers don't depend on kaos-content's internal Inline imports
    for the most common case (free-form section body).
    """
    return Paragraph(children=(Text(value=text),))
