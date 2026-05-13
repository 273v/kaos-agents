"""``compose_tabular`` — build a TabularDeliverable from FINDINGS cells.

Pure orchestration:

1. :func:`walk_cells` over memory to collect every typed
   :class:`ExtractionCell` written into FINDINGS by an extract-corpus
   recipe.
2. Filter to the target schema's column ids (cells outside the schema
   are ignored — they may belong to a different recipe in the same
   session).
3. :meth:`TabularDocument.from_cells` pivots into a typed table
   (``doc_id`` x column_id x cell.value).
4. :func:`aggregate_citations` walks Claims for the deliverable-wide
   span list.

The composer reuses ``TabularDocument.from_cells`` verbatim — it's
the canonical pivot already shipping in kaos-content. We don't
reinvent that machinery.

Returns a :class:`TabularDeliverable` carrying the typed table + the
deduplicated span list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_content import TabularDocument
from kaos_content.model.tabular import ColumnType
from kaos_core.logging import get_logger

from kaos_agents.output.citations import aggregate_citations
from kaos_agents.output.deliverables import TabularDeliverable
from kaos_agents.output.walks import walk_cells

if TYPE_CHECKING:
    from kaos_llm_core.signatures.extraction import ColumnSpec, ExtractionSchema

    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)


def compose_tabular(
    memory: SessionMemory,
    schema: ExtractionSchema,
    *,
    table_name: str = "extraction",
) -> TabularDeliverable:
    """Compose a :class:`TabularDeliverable` from cells in memory.

    Args:
        memory: Session memory whose FINDINGS section holds typed
            :class:`ExtractionCell` payloads (written via the
            extract-corpus recipe).
        schema: The target :class:`ExtractionSchema` that defines
            which columns to pivot into. Cells whose ``column_id``
            isn't in the schema are skipped.
        table_name: Name for the produced
            :class:`kaos_content.model.tabular.Table`.

    Returns:
        A :class:`TabularDeliverable` with one
        :class:`kaos_content.TabularDocument` table containing the
        pivoted cells, plus the deduplicated supporting-span list.
    """
    schema_column_ids = {col.id for col in schema.columns}
    cells_in_schema = [c for c in walk_cells(memory) if c.column_id in schema_column_ids]
    logger.debug(
        "compose_tabular: schema_id=%s columns=%d cells_seen=%d cells_in_schema=%d",
        schema.id,
        len(schema_column_ids),
        sum(1 for _ in walk_cells(memory)),
        len(cells_in_schema),
    )

    column_specs: tuple[tuple[str, ColumnType], ...] = tuple(
        (col.id, _column_type_for(col)) for col in schema.columns
    )
    document = TabularDocument.from_cells(
        cells_in_schema,
        column_specs=column_specs,
        table_name=table_name,
    )
    spans = tuple(aggregate_citations(memory))
    return TabularDeliverable(document=document, spans=spans)


def _column_type_for(col_spec: ColumnSpec) -> ColumnType:
    """Return the kaos-content :class:`ColumnType` for an ExtractionSchema column.

    The schema's ``column_type`` field is a Literal name (``"text"``,
    ``"number"``, etc.); :meth:`TabularDocument.from_cells` wants a
    :class:`kaos_content.model.tabular.ColumnType` enum instance.
    Resolves by name lookup; falls back to ``ColumnType.TEXT`` on
    mismatch (no schema column should ever fail this lookup, but
    defaulting beats raising on a future column-type addition).
    """
    name = (col_spec.column_type or "").upper()
    try:
        return ColumnType[name]
    except KeyError:
        logger.debug(
            "_column_type_for: unknown column_type=%s — falling back to TEXT",
            col_spec.column_type,
        )
        return ColumnType.TEXT
