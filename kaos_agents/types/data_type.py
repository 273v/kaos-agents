"""Tool I/O data-type taxonomy — Track 4 chunk T4-1.

Ports kelvin-agent's :class:`DataType` enum (kelvin/agent/models/data_types.py)
into the KAOS tool model. Used to tag the input/output formats every
tool consumes and emits, so agents (and group-discovery surfaces)
can filter tools by what they read or write.

The kaos-core :class:`ToolMetadata` doesn't carry I/O typing today; we
keep the typing on the kaos-agents side via a parallel registry
(``ToolDataTypeRegistry``) keyed by tool name, mirroring the
:class:`AgentToolSpec` pattern from Track 3 chunk A1. This avoids a
breaking schema change to kaos-core and lets callers opt into typing
per tool.

Design notes:
- Mirrors kelvin's 9-value enum (TEXT/MARKDOWN/JSON/JSONL/HTML/CSV/
  TABLE/RDF/BINARY) — these are the formats agent actions actually
  emit/consume in production.
- ``ToolDataTypeSpec`` is a frozen dataclass with ``input_type`` and
  ``output_type``; either may be ``None`` when the tool's I/O is
  polymorphic or the tagging hasn't been done yet.
- The registry returns an empty spec (None/None) for unregistered
  tools — same null-safe pattern as AgentToolSpecRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique


@unique
class DataType(StrEnum):
    """I/O format for tool inputs and outputs.

    Values mirror kelvin-agent's enum so call sites don't need to
    re-learn the taxonomy. Agents inspect these to discover compatible
    tools (see :meth:`ToolDataTypeRegistry.tools_by_input_type`).
    """

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    HTML = "html"
    CSV = "csv"
    TABLE = "table"  # JSON-validated list-of-lists (rows of cells)
    RDF = "rdf"
    BINARY = "binary"


@dataclass(frozen=True, slots=True)
class ToolDataTypeSpec:
    """Per-tool I/O type tag.

    ``input_type`` and ``output_type`` are independently optional —
    a tool that reads JSON and emits Markdown sets both; one whose
    input is polymorphic leaves ``input_type=None`` and only declares
    its output format.

    Frozen + slotted to match the rest of the kaos-agents value-type
    discipline.
    """

    input_type: DataType | None = None
    output_type: DataType | None = None

    @property
    def is_empty(self) -> bool:
        """True when neither I/O type is declared.

        Used by the registry to short-circuit an empty-spec lookup
        without allocating; mirrors :attr:`AgentToolSpec.is_empty`.
        """
        return self.input_type is None and self.output_type is None

    @classmethod
    def empty(cls) -> ToolDataTypeSpec:
        """Identity-stable empty spec (returned for unknown tools).

        Returning a singleton keeps callers from allocating a fresh
        spec on every lookup miss.
        """
        return _EMPTY_SPEC


_EMPTY_SPEC = ToolDataTypeSpec()


__all__ = [
    "DataType",
    "ToolDataTypeSpec",
]
