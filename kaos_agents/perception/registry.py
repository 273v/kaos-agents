"""Read-only-tool view over the existing kaos-agents tool registry.

The plan calls for a ToolRegistry filtered by ``readOnlyHint=True`` to
enforce "perceive vs act" at the registry boundary. This module does
NOT add a parallel registry — it is a thin filter helper over the
existing :mod:`kaos_agents.registry` / :mod:`kaos_core.base.tool`
surface.

The Actor subsystem (Phase 1.C) similarly works through the unfiltered
registry but consults reversibility to decide what to do.

Duck-typing matrix (verified against the codebase):

- :class:`kaos_core.base.tool.KaosTool` — exposes ``tool.metadata``
  (a ``ToolMetadata`` whose ``annotations`` is a
  ``ToolAnnotations | None``).
- :class:`kaos_llm_core.programs.tool.Tool` — wraps a
  ``ToolDefinition`` (no annotations); always treated as read-only
  is unsafe, so tools without annotations are skipped silently.
- ``dict`` — supports either an ``"annotations"`` key (mirroring
  ``ToolMetadata.annotations``) or a top-level ``"readOnlyHint"``
  flag (the wire shape used in MCP tool definitions).

Tools that don't fit any of these shapes are silently skipped — the
filter is intentionally fail-closed (omitting a tool is safer than
admitting an unknown one to the read-only set).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def read_only_tools(tools: Iterable[Any]) -> tuple[Any, ...]:
    """Return only the tools whose annotations declare ``readOnlyHint=True``.

    Accepts duck-typed inputs:

    - kaos-llm-core ``Tool`` objects (no annotations surface — skipped).
    - kaos-agents / kaos-core ``KaosTool`` objects (have a
      ``metadata.annotations`` chain).
    - dicts with an ``"annotations"`` key or a top-level
      ``"readOnlyHint"`` flag.

    Skips entries whose ``readOnlyHint`` is missing, ``False``, or
    not a boolean. Returns a tuple to match the input convention of
    Phase 1.B's value types (frozen, slottable).
    """
    selected: list[Any] = []
    for tool in tools:
        ann = _read_annotations(tool)
        if ann is None:
            continue
        if _read_only_hint(ann):
            selected.append(tool)
    return tuple(selected)


def _read_annotations(tool: Any) -> Any | None:
    """Extract an annotations object/dict from various tool shapes.

    Returns ``None`` when the tool has no annotations surface — the
    caller treats this as "skip silently."
    """
    # Dict shape — either nested under "annotations" or flat
    # ``readOnlyHint`` on the dict itself.
    if isinstance(tool, dict):
        if "annotations" in tool:
            return tool["annotations"]
        if "readOnlyHint" in tool:
            return tool
        return None

    # KaosTool: tool.metadata.annotations (ToolAnnotations | None)
    metadata = getattr(tool, "metadata", None)
    if metadata is not None:
        ann = getattr(metadata, "annotations", None)
        if ann is not None:
            return ann

    # Direct .annotations attribute (e.g., a duck-typed object).
    direct = getattr(tool, "annotations", None)
    if direct is not None:
        return direct

    return None


def _read_only_hint(annotations: Any) -> bool:
    """Read the ``readOnlyHint`` flag from an annotations object/dict.

    Robust to:

    - Pydantic models with a ``readOnlyHint`` field.
    - Plain dicts.
    - Objects whose ``readOnlyHint`` is non-bool (truthy strings etc.) —
      treated as "not read-only" because the contract is a strict bool.
    """
    if isinstance(annotations, dict):
        value = annotations.get("readOnlyHint")
    else:
        value = getattr(annotations, "readOnlyHint", None)
    return value is True


__all__ = ["read_only_tools"]
