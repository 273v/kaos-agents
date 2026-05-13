"""FieldSet — size-tiered serialization for prompt context. Track 4 chunk T4-3.

Ports kelvin-agent's :class:`FieldSet` enum (SMALL/MEDIUM/LARGE/ALL).
Lets the agent render the same metadata at different verbosity tiers
depending on context budget — a SMALL tool catalogue lists names
only, a MEDIUM one adds descriptions, LARGE adds I/O schemas, ALL
dumps the full record.

Design choice — per-class **dispatch functions**, not class-level
``FIELD_SETS`` constants:
- ToolMetadata lives in kaos-core; we can't add a constant to it
  without breaking layering.
- AgentMetadata / PatternMetadata / RecipeMetadata are pydantic
  KaosModels in kaos-agents — the same machinery wants to handle
  every metadata kind.
- :class:`ToolGroup` is a frozen dataclass — different shape again.

A single :func:`project` entry point dispatches by ``isinstance`` to
the right per-type projector. Out-of-tree types can register their
own projector via :func:`register_projector` (mirrors the
event-registry plug-in pattern from chunk 3).

The projection is **lossy by design** — SMALL is a deliberate
truncation. ``ALL`` is the only tier that round-trips through the
type's own ``model_dump`` / ``asdict``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from enum import StrEnum, unique
from typing import Any


@unique
class FieldSet(StrEnum):
    """Verbosity tier for metadata serialization.

    The four tiers form a strict subset relation:
    ``SMALL ⊆ MEDIUM ⊆ LARGE ⊆ ALL``. A consumer choosing the right
    tier balances prompt-token cost against information density.
    """

    # name only — for compact lists and short prompts
    SMALL = "small"
    # name + description — typical tool catalogue entry
    MEDIUM = "medium"
    # name + description + I/O schema — full ReAct prompt context
    LARGE = "large"
    # everything — debugging, audit, persistence
    ALL = "all"


# Per-type projector signature: (obj, field_set) → dict
_Projector = Callable[[Any, FieldSet], dict[str, Any]]

# Registry of projectors keyed by *type*. Populated below for built-in
# metadata classes; out-of-tree code can register more via
# :func:`register_projector`.
_PROJECTORS: dict[type, _Projector] = {}


def register_projector(typ: type, projector: _Projector) -> None:
    """Register a per-type FieldSet projector.

    Replaces any prior registration for the same type. Out-of-tree
    metadata classes (custom KaosModels, custom dataclasses) opt
    into FieldSet projection by registering here.
    """
    _PROJECTORS[typ] = projector


def project(obj: Any, field_set: FieldSet) -> dict[str, Any]:
    """Project ``obj`` to a dict at the requested verbosity tier.

    Dispatches by ``type(obj)`` to a registered projector. Falls back
    to a generic projector that:
    - calls ``model_dump()`` for pydantic models (ALL tier always);
      for SMALL/MEDIUM/LARGE returns ``{"name": getattr(obj, "name")}``
      when a ``name`` field exists, else ``{}``
    - calls ``dataclasses.asdict()`` for frozen dataclasses (ALL only)

    The fallback is intentionally minimal — types that want
    meaningful tiered projection should register their own projector.
    """
    projector = _PROJECTORS.get(type(obj))
    if projector is not None:
        return projector(obj, field_set)

    # Generic fallback for unregistered types
    return _fallback_project(obj, field_set)


def _fallback_project(obj: Any, field_set: FieldSet) -> dict[str, Any]:
    """Generic catch-all when no projector is registered for the type.

    Prefers full dump at ALL tier; for smaller tiers, surfaces a
    ``name`` field if present so the consumer at least has an
    identifier.
    """
    if field_set == FieldSet.ALL:
        # pydantic-style models
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        # dataclass
        try:
            return asdict(obj)
        except TypeError:
            pass
        # last-resort: vars()
        try:
            return dict(vars(obj))
        except TypeError:
            return {}

    # SMALL / MEDIUM / LARGE without a registered projector — at least
    # surface a name so the caller has an identifier to render.
    name = getattr(obj, "name", None)
    return {"name": name} if name else {}


# ---------------------------------------------------------------------------
# Built-in projectors
# ---------------------------------------------------------------------------


def _project_tool_metadata(meta: Any, field_set: FieldSet) -> dict[str, Any]:
    """Project :class:`kaos_core.types.metadata.ToolMetadata` per tier.

    Tiers:
    - SMALL: ``{"name"}``
    - MEDIUM: ``{"name", "display_name", "description"}``
    - LARGE: SMALL/MEDIUM + ``category``, ``capability``,
      ``input_schema`` (compact: parameter names only)
    - ALL: full ``model_dump()`` round-trip
    """
    if field_set == FieldSet.ALL:
        return meta.model_dump(mode="json")

    out: dict[str, Any] = {"name": meta.name}
    if field_set in (FieldSet.MEDIUM, FieldSet.LARGE):
        out["display_name"] = getattr(meta, "display_name", "")
        out["description"] = getattr(meta, "description", "")
    if field_set == FieldSet.LARGE:
        out["category"] = getattr(meta, "category", None)
        out["capability"] = getattr(meta, "capability", None)
        out["category"] = (
            out["category"].value if hasattr(out["category"], "value") else out["category"]
        )
        out["capability"] = (
            out["capability"].value if hasattr(out["capability"], "value") else out["capability"]
        )
        # Compact input_schema: just the parameter names. Full schemas
        # blow up the prompt — agents don't usually need types until
        # they're about to call the tool.
        params = getattr(meta, "input_schema", ()) or ()
        out["parameter_names"] = [p.name for p in params if hasattr(p, "name")]
    return out


def _project_tool_group(group: Any, field_set: FieldSet) -> dict[str, Any]:
    """Project :class:`ToolGroup` per tier.

    Tiers:
    - SMALL: ``{"name"}``
    - MEDIUM: ``{"name", "description"}``
    - LARGE: SMALL/MEDIUM + ``tool_count``, ``tool_names``
    - ALL: full ``asdict()`` round-trip including tags
    """
    if field_set == FieldSet.ALL:
        return asdict(group)

    out: dict[str, Any] = {"name": group.name}
    if field_set in (FieldSet.MEDIUM, FieldSet.LARGE):
        out["description"] = group.description
    if field_set == FieldSet.LARGE:
        out["tool_count"] = len(group.tool_names)
        out["tool_names"] = list(group.tool_names)
    return out


def _project_agent_metadata(meta: Any, field_set: FieldSet) -> dict[str, Any]:
    """Project :class:`AgentMetadata` per tier.

    Tiers:
    - SMALL: ``{"name"}``
    - MEDIUM: ``{"name", "description", "pattern"}``
    - LARGE: SMALL/MEDIUM + ``tags``
    - ALL: full ``model_dump()``
    """
    if field_set == FieldSet.ALL:
        return meta.model_dump(mode="json")

    out: dict[str, Any] = {"name": meta.name}
    if field_set in (FieldSet.MEDIUM, FieldSet.LARGE):
        out["description"] = getattr(meta, "description", "")
        out["pattern"] = getattr(meta, "pattern", "")
    if field_set == FieldSet.LARGE:
        out["tags"] = list(getattr(meta, "tags", ()) or ())
    return out


# Wire built-in projectors. The lazy-import dance avoids a circular
# import — types/__init__.py imports field_set.py during package init,
# and it would re-enter via ToolMetadata if we imported it at module
# level.
def _register_builtin_projectors() -> None:
    from kaos_core.types.metadata import ToolMetadata as _ToolMetadata

    from kaos_agents.types.metadata import AgentMetadata as _AgentMetadata
    from kaos_agents.types.tool_group import ToolGroup as _ToolGroup

    register_projector(_ToolMetadata, _project_tool_metadata)
    register_projector(_ToolGroup, _project_tool_group)
    register_projector(_AgentMetadata, _project_agent_metadata)


_register_builtin_projectors()


__all__ = [
    "FieldSet",
    "project",
    "register_projector",
]
