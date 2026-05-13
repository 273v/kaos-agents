"""Frozen metadata records for kaos-agents core concepts.

Mirrors ``kaos_core.types.metadata`` (``ToolMetadata`` etc.) — every
concept's ABC carries a frozen pydantic ``KaosModel`` describing its
identity (name, description, version, tags), and the ABC's
``@property metadata`` returns one of these instances.

The five concepts:

- :class:`AgentMetadata` — describes a :class:`KaosAgent` subclass
- :class:`EventMetadata` — describes a :class:`KaosEvent` subclass
- :class:`HookMetadata` — describes a :class:`KaosHook` subclass
- :class:`PatternMetadata` — describes a :class:`KaosPattern` subclass
- :class:`RecipeMetadata` — describes a :class:`KaosRecipe` subclass

Naming conventions
------------------

Names use a relaxed pattern relative to ``ToolMetadata`` because these
are *internal* concepts (events, hooks, patterns are not MCP-discoverable
on the wire — only :class:`kaos_core.types.metadata.ToolMetadata` is).

The pattern is ``[a-z0-9][a-z0-9_.-]*`` — lowercase, with optional
underscores/dots/hyphens. Empty names and uppercase are rejected. This
lets us write ``turn_start``, ``tool_call.result``, ``plan-execute``
without forcing artificial ``kaos-{module}-{action}`` segments.
"""

from __future__ import annotations

import re

from kaos_core.types.content import KaosModel
from pydantic import ConfigDict, Field, field_validator

# Relaxed identifier pattern. Stricter MCP-style validation lives on
# ``ToolMetadata.name`` in kaos-core; the metadata records here describe
# internal concepts that never cross the MCP wire.
_IDENT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")


class _BaseConceptMetadata(KaosModel):
    """Common fields shared by every kaos-agents concept metadata.

    Subclasses add concept-specific fields. ``frozen=True`` makes the
    instances safe to cache across awaits and use as dict keys (when
    fields are themselves hashable).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    name: str
    description: str
    module_name: str = "kaos_agents"
    version: str = "1.0.0"
    tags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _IDENT_PATTERN.match(value):
            msg = (
                "metadata name must match [a-z0-9][a-z0-9_.-]*; "
                f"got {value!r}. Use lowercase letters, digits, '_', '.', or '-'."
            )
            raise ValueError(msg)
        return value


class AgentMetadata(_BaseConceptMetadata):
    """Describes a :class:`kaos_agents.base.agent.KaosAgent` subclass."""

    pattern: str = Field(
        default="chat",
        description="Pattern name this agent uses (chat / plan_execute / research / ...).",
    )
    supports_streaming: bool = True


class EventMetadata(_BaseConceptMetadata):
    """Describes a :class:`kaos_agents.base.event.KaosEvent` subclass.

    The ``name`` field doubles as the wire-format ``type`` discriminator
    for serialization. ``category`` groups events for UI/observability
    consumers (lifecycle, stream, tool, plan, memory, delegation).
    """

    category: str = Field(
        default="lifecycle",
        description="Semantic group: lifecycle, stream, tool, plan, memory, delegation.",
    )


class HookMetadata(_BaseConceptMetadata):
    """Describes a :class:`kaos_agents.base.hook.KaosHook` subclass.

    ``listens_to`` is the set of event ``name``/``type`` discriminators
    this hook reacts to. An empty tuple means *all events*.
    """

    listens_to: tuple[str, ...] = Field(default_factory=tuple)


class PatternMetadata(_BaseConceptMetadata):
    """Describes a :class:`kaos_agents.base.pattern.KaosPattern` subclass.

    ``supports_intents`` is the set of :class:`IntentType` values this
    pattern dispatches natively.
    """

    supports_intents: tuple[str, ...] = Field(default_factory=tuple)


class RecipeMetadata(_BaseConceptMetadata):
    """Describes a :class:`kaos_agents.base.recipe.KaosRecipe` subclass.

    Extraction recipes optionally carry a ``harvey_recall_floor`` so the
    competitive baseline is part of the recipe's identity, and a
    ``schema_id`` pointing into the extraction schema registry.
    """

    harvey_recall_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    schema_id: str | None = None
