"""KaosEvent — the universal event ABC for streaming agent execution.

Every consumer surface (Python SDK, SSE, JSONL, MCP, WebSocket) ships and
receives the same KaosEvent stream. Events fall into two conceptual
categories distinguished by inheritance + ``isinstance``:

- **Stream deltas** (TextDelta, ThinkingDelta, ToolCallArgsDelta) —
  low-level, real-time, token-by-token events.
- **Lifecycle events** (TurnStart, ToolCallStart, StepComplete, ...) —
  high-level, semantic events for UI state machines, hooks, audit.

Type discipline (mirrors :class:`kaos_core.base.tool.KaosTool` ↔
:class:`kaos_core.types.metadata.ToolMetadata`):

- :class:`KaosEvent` extends :class:`kaos_core.types.content.KaosModel`
  (frozen pydantic ``BaseModel``). All wire-facing value types in KAOS
  are ``KaosModel`` subclasses; events are value types.
- Subclasses set ``type: ClassVar[str]`` as the wire discriminator
  (snake_case). Default: snake-cased class name.
- Identity is supplied by three classmethod hooks subclasses can override:
    * :meth:`event_type` → wire discriminator. Default: snake-cased
      class name. Subclasses override by setting ``type`` ClassVar.
    * :meth:`event_category` → semantic group ("lifecycle" / "stream" /
      "tool" / "plan" / "memory" / "delegation"). Default: "lifecycle".
      Intermediate subclasses (e.g. ``StreamDelta``) override.
    * :meth:`metadata` → frozen :class:`EventMetadata` aggregating the
      identity. Default builds from the other two hooks + ``__doc__``.

Why pydantic ``KaosModel`` rather than ``@dataclass``: kaos-core's
convention is pydantic for every value type — ``ToolMetadata``,
``ToolResult``, ``StreamingChunk``, ``ParameterSchema``, ``TextContent``
all are ``KaosModel`` subclasses. Events are value types, and the
alignment buys: discriminated-union deserialization for free, JSON
schema generation, validation, and a single serde mechanism across the
codebase. The performance delta (~1.5 μs per event) is invisible at
the 20-50 events per turn rate.

Why classmethod hooks rather than required fields: ClassVar ``name``-style
identity collides with subclass instance fields (``StepStart`` has a
``description`` payload field — a ClassVar of the same name on the base
breaks pydantic field collection). Classmethod hooks have no namespace
overlap with field annotations.
"""

from __future__ import annotations

import re
from typing import ClassVar

from kaos_core.types.content import KaosModel
from pydantic import ConfigDict

from kaos_agents.types.metadata import EventMetadata

# CamelCase → snake_case for default type discriminator.
_CAMEL_TO_SNAKE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _default_type(cls: type) -> str:
    """Resolve a snake_case discriminator from a class name.

    ``TurnStart`` → ``turn_start``; ``ToolCallResult`` → ``tool_call_result``.
    """
    return _CAMEL_TO_SNAKE.sub("_", cls.__name__).lower()


def _first_doc_line(cls: type) -> str:
    """Pull the first non-empty line of a class's docstring, or fallback."""
    doc = cls.__doc__ or ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "(no description)"


class KaosEvent(KaosModel):
    """Base for all agent events.

    Subclasses extend :class:`KaosEvent` and add their own payload
    fields. The five base fields below are common to every event and
    auto-filled by :class:`kaos_agents.events.EventEmitter`.

    Attributes:
        timestamp: ``time.monotonic()`` at emission. Monotonic, not wall
            clock, so it's safe for ordering within a process even
            across system clock changes.
        sequence: Monotonic counter within a run. The canonical ordering
            key for replay.
        session_id: Session this event belongs to.
        run_id: Run (turn) this event belongs to.
        agent_id: Which agent emitted this event. ``None`` = root agent.
            Set when events come from sub-agents or delegated agents.
    """

    # frozen=True: events are immutable post-construction (replay-safe).
    # extra="ignore": forward-compatible deserialization. Producers may
    #   add new fields that older consumers don't know about; events are
    #   on the wire, so we don't want strict rejection. (Compare to most
    #   KaosModels which use ``extra="forbid"`` to catch construction
    #   typos at the Python boundary.)
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    timestamp: float
    sequence: int
    session_id: str
    run_id: str
    agent_id: str | None = None

    # --- Identity hooks (override in subclasses) ---------------------

    @classmethod
    def event_type(cls) -> str:
        """The wire discriminator for this event class.

        Default: snake-cased class name. Override by setting a ``type``
        ClassVar on the subclass, or by overriding this method directly.
        """
        explicit = getattr(cls, "type", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        return _default_type(cls)

    @classmethod
    def event_category(cls) -> str:
        """Semantic group: ``lifecycle`` / ``stream`` / ``tool`` / ...

        Default: ``"lifecycle"``. ``StreamDelta`` overrides to ``"stream"``;
        subclasses inherit through normal MRO.
        """
        return "lifecycle"

    @classmethod
    def metadata(cls) -> EventMetadata:
        """Frozen metadata describing this event subclass.

        Built from :meth:`event_type` + :meth:`event_category` +
        the first line of ``cls.__doc__``. Override for richer metadata.
        """
        return EventMetadata(
            name=cls.event_type(),
            description=_first_doc_line(cls),
            category=cls.event_category(),
        )

    # ClassVar declared on the base so subclasses can set ``type = "..."``
    # at class-body level without pydantic treating it as an instance field.
    # Subclasses that omit a ``type`` ClassVar get the snake-cased default
    # via :meth:`event_type`.
    type: ClassVar[str] = ""
