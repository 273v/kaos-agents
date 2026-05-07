"""Memory mutation events.

Replaces the previous overloaded ``MemoryUpdated`` (which used a string
``action`` field for "add" / "evict" / "summarize" / ...) with a single
typed :class:`MemoryEvent` whose ``kind`` is a
:class:`MemoryEventKind` enum. Same surface area, type-checked
discrimination, and additional kinds (``HYDRATED`` / ``SEARCHED``) for
events the old taxonomy didn't fire.
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Any

from pydantic import Field

from kaos_agents.events._intermediates import LifecycleEvent


@unique
class MemoryEventKind(StrEnum):
    """What happened to memory."""

    ADDED = "added"
    EVICTED = "evicted"
    SUMMARIZED = "summarized"
    HYDRATED = "hydrated"  # Session loaded from VFS
    PERSISTED = "persisted"  # Session saved to VFS
    SEARCHED = "searched"  # BM25 query against memory


class MemoryEvent(LifecycleEvent):
    """A memory section mutated.

    ``section`` is a :class:`MemoryType` value (the section name).
    ``item_count`` is the number of items affected by this event
    (added, evicted, summarized, etc.).

    ``attributes`` is for kind-specific extras — e.g.,
    ``{"query": "filing fee", "top_k": 5}`` for ``SEARCHED``,
    ``{"new_total_tokens": 8192}`` for ``ADDED``.
    """

    kind: MemoryEventKind
    section: str = ""  # MemoryType value
    item_count: int = 0
    attributes: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "MemoryEvent",
    "MemoryEventKind",
]
