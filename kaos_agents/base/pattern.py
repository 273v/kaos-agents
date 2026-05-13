"""KaosPattern — dispatch-strategy ABC.

A :class:`KaosPattern` is a strategy for handling an :class:`IntentResult`
within an agent turn. The :class:`KaosAgent` outer loop classifies the
user's intent then dispatches to a pattern that yields events for that
intent. Concrete patterns (ChatAgent / PlanExecuteAgent / ResearchAgent
in :mod:`kaos_agents.patterns`) implement :meth:`dispatch`.

The split:

- :class:`KaosAgent` (in :mod:`kaos_agents.base.agent`) — *outer loop*:
  hydrate memory, classify intent, persist, emit lifecycle events.
- :class:`KaosPattern` — *inner loop*: given an intent, do the work
  (single-turn response, tool-use ReAct, plan-execute, research RAG)
  and yield events.

Today's :class:`kaos_agents.runtime.agent.BaseAgent` and its subclasses
conflate the two — BaseAgent owns the outer loop AND implements
respond-style dispatch; ChatAgent / PlanExecuteAgent / ResearchAgent
inherit and override the dispatch handlers. Track 3 (Patterns) carves
patterns/ into pure :class:`KaosPattern` implementations decoupled
from BaseAgent's outer loop. For Track 2, this ABC is added so the
pattern registry (chunk 4) has a contract to register against, even
though the existing pattern subclasses still inherit BaseAgent.

Auto-registration: every concrete subclass auto-registers in
:data:`kaos_agents.registry.pattern_registry.default_pattern_registry`
via ``__init_subclass__`` (added in Track 2 chunk 4). Subclasses can
opt out with ``register=False``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kaos_agents.base.event import KaosEvent
from kaos_agents.types.metadata import PatternMetadata

if TYPE_CHECKING:
    from kaos_agents.events.emitter import EventEmitter
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.intents import IntentResult
    from kaos_agents.types.memory import MemoryType


_CAMEL_TO_UNDERSCORE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _default_pattern_name(cls: type) -> str:
    """Snake-cased class name as the default pattern discriminator.

    ``ChatAgent`` -> ``chat_agent``; ``PlanExecuteAgent`` -> ``plan_execute_agent``.
    Strips leading underscores so private/test-private subclasses still
    produce a valid ``PatternMetadata.name``.
    """
    name = cls.__name__.lstrip("_")
    return _CAMEL_TO_UNDERSCORE.sub("_", name).lower() if name else "pattern"


def _first_doc_line(cls: type) -> str:
    """First non-empty line of a class's docstring, or fallback."""
    doc = cls.__doc__ or ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "(no description)"


class KaosPattern(ABC):
    """ABC for an intent-dispatch strategy.

    Concrete patterns (chat / plan_execute / research) implement
    :meth:`dispatch` to handle a classified intent and yield events.

    Type discipline mirrors :class:`KaosAgent`: classmethod metadata,
    abstract async-iterator dispatch.

    Note: today's pattern classes (ChatAgent / PlanExecuteAgent /
    ResearchAgent) inherit from BaseAgent rather than KaosPattern
    directly. Track 3 decouples them. For Track 2, this ABC primarily
    serves as the contract :class:`PatternRegistry` registers against,
    plus an aspirational target for the Track 3 refactor.
    """

    @classmethod
    def metadata(cls) -> PatternMetadata:
        """Frozen identity record for this pattern class.

        Subclasses override to set ``supports_intents`` (which intents
        this pattern handles natively), ``tags``, ``version``.
        """
        return PatternMetadata(
            name=_default_pattern_name(cls),
            description=_first_doc_line(cls),
        )

    @abstractmethod
    def dispatch(
        self,
        intent: IntentResult,
        message: str,
        memory: SessionMemory,
        context_items: dict[MemoryType, list[Any]],
        emitter: EventEmitter,
    ) -> AsyncIterator[KaosEvent]:
        """Handle a classified intent, yielding events progressively.

        Args:
            intent: The classified user intent.
            message: The user's raw message.
            memory: Hydrated :class:`SessionMemory` for the session.
            context_items: Pre-assembled per-section context items
                from :func:`kaos_agents.context.assemble.assemble_context`.
            emitter: Auto-fills timestamp/sequence/session/run on
                emitted events.

        Yields:
            :class:`KaosEvent` instances in execution order. The outer
            :class:`KaosAgent` loop wraps these with TURN-level Span
            boundaries and a final TurnSummary.
        """
        ...


__all__ = ["KaosPattern"]
