"""KaosAgent — the runnable-agent ABC.

Every class that can drive an agent turn loop and emit
:class:`KaosEvent` objects conforms to :class:`KaosAgent`. The canonical
implementation is :class:`kaos_agents.runtime.agent.BaseAgent` (Track 2
chunk 2 moves it from ``kaos_agents/agent.py`` into ``runtime/``).
Concrete patterns (ChatAgent / PlanExecuteAgent / ResearchAgent)
inherit from BaseAgent and pick up the contract transitively.

Type discipline (mirrors :class:`kaos_core.base.tool.KaosTool` ↔
:class:`kaos_core.types.metadata.ToolMetadata`):

- :class:`KaosAgent` is a plain Python ABC (agents have *behavior*;
  only value types are pydantic).
- :meth:`KaosAgent.metadata` is a classmethod returning a frozen
  pydantic :class:`AgentMetadata`. Subclasses override for richer
  identity (e.g. set ``pattern="research"`` for the RAG pattern).

The two execution modes:

- :meth:`run` — streaming. Yields :class:`KaosEvent` objects
  progressively. Subclasses MUST implement.
- :meth:`turn` — blocking. Default implementation collects events from
  :meth:`run` and converts to an :class:`AgentResponse`. Subclasses
  may override for a faster non-streaming path.

Why classmethod metadata rather than instance metadata: per-class
identity is the common case. Each ChatAgent instance shares the same
identity ("chat agent, version X"). Subclasses with computed identity
(e.g. delegated sub-agents tagged with a session-specific name) can
still override at the instance level by setting an instance attribute
and overriding :meth:`metadata` to return it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from kaos_agents.base.event import KaosEvent
from kaos_agents.types.metadata import AgentMetadata

if TYPE_CHECKING:
    from kaos_agents.types.response import AgentResponse


_CAMEL_TO_HYPHEN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _default_agent_name(cls: type) -> str:
    """Kebab-cased class name as the default agent discriminator.

    ``ChatAgent`` -> ``chat-agent``; ``PlanExecuteAgent`` -> ``plan-execute-agent``.
    Strips leading underscores so private/test-private subclasses
    (``_ChatAgentLike``) still produce a valid ``AgentMetadata.name``
    (the validator requires the first char to be ``[a-z0-9]``).
    """
    name = cls.__name__.lstrip("_")
    return _CAMEL_TO_HYPHEN.sub("-", name).lower() if name else "agent"


def _first_doc_line(cls: type) -> str:
    """First non-empty line of a class's docstring, or fallback."""
    doc = cls.__doc__ or ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "(no description)"


class KaosAgent(ABC):
    """ABC for any class that runs an agent turn loop and emits events.

    Subclasses implement :meth:`run` (streaming) and inherit a default
    :meth:`turn` (blocking) that collects events from ``run`` and
    converts to an :class:`AgentResponse`.

    The default :meth:`metadata` builds an :class:`AgentMetadata` from
    the class name + first line of ``__doc__``. Override for richer
    identity (set ``pattern="research"``, ``supports_streaming=True``,
    ``tags=...``, etc.).
    """

    @classmethod
    def metadata(cls) -> AgentMetadata:
        """Frozen identity record for this agent class.

        Subclasses override to set ``pattern``, ``tags``, ``version``.
        Default builds from kebab-cased class name + ``__doc__``.
        """
        return AgentMetadata(
            name=_default_agent_name(cls),
            description=_first_doc_line(cls),
        )

    @abstractmethod
    def run(self, message: str, session_id: str) -> AsyncIterator[KaosEvent]:
        """Execute one agent turn, yielding events progressively.

        This is the primary streaming entry point. Concrete agents
        implement the 8-step turn loop here (see
        :class:`kaos_agents.runtime.agent.BaseAgent` for the canonical
        implementation).

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Yields:
            :class:`KaosEvent` subclass instances in execution order.
        """
        ...

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        """Execute one agent turn (blocking mode).

        Default implementation collects all events from :meth:`run`
        and converts via the standard event-stream-to-response
        conversion. Subclasses may override for a faster non-streaming
        path, but the default is correct for any
        :class:`run`-conforming subclass.

        Args:
            message: The user's message.
            session_id: Session identifier for memory persistence.

        Returns:
            :class:`AgentResponse` with the agent's reply and metadata.
        """
        # Lazy import to dodge events <-> base cycle on package load.
        from kaos_agents.runtime.events_to_response import events_to_response

        events: list[KaosEvent] = []
        async for event in self.run(message, session_id):
            events.append(event)
        return events_to_response(events, session_id)


__all__ = ["KaosAgent"]
