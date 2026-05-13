"""KaosHook — the lifecycle-hook ABC.

Hooks provide deterministic interception points around agent events.
Implement a hook to add logging, metrics, guardrails, approval UI,
cost tracking, or audit trails — without modifying the agent itself.

Two ways to use hooks:

1. **Subclass** :class:`KaosHook` and override the methods you care about.
   Unoverridden methods are no-ops. This is the recommended pattern.
2. **Implement the structural typing contract** directly — any object
   with the right ``async def on_*`` methods works, no inheritance needed.

Tier-1 (decorator) on-ramp lands in chunk 6 (``@hook`` wraps a
function as a FunctionHook).

Type discipline (mirrors :class:`kaos_core.base.tool.KaosTool` ↔
:class:`kaos_core.types.metadata.ToolMetadata`):

- :class:`KaosHook` is a plain Python ABC (hooks have *behavior*; only
  value types are pydantic).
- :meth:`KaosHook.metadata` is a classmethod returning a frozen pydantic
  :class:`HookMetadata`.
- Subclasses can override ``metadata()`` for richer identity
  (``listens_to``, custom ``name``); the default builds from the class
  name + docstring.

Design grounded in competitive research:
- OpenAI Agents SDK: RunHooks with typed per-event callbacks
- Anthropic Claude SDK: 10 hook event types with typed inputs
- CrewAI: ~40 event types via pub/sub event bus

We chose the OpenAI pattern (typed callbacks per event, compose in order)
over the CrewAI pattern (pub/sub) for simplicity and type safety.
"""

from __future__ import annotations

import re
from enum import StrEnum, unique
from typing import TYPE_CHECKING

from kaos_agents.types.metadata import HookMetadata

if TYPE_CHECKING:
    from kaos_agents.events import (
        CitationFound,
        EvidenceInsufficient,
        IntentClassified,
        KaosEvent,
        MemoryEvent,
        PlanProposed,
        RunError,
        Span,
        TextDelta,
        TurnSummary,
    )


_CAMEL_TO_SNAKE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _default_hook_name(cls: type) -> str:
    """Snake-cased class name as the default hook discriminator."""
    return _CAMEL_TO_SNAKE.sub("-", cls.__name__).lower()


def _first_doc_line(cls: type) -> str:
    """Pull the first non-empty line of a class's docstring, or fallback."""
    doc = cls.__doc__ or ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "(no description)"


@unique
class HookAction(StrEnum):
    """What the hook tells the Runner to do after processing an event.

    Returned by ``on_tool_call_start`` to control tool execution.
    Other hook methods return ``None`` (they observe but don't control).
    """

    CONTINUE = "continue"  # Proceed normally
    SKIP = "skip"  # Skip this tool call (don't execute)
    REQUIRE_APPROVAL = "approval"  # Pause for human approval


class KaosHook:
    """Base hook with no-op defaults for all lifecycle events.

    Subclass and override only the events you care about. All methods
    are async to support I/O-bound hooks (logging to remote services,
    database writes, HTTP callbacks).

    Hook methods receive the event that just occurred. Most return
    ``None`` (observe only). ``on_tool_call_start`` returns a
    :class:`HookAction` to control whether the tool executes.

    Hooks are called in registration order. If multiple hooks return
    different ``HookAction`` values, the most restrictive wins:
    ``REQUIRE_APPROVAL > SKIP > CONTINUE``.
    """

    @classmethod
    def metadata(cls) -> HookMetadata:
        """Frozen identity record for this hook.

        Default builds from the snake-cased class name + first line of
        ``__doc__``. Override for richer metadata — e.g. to set
        ``listens_to=("tool_call",)`` so registries / dashboards can
        surface the events this hook reacts to.
        """
        return HookMetadata(
            name=_default_hook_name(cls),
            description=_first_doc_line(cls),
        )

    async def on_turn_start(self, event: Span) -> None:
        """Called at the beginning of a turn (``Span(TURN, START)``)."""

    async def on_intent_classified(self, event: IntentClassified) -> None:
        """Called after intent classification."""

    async def on_tool_call_start(self, event: Span) -> HookAction:
        """Called before a tool is executed (``Span(TOOL_CALL, START)``).

        Return action to control execution.
        """
        return HookAction.CONTINUE

    async def on_tool_call_result(self, event: Span) -> None:
        """Called after a tool completes (``Span(TOOL_CALL, COMPLETE)``)."""

    async def on_text_delta(self, event: TextDelta) -> None:
        """Called for each streaming text chunk."""

    async def on_step_start(self, event: Span) -> None:
        """Called when a plan step begins (``Span(STEP, START)``)."""

    async def on_step_complete(self, event: Span) -> None:
        """Called when a plan step completes (``Span(STEP, COMPLETE)``)."""

    async def on_plan_proposed(self, event: PlanProposed) -> None:
        """Called when a plan is proposed before execution."""

    async def on_citation_found(self, event: CitationFound) -> None:
        """Called when RAG verifies a citation."""

    async def on_evidence_insufficient(self, event: EvidenceInsufficient) -> None:
        """Called when RAG cannot find sufficient evidence."""

    async def on_memory_event(self, event: MemoryEvent) -> None:
        """Called when a memory section is modified (added / evicted / etc.)."""

    async def on_turn_complete(self, event: Span) -> None:
        """Called at the end of a turn (``Span(TURN, COMPLETE)``)."""

    async def on_turn_summary(self, event: TurnSummary) -> None:
        """Called with the typed turn-end aggregate (text / intent / usage)."""

    async def on_error(self, event: RunError) -> None:
        """Called when an error occurs during execution."""

    async def on_event(self, event: KaosEvent) -> None:
        """Called for *every* event — the catch-all fallback.

        Use this for hooks that don't care about event type (audit log,
        opaque metrics forwarding). Called *in addition to* the typed
        callbacks above; if you need to pick one or the other, override
        ``on_event`` and skip the typed methods.

        Default no-op so existing hooks aren't disrupted.
        """


__all__ = [
    "HookAction",
    "KaosHook",
]
