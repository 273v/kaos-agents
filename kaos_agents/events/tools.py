"""Tool-execution payloads + the one tool event that isn't a span.

Tool-call lifecycle (start / complete / error) is now :class:`Span`
events with ``subject=SpanSubject.TOOL_CALL`` — see
:mod:`kaos_agents.events.spans`. This module retains:

- :class:`ToolCallSummary` — the compact per-tool aggregate that ships
  inside :class:`TurnSummary.tool_calls` for end-of-turn telemetry.
- :class:`ToolCallApprovalRequired` — the one tool-related event that
  isn't a phase boundary: it's a *control-flow signal* that pauses the
  run and surfaces a ``run_state_ref`` for resume. Pause/resume is a
  distinct concept from start/complete, so it stays a typed event.
"""

from __future__ import annotations

from kaos_core.types.content import KaosModel
from pydantic import ConfigDict

from kaos_agents.events._intermediates import LifecycleEvent


class ToolCallSummary(KaosModel):
    """Compact summary of a tool call for :class:`TurnSummary`.

    ``cost_usd`` aggregates the LLM cost attributable to this tool
    invocation when the tool itself drove an LLM call (e.g. RAG's
    rag-query verifier, or a delegated sub-agent). Plain tools that
    don't call an LLM stay at ``0.0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tool_name: str
    call_id: str
    is_error: bool = False
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class ToolCallApprovalRequired(LifecycleEvent):
    """The runtime needs human approval before executing a tool.

    The run pauses after yielding this event. Resume via
    ``Runner.resume()`` with the ``run_state_ref``. See Phase 6 of the
    streaming roadmap.

    This is *not* a :class:`Span` because pause/resume is a control-flow
    signal with persistence semantics — distinct from a generic phase
    transition. Consumers handle it specifically.
    """

    call_id: str = ""
    tool_name: str = ""
    arguments: tuple[tuple[str, str], ...] = ()
    reason: str = ""
    run_state_ref: str = ""  # VFS path or ID for serialized RunState


__all__ = [
    "ToolCallApprovalRequired",
    "ToolCallSummary",
]
