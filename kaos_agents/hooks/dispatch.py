"""Hook dispatch — Runner-side helper that fans an event out to all hooks.

Lives in its own module so :class:`KaosHook` (in ``base.py``) doesn't
have to know about every concrete event type. The dispatcher walks the
hook tuple in registration order, routes each event to the matching
typed callback (plus the catch-all :meth:`KaosHook.on_event`), and
collects the most-restrictive :class:`HookAction` for events that
support tool-call gating.

Hook exceptions are logged and swallowed — hooks never affect the
control flow of the agent loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.hooks.base import HookAction, KaosHook

if TYPE_CHECKING:
    from kaos_agents.events import KaosEvent

logger = get_logger(__name__)


# Priority order for actions (higher index = more restrictive).
# Module-level constant — avoids re-creating on every dispatch_hook call.
_ACTION_PRIORITY: dict[HookAction, int] = {
    HookAction.CONTINUE: 0,
    HookAction.SKIP: 1,
    HookAction.REQUIRE_APPROVAL: 2,
}


async def dispatch_hook(
    hooks: tuple[KaosHook, ...],
    event: KaosEvent,
) -> HookAction:
    """Run all hooks for an event, returning the most restrictive action.

    For events that return HookAction (on_tool_call_start), the most
    restrictive action wins: REQUIRE_APPROVAL > SKIP > CONTINUE.

    For events that return None, this always returns CONTINUE.

    Hook exceptions are logged and swallowed — hooks never affect
    the control flow of the agent loop.
    """
    # Lazy import to avoid the events ↔ hooks circular at module load time.
    from kaos_agents.events import (
        CitationFound,
        EvidenceInsufficient,
        IntentClassified,
        MemoryEvent,
        PlanProposed,
        RunError,
        Span,
        SpanPhase,
        SpanSubject,
        TextDelta,
        TurnSummary,
    )

    most_restrictive = HookAction.CONTINUE

    for hook in hooks:
        try:
            # Catch-all first — hooks can opt to observe everything via on_event.
            await hook.on_event(event)

            if isinstance(event, Span):
                # Route by (subject, phase). Subjects/phases not listed
                # below (PROGRESS, CANCELLED, RUN, SUBAGENT, HANDOFF,
                # LLM_CALL, ERROR phases) are intentionally not routed
                # to typed KaosHook methods — consumers needing them
                # implement on_event() to observe them generically.
                if event.subject == SpanSubject.TURN:
                    if event.phase == SpanPhase.START:
                        await hook.on_turn_start(event)
                    elif event.phase == SpanPhase.COMPLETE:
                        await hook.on_turn_complete(event)
                elif event.subject == SpanSubject.TOOL_CALL:
                    if event.phase == SpanPhase.START:
                        action = await hook.on_tool_call_start(event)
                        if _ACTION_PRIORITY.get(action, 0) > _ACTION_PRIORITY[most_restrictive]:
                            most_restrictive = action
                    elif event.phase == SpanPhase.COMPLETE:
                        await hook.on_tool_call_result(event)
                elif event.subject == SpanSubject.STEP:
                    if event.phase == SpanPhase.START:
                        await hook.on_step_start(event)
                    elif event.phase == SpanPhase.COMPLETE:
                        await hook.on_step_complete(event)
            elif isinstance(event, IntentClassified):
                await hook.on_intent_classified(event)
            elif isinstance(event, TextDelta):
                await hook.on_text_delta(event)
            elif isinstance(event, PlanProposed):
                await hook.on_plan_proposed(event)
            elif isinstance(event, CitationFound):
                await hook.on_citation_found(event)
            elif isinstance(event, EvidenceInsufficient):
                await hook.on_evidence_insufficient(event)
            elif isinstance(event, MemoryEvent):
                await hook.on_memory_event(event)
            elif isinstance(event, TurnSummary):
                await hook.on_turn_summary(event)
            elif isinstance(event, RunError):
                await hook.on_error(event)
        except Exception as exc:
            logger.warning(
                "Hook %s raised %s on %s (swallowed): %s",
                type(hook).__name__,
                type(exc).__name__,
                type(event).__name__,
                exc,
            )

    return most_restrictive


__all__ = ["dispatch_hook"]
