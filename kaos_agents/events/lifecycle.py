"""Turn-level value events — facts that carry typed payload beyond a phase.

Phase boundaries (TurnStart / TurnComplete) are now :class:`Span` events
in :mod:`kaos_agents.events.spans`. This module owns events that report
*facts* the agent observed at the turn level — the intent it inferred,
an LLM invocation's token usage, a fatal error, and the typed aggregate
that fires alongside ``Span(TURN, COMPLETE)``.
"""

from __future__ import annotations

from kaos_agents.events._intermediates import LifecycleEvent
from kaos_agents.events.tools import ToolCallSummary


class TurnSummary(LifecycleEvent):
    """The typed aggregate that ships alongside ``Span(TURN, COMPLETE)``.

    Carries the final response text, the inferred intent, the
    per-tool-call summaries, and the cumulative token+cost numbers for
    the turn. Consumers that need to record "the turn ended with X"
    listen for ``TurnSummary``; consumers that need pure span structure
    (e.g. OTel exporters) listen for ``Span(TURN, COMPLETE)``.

    Pre-Span this was ``TurnComplete`` — the rename signals the new
    role: it's a *fact* about the turn's outcome, not a phase marker.
    """

    text: str = ""
    intent: str = ""
    tool_calls: tuple[ToolCallSummary, ...] = ()
    tokens_used: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class UsageObserved(LifecycleEvent):
    """Emitted each time an LLM invocation completes during a turn.

    Streaming handlers yield one ``UsageObserved`` per completed
    ``Call`` / ``Program`` with its ``Invocation.usage`` surfaced verbatim.
    The turn loop accumulates these into the aggregate fields on the
    final ``TurnSummary``. Downstream hooks (OTel, cost-aware routing,
    budget enforcement) can consume individual events for fine-grained
    attribution — which program fired, at which step, for how much.

    ``source`` is a short string the emitter chose to distinguish
    concurrent sub-invocations (e.g. ``"react"``, ``"respond"``,
    ``"classify"``, or the name of a delegated subagent). It's
    informational, not a routing key.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    source: str = ""


class IntentClassified(LifecycleEvent):
    """Emitted after intent classification completes.

    The ``intent`` is an :class:`IntentType` value (one of ``respond``,
    ``tool_use``, ``research``, ``plan``, ``clarify``).
    """

    intent: str = ""
    confidence: float = 0.0
    reasoning: str = ""


class RunError(LifecycleEvent):
    """A fatal, unrecoverable error during execution.

    For typed error subtypes that the runtime can react to (e.g.
    :class:`kaos_agents.events.budget.BudgetExceeded`,
    :class:`GroundingRefusalTriggered`), see those classes. ``RunError``
    is the generic terminal-error fallback — the runtime also raises
    it as an exception so in-process consumers can catch it either
    via the event stream or the exception path.

    Fields follow the KAOS agent-friendly error pattern:
    (1) what went wrong, (2) how to fix it, (3) alternative approach.
    """

    error_type: str = ""
    message: str = ""
    recovery_hint: str = ""


class PatternMismatch(LifecycleEvent):
    """The intent classifier picked a pattern the running agent can't dispatch.

    Emitted by :meth:`BaseAgent._handle_plan` /
    :meth:`BaseAgent._handle_research` when the session-level
    ``pattern`` field doesn't include the dispatch handler the
    per-turn intent demands — most commonly ``pattern="chat"`` on
    the session but the per-turn :class:`~kaos_agents.intent.types.IntentResult`
    returned :attr:`~kaos_agents.types.intents.IntentType.PLAN` or
    :attr:`~kaos_agents.types.intents.IntentType.RESEARCH`.

    Pre-0.1.0a10 ``BaseAgent`` silently degraded to
    :meth:`_handle_respond` in that case, producing training-data
    answers without tools and masking the dispatch hole behind a
    confident-looking response. That fall-through is the root cause
    of the R1-REAL v2-matrix Tests 3 + 7 regression — see
    ``kaos-modules/docs/plans/kaos-agents-autonomy-improvement-1.md``
    for the diagnosis.

    Observers should treat this event as a strong "this turn is
    going to under-perform" signal — the fallback path
    (:meth:`_handle_tool_use` or :meth:`_handle_respond`) is a
    degraded mode, not the intended one. Callers can react by
    re-opening the session with the recommended pattern (e.g., a
    future ``pattern="auto"`` wrapper would consume this event to
    switch agents mid-run).

    Fields:

    * ``classified_intent`` — :attr:`~kaos_agents.types.intents.IntentType.value`
      the classifier returned (e.g., ``"plan"``, ``"research"``).
    * ``agent_pattern`` — :attr:`~kaos_agents.config.AgentPattern.value`
      of the running agent (e.g., ``"chat"``).
    * ``recommended_pattern`` — :attr:`~kaos_agents.config.AgentPattern.value`
      the agent recommends opening for this kind of intent
      (e.g., ``"plan"``).
    * ``fallback_handler`` — name of the handler the runtime fell
      through to (e.g., ``"_handle_tool_use"``,
      ``"_handle_respond"``).
    * ``rationale`` — one-sentence human-readable explanation
      suitable for surfacing in a UI or log.
    """

    classified_intent: str = ""
    agent_pattern: str = ""
    recommended_pattern: str = ""
    fallback_handler: str = ""
    rationale: str = ""


__all__ = [
    "IntentClassified",
    "PatternMismatch",
    "RunError",
    "TurnSummary",
    "UsageObserved",
]
