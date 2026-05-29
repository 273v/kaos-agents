"""Intent classification — routes user messages to the right dispatch pattern.

Uses a lightweight Call with a classification Signature. The classifier
examines the assembled memory context (recent messages, available tools,
loaded documents) to determine the user's intent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from kaos_core.logging import get_logger
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents._constants import (
    HEURISTIC_CONFIDENCE_DEFAULT,
    HEURISTIC_CONFIDENCE_GREETING,
    HEURISTIC_CONFIDENCE_PLAN,
    HEURISTIC_CONFIDENCE_RESEARCH,
    HEURISTIC_CONFIDENCE_TOOL_USE,
)
from kaos_agents.settings import DEFAULT_MODEL
from kaos_agents.types import IntentResult, IntentType, InvocationUsage

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryItem

logger = get_logger(__name__)

# Heuristic keyword lists for fallback classification.
# These are module constants (not hardcoded in function bodies) so they
# can be audited, tested, and potentially made configurable.
_GREETING_WORDS = frozenset({"hello", "hi", "thanks", "ok", "yes", "no", "bye"})
_QUESTION_WORDS = frozenset({"what", "how", "why", "when", "where", "who"})
_ACTION_WORDS = frozenset(
    {"extract", "search", "find", "get", "fetch", "download", "parse", "analyze"}
)
_PLAN_PHRASES = ("then", "after that", "first", "step by step", "steps to")


class ClassifyIntentSignature(Signature):
    """Classify the user's intent for routing to the appropriate handler.

    Choose ONE category:

    * **respond** — simple conversational response, greeting, or
      acknowledgement. No tools needed AND no real-world fact-
      checking needed.
    * **tool_use** — the user wants to perform an action that
      requires calling tools, or wants a verifiable answer to a
      question about a real-world entity / fact that a registered
      tool category can address.
    * **research** — the user is asking a question about loaded
      documents that requires retrieval and reasoning over
      document content.
    * **plan** — the user wants a multi-step workflow (analyze a
      document, then extract specific data, then summarize
      findings).
    * **clarify** — the user's request is ambiguous and you need
      more information before proceeding.

    Routing heuristics:

    * Read ``available_tool_categories`` to see what tool
      categories are registered THIS turn. Each non-empty line is
      one category — ``"<name>: <one-sentence purpose>"``. If a
      category covers the user's question (e.g. a category whose
      purpose mentions the kind of entity / domain the user asked
      about), prefer ``tool_use`` (or ``research`` when the
      relevant category is about reasoning over loaded documents).
      Memory-only ``respond`` is unsafe when a relevant category
      is in the catalog — the catalog exists precisely so you can
      verify rather than guess from training data.
    * When the question is about a CURRENT real-world entity
      (current officeholder, price, market, rule status, filing,
      organizational fact) AND any relevant tool category is
      listed in ``available_tool_categories``, prefer
      ``tool_use`` over ``respond``. Answering from training
      memory is unsafe for current-state questions.
    * When relevant catalog categories exist and documents are
      also loaded, and the question relates to those documents'
      content, prefer ``research`` — the document-retrieval
      categories are still the catalog category that fits.
    * If the user mentions specific tools or explicit actions,
      prefer ``tool_use``.
    * If the request involves multiple sequential steps, prefer
      ``plan``.
    * When ``available_tool_categories`` is empty (no tools
      registered this turn), the catalog-aware bullets above are
      no-ops. Fall back to the message-shape signal: pure
      conversational small-talk → ``respond``; everything else
      → ``tool_use`` (it remains the safest general default).
    * When in doubt between ``tool_use`` and ``research``, prefer
      ``tool_use`` (it's more general).

    The catalog-aware rules are deliberately abstract — you read
    the registered categories at classification time and pick the
    routing category whose semantics fit. The Signature carries
    no hardcoded tool names; that's why ``available_tool_categories``
    is an input rather than baked into this prompt.
    """

    message: str = InputField(description="The user's message to classify.")
    conversation_context: str = InputField(
        description="Recent conversation history and assembled memory context."
    )
    available_tool_categories: str = InputField(
        default="",
        description=(
            "Newline-separated list of tool categories registered on "
            "the runtime THIS turn. Each non-empty line is one "
            'category in the shape ``"<name>: <one-sentence '
            'purpose>"``. The classifier uses this to decide '
            "whether a relevant tool category fits the user's "
            "question — see the routing-heuristics block in the "
            "class docstring. Default empty string preserves the "
            "pre-fix path: with no registered categories the "
            "catalog-aware bullets are no-ops and routing falls "
            "back to message-shape heuristics only."
        ),
    )
    intent: Literal["respond", "tool_use", "research", "plan", "clarify"] = OutputField(
        description=(
            "Routing category. Constrained to the five values above so "
            "malformed model output triggers codec validation retry "
            "rather than silent fallback to RESPOND."
        )
    )
    confidence: float = OutputField(
        description="Confidence in the classification, in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = OutputField(description="One-sentence explanation of the classification.")


async def classify_intent(
    user_message: str,
    memory: SessionMemory,
    *,
    model: str = DEFAULT_MODEL,
    context_items: dict[str, list[MemoryItem]] | None = None,
    context_text: str = "",
    available_tool_categories: str = "",
) -> IntentResult:
    """Classify user intent using an LLM.

    Args:
        user_message: The user's message to classify.
        memory: The session memory for context.
        model: LLM model to use for classification.
        context_items: Pre-assembled context (if already computed).
        context_text: Pre-assembled context as text (passed to LLM).
        available_tool_categories: Newline-separated catalog of tool
            categories registered on the live runtime, in
            ``"<name>: <purpose>"`` form. Passed straight through to
            the :class:`ClassifyIntentSignature` so the LLM can reason
            about which category fits the user's question. Default
            ``""`` keeps backward compatibility — callers that don't
            populate this preserve the pre-0.1.0a17 routing behavior.

    Returns:
        IntentResult with intent type, confidence, and reasoning.

    Raises:
        Exception: When the LLM raised an auth-failure, rate-limit,
            service-unavailable, transport, or context-too-large error.
            These are **actionable infrastructure failures** that the
            runtime must surface as :class:`~kaos_agents.events.RunError`
            rather than silently fall back to the heuristic — otherwise
            the agent would claim "successful" classification when no
            LLM is reachable, producing a ``ToolResult(isError=False,
            text="")`` that breaks SOC2 alerting. The runtime's
            ``_run_inner`` catches this and emits a structured RunError.

    For non-actionable LLM failures (e.g. a transient codec decode
    failure, an unexpected provider response shape, or an output
    validation retry exhaustion), this function falls back to the
    heuristic classifier — those are recoverable from a UX perspective.
    """
    from kaos_agents.errors import classify_agent_failure

    try:
        return await _classify_with_llm(
            user_message,
            memory,
            model=model,
            context_text=context_text,
            available_tool_categories=available_tool_categories,
        )
    except Exception as exc:
        failure = classify_agent_failure(exc)
        if failure is not None and failure.is_surfacing:
            # Auth / rate-limit / service-unavailable / transport /
            # context-too-large: do NOT silently fall back. The whole
            # point of probe 4b's fix is that a missing API key MUST
            # surface as an error, not as a heuristic-classified
            # "successful" turn that produces an empty response.
            logger.warning(
                "classify_intent: actionable LLM failure (%s, kind=%s, provider=%s); "
                "re-raising for RunError surfacing",
                type(exc).__name__,
                failure.kind,
                failure.provider or "unknown",
            )
            raise
        logger.warning(
            "classify_intent: LLM classification failed (%s: %s), using heuristic fallback",
            type(exc).__name__,
            exc,
        )
        return _classify_heuristic(user_message, memory)


async def _classify_with_llm(
    user_message: str,
    memory: SessionMemory,
    *,
    model: str,
    context_text: str = "",
    available_tool_categories: str = "",
) -> IntentResult:
    """LLM-based intent classification.

    Uses ``Call.invoke()`` (not ``Call.__call__``) so the per-classification
    cost/usage flows through the standard ``Invocation`` path. The bare
    ``__call__`` form would silently discard ``Invocation.usage``, leaving a
    hole in the ``--max-cost`` ceiling that fires every turn.
    """
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core import Call

    if not context_text:
        from kaos_agents.types.memory import MemoryType

        recent = memory.get_recent(MemoryType.MESSAGES, 5)
        context_text = (
            "\n".join(item.content for item in recent) if recent else "(no prior messages)"
        )

    from kaos_agents._examples import load_examples

    call = Call(ClassifyIntentSignature, model=model, examples=load_examples("classify_intent"))
    invocation = await call.invoke(
        message=user_message,
        conversation_context=context_text,
        available_tool_categories=available_tool_categories,
    )

    out = invocation.output
    # `intent` is now Literal-typed at the Signature level; the codec's
    # validation retry path enforces the constraint upstream. We still
    # do `IntentType(...)` to lift str → enum, but we no longer need a
    # silent except-fallback — a mis-typed value would have raised
    # `ValidationRetryExhaustedError` in `call.invoke`.
    intent_type = IntentType(out.intent)
    confidence = max(0.0, min(1.0, float(out.confidence)))

    return IntentResult(
        intent=intent_type,
        confidence=confidence,
        reasoning=out.reasoning,
        # Plumb per-classification usage through to the agent's
        # `TurnComplete.usage` / the session cost ceiling. Was missing
        # before — every classification's tokens went unattributed,
        # leaving a hole in `--max-cost` enforcement that fired every
        # turn.
        usage=InvocationUsage.from_invocation(invocation),
    )


def _classify_heuristic(user_message: str, memory: SessionMemory) -> IntentResult:
    """Simple keyword-based fallback when LLM classification is unavailable.

    This is a last-resort fallback — not a replacement for LLM classification.
    """
    msg_lower = user_message.lower().strip()

    # Greetings and simple responses
    # Use word set intersection (not substring) to avoid "hi" matching "this"
    words = set(msg_lower.split())
    if len(words) <= 3 and words & _GREETING_WORDS:
        return IntentResult(
            intent=IntentType.RESPOND,
            confidence=HEURISTIC_CONFIDENCE_GREETING,
            reasoning="Short greeting/acknowledgment detected (heuristic).",
        )

    # Question words with loaded documents → research
    from kaos_agents.types.memory import MemoryType

    has_docs = (
        memory.has_section(MemoryType.DOCUMENTS)
        and memory.section_item_count(MemoryType.DOCUMENTS) > 0
    )
    if has_docs and words & _QUESTION_WORDS:
        return IntentResult(
            intent=IntentType.RESEARCH,
            confidence=HEURISTIC_CONFIDENCE_RESEARCH,
            reasoning="Question word with loaded documents (heuristic).",
        )

    # Multi-step indicators → plan (check before action words)
    if any(phrase in msg_lower for phrase in _PLAN_PHRASES):
        return IntentResult(
            intent=IntentType.PLAN,
            confidence=HEURISTIC_CONFIDENCE_PLAN,
            reasoning="Multi-step language detected (heuristic).",
        )

    # Action words → tool_use (word boundary match, not substring)
    if words & _ACTION_WORDS:
        return IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=HEURISTIC_CONFIDENCE_TOOL_USE,
            reasoning="Action keyword detected (heuristic).",
        )

    # Default: tool_use (most general)
    return IntentResult(
        intent=IntentType.TOOL_USE,
        confidence=HEURISTIC_CONFIDENCE_DEFAULT,
        reasoning="No strong signal — defaulting to tool_use (heuristic).",
    )
