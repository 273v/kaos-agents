"""Intent classification — routes user messages to the right dispatch pattern.

Uses a lightweight Call with a classification Signature. The classifier
examines the assembled memory context (recent messages, available tools,
loaded documents) to determine the user's intent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.models import IntentResult, IntentType

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryItem

logger = get_logger(__name__)

# Default model for classification (cheap, fast).
# In production, this is overridden by KaosAgentSettings.default_llm_model.
_DEFAULT_CLASSIFY_MODEL = "anthropic:claude-haiku-4-5"

# Heuristic keyword lists for fallback classification.
# These are module constants (not hardcoded in function bodies) so they
# can be audited, tested, and potentially made configurable.
_GREETING_WORDS = frozenset({"hello", "hi", "thanks", "ok", "yes", "no", "bye"})
_QUESTION_WORDS = frozenset({"what", "how", "why", "when", "where", "who"})
_ACTION_WORDS = frozenset(
    {"extract", "search", "find", "get", "fetch", "download", "parse", "analyze"}
)
_PLAN_PHRASES = ("then", "after that", "first", "step by step", "steps to")

# System instruction for the classifier
_CLASSIFY_INSTRUCTION = """Classify the user's intent into one of these categories:

- respond: Simple conversational response, greeting, or acknowledgment. No tools needed.
- tool_use: The user wants to perform an action that requires calling tools (extract data, search the web, analyze a file, etc.).
- research: The user is asking a question about loaded documents that requires retrieval and reasoning over document content.
- plan: The user wants a multi-step workflow (analyze a document, then extract specific data, then summarize findings).
- clarify: The user's request is ambiguous and you need more information before proceeding.

Consider the conversation history and available context when classifying.
If documents are loaded and the question relates to their content, prefer "research".
If the user mentions specific tools or actions, prefer "tool_use".
If the request involves multiple sequential steps, prefer "plan".
When in doubt between tool_use and research, prefer tool_use (it's more general).
"""


async def classify_intent(
    user_message: str,
    memory: SessionMemory,
    *,
    model: str = _DEFAULT_CLASSIFY_MODEL,
    context_items: dict[str, list[MemoryItem]] | None = None,
    context_text: str = "",
) -> IntentResult:
    """Classify user intent using an LLM.

    Args:
        user_message: The user's message to classify.
        memory: The session memory for context.
        model: LLM model to use for classification.
        context_items: Pre-assembled context (if already computed).
        context_text: Pre-assembled context as text (passed to LLM).

    Returns:
        IntentResult with intent type, confidence, and reasoning.
    """
    try:
        return await _classify_with_llm(
            user_message, memory, model=model, context_text=context_text
        )
    except Exception as exc:
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
) -> IntentResult:
    """LLM-based intent classification."""
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core import Call, InputField, OutputField, Signature

    class ClassifyIntent(Signature):
        """Classify the user's intent for routing to the appropriate handler."""

        message: str = InputField(description="The user's message to classify")
        conversation_context: str = InputField(
            description="Recent conversation history and assembled memory context"
        )
        intent: str = OutputField(description="One of: respond, tool_use, research, plan, clarify")
        confidence: float = OutputField(description="Confidence score 0.0 to 1.0")
        reasoning: str = OutputField(description="Brief explanation of the classification")

    # Use provided context, or build from memory if not provided
    if not context_text:
        from kaos_agents.memory.types import MemoryType

        recent = memory.get_recent(MemoryType.MESSAGES, 5)
        context_text = (
            "\n".join(item.content for item in recent) if recent else "(no prior messages)"
        )

    call = Call(ClassifyIntent, model=model, instructions=_CLASSIFY_INSTRUCTION)
    result = await call(message=user_message, conversation_context=context_text)

    # Parse the intent string with validation logging
    raw_intent = result.intent.lower().strip()
    try:
        intent_type = IntentType(raw_intent)
    except ValueError:
        logger.warning(
            "classify_intent: LLM returned invalid intent '%s', falling back to RESPOND",
            raw_intent,
        )
        intent_type = IntentType.RESPOND

    raw_confidence = float(result.confidence)
    if not 0.0 <= raw_confidence <= 1.0:
        logger.debug("classify_intent: LLM confidence %.2f out of [0,1], clamping", raw_confidence)
    confidence = max(0.0, min(1.0, raw_confidence))

    return IntentResult(
        intent=intent_type,
        confidence=confidence,
        reasoning=result.reasoning,
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
            confidence=0.8,
            reasoning="Short greeting/acknowledgment detected (heuristic).",
        )

    # Question words with loaded documents → research
    from kaos_agents.memory.types import MemoryType

    has_docs = (
        memory.has_section(MemoryType.DOCUMENTS)
        and memory.section_item_count(MemoryType.DOCUMENTS) > 0
    )
    if has_docs and words & _QUESTION_WORDS:
        return IntentResult(
            intent=IntentType.RESEARCH,
            confidence=0.6,
            reasoning="Question word with loaded documents (heuristic).",
        )

    # Multi-step indicators → plan (check before action words)
    if any(phrase in msg_lower for phrase in _PLAN_PHRASES):
        return IntentResult(
            intent=IntentType.PLAN,
            confidence=0.5,
            reasoning="Multi-step language detected (heuristic).",
        )

    # Action words → tool_use (word boundary match, not substring)
    if words & _ACTION_WORDS:
        return IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.6,
            reasoning="Action keyword detected (heuristic).",
        )

    # Default: tool_use (most general)
    return IntentResult(
        intent=IntentType.TOOL_USE,
        confidence=0.4,
        reasoning="No strong signal — defaulting to tool_use (heuristic).",
    )
