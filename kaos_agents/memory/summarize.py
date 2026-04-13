"""Memory summarization — compress section contents using LLM.

Summarization is triggered by:
- ON_OVERFLOW: When eviction would drop items, summarize them first
- ON_TURN: At end of each turn, summarize the entire section

The summarizer uses kaos-llm-core Call to compress a list of memory items
into a single summary item that preserves key information within fewer tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.settings import DEFAULT_MODEL

if TYPE_CHECKING:
    from kaos_agents.memory.types import MemoryItem, MemoryType

logger = get_logger(__name__)


async def summarize_items(
    items: list[MemoryItem],
    section_type: MemoryType,
    *,
    model: str = DEFAULT_MODEL,
    target_tokens: int = 200,
    chars_per_token: float = 4.0,
) -> str:
    """Summarize a list of memory items into a compact text.

    Uses Call(SummarizeSig) to compress items while preserving key information.

    Args:
        items: Items to summarize.
        section_type: What kind of section these items are from (for prompt context).
        model: LLM model for summarization.
        target_tokens: Approximate target length of the summary in tokens.
        chars_per_token: Character-to-token ratio for target length guidance.

    Returns:
        Summary text. If LLM call fails, falls back to a mechanical concatenation
        of item first lines.
    """
    if not items:
        return ""

    # Build the content to summarize
    content = "\n---\n".join(item.content for item in items)
    target_chars = int(target_tokens * chars_per_token)

    try:
        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core import Call, InputField, OutputField, Signature

        class SummarizeMemory(Signature):
            """Summarize memory items into a concise summary preserving key facts."""

            content: str = InputField(description="The content to summarize")
            section_type: str = InputField(description="What kind of memory this is")
            target_length: str = InputField(description="Target length guidance")
            summary: str = OutputField(description="Concise summary preserving key facts")

        call = Call(
            SummarizeMemory,
            model=model,
            instructions=(
                "Summarize the following memory items concisely. "
                "Preserve key facts, names, dates, numbers, and decisions. "
                "Drop greetings, filler, and redundant phrasing. "
                "Use bullet points for multiple distinct facts."
            ),
        )

        result = await call(
            content=content,
            section_type=section_type.value,
            target_length=f"approximately {target_chars} characters ({target_tokens} tokens)",
        )

        summary = str(result.summary)
        logger.debug(
            "summarize: section=%s items=%d input_chars=%d output_chars=%d",
            section_type.value,
            len(items),
            len(content),
            len(summary),
        )
        return summary

    except Exception as exc:
        logger.warning("summarize: LLM failed (%s), using mechanical fallback", exc)
        return _fallback_summarize(items, target_chars)


def _fallback_summarize(items: list[MemoryItem], target_chars: int) -> str:
    """Mechanical fallback: first line of each item, truncated to target length."""
    lines = []
    total = 0
    for item in items:
        first_line = item.content.split("\n", 1)[0]
        if len(first_line) > 100:
            first_line = first_line[:97] + "..."
        if total + len(first_line) > target_chars:
            lines.append("...")
            break
        lines.append(f"- {first_line}")
        total += len(first_line) + 4  # account for "- " prefix and newline
    return "[Summary of prior items]\n" + "\n".join(lines)
