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
from kaos_llm_core import InputField, OutputField, Signature

from kaos_agents._constants import (
    FALLBACK_LINE_OVERHEAD,
    FALLBACK_SUMMARY_LINE_MAX,
    SUMMARIZE_FALLBACK_TOKENS,
)
from kaos_agents.settings import DEFAULT_MODEL

if TYPE_CHECKING:
    from kaos_agents.types.memory import MemoryItem, MemoryType

logger = get_logger(__name__)


class SummarizeMemorySignature(Signature):
    """Summarize memory items into a concise summary preserving key facts.

    Compress the input content while preserving:

    * Key facts, names, dates, numbers, and decisions
    * Distinct claims that downstream context assembly needs

    Drop greetings, filler, and redundant phrasing. Use bullet points
    for multiple distinct facts. Target the requested length — pack
    information density rather than padding to hit it.
    """

    content: str = InputField(description="The content to summarize.")
    section_type: str = InputField(description="What kind of memory this is.")
    target_length: str = InputField(description="Target length guidance for the summary.")
    summary: str = OutputField(description="Concise summary preserving key facts.")


async def summarize_items(
    items: list[MemoryItem],
    section_type: MemoryType,
    *,
    model: str = DEFAULT_MODEL,
    target_tokens: int = SUMMARIZE_FALLBACK_TOKENS,
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
        from kaos_llm_core import Call

        from kaos_agents._examples import load_examples

        call = Call(
            SummarizeMemorySignature, model=model, examples=load_examples("summarize_memory")
        )
        # Use ``invoke`` (not bare ``__call__``) so per-summarisation
        # token cost flows through ``Invocation.usage``. ON_OVERFLOW +
        # ON_TURN summarisations fire repeatedly across a session;
        # bare-call form silently breaches the ``--max-cost`` ceiling.
        invocation = await call.invoke(
            content=content,
            section_type=section_type.value,
            target_length=f"approximately {target_chars} characters ({target_tokens} tokens)",
        )
        result = invocation.output
        usage = invocation.usage

        summary = str(result.summary)
        logger.debug(
            "summarize: section=%s items=%d input_chars=%d output_chars=%d cost_usd=%.6f",
            section_type.value,
            len(items),
            len(content),
            len(summary),
            float(getattr(usage, "cost_usd", 0.0) or 0.0),
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
        if len(first_line) > FALLBACK_SUMMARY_LINE_MAX:
            first_line = first_line[: FALLBACK_SUMMARY_LINE_MAX - 3] + "..."
        if total + len(first_line) > target_chars:
            lines.append("...")
            break
        lines.append(f"- {first_line}")
        # account for "- " prefix and newline
        total += len(first_line) + FALLBACK_LINE_OVERHEAD
    return "[Summary of prior items]\n" + "\n".join(lines)
