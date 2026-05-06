"""Recall primitive — reads from memory to build context for other primitives.

This is the primitive that closes the Reflexion feedback loop. Without
explicit recall, the agent writes failures to memory but never reads them
back into planning. Making recall a first-class primitive ensures:
- Context assembly is testable in isolation
- Memory reads are auditable (which items from which sections contributed)
- Budget-awareness (respects token limits on assembled context)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.memory.types import MemoryType

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Context:
    """The result of a Recall — assembled text with provenance."""

    text: str  # Assembled context for injection into LLM prompts
    items: tuple[str, ...] = ()  # Item IDs that contributed
    sections_used: tuple[str, ...] = ()  # Section names that contributed
    token_count: int = 0


def recall(
    memory: SessionMemory,
    sections: list[MemoryType],
    *,
    # 64K is the per-recall budget, sized to give the agent a generous
    # slice of available context (out of the 200K
    # default_context_budget_tokens). Was 8K — fine for GPT-3.5, way
    # too small for 2026 frontier models with 200K-1M contexts.
    budget_tokens: int = 64_000,
    priority_order: list[MemoryType] | None = None,
) -> Context:
    """Retrieve context from memory for use by other primitives.

    Assembles text from requested memory sections within a token budget.
    Sections are included in priority order; lowest-priority sections
    are trimmed first if the total exceeds the budget.

    Args:
        memory: The session memory to read from.
        sections: Which memory sections to include.
        budget_tokens: Maximum total tokens in the assembled context.
        priority_order: Sections listed first are highest priority (trimmed last).

    Returns:
        Context with assembled text and provenance metadata.
    """
    section_items = memory.get_sections(
        sections,
        total_budget_tokens=budget_tokens,
        priority_order=priority_order,
    )

    parts: list[str] = []
    all_item_ids: list[str] = []
    all_sections: list[str] = []
    total_tokens = 0

    for section_type, items in section_items.items():
        if not items:
            continue
        all_sections.append(section_type.value)
        section_text = "\n".join(item.content for item in items)
        parts.append(f"=== {section_type.value.upper()} ===\n{section_text}")
        for item in items:
            all_item_ids.append(item.id)
            total_tokens += item.token_count

    text = "\n\n".join(parts) if parts else ""

    logger.debug(
        "recall: %d sections, %d items, %d tokens (budget %d)",
        len(all_sections),
        len(all_item_ids),
        total_tokens,
        budget_tokens,
    )

    return Context(
        text=text,
        items=tuple(all_item_ids),
        sections_used=tuple(all_sections),
        token_count=total_tokens,
    )
