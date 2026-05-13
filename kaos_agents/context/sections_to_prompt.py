"""Bind memory sections to a :class:`Signature`'s input fields.

Translates kelvin-agent's ``MemoryInstructAction.get_sections()`` /
``get_prompt()`` machinery (kelvin/agent/actions/patterns/memory_instruct.py:69-401)
into a kaos-llm-core-aware helper. Where kelvin assembled a
``JSONInstructTemplate`` string from declared section names, kaos
binds the rendered section content to typed Signature input fields so
``Call.invoke(**bindings)`` consumes them directly.

Pairs with chunk A1's :class:`AgentToolSpec.memory_sections` — a tool
declares which memory sections it wants, the bridge calls
:func:`bind_sections` with a section-to-field mapping, and the result
goes straight into the LLM call. No string concatenation gymnastics.

Example::

    from kaos_agents.context.sections_to_prompt import bind_sections
    from kaos_agents.types import MemoryType

    bindings = bind_sections(
        memory,
        section_to_field={
            MemoryType.MESSAGES: "conversation_history",
            MemoryType.FINDINGS: "verified_findings",
        },
        most_recent=5,
    )
    response = await call.invoke(message="What's new?", **bindings)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory
    from kaos_agents.types.memory import MemoryType


_EMPTY_SECTION_PLACEHOLDER = "(empty)"


def bind_sections(
    memory: SessionMemory,
    section_to_field: dict[MemoryType, str],
    *,
    most_recent: int | None = None,
    separator: str = "\n",
    empty_placeholder: str = _EMPTY_SECTION_PLACEHOLDER,
    item_prefix: str = "",
) -> dict[str, str]:
    """Render memory sections into a Signature-bindable kwargs dict.

    For each ``(section, field_name)`` pair in ``section_to_field``,
    fetches the section's items and joins their ``.content`` strings
    with ``separator``. Returns a ``dict[field_name, str]`` ready to
    unpack into ``Signature.invoke(**bindings)``.

    Args:
        memory: The session memory to read from.
        section_to_field: Mapping from :class:`MemoryType` to the
            Signature's InputField name. Each key is a section to
            include; each value is the field name on the Signature.
        most_recent: When set, only the N most recent items per
            section are rendered. ``None`` = all items in the section
            (subject to budget; sections enforce their own caps).
        separator: String inserted between items. Default ``"\\n"``;
            pass ``"\\n\\n"`` for paragraph spacing.
        empty_placeholder: Rendered string for empty sections. Default
            ``"(empty)"``. Pass ``""`` to render empty sections as
            empty strings (the LLM may omit them entirely).
        item_prefix: Optional prefix prepended to every item before
            joining (e.g., ``"- "`` for bulleted rendering). Default
            empty.

    Returns:
        ``dict[str, str]`` mapping each requested field name to the
        rendered section content. Field names not in the mapping are
        absent — the caller is expected to know which Signature fields
        they're filling.

    Notes:
        - Item content is rendered as-is via ``MemoryItem.content``.
          For richer rendering (size-tiered field projection), pair
          with the :class:`FieldSet` work in Track 6.
        - Empty memory or unconfigured sections produce the placeholder
          rather than raising — this is the prompt-assembly path, not
          the validation path. Sections that need to exist are
          declared in :class:`SectionConfig` and fail loudly there.
    """
    out: dict[str, str] = {}

    for section, field_name in section_to_field.items():
        items = (
            memory.get_recent(section, most_recent)
            if most_recent is not None
            else memory.get(section)
        )

        if not items:
            out[field_name] = empty_placeholder
            continue

        parts = [f"{item_prefix}{item.content}" for item in items]
        out[field_name] = separator.join(parts)

    return out


__all__ = ["bind_sections"]
