"""Tool-filter helper — applies a :class:`SessionToolSet` to a tool list.

Track 4 chunk T4-5. Pure function; no I/O, no module-level state. The
bridge layer (``bridge_tools`` in T6) will call this when assembling
the per-session tool surface; for now, it's a callable that any agent
plumbing can use to enforce the session's allow / deny rules.

Filter rules (applied in order, see :class:`SessionToolSet` docstring):

1. Tool blocked by ``denied_tools`` → drop (always wins).
2. ``allowed_tools`` non-empty + tool not listed → drop.
3. ``allowed_groups`` non-empty + tool not in any allowed group → drop.
4. Otherwise → keep.

Tools without a registered group still pass step 3 *iff* they passed
step 2 (i.e. were explicitly allowed). This avoids a session config
silently dropping freshly-registered tools that haven't been grouped
yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from kaos_agents.registry import default_tool_group_registry

if TYPE_CHECKING:
    from kaos_agents.registry.tool_group_registry import ToolGroupRegistry
    from kaos_agents.types.session_tool_set import SessionToolSet


def _tool_name(t: Any) -> str | None:
    """Extract the tool's name from a KaosTool instance or ToolMetadata.

    Mirrors :func:`kaos_agents.context.tool_catalog._tool_metadata_iter`
    — accepts tools by either ``.metadata.name`` or ``.name`` shape.
    """
    meta = getattr(t, "metadata", None)
    if meta is not None:
        if callable(meta):
            meta = meta()
        return getattr(meta, "name", None)
    return getattr(t, "name", None)


def filter_tools(
    tools: Sequence[Any],
    tool_set: SessionToolSet,
    *,
    group_registry: ToolGroupRegistry | None = None,
) -> list[Any]:
    """Return only the tools the session is permitted to use.

    Args:
        tools: Sequence of KaosTool or ToolMetadata instances.
        tool_set: Per-session allow / deny config.
        group_registry: Where to resolve ``allowed_groups`` membership.
            Default is :data:`default_tool_group_registry`.

    Returns:
        New list (never mutates input). Order from ``tools`` preserved.
        When ``tool_set`` is unrestricted, returns ``list(tools)``.

    Tools whose name cannot be extracted are dropped — better to fail
    closed than to silently leak unrecognised tool shapes.
    """
    if tool_set.is_unrestricted:
        return list(tools)

    registry = group_registry or default_tool_group_registry

    # Pre-compute group membership for every named tool. ToolGroupRegistry
    # already provides ``groups_for_tool`` so this is just a lookup.
    out: list[Any] = []
    for t in tools:
        name = _tool_name(t)
        if name is None:
            continue
        if name in tool_set.denied_tools:
            continue

        # Step 2: explicit tool allow-list. When this list is set
        # AND the tool is in it, accept; otherwise fall through to
        # the group check (which may still allow via group membership).
        if tool_set.allowed_tools and name in tool_set.allowed_tools:
            out.append(t)
            continue

        # Step 3: group allow-list
        if tool_set.allowed_groups:
            tool_groups = set(registry.groups_for_tool(name))
            if tool_groups & tool_set.allowed_groups:
                out.append(t)
            # else: blocked by group filter
            continue

        # Neither allowed_tools nor allowed_groups specified → allow
        # (denied_tools already filtered above).
        if not tool_set.allowed_tools and not tool_set.allowed_groups:
            out.append(t)

    return out


__all__ = ["filter_tools"]
