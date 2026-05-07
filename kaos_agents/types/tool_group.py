"""ToolGroup — collection-of-tools value type. Track 4 chunk T4-2.

Ports kelvin-agent's :class:`ActionGroup` pattern: a named bundle of
related tools with a description, used as a discovery surface so the
agent first picks a *group* (e.g. "extraction", "graph") and then
drills into the individual tools inside.

This replaces a flat tool catalogue with a 2-level hierarchy that
matches how lawyers / data scientists already reason about
capabilities ("the extraction tools" / "the graph tools" /
"the memory tools").

Holds tool **names** rather than instances so the value type is
JSON-round-trippable, the kaos-core :class:`KaosTool` doesn't have
to leak into the data layer, and one tool can belong to multiple
groups without identity headaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ToolGroup:
    """Named bundle of related tools.

    Identity is the ``name``; two groups with the same name are equal
    iff their tool lists and description match.

    Args:
        name: Stable identifier (kebab-case, e.g. "graph", "extraction").
        description: Human-readable summary — surfaced in the prompt
            when the agent renders a 2-level catalogue.
        tool_names: Tuple of tool names that belong to this group. Order
            is significant — used as the rendering order in catalogues.
        tags: Free-form tags for cross-cutting filters (``("readonly",)``,
            ``("destructive",)``, etc.). Don't affect lookup.
    """

    name: str
    description: str
    tool_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolGroup.name must be a non-empty string")

    def __len__(self) -> int:
        return len(self.tool_names)

    def __contains__(self, tool_name: object) -> bool:
        return isinstance(tool_name, str) and tool_name in self.tool_names

    def __iter__(self):
        return iter(self.tool_names)


__all__ = ["ToolGroup"]
