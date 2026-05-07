"""ToolGroupRegistry — group-name → :class:`ToolGroup` catalogue.

Track 4 chunk T4-2. Mirrors :class:`AgentToolSpecRegistry` and
:class:`ToolDataTypeRegistry` shapes; this one holds the 2-level
discovery surface (group → tools).

Lookup semantics: unknown group names return ``None`` (not an empty
group), because callers normally want to know whether the group
exists before drilling in. The empty-spec pattern from the other
registries doesn't fit here — an empty group with the wrong name
would lie about its identity.

A tool can appear in multiple groups; the registry doesn't enforce
single-membership. Use :meth:`groups_for_tool` to discover which
groups a given tool name participates in.
"""

from __future__ import annotations

from kaos_core.exceptions import RegistryError

from kaos_agents.types.tool_group import ToolGroup


class ToolGroupRegistry:
    """Catalogue of tool groups keyed by group name.

    Typical usage::

        from kaos_agents.registry import default_tool_group_registry
        from kaos_agents.types import ToolGroup

        default_tool_group_registry.register(
            ToolGroup(
                name="extraction",
                description="Schema-driven structured extraction tools.",
                tool_names=("kaos-extract-schema", "kaos-extract-corpus"),
            )
        )

        # 2-level discovery
        for group in registry.groups():
            print(group.name, group.description)
            for tool_name in group.tool_names:
                print("  -", tool_name)
    """

    __slots__ = ("_by_name",)

    def __init__(self) -> None:
        self._by_name: dict[str, ToolGroup] = {}

    # --- Mutation ---------------------------------------------------

    def register(self, group: ToolGroup, *, force: bool = False) -> None:
        """Register a tool group.

        Args:
            group: The group to register. ``group.name`` is the key.
            force: When ``True``, replaces an existing group. Default
                ``False`` raises if a *different* group with the same
                name is already registered.

        Raises:
            RegistryError: On name conflict without ``force``.
        """
        existing = self._by_name.get(group.name)
        if existing is not None and existing != group and not force:
            raise RegistryError(
                "Tool group already registered with this name",
                group_name=group.name,
            )
        self._by_name[group.name] = group

    def unregister(self, group_name: str) -> ToolGroup | None:
        """Remove a group; returns the removed group or None."""
        return self._by_name.pop(group_name, None)

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_name.clear()

    # --- Lookup -----------------------------------------------------

    def get(self, group_name: str) -> ToolGroup | None:
        """Resolve a group by name; returns None when unregistered.

        Unlike the other registries' empty-spec fallback, an unknown
        group is genuinely absent — pretending otherwise would
        misrepresent the catalogue.
        """
        return self._by_name.get(group_name)

    def has(self, group_name: str) -> bool:
        """Whether a group is registered under ``group_name``."""
        return group_name in self._by_name

    def list_names(self) -> list[str]:
        """All registered group names, sorted."""
        return sorted(self._by_name)

    def groups(self) -> tuple[ToolGroup, ...]:
        """All registered groups in registration order."""
        return tuple(self._by_name.values())

    def tools_in(self, group_name: str) -> tuple[str, ...]:
        """Tool names belonging to a group; empty tuple when unknown."""
        group = self._by_name.get(group_name)
        return group.tool_names if group is not None else ()

    def groups_for_tool(self, tool_name: str) -> tuple[str, ...]:
        """Names of groups that contain ``tool_name``, sorted.

        Tools may belong to multiple groups. Returns sorted output for
        deterministic prompts when the agent renders membership.
        """
        return tuple(sorted(name for name, group in self._by_name.items() if tool_name in group))

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, group_name: object) -> bool:
        return isinstance(group_name, str) and group_name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __repr__(self) -> str:
        return f"ToolGroupRegistry({len(self._by_name)} groups: {sorted(self._by_name)})"


# Module-level default. Empty until tool integrations register their
# groups (typically in register_agent_tools).
default_tool_group_registry = ToolGroupRegistry()


__all__ = ["ToolGroupRegistry", "default_tool_group_registry"]
