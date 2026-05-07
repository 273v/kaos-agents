"""ToolDataTypeRegistry — tool-name → :class:`ToolDataTypeSpec` catalogue.

Track 4 chunk T4-1. Mirrors :class:`AgentToolSpecRegistry` but holds
the I/O type tagging from :class:`DataType`. Lookups return an empty
spec for unregistered tools, so consumers never None-check.

Type-driven discovery (kelvin's ``get_actions_by_input_type`` pattern)
is the differentiator: agents that need a tool consuming JSON or
emitting Markdown can ask the registry for matches without scanning
the whole catalogue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.exceptions import RegistryError

from kaos_agents.types.data_type import DataType, ToolDataTypeSpec

if TYPE_CHECKING:
    from kaos_core.base.tool import KaosTool


class ToolDataTypeRegistry:
    """Catalogue of per-tool I/O type tags.

    Out-of-band tagging — tools don't carry I/O type metadata on their
    :class:`KaosTool` subclass; the registration is owned by whichever
    module assembles the tool surface. This keeps :class:`KaosTool`
    minimal and lets multiple integrations declare different specs
    for the same tool name (e.g. a "research" profile vs a "billing"
    profile).
    """

    __slots__ = ("_by_name",)

    def __init__(self) -> None:
        self._by_name: dict[str, ToolDataTypeSpec] = {}

    # --- Mutation ---------------------------------------------------

    def register(
        self,
        tool_name: str,
        spec: ToolDataTypeSpec,
        *,
        force: bool = False,
    ) -> None:
        """Register I/O type tags for ``tool_name``.

        Raises:
            RegistryError: If a different spec is already registered
                and ``force`` is False.
        """
        existing = self._by_name.get(tool_name)
        if existing is not None and existing != spec and not force:
            raise RegistryError(
                "Tool data-type spec already registered with this name",
                tool_name=tool_name,
            )
        self._by_name[tool_name] = spec

    def unregister(self, tool_name: str) -> ToolDataTypeSpec | None:
        """Remove a registration; returns the removed spec or None."""
        return self._by_name.pop(tool_name, None)

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_name.clear()

    # --- Lookup -----------------------------------------------------

    def get(self, tool_name: str) -> ToolDataTypeSpec:
        """Resolve a spec; returns the empty spec when unregistered.

        The empty-spec fallback means callers can always read
        ``spec.input_type`` / ``spec.output_type`` without None-checks
        on the spec object itself (the *fields* are still typed
        ``DataType | None``).
        """
        return self._by_name.get(tool_name, ToolDataTypeSpec.empty())

    def get_for_tool(self, kaos_tool: KaosTool) -> ToolDataTypeSpec:
        """Convenience wrapper — fetch by ``kaos_tool.metadata.name``."""
        return self.get(kaos_tool.metadata.name)

    def has(self, tool_name: str) -> bool:
        """Whether a non-empty spec is registered."""
        return tool_name in self._by_name

    def list_names(self) -> list[str]:
        """All registered tool names, sorted."""
        return sorted(self._by_name)

    # --- Type-driven discovery (kelvin pattern) ---------------------

    def tools_by_input_type(self, input_type: DataType) -> tuple[str, ...]:
        """Names of registered tools whose declared ``input_type`` matches.

        Used by the agent to discover compatible tools given a typed
        artifact in hand: "I have JSON; which tools accept JSON?".
        """
        return tuple(
            sorted(name for name, spec in self._by_name.items() if spec.input_type == input_type)
        )

    def tools_by_output_type(self, output_type: DataType) -> tuple[str, ...]:
        """Names of registered tools whose declared ``output_type`` matches.

        Used to plan: "I need to produce a CSV; which tools emit CSV?".
        """
        return tuple(
            sorted(name for name, spec in self._by_name.items() if spec.output_type == output_type)
        )

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, tool_name: object) -> bool:
        return isinstance(tool_name, str) and tool_name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __repr__(self) -> str:
        return f"ToolDataTypeRegistry({len(self._by_name)} specs: {sorted(self._by_name)})"


# Module-level default. Empty until tool integrations register their
# I/O types. Out-of-tree code can register against this registry, or
# build its own.
default_tool_data_type_registry = ToolDataTypeRegistry()


__all__ = ["ToolDataTypeRegistry", "default_tool_data_type_registry"]
