"""AgentToolSpecRegistry — tool-name → AgentToolSpec catalogue.

Mirrors the existing :class:`EventRegistry` / :class:`HookRegistry` /
:class:`PatternRegistry` shape. Holds the declarative dependency
contracts for tools the agent will dispatch.

Tools register out-of-band rather than declaring the spec on their
KaosTool subclass — see :mod:`kaos_agents.types.agent_tool_spec`
for the rationale.

Lookup semantics: a tool with no registration returns
:meth:`AgentToolSpec.empty`. This means *every* tool has a spec
queryable; "did this tool register?" reduces to ``spec.is_empty``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.exceptions import RegistryError

from kaos_agents.types.agent_tool_spec import (
    AgentToolRegistration,
    AgentToolSpec,
)

if TYPE_CHECKING:
    from kaos_core.base.tool import KaosTool


class AgentToolSpecRegistry:
    """Catalogue of declarative tool specs keyed by tool name.

    Typical use::

        from kaos_agents.registry import default_tool_spec_registry
        from kaos_agents.types import AgentToolSpec, MemoryType, ModelRole

        default_tool_spec_registry.register(
            "kaos-agent-memory-search",
            AgentToolSpec(memory_sections=(MemoryType.FINDINGS, MemoryType.MESSAGES)),
        )

        spec = default_tool_spec_registry.get_for_tool(some_kaos_tool)
        if spec.memory_sections:
            ...

    The bridge (Track 3 chunk A2 onward) consults this registry at
    wiring time. For chunk A1 the registry is the contract — actual
    consumption is in subsequent chunks.
    """

    __slots__ = ("_by_name",)

    def __init__(self) -> None:
        self._by_name: dict[str, AgentToolRegistration] = {}

    # --- Mutation ---------------------------------------------------

    def register(
        self,
        tool_name: str,
        spec: AgentToolSpec,
        *,
        tags: tuple[str, ...] = (),
        force: bool = False,
    ) -> None:
        """Register a spec for a tool name.

        Args:
            tool_name: The tool's :class:`ToolMetadata.name` discriminator.
            spec: Declarative dependencies for this tool.
            tags: Optional free-form tags for grouping registrations
                (e.g. ``("research",)``, ``("memory",)``). Don't affect
                lookup; useful for inspection / dashboards.
            force: When ``True``, replaces an existing different
                registration. Default ``False`` raises on conflict.

        Raises:
            RegistryError: If a *different* spec is already registered
                under ``tool_name`` and ``force`` is False.
        """
        existing = self._by_name.get(tool_name)
        if existing is not None and existing.spec != spec and not force:
            raise RegistryError(
                "Tool spec already registered with this name",
                tool_name=tool_name,
            )
        self._by_name[tool_name] = AgentToolRegistration(tool_name=tool_name, spec=spec, tags=tags)

    def unregister(self, tool_name: str) -> AgentToolRegistration | None:
        """Remove a registration. Returns the removed record, or None."""
        return self._by_name.pop(tool_name, None)

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_name.clear()

    # --- Lookup -----------------------------------------------------

    def get(self, tool_name: str) -> AgentToolSpec:
        """Resolve a spec by tool name. Returns the empty spec when unregistered.

        The empty-spec fallback means callers don't need to None-check —
        every tool has a queryable spec, and ``spec.is_empty`` answers
        "did this tool register?".
        """
        registration = self._by_name.get(tool_name)
        return registration.spec if registration is not None else AgentToolSpec.empty()

    def get_for_tool(self, kaos_tool: KaosTool) -> AgentToolSpec:
        """Convenience wrapper — fetch by ``kaos_tool.metadata.name``."""
        return self.get(kaos_tool.metadata.name)

    def get_registration(self, tool_name: str) -> AgentToolRegistration | None:
        """Return the full registration record (with tags), or None."""
        return self._by_name.get(tool_name)

    def has(self, tool_name: str) -> bool:
        """Whether a non-empty spec is registered under ``tool_name``."""
        return tool_name in self._by_name

    def list_names(self) -> list[str]:
        """All registered tool names, sorted."""
        return sorted(self._by_name)

    def list_registrations(self) -> tuple[AgentToolRegistration, ...]:
        """All registrations in registration order."""
        return tuple(self._by_name.values())

    def list_by_tag(self, tag: str) -> tuple[AgentToolRegistration, ...]:
        """All registrations carrying ``tag``."""
        return tuple(reg for reg in self._by_name.values() if tag in reg.tags)

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, tool_name: object) -> bool:
        return isinstance(tool_name, str) and tool_name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __repr__(self) -> str:
        return f"AgentToolSpecRegistry({len(self._by_name)} specs: {sorted(self._by_name)})"


# Module-level default. Empty by default; built-in tool specs are
# registered explicitly by the modules that own them (e.g. memory
# tools register in tools.py via a module-level call). Out-of-tree
# code can register against this registry, or build its own.
default_tool_spec_registry = AgentToolSpecRegistry()


__all__ = ["AgentToolSpecRegistry", "default_tool_spec_registry"]
