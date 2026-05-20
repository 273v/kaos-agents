"""CapabilityRegistry — name → :class:`Capability` catalogue.

Plan §16 (Capability layer) of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``.

The registry mirrors the shape of :class:`ToolGroupRegistry`,
:class:`HookRegistry`, etc. — slotted, explicit ``register`` /
``unregister`` / ``clear`` / ``get`` / ``has``, plus a module-level
``default_capability_registry`` singleton.

In addition to name-keyed lookup the registry supports:

* ``list_by_kind(kind)`` — all capabilities with a given
  :class:`CapabilityKind`; the M1 fitness ranker / planner uses this to
  enumerate "all SEARCH capabilities" rather than walking every tool.
* ``backing_tools(name)`` — the set of concrete tool names that
  implement a capability. The runtime resolves a chosen Capability to
  this set at dispatch time.
* ``capabilities_for_tool(tool_name)`` — the inverse: which
  capabilities a tool participates in. A single tool can implement
  multiple capabilities (e.g., ``kaos-web-search`` is both SEARCH
  and READ depending on the caller's intent).

The registry does NOT validate against any tool registry; capability
declarations can name tools that aren't (yet) registered. The runtime
resolver is responsible for skipping unknown bindings at dispatch time.
"""

from __future__ import annotations

from kaos_core.exceptions import RegistryError
from kaos_core.types.capability import Capability, CapabilityKind


class CapabilityRegistry:
    """Catalogue of capabilities keyed by canonical name.

    Typical usage::

        from kaos_agents.registry import default_capability_registry
        from kaos_core.types import Capability, CapabilityKind, CostClass

        default_capability_registry.register(
            Capability(
                name="retrieve",
                kind=CapabilityKind.SEARCH,
                description=(
                    "Find pointers to information across configured "
                    "sources (web search, federal register, SEC EDGAR, "
                    "attached documents). Returns ranked manifests."
                ),
                inputs=("query", "source_filter", "max_n"),
                outputs=("RetrievalHit[]",),
                cost_class=CostClass.CHEAP,
                backing_tool_names=(
                    "kaos-web-search",
                    "kaos-source-fr-search",
                    "kaos-source-edgar-search",
                    "kaos-content-search-document",
                ),
            )
        )
    """

    __slots__ = ("_by_name", "_by_tool")

    def __init__(self) -> None:
        self._by_name: dict[str, Capability] = {}
        # Inverse index for ``capabilities_for_tool``. Updated lazily
        # on register / unregister so lookups are O(1).
        self._by_tool: dict[str, set[str]] = {}

    # --- Mutation ---------------------------------------------------

    def register(self, capability: Capability, *, force: bool = False) -> None:
        """Register a capability.

        Args:
            capability: The capability to register. ``capability.name``
                is the key.
            force: When ``True``, replaces an existing capability. Default
                ``False`` raises if a *different* capability with the
                same name is already registered.

        Raises:
            RegistryError: On name conflict without ``force``.
        """
        existing = self._by_name.get(capability.name)
        if existing is not None and existing != capability and not force:
            raise RegistryError(
                "Capability already registered with this name",
                capability_name=capability.name,
            )
        # If replacing, remove the old inverse-index entries first so
        # ``capabilities_for_tool`` doesn't return stale memberships.
        if existing is not None:
            for tool_name in existing.backing_tool_names:
                names = self._by_tool.get(tool_name)
                if names is not None:
                    names.discard(existing.name)
                    if not names:
                        del self._by_tool[tool_name]
        self._by_name[capability.name] = capability
        for tool_name in capability.backing_tool_names:
            self._by_tool.setdefault(tool_name, set()).add(capability.name)

    def unregister(self, name: str) -> Capability | None:
        """Remove a capability; returns the removed instance or None."""
        capability = self._by_name.pop(name, None)
        if capability is None:
            return None
        for tool_name in capability.backing_tool_names:
            names = self._by_tool.get(tool_name)
            if names is not None:
                names.discard(name)
                if not names:
                    del self._by_tool[tool_name]
        return capability

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_name.clear()
        self._by_tool.clear()

    # --- Lookup -----------------------------------------------------

    def get(self, name: str) -> Capability | None:
        """Resolve a capability by name; returns None when unregistered."""
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        """Whether a capability is registered under ``name``."""
        return name in self._by_name

    def list_names(self) -> list[str]:
        """All registered capability names, sorted."""
        return sorted(self._by_name)

    def capabilities(self) -> tuple[Capability, ...]:
        """All registered capabilities in registration order."""
        return tuple(self._by_name.values())

    def list_by_kind(self, kind: CapabilityKind) -> tuple[Capability, ...]:
        """All capabilities of a given :class:`CapabilityKind`.

        The planner uses this to enumerate "all SEARCH capabilities" or
        "all JUDGE capabilities" without walking every tool.
        """
        return tuple(c for c in self._by_name.values() if c.kind == kind)

    def backing_tools(self, name: str) -> tuple[str, ...]:
        """Tool names that implement the capability ``name``.

        Empty tuple when the capability is unknown OR purely
        declarative (a Persona, a UI surface) with no backing tools.
        """
        capability = self._by_name.get(name)
        return capability.backing_tool_names if capability is not None else ()

    def capabilities_for_tool(self, tool_name: str) -> tuple[str, ...]:
        """Names of capabilities backed by ``tool_name``, sorted.

        A single tool may implement multiple capabilities (e.g.,
        ``kaos-web-search`` participates in both SEARCH and READ
        depending on caller intent). Returns sorted output for
        deterministic prompt rendering.
        """
        return tuple(sorted(self._by_tool.get(tool_name, ())))

    # --- Dunder -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __repr__(self) -> str:
        return f"CapabilityRegistry({len(self._by_name)} capabilities: {sorted(self._by_name)})"


# Module-level default. Empty until consumers register their capabilities
# (typically alongside the corresponding ``register_*_tools`` calls).
default_capability_registry = CapabilityRegistry()


__all__ = [
    "CapabilityRegistry",
    "default_capability_registry",
]
