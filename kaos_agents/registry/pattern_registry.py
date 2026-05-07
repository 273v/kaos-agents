"""PatternRegistry — pattern-name → KaosAgent subclass.

Mirrors :class:`kaos_agents.registry.event_registry.EventRegistry` shape
but for pattern dispatch: looking up which :class:`KaosAgent` subclass
(``ChatAgent`` / ``PlanExecuteAgent`` / ``ResearchAgent``) to
instantiate given a pattern name from :class:`AgentPattern`.

Replaces the if/elif switch in :class:`Runner._build_internal_agent`
with a registry lookup, which:

- Lets out-of-tree code register custom patterns without touching Runner.
- Surfaces all known patterns for UI/CLI ("which patterns can I use?")
- Aligns with the rest of the kaos-agents registry layer
  (EventRegistry, HookRegistry).

Type discipline (Track 2 vintage): we hold ``type[KaosAgent]``,
because today's pattern classes (ChatAgent / PlanExecuteAgent /
ResearchAgent) inherit BaseAgent which is a KaosAgent. They do *not*
yet conform to :class:`KaosPattern` (the dispatch-strategy ABC) —
that's a Track 3 concern. This registry is intentionally permissive
about the shape of registered classes for that reason; once Track 3
splits dispatch strategies from agent shells, the type narrows.

Patterns auto-register via the explicit
:meth:`patterns.__init__.py` call (not via ``__init_subclass__``)
because:
- We want exactly the canonical 3 patterns in the default registry,
  not every BaseAgent subclass anyone happens to create.
- Tests subclass BaseAgent for fixtures; auto-registration would
  pollute the default registry with test-private classes.

For out-of-tree custom patterns, register explicitly::

    from kaos_agents.registry import default_pattern_registry
    default_pattern_registry.register("custom", MyCustomAgent)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.exceptions import RegistryError

if TYPE_CHECKING:
    from kaos_agents.base.agent import KaosAgent


class PatternRegistry:
    """Catalogue of pattern-name → :class:`KaosAgent` subclass.

    Holds one entry per registered pattern. Lookup is by pattern name
    (matching :class:`AgentPattern` enum values: ``"chat"``,
    ``"plan"``, ``"research"``).

    Typical use::

        from kaos_agents.registry import default_pattern_registry
        from kaos_agents.config import AgentPattern

        cls = default_pattern_registry.get(AgentPattern.CHAT.value)  # -> ChatAgent
        agent = cls(vfs=vfs, ...)
    """

    __slots__ = ("_by_name",)

    def __init__(self) -> None:
        self._by_name: dict[str, type[KaosAgent]] = {}

    # --- Mutation -----------------------------------------------------

    def register(
        self,
        name: str,
        cls: type[KaosAgent],
        *,
        force: bool = False,
    ) -> None:
        """Register a pattern class under ``name``.

        Args:
            name: Pattern discriminator. Should match an
                :class:`AgentPattern` enum value when registering
                built-in patterns.
            cls: A :class:`KaosAgent` subclass.
            force: When ``True``, replaces any existing registration
                for this name. Default ``False`` raises on conflict.

        Raises:
            RegistryError: If a different class is already registered
                under ``name`` and ``force`` is False.
        """
        existing = self._by_name.get(name)
        if existing is not None and existing is not cls and not force:
            raise RegistryError(
                "Pattern already registered with this name",
                tool_name=name,
            )
        self._by_name[name] = cls

    def unregister(self, name: str) -> type[KaosAgent] | None:
        """Remove a registration. Returns the removed class, or None."""
        return self._by_name.pop(name, None)

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_name.clear()

    # --- Lookup -------------------------------------------------------

    def get(self, name: str) -> type[KaosAgent] | None:
        """Resolve a pattern class by name. ``None`` if unknown."""
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        """Whether a pattern with this name is registered."""
        return name in self._by_name

    def list_names(self) -> list[str]:
        """All registered pattern names, sorted."""
        return sorted(self._by_name)

    def list_classes(self) -> list[type[KaosAgent]]:
        """All registered pattern classes, in registration order."""
        return list(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __repr__(self) -> str:
        return f"PatternRegistry({len(self._by_name)} patterns: {sorted(self._by_name)})"


# Module-level default. Populated by :mod:`kaos_agents.patterns` on
# import — see that module's :func:`__init__` for the explicit
# registration calls.
default_pattern_registry = PatternRegistry()


__all__ = ["PatternRegistry", "default_pattern_registry"]
