"""EventRegistry — wire-discriminator → KaosEvent class.

Mirrors :class:`kaos_core.registry.tool_registry.ToolRegistry` shape.
Where ``ToolRegistry`` keys by ``ToolMetadata.name``, this registry keys
by :meth:`kaos_agents.base.event.KaosEvent.event_type`.

Powers :func:`kaos_agents.events.deserialize_event` — given a
``{"type": "..."}`` dict from the wire, the registry resolves the
right :class:`KaosEvent` subclass to ``model_validate`` against.

Auto-registration
-----------------

:class:`KaosEvent` (chunk 2) uses :meth:`KaosEvent.__init_subclass__`
to register every subclass in the
:data:`default_event_registry` at class-creation time. That keeps
chunk-4's events/ subpackage split working without any explicit
registration boilerplate — importing the module triggers registration.

Custom registries
-----------------

For tests or out-of-tree extensions, callers may build their own
:class:`EventRegistry` and register custom subclasses. The default
serde path uses :data:`default_event_registry`; pass a custom one
via the optional ``registry`` parameter on serde helpers (planned
for Track 5 wire/transport).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.exceptions import RegistryError

if TYPE_CHECKING:
    from kaos_agents.base.event import KaosEvent


class EventRegistry:
    """Catalogue of :class:`KaosEvent` subclasses keyed by wire discriminator.

    Holds one entry per concrete event subclass. Lookup is by the
    snake-case ``type`` string emitted on the wire.

    Typical use::

        from kaos_agents.registry import default_event_registry
        from kaos_agents.events import TurnStart

        # Auto-registered at class creation:
        cls = default_event_registry.get("turn_start")  # -> TurnStart

        # Manual:
        custom = EventRegistry()
        custom.register(TurnStart)
        cls = custom.get("turn_start")
    """

    __slots__ = ("_by_type",)

    def __init__(self) -> None:
        self._by_type: dict[str, type[KaosEvent]] = {}

    # --- Mutation -----------------------------------------------------

    def register(self, cls: type[KaosEvent], *, force: bool = False) -> None:
        """Register an event subclass.

        Args:
            cls: A :class:`KaosEvent` subclass with a resolvable
                :meth:`event_type`.
            force: When ``True``, replaces any existing registration
                for the same discriminator. Default ``False`` raises
                :class:`RegistryError` on conflict — the
                kaos-core convention for catching accidental
                double-registration during refactors.

        Raises:
            RegistryError: If a different class is already registered
                under the same ``event_type`` and ``force`` is False.
        """
        type_name = cls.event_type()
        existing = self._by_type.get(type_name)
        if existing is not None and existing is not cls and not force:
            raise RegistryError(
                "Event class already registered for this type discriminator",
                tool_name=type_name,
            )
        self._by_type[type_name] = cls

    def unregister(self, type_name: str) -> type[KaosEvent] | None:
        """Remove a registration. Returns the removed class, or None."""
        return self._by_type.pop(type_name, None)

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_type.clear()

    # --- Lookup -------------------------------------------------------

    def get(self, type_name: str) -> type[KaosEvent] | None:
        """Resolve an event class by wire discriminator. ``None`` if unknown."""
        return self._by_type.get(type_name)

    def has(self, type_name: str) -> bool:
        """Whether a discriminator is registered."""
        return type_name in self._by_type

    def list_types(self) -> list[str]:
        """All registered discriminator strings, sorted."""
        return sorted(self._by_type)

    def list_classes(self) -> list[type[KaosEvent]]:
        """All registered event classes, in registration order."""
        return list(self._by_type.values())

    def __len__(self) -> int:
        return len(self._by_type)

    def __contains__(self, type_name: object) -> bool:
        return isinstance(type_name, str) and type_name in self._by_type

    def __iter__(self):
        return iter(self._by_type)

    def __repr__(self) -> str:
        return f"EventRegistry({len(self._by_type)} types: {sorted(self._by_type)})"


# Module-level default. :meth:`KaosEvent.__init_subclass__` registers
# every concrete subclass here on creation, so importing
# ``kaos_agents.events`` populates this registry as a side effect.
default_event_registry = EventRegistry()
