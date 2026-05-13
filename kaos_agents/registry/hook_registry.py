"""HookRegistry — name → KaosHook *instance* catalogue.

Mirrors :class:`kaos_core.registry.tool_registry.ToolRegistry` shape,
with one important difference: where ``ToolRegistry`` keys *classes*
by ``ToolMetadata.name``, this registry keys *instances* by
``HookMetadata.name``.

Why instances rather than classes: hooks are typically stateful
(``CostTrackingHook`` accumulates totals, ``AuditHook`` holds a
``SessionStore`` reference). Different runs may want different hook
instances with different state — a class-level registry would lose that.

Two ways the registry is populated:

1. **Tier-1 (decorator)** — chunk 6's ``@hook`` decorator wraps a
   function as a FunctionHook instance and auto-registers into
   :data:`default_hook_registry`. Importing the module triggers
   registration.
2. **Tier-3 (manual)** — applications build their own registry and
   register hook instances explicitly: ``reg.register(MyHook())``.

Runner accepts hooks via the ``hooks`` parameter directly — the
registry is for hook *discovery* by name, not for Runner construction.
Future Runner enhancements may grow a ``hook_registry`` parameter that
expands all registered hooks at run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_core.exceptions import RegistryError

if TYPE_CHECKING:
    from kaos_agents.hooks.base import KaosHook


class HookRegistry:
    """Catalogue of :class:`KaosHook` instances keyed by metadata.name.

    Holds one entry per registered hook instance. Lookup is by the hook's
    ``metadata().name`` discriminator.

    Typical use::

        from kaos_agents.registry import default_hook_registry
        from kaos_agents.hooks import LoggingHook

        default_hook_registry.register(LoggingHook())
        hook = default_hook_registry.get("kaos-agents-logging-hook")

        # Inspect what's listening for a specific event:
        for h in default_hook_registry.list_listeners("turn_summary"):
            ...
    """

    __slots__ = ("_by_name",)

    def __init__(self) -> None:
        self._by_name: dict[str, KaosHook] = {}

    # --- Mutation -----------------------------------------------------

    def register(self, hook: KaosHook, *, force: bool = False) -> None:
        """Register a hook instance.

        Args:
            hook: A :class:`KaosHook` instance with a resolvable ``metadata()``.
            force: When ``True``, replaces any existing registration for
                the same name. Default ``False`` raises
                :class:`RegistryError` on conflict.

        Raises:
            RegistryError: If a different instance is already registered
                under the same ``metadata().name`` and ``force`` is False.
        """
        name = hook.metadata().name
        existing = self._by_name.get(name)
        if existing is not None and existing is not hook and not force:
            raise RegistryError(
                "Hook already registered with this name",
                tool_name=name,
            )
        self._by_name[name] = hook

    def unregister(self, name: str) -> KaosHook | None:
        """Remove a registration. Returns the removed instance, or None."""
        return self._by_name.pop(name, None)

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_name.clear()

    # --- Lookup -------------------------------------------------------

    def get(self, name: str) -> KaosHook | None:
        """Resolve a hook instance by name. ``None`` if unknown."""
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        """Whether a hook with this name is registered."""
        return name in self._by_name

    def list_names(self) -> list[str]:
        """All registered hook names, sorted."""
        return sorted(self._by_name)

    def list_hooks(self) -> tuple[KaosHook, ...]:
        """All registered hooks, in registration order.

        Returned as a tuple so callers can pass it directly to
        ``Runner(... hooks=registry.list_hooks())``.
        """
        return tuple(self._by_name.values())

    def list_listeners(self, event_type: str) -> tuple[KaosHook, ...]:
        """All hooks whose ``metadata().listens_to`` includes ``event_type``.

        An empty ``listens_to`` tuple means "listens to all events" —
        those hooks are included in every query.
        """
        out: list[KaosHook] = []
        for hook in self._by_name.values():
            listens_to = hook.metadata().listens_to
            if not listens_to or event_type in listens_to:
                out.append(hook)
        return tuple(out)

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __repr__(self) -> str:
        return f"HookRegistry({len(self._by_name)} hooks: {sorted(self._by_name)})"


# Module-level default. Empty by default; chunk 6's ``@hook`` decorator
# auto-registers into this. Applications building their own registries
# should construct a fresh :class:`HookRegistry` rather than mutating
# this global.
default_hook_registry = HookRegistry()


__all__ = ["HookRegistry", "default_hook_registry"]
