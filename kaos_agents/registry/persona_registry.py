"""PersonaRegistry — name → :class:`KaosPersona` catalogue.

Plan §12 of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``.

Mirrors :class:`ToolGroupRegistry` and :class:`CapabilityRegistry` —
slotted, explicit ``register`` / ``unregister`` / ``clear`` / ``get``
/ ``has`` with a module-level :data:`default_persona_registry`
singleton.

Persona names are NOT hardcoded in business logic anywhere — the
registry is the only authority on what a "research" or "drafting"
persona means in a given deployment. The seeded canonical trio lives
in :mod:`kaos_agents.personas.builtin` and is registered into the
default registry on import; downstream deployments may register
additional personas, or clear and re-seed with their own.
"""

from __future__ import annotations

from kaos_core.exceptions import RegistryError

from kaos_agents.types.persona import KaosPersona


class PersonaRegistry:
    """Catalogue of personas keyed by canonical name.

    Typical usage::

        from kaos_agents.registry import default_persona_registry
        from kaos_agents.types import KaosPersona

        default_persona_registry.register(
            KaosPersona.build(
                name="diligence",
                description="M&A due-diligence flow.",
                allowed_groups=("documents", "citations", "vfs"),
            )
        )

        persona = default_persona_registry.get("diligence")
        for group in persona.allowed_groups:
            ...
    """

    __slots__ = ("_by_name",)

    def __init__(self) -> None:
        self._by_name: dict[str, KaosPersona] = {}

    # --- Mutation ---------------------------------------------------

    def register(self, persona: KaosPersona, *, force: bool = False) -> None:
        """Register a persona.

        Args:
            persona: The persona to register. ``persona.name`` is the key.
            force: When ``True``, replaces an existing persona. Default
                ``False`` raises if a *different* persona with the same
                name is already registered.

        Raises:
            RegistryError: On name conflict without ``force``.
        """
        existing = self._by_name.get(persona.name)
        if existing is not None and existing != persona and not force:
            raise RegistryError(
                "Persona already registered with this name",
                persona_name=persona.name,
            )
        self._by_name[persona.name] = persona

    def unregister(self, name: str) -> KaosPersona | None:
        """Remove a persona; returns the removed instance or None."""
        return self._by_name.pop(name, None)

    def clear(self) -> None:
        """Drop all registrations. Primarily for tests."""
        self._by_name.clear()

    # --- Lookup -----------------------------------------------------

    def get(self, name: str) -> KaosPersona | None:
        """Resolve a persona by name; returns None when unregistered."""
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        """Whether a persona is registered under ``name``."""
        return name in self._by_name

    def list_names(self) -> list[str]:
        """All registered persona names, sorted."""
        return sorted(self._by_name)

    def personas(self) -> tuple[KaosPersona, ...]:
        """All registered personas in registration order."""
        return tuple(self._by_name.values())

    # --- Dunder -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __repr__(self) -> str:
        return f"PersonaRegistry({len(self._by_name)} personas: {sorted(self._by_name)})"


# Module-level default. Populated by importing
# :mod:`kaos_agents.personas.builtin`, which calls
# :func:`register_builtin_personas` on the singleton.
default_persona_registry = PersonaRegistry()


__all__ = ["PersonaRegistry", "default_persona_registry"]
