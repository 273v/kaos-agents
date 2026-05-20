"""Persona presets for the KAOS Runner.

Public re-exports:

- :class:`~kaos_agents.types.persona.KaosPersona` — the value type
- :class:`~kaos_agents.registry.persona_registry.PersonaRegistry` —
  the registry shape
- :data:`~kaos_agents.registry.persona_registry.default_persona_registry`
  — the module-level singleton, pre-seeded on import with the three
  canonical personas (Research / Drafting / Forensics) via
  :func:`register_builtin_personas`.

Downstream code should resolve personas through the registry, not by
referencing the seeded ``RESEARCH`` / ``DRAFTING`` / ``FORENSICS``
module attributes directly — that keeps the seeded names from leaking
into business logic as hardcoded identifiers.
"""

from __future__ import annotations

from kaos_agents.personas.builtin import (
    DRAFTING,
    FORENSICS,
    RESEARCH,
    register_builtin_personas,
)
from kaos_agents.registry.persona_registry import (
    PersonaRegistry,
    default_persona_registry,
)
from kaos_agents.types.persona import KaosPersona

# Eagerly seed the canonical trio into the default singleton on import.
# Idempotent via ``force=True`` so re-import (e.g. across test sessions)
# stays a no-op against the existing registrations.
register_builtin_personas(default_persona_registry)


__all__ = [
    "DRAFTING",
    "FORENSICS",
    "RESEARCH",
    "KaosPersona",
    "PersonaRegistry",
    "default_persona_registry",
    "register_builtin_personas",
]
