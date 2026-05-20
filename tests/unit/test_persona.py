"""Unit tests for :class:`kaos_agents.types.persona.KaosPersona` and
:class:`kaos_agents.registry.persona_registry.PersonaRegistry`.

Covers Step 5a of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-progress.md``:

- Value-type invariants (frozen, slots, hashable, equality)
- Construction validation (empty/whitespace name, ceiling-superset)
- Registry register/get/has/clear/list/unregister + name-conflict
- The three seeded builtin personas register cleanly and carry the
  expected canonical group sets.
"""

from __future__ import annotations

import dataclasses

import pytest
from kaos_core.exceptions import RegistryError

from kaos_agents.personas import (
    DRAFTING,
    FORENSICS,
    RESEARCH,
    default_persona_registry,
    register_builtin_personas,
)
from kaos_agents.registry import PersonaRegistry
from kaos_agents.types import KaosPersona


def _make(
    *,
    name: str = "diligence",
    description: str = "Test persona",
    allowed_groups: tuple[str, ...] = ("documents", "vfs"),
    ceiling_groups: tuple[str, ...] | None = None,
) -> KaosPersona:
    """Test helper for the common builder path."""
    return KaosPersona.build(
        name=name,
        description=description,
        allowed_groups=allowed_groups,
        ceiling_groups=ceiling_groups if ceiling_groups is not None else allowed_groups,
        model_role_preferences={"respond": "anthropic:claude-sonnet-4-6"},
        default_budgets={"max_loop_cost_usd": 0.5},
    )


# ─── Value-type invariants ─────────────────────────────────────────────


class TestValueTypeInvariants:
    def test_is_frozen(self) -> None:
        persona = _make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            persona.name = "mutated"  # ty: ignore[invalid-assignment]

    def test_uses_slots(self) -> None:
        # ``slots=True`` removes the per-instance ``__dict__``.
        persona = _make()
        assert not hasattr(persona, "__dict__")
        # And exposes ``__slots__`` on the class.
        assert hasattr(KaosPersona, "__slots__")

    def test_is_hashable(self) -> None:
        persona = _make()
        # Sanity: actually compute the hash and use it in a set.
        assert {persona, persona} == {persona}
        assert hash(persona) == hash(_make())

    def test_equality_is_field_wise(self) -> None:
        a = _make()
        b = _make()
        assert a == b
        c = _make(name="other")
        assert a != c

    def test_dict_inputs_normalize_to_sorted_pair_tuples(self) -> None:
        # Same dict content, different insertion order — same persona.
        a = KaosPersona.build(
            name="x",
            description="x",
            allowed_groups=("vfs",),
            model_role_preferences={"respond": "r", "classify": "c"},
            default_budgets={"b": 1.0, "a": 0.5},
        )
        b = KaosPersona.build(
            name="x",
            description="x",
            allowed_groups=("vfs",),
            model_role_preferences={"classify": "c", "respond": "r"},
            default_budgets={"a": 0.5, "b": 1.0},
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_round_trip_accessors(self) -> None:
        persona = _make()
        assert persona.model_for_role("respond") == "anthropic:claude-sonnet-4-6"
        assert persona.model_for_role("nonexistent") is None
        assert persona.budget_for("max_loop_cost_usd") == 0.5
        assert persona.budget_for("nonexistent") is None
        assert persona.model_role_preferences_as_dict() == {
            "respond": "anthropic:claude-sonnet-4-6",
        }
        assert persona.default_budgets_as_dict() == {"max_loop_cost_usd": 0.5}


# ─── Construction validation ──────────────────────────────────────────


class TestConstructionValidation:
    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            KaosPersona(name="", description="x")

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="whitespace"):
            KaosPersona(name="   ", description="x")

    def test_ceiling_must_be_superset_of_allowed(self) -> None:
        with pytest.raises(ValueError, match="superset"):
            KaosPersona(
                name="bad",
                description="x",
                allowed_groups=("web", "browser"),
                ceiling_groups=("web",),  # missing 'browser'
            )

    def test_ceiling_equal_to_allowed_is_valid(self) -> None:
        persona = KaosPersona(
            name="tight",
            description="x",
            allowed_groups=("vfs",),
            ceiling_groups=("vfs",),
        )
        assert persona.allows("vfs")
        assert persona.within_ceiling("vfs")

    def test_ceiling_strict_superset_is_valid(self) -> None:
        persona = KaosPersona(
            name="elevatable",
            description="x",
            allowed_groups=("documents",),
            ceiling_groups=("documents", "web"),
        )
        assert persona.allows("documents")
        assert not persona.allows("web")
        assert persona.within_ceiling("web")

    def test_empty_groups_are_valid(self) -> None:
        # A persona with no groups (allow-nothing) is structurally valid
        # — the registry doesn't dictate semantics.
        persona = KaosPersona(name="empty", description="x")
        assert persona.allowed_groups == ()
        assert persona.ceiling_groups == ()


# ─── Registry semantics ───────────────────────────────────────────────


class TestRegistry:
    def test_register_get_has(self) -> None:
        r = PersonaRegistry()
        p = _make(name="diligence")
        r.register(p)
        assert r.get("diligence") is p
        assert r.has("diligence")
        assert "diligence" in r
        assert len(r) == 1

    def test_get_unknown_returns_none(self) -> None:
        r = PersonaRegistry()
        assert r.get("missing") is None
        assert not r.has("missing")

    def test_list_names_sorted(self) -> None:
        r = PersonaRegistry()
        r.register(_make(name="zeta"))
        r.register(_make(name="alpha"))
        r.register(_make(name="mu"))
        assert r.list_names() == ["alpha", "mu", "zeta"]

    def test_personas_preserves_insertion_order(self) -> None:
        r = PersonaRegistry()
        order = ["zeta", "alpha", "mu"]
        for name in order:
            r.register(_make(name=name))
        assert [p.name for p in r.personas()] == order

    def test_register_identical_is_idempotent(self) -> None:
        r = PersonaRegistry()
        p = _make()
        r.register(p)
        r.register(p)
        assert len(r) == 1

    def test_register_different_raises_without_force(self) -> None:
        r = PersonaRegistry()
        r.register(_make(name="x", allowed_groups=("documents",)))
        with pytest.raises(RegistryError):
            r.register(_make(name="x", allowed_groups=("vfs",)))

    def test_register_with_force_replaces(self) -> None:
        r = PersonaRegistry()
        r.register(_make(name="x", allowed_groups=("documents",)))
        r.register(_make(name="x", allowed_groups=("vfs",)), force=True)
        replaced = r.get("x")
        assert replaced is not None
        assert replaced.allowed_groups == ("vfs",)

    def test_unregister_and_clear(self) -> None:
        r = PersonaRegistry()
        r.register(_make(name="a"))
        r.register(_make(name="b"))
        removed = r.unregister("a")
        assert removed is not None and removed.name == "a"
        assert r.unregister("missing") is None
        r.clear()
        assert len(r) == 0
        assert r.list_names() == []


# ─── Seeded builtin personas ──────────────────────────────────────────


class TestBuiltinPersonas:
    def test_all_three_register_in_default_registry(self) -> None:
        # Module import side-effect has already registered them.
        names = default_persona_registry.list_names()
        assert "research" in names
        assert "drafting" in names
        assert "forensics" in names

    def test_register_builtin_personas_is_idempotent(self) -> None:
        r = PersonaRegistry()
        register_builtin_personas(r)
        register_builtin_personas(r)
        assert sorted(r.list_names()) == ["drafting", "forensics", "research"]

    def test_research_persona_shape(self) -> None:
        assert RESEARCH.name == "research"
        assert RESEARCH.allows("web")
        assert RESEARCH.allows("documents")
        assert RESEARCH.allows("retrieval")
        # No drafting / forensics-only surfaces.
        assert not RESEARCH.allows("authoring")
        assert not RESEARCH.allows("forensics")
        # Model preferences populated.
        assert RESEARCH.model_for_role("respond") is not None
        assert RESEARCH.model_for_role("classify") is not None

    def test_drafting_extends_research(self) -> None:
        assert DRAFTING.name == "drafting"
        # Strict superset of Research.
        research_set = set(RESEARCH.allowed_groups)
        drafting_set = set(DRAFTING.allowed_groups)
        assert research_set.issubset(drafting_set)
        # Adds authoring + programs as the deliverable scope.
        assert DRAFTING.allows("authoring")
        assert DRAFTING.allows("programs")

    def test_forensics_is_tight_lane(self) -> None:
        assert FORENSICS.name == "forensics"
        assert FORENSICS.allows("forensics")
        assert FORENSICS.allows("documents")
        assert FORENSICS.allows("citations")
        assert FORENSICS.allows("vfs")
        # No outbound surfaces — by deliberate design.
        assert not FORENSICS.allows("web")
        assert not FORENSICS.allows("browser")
        assert not FORENSICS.allows("sources")
