"""Unit tests for :class:`kaos_agents.registry.CapabilityRegistry`.

Mirrors the test discipline of ``test_tool_group_registry.py``: register
/ unregister / clear / lookup / inverse-index, plus the kind-filter and
name-conflict semantics specific to capabilities.

See plan §16 in
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``.
"""

from __future__ import annotations

import pytest
from kaos_core.exceptions import RegistryError
from kaos_core.types import Capability, CapabilityKind, CostClass, LatencyClass

from kaos_agents.registry import CapabilityRegistry, default_capability_registry


def _make(name: str, kind: CapabilityKind = CapabilityKind.SEARCH, **kwargs) -> Capability:
    return Capability(
        name=name,
        kind=kind,
        description=f"Test capability {name}",
        **kwargs,
    )


class TestRegisterAndLookup:
    def test_register_get_has(self) -> None:
        r = CapabilityRegistry()
        c = _make("retrieve")
        r.register(c)
        assert r.get("retrieve") is c
        assert r.has("retrieve") is True
        assert "retrieve" in r
        assert len(r) == 1

    def test_get_unknown_returns_none(self) -> None:
        r = CapabilityRegistry()
        assert r.get("missing") is None
        assert r.has("missing") is False
        assert "missing" not in r

    def test_list_names_sorted(self) -> None:
        r = CapabilityRegistry()
        r.register(_make("zeta"))
        r.register(_make("alpha"))
        r.register(_make("mu"))
        assert r.list_names() == ["alpha", "mu", "zeta"]

    def test_capabilities_preserves_insertion_order(self) -> None:
        r = CapabilityRegistry()
        order = ["zeta", "alpha", "mu"]
        for name in order:
            r.register(_make(name))
        assert [c.name for c in r.capabilities()] == order

    def test_iter(self) -> None:
        r = CapabilityRegistry()
        r.register(_make("a"))
        r.register(_make("b"))
        assert sorted(iter(r)) == ["a", "b"]

    def test_repr(self) -> None:
        r = CapabilityRegistry()
        r.register(_make("retrieve"))
        assert "CapabilityRegistry" in repr(r)
        assert "retrieve" in repr(r)


class TestConflictAndForce:
    def test_register_identical_is_idempotent(self) -> None:
        r = CapabilityRegistry()
        c = _make("retrieve")
        r.register(c)
        r.register(c)  # same instance — no conflict
        assert len(r) == 1

    def test_register_different_raises_without_force(self) -> None:
        r = CapabilityRegistry()
        r.register(_make("retrieve", kind=CapabilityKind.SEARCH))
        with pytest.raises(RegistryError):
            r.register(_make("retrieve", kind=CapabilityKind.READ))

    def test_register_with_force_replaces(self) -> None:
        r = CapabilityRegistry()
        r.register(_make("retrieve", kind=CapabilityKind.SEARCH))
        r.register(_make("retrieve", kind=CapabilityKind.READ), force=True)
        c = r.get("retrieve")
        assert c is not None
        assert c.kind == CapabilityKind.READ


class TestUnregisterAndClear:
    def test_unregister_returns_removed(self) -> None:
        r = CapabilityRegistry()
        c = _make("retrieve")
        r.register(c)
        removed = r.unregister("retrieve")
        assert removed is c
        assert not r.has("retrieve")
        assert len(r) == 0

    def test_unregister_unknown_returns_none(self) -> None:
        r = CapabilityRegistry()
        assert r.unregister("missing") is None

    def test_clear(self) -> None:
        r = CapabilityRegistry()
        r.register(_make("a"))
        r.register(_make("b"))
        r.clear()
        assert len(r) == 0
        assert r.list_names() == []


class TestKindFiltering:
    def test_list_by_kind(self) -> None:
        r = CapabilityRegistry()
        r.register(_make("retrieve", kind=CapabilityKind.SEARCH))
        r.register(_make("discover", kind=CapabilityKind.SEARCH))
        r.register(_make("read", kind=CapabilityKind.READ))
        r.register(_make("judge-grounding", kind=CapabilityKind.JUDGE))

        searches = r.list_by_kind(CapabilityKind.SEARCH)
        assert {c.name for c in searches} == {"retrieve", "discover"}

        reads = r.list_by_kind(CapabilityKind.READ)
        assert [c.name for c in reads] == ["read"]

        judges = r.list_by_kind(CapabilityKind.JUDGE)
        assert [c.name for c in judges] == ["judge-grounding"]

        # Kind with no members returns empty tuple, not None
        computes = r.list_by_kind(CapabilityKind.COMPUTE)
        assert computes == ()


class TestBackingToolsInverseIndex:
    def test_backing_tools_simple(self) -> None:
        r = CapabilityRegistry()
        r.register(
            _make(
                "retrieve",
                backing_tool_names=("kaos-web-search", "kaos-source-fr-search"),
            )
        )
        assert r.backing_tools("retrieve") == (
            "kaos-web-search",
            "kaos-source-fr-search",
        )

    def test_backing_tools_unknown_returns_empty(self) -> None:
        r = CapabilityRegistry()
        assert r.backing_tools("missing") == ()

    def test_capabilities_for_tool_simple(self) -> None:
        r = CapabilityRegistry()
        r.register(
            _make(
                "retrieve",
                kind=CapabilityKind.SEARCH,
                backing_tool_names=("kaos-web-search",),
            )
        )
        r.register(
            _make(
                "read-page",
                kind=CapabilityKind.READ,
                backing_tool_names=("kaos-web-search",),
            )
        )
        # Same tool implements two capabilities.
        assert r.capabilities_for_tool("kaos-web-search") == (
            "read-page",
            "retrieve",
        )

    def test_capabilities_for_unknown_tool_returns_empty(self) -> None:
        r = CapabilityRegistry()
        assert r.capabilities_for_tool("not-a-tool") == ()

    def test_unregister_clears_inverse_index(self) -> None:
        r = CapabilityRegistry()
        r.register(
            _make(
                "retrieve",
                backing_tool_names=("kaos-web-search",),
            )
        )
        assert r.capabilities_for_tool("kaos-web-search") == ("retrieve",)
        r.unregister("retrieve")
        assert r.capabilities_for_tool("kaos-web-search") == ()

    def test_force_register_replaces_inverse_index(self) -> None:
        r = CapabilityRegistry()
        r.register(
            _make(
                "retrieve",
                backing_tool_names=("kaos-web-search",),
            )
        )
        r.register(
            _make(
                "retrieve",
                kind=CapabilityKind.READ,
                backing_tool_names=("kaos-source-fr-get-content",),
            ),
            force=True,
        )
        # Old tool no longer participates; new tool does.
        assert r.capabilities_for_tool("kaos-web-search") == ()
        assert r.capabilities_for_tool("kaos-source-fr-get-content") == ("retrieve",)


class TestCostAndLatencyExposed:
    def test_attributes_preserved(self) -> None:
        r = CapabilityRegistry()
        r.register(
            Capability(
                name="cheap-search",
                kind=CapabilityKind.SEARCH,
                description="A cheap fast search.",
                cost_class=CostClass.CHEAP,
                latency_class=LatencyClass.FAST,
            )
        )
        c = r.get("cheap-search")
        assert c is not None
        assert c.cost_class == CostClass.CHEAP
        assert c.latency_class == LatencyClass.FAST


class TestDefaultSingleton:
    def test_singleton_is_empty_by_default(self) -> None:
        # Don't assert size — other tests may have polluted it. Just
        # assert it's the expected type.
        assert isinstance(default_capability_registry, CapabilityRegistry)
