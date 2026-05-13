"""MatterClientGuard unit tests — Phase 4.A."""

from __future__ import annotations

from typing import Any, cast

import pytest

from kaos_agents.memory.isolation import MatterClientGuard, MatterIsolationError


class TestUnboundGuard:
    def test_unbound_accepts_any_tuple(self) -> None:
        g = MatterClientGuard()
        # No exceptions
        g.assert_readable(("m1", "c1"))
        g.assert_writable(("m2", "c2"))


class TestBoundGuard:
    def test_bound_accepts_matching_namespace(self) -> None:
        g = MatterClientGuard().bind(("m1", "c1"))
        g.assert_readable(("m1", "c1"))
        g.assert_writable(("m1", "c1"))

    def test_bound_blocks_different_namespace_on_read(self) -> None:
        g = MatterClientGuard().bind(("m1", "c1"))
        with pytest.raises(MatterIsolationError) as exc:
            g.assert_readable(("m2", "c2"))
        msg = str(exc.value)
        assert "Cross-namespace read blocked" in msg
        assert "Fix:" in msg
        assert "Alternative:" in msg

    def test_bound_blocks_different_namespace_on_write(self) -> None:
        g = MatterClientGuard().bind(("m1", "c1"))
        with pytest.raises(MatterIsolationError) as exc:
            g.assert_writable(("rogue", "actor"))
        msg = str(exc.value)
        assert "Cross-namespace write blocked" in msg


class TestNamespaceShape:
    def test_invalid_shape_raises_helpful_message(self) -> None:
        g = MatterClientGuard()
        bad: Any = "not-a-tuple"
        with pytest.raises(MatterIsolationError) as exc:
            g.assert_readable(cast("tuple[str, str]", bad))
        msg = str(exc.value)
        assert "Invalid matter_client" in msg
        assert "Fix:" in msg
        assert "Alternative:" in msg

    def test_wrong_arity_tuple_raises(self) -> None:
        g = MatterClientGuard()
        bad: Any = ("only-one",)
        with pytest.raises(MatterIsolationError):
            g.assert_writable(cast("tuple[str, str]", bad))


class TestBindingReturnsNewGuard:
    def test_bind_returns_independent_instance(self) -> None:
        g = MatterClientGuard()
        g_bound = g.bind(("m1", "c1"))
        # Original guard is still unbound and accepts anything.
        g.assert_readable(("anything", "goes"))
        # The bound guard rejects mismatches.
        with pytest.raises(MatterIsolationError):
            g_bound.assert_readable(("anything", "goes"))
