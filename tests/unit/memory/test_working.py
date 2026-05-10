"""WorkingMemory unit tests — Phase 4.A."""

from __future__ import annotations

import pytest

from kaos_agents.core.invocation import TurnInvocation, _active_turn_var
from kaos_agents.memory.working import WorkingMemory


class TestWorkingMemoryConstruction:
    def test_no_active_turn_raises_runtime_error(self) -> None:
        # Ensure no active turn for this task.
        token = _active_turn_var.set(None)
        try:
            with pytest.raises(RuntimeError) as exc:
                WorkingMemory()
            msg = str(exc.value)
            assert "TurnInvocation" in msg
            assert "current_turn" in msg
        finally:
            _active_turn_var.reset(token)

    def test_explicit_invocation_works(self) -> None:
        tin = TurnInvocation()
        wm = WorkingMemory(invocation=tin)
        assert "working" in tin.extras
        assert tin.extras["working"] == {}
        assert wm is not None

    def test_picks_up_active_turn_via_contextvar(self) -> None:
        tin = TurnInvocation()
        token = _active_turn_var.set(tin)
        try:
            wm = WorkingMemory()  # no explicit invocation arg
            wm.set("alpha", 1)
            assert tin.extras["working"]["alpha"] == 1
        finally:
            _active_turn_var.reset(token)


class TestWorkingMemoryAccess:
    def _wm(self) -> tuple[WorkingMemory, TurnInvocation]:
        tin = TurnInvocation()
        return WorkingMemory(invocation=tin), tin

    def test_set_get_round_trip(self) -> None:
        wm, _ = self._wm()
        wm.set("foo", "bar")
        assert wm.get("foo") == "bar"

    def test_get_default(self) -> None:
        wm, _ = self._wm()
        assert wm.get("missing") is None
        assert wm.get("missing", default=42) == 42

    def test_increment_default_zero_to_one(self) -> None:
        wm, _ = self._wm()
        assert wm.increment("ticks") == 1
        assert wm.increment("ticks") == 2
        assert wm.increment("ticks", by=3) == 5
        assert wm.get("ticks") == 5

    def test_update_merges_keys(self) -> None:
        wm, _ = self._wm()
        wm.set("a", 1)
        wm.update(b=2, c=3)
        assert wm.get("a") == 1
        assert wm.get("b") == 2
        assert wm.get("c") == 3

    def test_clear_empties_slot(self) -> None:
        wm, tin = self._wm()
        wm.set("x", 1)
        wm.set("y", 2)
        assert len(wm) == 2
        wm.clear()
        assert len(wm) == 0
        assert tin.extras["working"] == {}

    def test_contains_and_keys(self) -> None:
        wm, _ = self._wm()
        wm.set("present", 1)
        assert "present" in wm
        assert "absent" not in wm
        keys = list(wm.keys())
        assert keys == ["present"]
