"""Tests for kaos_agents.registry.event_registry.

Covers:
- Auto-registration at KaosEvent subclass creation
- Manual register / get / unregister / clear
- Conflict detection (double-registration without force=True)
- Round-trip via the default registry path used by deserialize_event
"""

from __future__ import annotations

import pytest
from kaos_core.exceptions import RegistryError

from kaos_agents.base.event import KaosEvent
from kaos_agents.events import (
    ALL_EVENT_TYPES,
    StepStart,
    TurnStart,
)
from kaos_agents.registry.event_registry import (
    EventRegistry,
    default_event_registry,
)


@pytest.mark.unit
class TestAutoRegistration:
    def test_default_registry_populated_for_all_event_types(self) -> None:
        """Every concrete event in ALL_EVENT_TYPES is auto-registered."""
        for cls in ALL_EVENT_TYPES:
            registered = default_event_registry.get(cls.event_type())
            assert registered is cls, (
                f"{cls.__name__} (type={cls.event_type()!r}) not in default registry"
            )

    def test_intermediate_classes_also_registered(self) -> None:
        """StreamDelta + LifecycleEvent auto-register too — downstream code
        can target them by name."""
        from kaos_agents.events import LifecycleEvent, StreamDelta

        assert default_event_registry.get(StreamDelta.event_type()) is StreamDelta
        assert default_event_registry.get(LifecycleEvent.event_type()) is LifecycleEvent

    def test_register_false_opts_out(self) -> None:
        """A subclass declared with register=False stays out of the registry."""
        before = set(default_event_registry.list_types())

        class _NotRegistered(KaosEvent, register=False):
            """Not registered with the default registry."""

        after = set(default_event_registry.list_types())
        assert after == before, "register=False should suppress auto-registration"


@pytest.mark.unit
class TestEventRegistry:
    def test_register_and_get(self) -> None:
        reg = EventRegistry()
        reg.register(TurnStart)
        assert reg.get("turn_start") is TurnStart

    def test_get_unknown_returns_none(self) -> None:
        reg = EventRegistry()
        assert reg.get("not_a_real_event") is None

    def test_double_register_raises(self) -> None:
        """The kaos-core convention is to surface accidental double-registration."""
        reg = EventRegistry()
        reg.register(TurnStart)

        # Same class, no force: idempotent (existing == cls).
        reg.register(TurnStart)

        # Different class same key: error.
        class _AltTurnStart(KaosEvent, register=False):
            """An alternate TurnStart — same discriminator."""

            @classmethod
            def event_type(cls) -> str:
                return "turn_start"

        with pytest.raises(RegistryError):
            reg.register(_AltTurnStart)

    def test_force_replaces(self) -> None:
        reg = EventRegistry()
        reg.register(TurnStart)

        class _AltTurnStart(KaosEvent, register=False):
            """Replaces TurnStart for the turn_start discriminator."""

            @classmethod
            def event_type(cls) -> str:
                return "turn_start"

        reg.register(_AltTurnStart, force=True)
        assert reg.get("turn_start") is _AltTurnStart

    def test_unregister(self) -> None:
        reg = EventRegistry()
        reg.register(TurnStart)
        removed = reg.unregister("turn_start")
        assert removed is TurnStart
        assert reg.get("turn_start") is None

    def test_clear(self) -> None:
        reg = EventRegistry()
        reg.register(TurnStart)
        reg.register(StepStart)
        reg.clear()
        assert len(reg) == 0

    def test_membership(self) -> None:
        reg = EventRegistry()
        reg.register(TurnStart)
        assert "turn_start" in reg
        assert "not_a_real_event" not in reg
        assert 42 not in reg  # non-string keys are False, never raise

    def test_list_types_sorted(self) -> None:
        reg = EventRegistry()
        reg.register(StepStart)
        reg.register(TurnStart)
        # Sorted alphabetically — "step_start" < "turn_start".
        assert reg.list_types() == ["step_start", "turn_start"]

    def test_iter_yields_keys(self) -> None:
        reg = EventRegistry()
        reg.register(TurnStart)
        reg.register(StepStart)
        assert set(reg) == {"turn_start", "step_start"}


@pytest.mark.unit
class TestSerdeViaRegistry:
    """Confirm the registry is the live dispatch table for serialize/deserialize."""

    def test_round_trip_uses_default_registry(self) -> None:
        from kaos_agents.events import deserialize_event, serialize_event

        e = TurnStart(timestamp=1.0, sequence=0, session_id="s", run_id="r", turn_number=3)
        round_tripped = deserialize_event(serialize_event(e))
        assert isinstance(round_tripped, TurnStart)
        assert round_tripped.turn_number == 3

    def test_unknown_type_uses_registry_for_error_message(self) -> None:
        from kaos_agents.errors import EventDeserializationError
        from kaos_agents.events import deserialize_event

        with pytest.raises(EventDeserializationError, match="Valid types"):
            deserialize_event({"type": "bogus_event_name"})
