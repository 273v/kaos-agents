"""Unit tests for the :class:`TriggerSource` ABC contract.

Phase 2.A ships only the contract; concrete implementations land in
Phase 4+. These tests pin down the abstract surface so subclasses
that arrive later have a stable interface to satisfy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from kaos_agents.triggers.base import Trigger, TriggerSource


class TestTriggerSourceContract:
    def test_abstract_cannot_instantiate(self) -> None:
        """Plain :class:`TriggerSource` raises ``TypeError`` on construction."""
        with pytest.raises(TypeError):
            TriggerSource()  # type: ignore[abstract]

    def test_partial_implementation_cannot_instantiate(self) -> None:
        """A subclass that only implements one abstract method still cannot
        be constructed."""

        class OnlyAiter(TriggerSource):
            def __aiter__(self) -> AsyncIterator[Trigger]:  # type: ignore[override]
                raise NotImplementedError

        with pytest.raises(TypeError):
            OnlyAiter()  # type: ignore[abstract]

    def test_full_implementation_can_instantiate(self) -> None:
        """A subclass that implements both abstract methods can be
        constructed and conforms to the protocol."""

        class _MinimalSource(TriggerSource):
            def __init__(self) -> None:
                self.closed = False

            async def __aiter__(self) -> AsyncIterator[Trigger]:  # type: ignore[override]
                # Empty async generator — yields no triggers, just satisfies
                # the protocol shape so we can verify constructability.
                if False:
                    yield Trigger.mcp("never")  # pragma: no cover

            async def close(self) -> None:
                self.closed = True

        src = _MinimalSource()
        assert isinstance(src, TriggerSource)
        assert src.closed is False
