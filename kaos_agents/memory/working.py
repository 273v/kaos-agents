"""WorkingMemory — paper §6.1 "the desk".

Per-turn scratch that lives on :class:`TurnInvocation.extras` and is
garbage-collected at turn end. Distinct from session memory (multi-turn
conversation/findings) and institutional memory (across-session
knowledge).

Phase 4.A ships a thin typed wrapper over the existing extras dict so
planners and judges can read/write per-turn scratch without scribbling
on extras directly.
"""

from __future__ import annotations

from typing import Any

from kaos_agents.core.invocation import TurnInvocation, current_turn


class WorkingMemory:
    """Typed accessor over TurnInvocation.extras["working"].

    Initialised on first access; cleared at turn finalisation by the
    AgentLoop. Subclassing the dict instead of wrapping it would tie
    WorkingMemory to TurnInvocation's mutation discipline; we wrap so
    we can validate keys / surface common patterns (counters, latest
    perception result, etc.).
    """

    _SLOT = "working"

    def __init__(self, invocation: TurnInvocation | None = None) -> None:
        resolved = invocation or current_turn()
        if resolved is None:
            raise RuntimeError(
                "WorkingMemory requires an active TurnInvocation. "
                "Construct one explicitly via WorkingMemory(invocation=...) "
                "or call current_turn() inside AgentLoop.forward()."
            )
        self._invocation: TurnInvocation = resolved
        if self._SLOT not in self._invocation.extras:
            self._invocation.extras[self._SLOT] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._scratch.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._scratch[key] = value

    def update(self, **kwargs: Any) -> None:
        self._scratch.update(kwargs)

    def increment(self, counter: str, by: int = 1) -> int:
        cur = int(self._scratch.get(counter, 0))
        new = cur + by
        self._scratch[counter] = new
        return new

    def __contains__(self, key: str) -> bool:
        return key in self._scratch

    def __len__(self) -> int:
        return len(self._scratch)

    def keys(self) -> Any:
        return self._scratch.keys()

    def clear(self) -> None:
        self._scratch.clear()

    @property
    def _scratch(self) -> dict[str, Any]:
        return self._invocation.extras[self._SLOT]


__all__ = ["WorkingMemory"]
