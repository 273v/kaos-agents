"""Budget-related error events.

:class:`BudgetExceeded` is a typed error subtype distinct from
:class:`RunError` so plan-execute logic can react specifically — for
example, replan with a tighter scope or ask the user to lift the
ceiling. Generic terminal errors stay on ``RunError``.
"""

from __future__ import annotations

from kaos_agents.events._intermediates import LifecycleEvent


class BudgetExceeded(LifecycleEvent):
    """A configured budget was exceeded.

    ``kind`` is one of ``"cost"``, ``"tokens"``, ``"steps"``,
    ``"replans"``, ``"wall_clock"`` — matches
    :class:`kaos_agents.types.plan.StopReason` values.

    Listeners (plan strategies, UI, hooks) decide whether the run
    stops, replans tighter, or escalates to the user.
    """

    kind: str = ""
    limit: float = 0.0
    actual: float = 0.0
    reason: str = ""


__all__ = ["BudgetExceeded"]
