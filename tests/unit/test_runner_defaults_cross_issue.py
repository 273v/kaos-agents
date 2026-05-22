"""Cross-issue invariant tests for ``Runner`` default-hook install.

This file is the integration glue between the launch-blocker plan's
Issue 2 (matter isolation) and Issue 5 (CircuitBreaker default
install). Each issue has its own per-hook test
(``test_runner_default_matter_isolation_hook.py`` and
``test_runner_default_circuit_breaker.py``), but those tests
construct their Runners with the OTHER hook absent. That hides a
regression class: someone refactors the default-hook installer and
accidentally drops one of the two — the per-issue tests still pass
because each only checks its own hook.

These tests pin the cross-issue invariant: a Runner constructed
with no overrides and ``matter_id="..."`` installs BOTH a
:class:`MatterIsolationHook` AND a :class:`CircuitBreaker` by
default. A regression that drops either trips this test even when
the per-issue suites stay green.

Plan: ``kaos-modules/docs/plans/2026-05-22-launch-blocker-top-10.md``
§Issue 2 + §Issue 5.
"""

from __future__ import annotations

import pytest

from kaos_agents.action.circuit import CircuitBreaker
from kaos_agents.config import Agent
from kaos_agents.memory.isolation import MatterIsolationHook
from kaos_agents.runtime.runner import Runner


def _hooks_of_type(runner: Runner, hook_type: type) -> list:
    """Return the installed hooks of a given type."""
    return [h for h in runner._hooks if isinstance(h, hook_type)]


@pytest.fixture
def agent() -> Agent:
    return Agent(name="test", instructions="test")


@pytest.mark.unit
def test_default_runner_installs_both_circuit_breaker_and_matter_hook(agent: Agent) -> None:
    """The load-bearing cross-issue invariant.

    A Runner built with ``matter_id="abc-2026-0042"`` and no other
    overrides MUST carry exactly one CircuitBreaker AND exactly one
    MatterIsolationHook. If a future refactor drops either, this
    test fails loudly."""
    runner = Runner(agent, matter_id="abc-2026-0042")
    cbs = _hooks_of_type(runner, CircuitBreaker)
    mhs = _hooks_of_type(runner, MatterIsolationHook)
    assert len(cbs) == 1, (
        f"expected exactly 1 CircuitBreaker installed by default, "
        f"got {len(cbs)}. This is the Issue 5 / B1.1 regression class."
    )
    assert len(mhs) == 1, (
        f"expected exactly 1 MatterIsolationHook installed by default, "
        f"got {len(mhs)}. This is the Issue 2 regression class."
    )
    # The MatterIsolationHook must carry the matter_id passed at
    # construction time — a regression that wires it with None would
    # silently degrade the hook to no-op mode and silently allow
    # cross-matter access.
    assert mhs[0].matter_id == "abc-2026-0042"


@pytest.mark.unit
def test_default_runner_without_matter_id_still_has_circuit_breaker(agent: Agent) -> None:
    """Legacy session path: ``matter_id=None`` (default) still gets
    the CircuitBreaker — Issue 5 is unconditional — and any installed
    MatterIsolationHook is in no-op mode (matter_id=None), so the
    dispatch layer doesn't fail on legacy sessions."""
    runner = Runner(agent)  # no matter_id
    cbs = _hooks_of_type(runner, CircuitBreaker)
    mhs = _hooks_of_type(runner, MatterIsolationHook)
    assert len(cbs) == 1, "CircuitBreaker default install must be unconditional"
    # The hook may be installed (no-op mode) or omitted entirely;
    # accept either. The safety invariant is that any installed hook
    # carries matter_id=None so it's a no-op.
    if mhs:
        assert mhs[0].matter_id is None


@pytest.mark.unit
def test_unsafe_bypass_disables_both_default_hooks(agent: Agent) -> None:
    """The escape hatch: ``unsafe_bypass=True`` skips BOTH default
    installs. Test + benchmark callers that need the raw loop rely
    on this — a regression that lets one default hook through is a
    correctness blocker for the bench harness."""
    runner = Runner(agent, matter_id="x", unsafe_bypass=True)
    cbs = _hooks_of_type(runner, CircuitBreaker)
    mhs = _hooks_of_type(runner, MatterIsolationHook)
    assert len(cbs) == 0, "unsafe_bypass=True must skip CircuitBreaker default install"
    assert len(mhs) == 0, "unsafe_bypass=True must skip MatterIsolationHook default install"


@pytest.mark.unit
def test_caller_supplied_matter_hook_is_not_double_installed(agent: Agent) -> None:
    """Idempotency mirror on the Issue 2 hook: caller-supplied
    hook survives unchanged, no double-install. Pre-existing
    consumers (e.g. a custom hook bound to a parent matter for a
    nested sub-matter audit chain) MUST keep their wiring."""
    custom_hook = MatterIsolationHook(matter_id="custom-matter")
    runner = Runner(agent, matter_id="other-matter", hooks=(custom_hook,))
    mhs = _hooks_of_type(runner, MatterIsolationHook)
    assert len(mhs) == 1
    assert mhs[0] is custom_hook
    # The caller's hook keeps its own matter_id — Runner does NOT
    # rebind it from the ctor kwarg. Intentional: a caller who
    # supplied a custom hook is opting out of default-binding
    # semantics.
    assert mhs[0].matter_id == "custom-matter"


@pytest.mark.unit
def test_per_issue_opt_out_kwargs_compose(agent: Agent) -> None:
    """Each per-issue opt-out kwarg disables only its own hook.

    The current Runner exposes ``install_default_circuit_breaker``
    for the Issue 5 default install. The Issue 2 MatterIsolationHook
    install is gated on the matter_id presence — passing
    ``matter_id=None`` skips it. Confirm both can be controlled
    independently."""
    runner = Runner(
        agent,
        matter_id="abc",
        install_default_circuit_breaker=False,
    )
    cbs = _hooks_of_type(runner, CircuitBreaker)
    mhs = _hooks_of_type(runner, MatterIsolationHook)
    # CircuitBreaker disabled via its own kwarg.
    assert len(cbs) == 0
    # MatterIsolationHook still installed because matter_id is set.
    assert len(mhs) == 1
    assert mhs[0].matter_id == "abc"
