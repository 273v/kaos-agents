"""Unit tests for the Runner-level MatterIsolationHook install (plan §Issue 2).

Mirrors the structure of ``test_runner_default_circuit_breaker.py``:
when ``matter_id=...`` is passed, the Runner idempotently appends a
:class:`MatterIsolationHook` to its hooks tuple. ``matter_id=None``
(default) leaves the hooks untouched so legacy callers keep working.
"""

from __future__ import annotations

import pytest

from kaos_agents.action.circuit import CircuitBreaker
from kaos_agents.config import Agent
from kaos_agents.hooks.base import KaosHook
from kaos_agents.memory.isolation import MatterIsolationHook
from kaos_agents.runtime.runner import Runner


@pytest.fixture
def stub_agent() -> Agent:
    return Agent(name="test", instructions="test")


@pytest.mark.unit
def test_no_matter_id_means_no_isolation_hook_installed(stub_agent: Agent) -> None:
    """The default-None case: no MatterIsolationHook installed.
    Pre-existing Runners construct with their pre-Issue-2 hook
    set, unchanged."""
    runner = Runner(stub_agent)
    has_hook = any(isinstance(h, MatterIsolationHook) for h in runner._hooks)
    assert has_hook is False


@pytest.mark.unit
def test_matter_id_installs_isolation_hook(stub_agent: Agent) -> None:
    """Constructing with ``matter_id=...`` appends a MatterIsolationHook
    bound to that matter."""
    runner = Runner(stub_agent, matter_id="ABC-2026-0001")
    hooks = [h for h in runner._hooks if isinstance(h, MatterIsolationHook)]
    assert len(hooks) == 1
    assert hooks[0].matter_id == "ABC-2026-0001"


@pytest.mark.unit
def test_matter_id_install_is_idempotent_when_caller_supplied(
    stub_agent: Agent,
) -> None:
    """If the caller already installed a MatterIsolationHook, the
    Runner does NOT double-install — even when the bound matter
    differs from the caller's. The caller's hook wins."""
    caller_hook = MatterIsolationHook(matter_id="CALLER-MATTER")
    runner = Runner(
        stub_agent,
        matter_id="ARG-MATTER",
        hooks=(caller_hook,),
    )
    hooks = [h for h in runner._hooks if isinstance(h, MatterIsolationHook)]
    assert len(hooks) == 1
    assert hooks[0] is caller_hook  # caller's instance preserved


@pytest.mark.unit
def test_unsafe_bypass_skips_isolation_install(stub_agent: Agent) -> None:
    """``unsafe_bypass=True`` is the test-only escape hatch. It
    skips the isolation install (and the circuit breaker default)
    so test fixtures can exercise the raw loop."""
    runner = Runner(stub_agent, matter_id="MATTER-X", unsafe_bypass=True)
    has_hook = any(isinstance(h, MatterIsolationHook) for h in runner._hooks)
    assert has_hook is False


@pytest.mark.unit
def test_isolation_hook_coexists_with_circuit_breaker(stub_agent: Agent) -> None:
    """The two defaults are independent: matter_id installs both
    by default (CircuitBreaker per Issue 5 / B1.1 + MatterIsolationHook
    per Issue 2). Order: CircuitBreaker first, then MatterIsolationHook,
    matching the chronology in Runner.__init__."""
    runner = Runner(stub_agent, matter_id="MATTER-A")
    matter_hooks = [h for h in runner._hooks if isinstance(h, MatterIsolationHook)]
    breaker_hooks = [h for h in runner._hooks if isinstance(h, CircuitBreaker)]
    assert len(matter_hooks) == 1
    assert len(breaker_hooks) == 1


@pytest.mark.unit
def test_runner_exposes_bound_matter_id(stub_agent: Agent) -> None:
    """The ``_matter_id`` attribute lets audit / observability layers
    correlate the Runner instance with its tenant scope."""
    runner = Runner(stub_agent, matter_id="ABC-2026-0001")
    assert runner._matter_id == "ABC-2026-0001"

    none_runner = Runner(stub_agent)
    assert none_runner._matter_id is None


@pytest.mark.unit
def test_other_hooks_preserved_when_matter_id_set(stub_agent: Agent) -> None:
    """Adding the MatterIsolationHook MUST NOT drop other hooks
    the caller supplied. Critical because Runners often have an
    audit hook + OTel hook + permissions hook already wired."""

    class _DummyAudit(KaosHook):
        pass

    audit = _DummyAudit()
    runner = Runner(stub_agent, matter_id="MATTER-A", hooks=(audit,))
    assert audit in runner._hooks
    # And the isolation hook is still appended.
    assert any(isinstance(h, MatterIsolationHook) for h in runner._hooks)
