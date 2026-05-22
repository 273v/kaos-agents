"""Unit tests for Runner.__init__ default CircuitBreaker install
(plan §Issue 5 / launch-blocker B1.1).

Pre-0.1.8 the CircuitBreaker hook was only wired by the API server
(``kaos_agents/api/server.py:459/590``) and by the SPA. Direct
``Runner(...)`` callers — CLI agents, MCP tool wrappers, kaos-agents-
bench scenarios, benchmarks — ran without protection against the
N-consecutive-empty-results pathology that powered session
01KS2DEBYT341F1F16B3BRQRV0 (12 zero-result web searches in a row, no
breaker, runaway cost).

Tests:

* default ctor → exactly one CircuitBreaker in ``_hooks``
* opt-out via ``install_default_circuit_breaker=False`` → zero
* idempotent: caller already passed a CircuitBreaker → still one
* unsafe_bypass=True → opt-out cascades through (no breaker)
* extra hooks alongside the default install are preserved
"""

from __future__ import annotations

import pytest

from kaos_agents.action.circuit import CircuitBreaker
from kaos_agents.config import Agent
from kaos_agents.hooks.base import KaosHook
from kaos_agents.runtime.runner import Runner


def _count_breakers(runner: Runner) -> int:
    return sum(isinstance(h, CircuitBreaker) for h in runner._hooks)


@pytest.fixture
def agent() -> Agent:
    return Agent(name="test", instructions="test")


@pytest.mark.unit
def test_default_install_adds_circuit_breaker(agent: Agent) -> None:
    """Pre-0.1.8 callers got an empty hooks tuple; with the default
    install they now get one CircuitBreaker without changing any
    other call shape."""
    runner = Runner(agent)
    assert _count_breakers(runner) == 1


@pytest.mark.unit
def test_opt_out_via_install_kwarg_skips_breaker(agent: Agent) -> None:
    """Tests + benches that need the raw loop can disable the
    auto-install via ``install_default_circuit_breaker=False``."""
    runner = Runner(agent, install_default_circuit_breaker=False)
    assert _count_breakers(runner) == 0


@pytest.mark.unit
def test_unsafe_bypass_also_skips_default_breaker(agent: Agent) -> None:
    """``unsafe_bypass=True`` is the existing "raw loop" escape hatch
    for tests + internal benchmarks. The default-breaker install must
    honor it so bench scenarios aren't accidentally protected."""
    runner = Runner(agent, unsafe_bypass=True)
    assert _count_breakers(runner) == 0


@pytest.mark.unit
def test_idempotent_when_caller_supplies_breaker(agent: Agent) -> None:
    """If the caller already passed a CircuitBreaker (e.g. the API
    server's existing wiring), the default install MUST NOT duplicate
    — two breakers in the hook chain double-counts consecutive
    failures and would trip far earlier than intended."""
    caller_supplied = CircuitBreaker()
    runner = Runner(agent, hooks=(caller_supplied,))
    assert _count_breakers(runner) == 1
    # Identity check: the caller's instance is the one that survived
    # (we appended nothing; idempotent skip).
    assert runner._hooks[0] is caller_supplied


@pytest.mark.unit
def test_default_install_preserves_other_hooks(agent: Agent) -> None:
    """Hooks supplied by the caller (LoggingHook, AuditHook, etc.)
    must remain in the tuple — the default install appends a breaker,
    it does not replace the existing hooks."""

    class _SentinelHook(KaosHook):
        pass

    sentinel = _SentinelHook()
    runner = Runner(agent, hooks=(sentinel,))
    # Sentinel preserved + default breaker appended = 2 hooks total.
    assert len(runner._hooks) == 2
    assert runner._hooks[0] is sentinel
    assert _count_breakers(runner) == 1
