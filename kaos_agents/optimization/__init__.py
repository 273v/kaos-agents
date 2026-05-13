"""kaos-agents optimization — evaluation harness + agent-aware optimizers.

Phase 5.B ships :func:`evaluate_agent`: the agent-layer counterpart to
:func:`kaos_llm_core.optimization.evaluation.evaluate`. Routes each
example through a fully-wired :class:`AgentLoop` (or a constructor
factory), captures the resulting :class:`TurnInvocation`, applies a
metric, and aggregates into a :class:`EvalResult` reusing the
kaos-llm-core type so existing optimizer machinery composes.

Cost attribution: every turn run inside an :func:`evaluate_agent`
trial scope publishes its TurnSummary tokens+cost to the active
:class:`~kaos_llm_core.optimization.trial_runner.TrialRunner` via the
Phase 5.A :class:`CostTrackingHook` integration. So an
:func:`evaluate_agent` run inside a ``TrialRunner.trial(...)`` context
charges the right accumulator without manual plumbing.
"""

from __future__ import annotations

from kaos_agents.optimization.evaluate import (
    AgentEvalResult,
    AgentExample,
    AgentMetricFn,
    evaluate_agent,
)

__all__ = [
    "AgentEvalResult",
    "AgentExample",
    "AgentMetricFn",
    "evaluate_agent",
]
