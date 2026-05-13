"""Action subsystem — paper Q4 ("how does the agent make things happen?").

Phase 1.C of the kaos-agents ground-up rewrite. Purely additive — no
existing module wires through it yet.

Public surface:

- :class:`Reversibility` — four-tier action classification.
- :class:`ActionPlan` / :class:`ActionResult` / :class:`ActionRefusal`
  — frozen value types for the propose → dispatch → result/refuse
  pipeline.
- :class:`Actor` — the gating Program; consumes an :class:`ActionPlan`
  and returns a result or a refusal.  **Lazy-imported** so that the
  base ``kaos-agents`` install (without the ``[llm]`` extra) can still
  ``import kaos_agents.action`` — :class:`Actor` itself subclasses
  ``kaos_llm_core.programs.base.Program`` and is only resolved on first
  attribute access (PEP 562).
- :class:`ApprovalWorkflow` — predicate + event builder for human
  approval.
- :class:`RateLimiter` / :class:`CircuitBreaker` /
  :class:`CircuitState` — :class:`KaosHook` subclasses with sync
  ``allow`` predicates for inline use by the Actor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_agents.action.approval import ApprovalWorkflow
from kaos_agents.action.circuit import CircuitBreaker, CircuitState
from kaos_agents.action.rate_limit import RateLimiter
from kaos_agents.action.reversibility import Reversibility, infer_reversibility
from kaos_agents.action.types import ActionPlan, ActionRefusal, ActionResult

if TYPE_CHECKING:  # pragma: no cover — typing-only re-export
    from kaos_agents.action.actor import Actor

__all__ = [
    "ActionPlan",
    "ActionRefusal",
    "ActionResult",
    "Actor",
    "ApprovalWorkflow",
    "CircuitBreaker",
    "CircuitState",
    "RateLimiter",
    "Reversibility",
    "infer_reversibility",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for optional-extra-dependent names.

    :class:`Actor` lives in :mod:`kaos_agents.action.actor`, which imports
    ``kaos_llm_core.programs.base.Program`` at module load.  ``kaos_llm_core``
    ships only with the ``[llm]`` extra, so eagerly importing it from this
    package's ``__init__`` would break the base install.

    Resolving :class:`Actor` lazily here keeps ``import kaos_agents.action``
    working without ``[llm]`` and only raises (with a clear install hint) at
    the point where a consumer actually touches the name.
    """
    if name == "Actor":
        try:
            from kaos_agents.action.actor import Actor as _Actor
        except ImportError as exc:  # pragma: no cover — exercised by smoke test
            raise ImportError(
                "kaos_agents.action.Actor requires the [llm] extra. "
                "Install with: `pip install 'kaos-agents[llm]'` (or "
                "`uv pip install 'kaos-agents[llm]'`). "
                f"Underlying ImportError: {exc}"
            ) from exc
        return _Actor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
