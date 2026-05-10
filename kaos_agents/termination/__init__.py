"""Termination subsystem (paper §Q7).

The :class:`TerminationJudge` answers "is this turn done?" for the
AgentLoop. It composes 5 axes — budget, failure, loop, quality,
graceful degradation — into a single :class:`Decision`. Each axis is
independently testable; the Judge is a thin orchestrator.

Phase 4.B (rewrite-plan-ten-questions.md §13 Resolved Decision #7):
loop detection uses fuzzy hashing over the last N step signatures.
The plan called for TLSH, but kaos-nlp-core ships CTPH (see
:mod:`kaos_nlp_core.hashing`); :class:`LoopDetector` adapts the
"distance ≤ 30" semantics to "Jaccard similarity ≥ 0.5" and falls
back to string equality when the kaos-nlp-core hashing surface is
unavailable.
"""

from __future__ import annotations

from kaos_agents.termination.degrade import DegradationOutcome, DegradationPolicy
from kaos_agents.termination.judge import TerminationJudge
from kaos_agents.termination.loop_detect import LoopDetector, LoopDetectorResult
from kaos_agents.termination.types import Decision, DecisionKind, SuccessCriteria

__all__ = [
    "Decision",
    "DecisionKind",
    "DegradationOutcome",
    "DegradationPolicy",
    "LoopDetector",
    "LoopDetectorResult",
    "SuccessCriteria",
    "TerminationJudge",
]
