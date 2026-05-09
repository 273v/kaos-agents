"""kaos_agents.intent — Phase 1.A Intent subsystem (Ten Questions Q2).

Answers the paper's Q2: how does the agent understand what's being asked?

Public surface:

* :class:`Goal`, :class:`Constraint`, :class:`ConstraintKind` —
  the typed pieces of an intent (paper §3.2 / §3.4).
* :class:`Ambiguity`, :class:`AmbiguityKind` — span-level pointers to
  unclear text + candidate interpretations (paper §3.3).
* :class:`IntentResult` — the new typed result that Phase 2's AgentLoop
  consumes. Distinct from the legacy
  :class:`kaos_agents.types.intents.IntentResult`, which stays untouched
  until cutover.
* :class:`IntentSignature` — kaos-llm-core Signature for the LLM-facing
  schema.
* :class:`IntentExtractor` — Program that drives the Signature and
  projects the result onto :class:`IntentResult`.
* :class:`ClarificationLoop` — composes a clarifying question for an
  IntentResult with ``requires_clarification=True``.

This package is purely additive — Phase 1.A does not wire it into the
runtime. Phase 2 will extend ``AgentLoop.prepare_turn`` to call the
extractor and route on the result.
"""

from kaos_agents.intent.clarify import ClarificationLoop
from kaos_agents.intent.extractor import IntentExtractor
from kaos_agents.intent.signature import IntentSignature
from kaos_agents.intent.types import (
    Ambiguity,
    AmbiguityKind,
    Constraint,
    ConstraintKind,
    Goal,
    IntentResult,
)

__all__ = [
    "Ambiguity",
    "AmbiguityKind",
    "ClarificationLoop",
    "Constraint",
    "ConstraintKind",
    "Goal",
    "IntentExtractor",
    "IntentResult",
    "IntentSignature",
]
