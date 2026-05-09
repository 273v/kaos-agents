"""IntentSignature — kaos-llm-core Signature for goal extraction.

A Signature is a Pydantic model that declares the inputs and outputs of
an LLM function. Fields are marked as ``InputField`` or ``OutputField``
via :mod:`kaos_llm_core.signatures.fields`. The class docstring becomes
the system instruction the codec plumbs into the prompt.

This Signature is the typed contract for one Call inside
:class:`~kaos_agents.intent.extractor.IntentExtractor`. The Pythonic
shape mirrors :class:`~kaos_agents.intent.types.IntentResult`; the
extractor projects ``IntentSignature`` outputs onto an ``IntentResult``
at the module boundary.

Mirrors paper §3.2-3.5. Optimizable by ``InstructionOptimizer`` /
``MiproV2Optimizer`` against a labeled intent corpus.
"""

from __future__ import annotations

from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature

from kaos_agents.config import AgentPattern
from kaos_agents.intent.types import Ambiguity, Constraint, Goal


class IntentSignature(Signature):
    """Extract typed intent from a trigger payload.

    You are an intent classifier for an agentic system. Read the user's
    message and prior context, then produce a typed intent.

    Decision rules:

    1. Set ``requires_clarification=true`` ONLY when the message is so
       ambiguous you cannot reasonably propose a goal. Otherwise propose
       a best-effort goal and surface ambiguities for the agent to
       handle later.
    2. Be conservative on ``confidence`` — a confident wrong
       classification is worse than a tentative correct one. Reserve
       ``confidence>=0.9`` for unambiguous, single-goal messages.
    3. Pick ``pattern`` from the message shape, not the topic:
       conversational small-talk → ``CHAT``; "do X then Y then Z" →
       ``PLAN``; "what does the corpus say about X" → ``RESEARCH``.
    4. Surface every ambiguity you detect in ``ambiguities``, even when
       ``requires_clarification=false``. Downstream code uses these
       hints to resolve references from memory.
    5. Constraints are about *how the goal must be satisfied*: deadlines,
       budgets, jurisdictions, output format, style, scope. Goal restating
       does not count.
    """

    # inputs
    message: str = InputField(description="The user's request.")
    recent_messages: str = InputField(
        default="",
        description=(
            "Recent conversation context — a flattened summary of the "
            "last N MESSAGES section items joined with newlines, or "
            "empty string when there is no prior conversation."
        ),
    )
    domain_examples: str = InputField(
        default="",
        description=(
            "Optional calibration examples for the domain (paper §3.5). "
            'Format: "input -> goal" strings, one per line.'
        ),
    )

    # outputs
    goal: Goal = OutputField(description="The primary objective extracted from the message.")
    constraints: tuple[Constraint, ...] = OutputField(
        default=(),
        description=(
            "Constraints on the goal — deadlines, budgets, jurisdictions, "
            "format, style, scope. Empty tuple when none are stated."
        ),
    )
    ambiguities: tuple[Ambiguity, ...] = OutputField(
        default=(),
        description=(
            "Span-level pointers to unclear text in the message + "
            "candidate interpretations. Empty when the message is fully clear."
        ),
    )
    requires_clarification: bool = OutputField(
        default=False,
        description=(
            "True only when the message is so ambiguous that no "
            "reasonable goal can be proposed without user input."
        ),
    )
    pattern: AgentPattern = OutputField(
        default=AgentPattern.CHAT,
        description=(
            "Which agent pattern to dispatch: CHAT (single-turn ReAct), "
            "PLAN (multi-step plan-execute), or RESEARCH (RAG over corpus)."
        ),
    )
    confidence: float = OutputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Self-rating of the classification, 0.0 to 1.0.",
    )
