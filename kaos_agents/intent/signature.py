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
    6. When ``corpus_attached=true`` (the session has documents in
       memory) and the message refers to those documents indirectly
       (pronouns like "that", "those", "these", "it"; or "the file",
       "the document", "that PDF", "the attached", "the corpus"; or
       short follow-ups like "summarize" / "what does it say" /
       "extract terms") → set ``pattern=RESEARCH`` (grounded
       Q&A / extraction over the corpus). The downstream agent will
       route to ``search_memory(DOCUMENTS)`` and produce cited
       findings. Do NOT pick ``CHAT`` for such references — the
       previous SPA regression (R1-REAL UX-C2, 2026-05-17) was
       exactly this: a follow-up "summarize that" with an attached
       PDF routed to CHAT, the agent had no corpus context in the
       prompt, and confidently answered from training data instead
       of the attached document.
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
    corpus_attached: bool = InputField(
        default=False,
        description=(
            "True when the session has one or more documents attached "
            "in memory (SessionMemory.DOCUMENTS section). The agent "
            "loop computes this from the live memory snapshot and "
            "passes it in so the classifier can resolve indirect "
            "document references (rule 6). Default ``False`` keeps "
            "behavior unchanged for sessions with no attached corpus."
        ),
    )
    corpus_size: int = InputField(
        default=0,
        ge=0,
        description=(
            "Number of documents currently in SessionMemory.DOCUMENTS. "
            "Calibration signal for rule 6 — single-document sessions "
            "tend to use 'the file' / 'it'; multi-document corpora "
            "tend to use 'the docs' / 'these'. Default ``0`` (no "
            "corpus) keeps behavior unchanged."
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
