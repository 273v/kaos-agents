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
    7. Populate ``targets`` with the corpus items the message points
       at, drawn VERBATIM from ``corpus_headlines``. The downstream
       planner uses this to scope work and the critic uses it to
       verify the answer covered the right files. Rules:

       - Specific named files (``"compare EMNA and Acme"``) →
         ``targets=["EMNA Mutual NDA.docx", "MNDA - Acme.docx"]``.
         Just the files the user named, not everything attached.
       - General reference to the attached set (``"summarize these"``,
         ``"what's in the documents?"``, ``"GL on these 5"``) →
         expand ``targets`` to the full ``corpus_headlines`` list.
         No sentinels — emit the actual filenames.
       - Anaphora alone with no specific file (``"summarize that"``
         when prior turn touched one document) → ``targets`` is the
         single file the anaphor resolves to, when resolvable from
         ``recent_messages``; empty list otherwise.
       - No corpus reference (``"what's 2+2?"``) → ``targets=[]``
         AND ``corpus_attached=false``.
       - Corpus-adjacent but no specific target (``"draft me an
         employment agreement"`` in a session with NDAs attached) →
         ``targets=[]`` with ``corpus_attached=true``. The planner
         decides whether the corpus is in scope.

       Every value in ``targets`` MUST appear verbatim in
       ``corpus_headlines``. Do not paraphrase, abbreviate, or
       invent filenames; the extractor rejects unknown filenames.
       Surfacing anaphora resolution as a typed output makes it
       inspectable, lets the planner scope tool calls, and gives
       the critic a verification handle. See
       ``kaos-modules/docs/plans/persona-matrix-followups.md`` §6.
    8. Factual-external-entity bias. When the user asks about a
       *factual external entity* — a regulation, statute, case,
       agency rule, public filing, public-company fact, current
       status of a real-world thing — ALWAYS set
       ``requires_clarification=false`` and propose a best-effort
       goal that the downstream planner can route to research
       tools. The downstream agent has FR / eCFR / EDGAR / GovInfo
       / web-search and is responsible for verification; it is NOT
       this classifier's job to extract every parameter up-front.
       Even when jurisdiction / version / time-frame is genuinely
       ambiguous, propose the most likely reading (latest version,
       U.S. federal scope unless the prompt suggests otherwise,
       current status as of today) and surface alternatives via
       ``ambiguities``. The 2026-05-19 ``what is the latest diesel
       emission reg`` session is the canonical anti-pattern: 6
       rounds of clarification (``which jurisdiction?`` /
       ``US federal or state?`` / ``on-road or off-road?``) before
       the agent finally answered from training memory with zero
       tool calls. Pick a default, dispatch to FR / eCFR, and let
       the answer carry its own caveat instead of ping-ponging on
       clarification.
    9. Clarification-loop ceiling. If ``recent_messages`` already
       contains an assistant turn that asked for clarification on
       this same goal in the prior turn, do NOT ask again. Set
       ``requires_clarification=false`` and propose the strongest
       reading. Two rounds of clarification is the ceiling; any
       further round wastes the user's turn and is a documented
       UX-regression mode. (Companion rule to GoalCheckSignature
       clarification-loop ceiling.)
    10. Domain-conventional shorthand. When the user uses a
       domain-conventional abbreviation in context (e.g., "GL" in
       a contracts / M&A setting → governing law; "DD" in
       transactions → due diligence; "RFI" in procurement →
       request for information; "10-K" in finance → annual SEC
       filing), prefer the conventional reading and propose the
       goal. Only set ``requires_clarification=true`` if the
       candidate readings are genuinely incompatible. The
       2026-05-18 persona run had an agent refuse to interpret
       "GL on these 5" as governing-law review of attached NDAs —
       that's an unnecessary over-refusal when the domain context
       is unambiguous.
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
    corpus_headlines: str = InputField(
        default="",
        description=(
            "Newline-separated list of corpus item headlines (typically "
            "filenames, optionally annotated with size / mime). Used by "
            "rule 7 to resolve anaphoric / generic references to "
            "specific files. Each line is one corpus item; the first "
            "token before any separator is the canonical filename to "
            "emit in ``targets``. Empty string when the session has no "
            "attached corpus."
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
    targets: tuple[str, ...] = OutputField(
        default=(),
        description=(
            "Corpus item filenames the intent points at, drawn VERBATIM "
            "from ``corpus_headlines``. Empty when the question is "
            "non-corpus or when there is no specific corpus reference. "
            "See rule 7 in the class docstring for the full semantics. "
            "Downstream the extractor validates each entry against "
            "``corpus_headlines`` and rejects unknown filenames."
        ),
    )
