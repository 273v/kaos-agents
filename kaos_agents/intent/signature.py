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

    Read the user's message and prior context, produce a typed intent.

    1. Set ``requires_clarification=true`` ONLY when the message is so
       ambiguous you cannot propose a goal. Otherwise propose a
       best-effort goal and surface ambiguities in ``ambiguities``.

    2. Be conservative on ``confidence``. Reserve ``confidence>=0.9``
       for unambiguous single-goal messages.

    3. Pick ``pattern`` from the message shape, not the topic:
       conversational small-talk → ``CHAT``; "do X then Y then Z" →
       ``PLAN``; "what does the corpus say about X" → ``RESEARCH``.

    4. Surface every ambiguity in ``ambiguities``, even when
       ``requires_clarification=false``.

    5. Constraints describe HOW the goal must be satisfied: deadlines,
       budgets, jurisdictions, output format, style, scope.

    6. When ``corpus_attached=true`` AND the message refers to attached
       documents (pronouns "that/those/these/it"; phrases "the file",
       "the document", "that PDF", "the attached", "the corpus"; short
       follow-ups "summarize", "what does it say", "extract terms") →
       set ``pattern=RESEARCH``. Do NOT pick ``CHAT``.

    7. Populate ``targets`` with corpus items the message points at,
       drawn VERBATIM from ``corpus_headlines``:

       - Named files ("compare EMNA and Acme") → just those files.
       - General reference ("summarize these", "what's in the docs",
         "GL on these 5") → the full ``corpus_headlines`` list.
       - Anaphora resolvable from ``recent_messages`` → the one file.
       - No corpus reference → ``targets=[]`` AND
         ``corpus_attached=false``.
       - Corpus-adjacent but no specific target → ``targets=[]`` with
         ``corpus_attached=true``.

       Every value in ``targets`` MUST appear verbatim in
       ``corpus_headlines``.

    8. Factual-external-entity bias. When the user asks about a
       regulation, statute, case, agency rule, public filing,
       public-company fact, or current status of a real-world thing,
       ALWAYS set ``requires_clarification=false`` and propose a
       best-effort goal. Even when jurisdiction / version / time-frame
       is ambiguous, propose the most likely reading (latest version,
       U.S. federal scope unless otherwise indicated, current status
       as of today) and surface alternatives via ``ambiguities``. When
       ``available_tool_groups`` lists a fitting group, dispatch with
       ``pattern=CHAT`` (single-turn tool-using) or
       ``pattern=RESEARCH`` (when the group reasons over loaded
       documents).

    9. Clarification ceiling. If ``recent_messages`` shows an
       assistant turn already asked for clarification on this same
       goal, do NOT ask again. Set ``requires_clarification=false``
       and propose the strongest reading.

    10. Domain-conventional shorthand. Interpret in context: "GL" in
        contracts → governing law; "DD" in deals → due diligence;
        "RFI" in procurement → request for information; "10-K" →
        annual SEC filing. Only set ``requires_clarification=true``
        when the candidate readings are genuinely incompatible.
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
    corpus_kinds: str = InputField(
        default="",
        description=(
            "Newline-separated sorted-distinct list of content-type "
            "*groups* present in the attached corpus, as classified by "
            "``kaos_nlp_core.content_type.detect()``. Vocabulary: "
            "``pdf`` / ``office-docx`` / ``office-xlsx`` / ``office-pptx`` / "
            "``office-doc`` / ``office-xls`` / ``office-ppt`` / ``image`` / "
            "``audio`` / ``video`` / ``archive`` / ``email`` / ``html`` / "
            "``text`` / ``font`` / ``binary`` / ``unknown``. "
            "Producers (kaos-ui upload handler, kaos-source materializer, "
            "any consumer that adds to ``SessionMemory.DOCUMENTS``) "
            "SHOULD set ``metadata['content_type_group']`` on each "
            "document item so this signal is populated; "
            "``_corpus_kinds_from_memory`` aggregates those values "
            "here. Empty string when no producer set the metadata. "
            "The classifier MAY use this to bias toward PDF / office / "
            "image extraction tools when those formats dominate the "
            "corpus — but should not refuse on the basis of corpus "
            "kinds alone."
        ),
    )
    available_tool_groups: str = InputField(
        default="",
        description=(
            "Newline-separated catalogue of tool groups registered on "
            "the runtime THIS turn. Each non-empty line is one group "
            'in the shape ``"<name>: <one-sentence purpose>"``. Rule 8 '
            "uses this to decide whether a relevant group covers a "
            "factual-external-entity question — when a fitting group "
            "is listed the classifier proposes a goal with the "
            "appropriate pattern instead of falling back to CHAT-with-"
            "no-tools. The classifier MUST NOT enumerate specific tool "
            "names in its proposed goal; it points the planner at the "
            "right pattern and lets the planner pick a tool. Default "
            '``""`` preserves the pre-0.1.0a17 routing path for '
            "callers that don't populate this input."
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
