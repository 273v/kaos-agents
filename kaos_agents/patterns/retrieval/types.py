"""Value types for the retrieval-planner Phase 1 surface.

Frozen + slotted dataclasses per the kaos-agents value-type convention.
StrEnum so the value is JSON-serializable on the wire (LLM emits the
strategy as a string literal; ``RetrievalStrategy("bm25")`` parses
it back).

These types are deliberately import-cheap. Lifting to kaos-llm-core in
Phase 2 (see ``plans/2026-05-26-retrieval-planner-and-findings-dispatch.md``)
moves them verbatim — no kaos-agents-specific dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, unique


@unique
class RetrievalStrategy(StrEnum):
    """How to narrow a corpus before grounding."""

    NONE = "none"
    """Pass the full DocumentView to the grounding step.

    Cheap-and-correct for small corpora (N < retrieval_plan_floor) and
    for short documents where ``every_sentence_selector`` would
    enumerate a tractable candidate set anyway. The planner emits
    this when the question's specificity matches the corpus size.
    """

    TOKEN = "token"
    """Lexical narrowing by single-word tokens.

    Run BM25 over the token list joined as one query. Cheapest LLM-
    free narrowing when the question carries distinctive nouns.
    """

    NGRAM = "ngram"
    """Lexical narrowing by multi-word phrases.

    Like ``TOKEN`` but the probe vocabulary is phrase-shaped. BM25
    tokenizes phrases naturally; an exact-phrase mode is future work
    in kaos-content if needed.
    """

    BM25 = "bm25"
    """Doc-level BM25 triage then per-doc enumeration.

    Two-step: ``triage_corpus`` narrows the DOCUMENTS section to the
    top-K docs by BM25 score against the question, then the planner
    rebuilds the merged DocumentView from those K docs.
    """

    EMBEDDING = "embed"
    """Dense-retrieval narrowing via ``kaos-nlp-transformers``.

    Routes to ``kaos_content.search.search_document(retrieval="embeddings",
    level="sentence")``. Requires the optional ``kaos-nlp-transformers``
    extra; applier falls back to BM25 with a logged warning when the
    extra isn't installed.
    """


@dataclass(frozen=True, slots=True)
class RetrievalPlanResult:
    """Typed plan returned by ``RetrievalPlanner.plan``.

    Mirrors the ``PlanRetrieval`` Signature's output fields exactly,
    plus a ``usage`` slot so the caller can plumb the planner's
    LLM cost into ``TurnSummary.cost_usd``. ``usage`` is ``None``
    when the planner short-circuited (e.g. corpus below the
    retrieval_plan_floor).
    """

    strategy: RetrievalStrategy
    """Which narrowing strategy to apply."""

    tokens: tuple[str, ...] = ()
    """Single-word tokens (for ``TOKEN`` strategy)."""

    ngrams: tuple[str, ...] = ()
    """Multi-word phrases (for ``NGRAM`` strategy)."""

    query: str = ""
    """Free-text query (for ``BM25`` / ``EMBEDDING`` strategy).

    Also used as the BM25 input when ``TOKEN`` / ``NGRAM`` need a
    fallback (empty probe list)."""

    top_k: int = 20
    """How many docs/sentences to keep after narrowing."""

    reasoning: str = ""
    """One-sentence rationale from the LLM (or the short-circuit reason
    when ``strategy=NONE`` was assigned without an LLM call)."""

    usage: object | None = None
    """``InvocationUsage`` from the planner Call, or ``None``.

    Typed as ``object`` to keep this module dep-free (the value type
    lives in ``kaos_agents.types.usage`` which itself is import-cheap;
    we keep this surface aligned with the Phase 2 lift where this
    field becomes ``InvocationUsage | None`` once the planner moves
    into ``kaos-llm-core`` next to that value type)."""


@dataclass(frozen=True, slots=True)
class RetrievalApplyResult:
    """Telemetry record from ``apply_retrieval_plan``.

    Surfaces in ``Span(SUBAGENT, "research.retrieval_apply")``
    attributes so the SPA Activity panel + the corpus-stress judge
    see exactly what the applier did. No LLM cost — pure mechanical
    narrowing.
    """

    strategy: RetrievalStrategy
    """The strategy that was applied (may differ from the planner's
    pick when an applier fell back — e.g. ``EMBEDDING`` requested
    without the ``kaos-nlp-transformers`` extra falls back to BM25)."""

    kept: int
    """Number of items kept after narrowing — docs for ``BM25``,
    sentences for ``TOKEN`` / ``NGRAM`` / ``EMBEDDING``, full corpus
    item count for ``NONE``."""

    dropped: int = 0
    """Items dropped by the narrowing. ``0`` for ``NONE``."""

    fallback_reason: str = ""
    """Non-empty when an applier degraded the planner's pick
    (e.g. ``"kaos-nlp-transformers not installed"``)."""

    warnings: tuple[str, ...] = field(default_factory=tuple)
    """Soft diagnostics — empty query / no matches / etc."""


__all__ = [
    "RetrievalApplyResult",
    "RetrievalPlanResult",
    "RetrievalStrategy",
]
