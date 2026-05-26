"""LLM-driven retrieval planner.

Mirrors ``kaos_llm_core.programs.query_expander.LLMQueryExpander`` shape:

* a ``Signature`` describing the typed I/O contract,
* a ``Protocol`` so callers can swap in heuristic / cached / fixture
  implementations without touching consumers,
* a thin ``Call``-wrapper class that runs the Signature.

This is the surface that lifts verbatim to
``kaos_llm_core/programs/retrieval_planner.py`` in Phase 2. Keep it
free of kaos-agents-specific imports so the port is mechanical.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kaos_core.logging import get_logger
from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures.fields import InputField, OutputField
from kaos_llm_core.signatures.signature import Signature
from kaos_llm_core.types import Example

from kaos_agents.patterns.retrieval.types import (
    RetrievalPlanResult,
    RetrievalStrategy,
)

logger = get_logger(__name__)


class PlanRetrieval(Signature):
    """Pick a corpus-narrowing strategy and emit typed probes.

    The planner sits between intent classification and the
    grounding step (FindingsAgent / RAG). It looks at the question
    and a one-line corpus summary, then picks the cheapest narrowing
    that preserves recall on the relevant evidence.

    Strategies, cheapest-to-most-expensive:

    * ``none`` -- skip narrowing entirely. Right when the corpus is
      small enough to scan end-to-end, or when the question is so
      broad that any token / BM25 filter would drop relevant
      evidence.
    * ``token`` -- emit 3-10 single-word search tokens. Right for
      questions that name a specific entity / number / acronym.
    * ``ngram`` -- emit 2-5 multi-word phrases. Right for questions
      about a named clause / section ("indemnification scope",
      "launch date").
    * ``bm25`` -- emit one free-text BM25 query and a doc-level
      top_k. Right for multi-needle questions over large heterogeneous
      corpora ("for each of Alpha, Bravo, Charlie report the
      planted fact").
    * ``embed`` -- emit one query for dense retrieval. Right for
      semantic / paraphrase questions where the answer's wording
      will differ from the question's.

    The probe lists you DO NOT use for the chosen strategy should be
    empty. ``query`` is only required for ``bm25`` and ``embed``;
    it can also serve as a BM25 fallback when ``token`` / ``ngram``
    probe lists are sparse.

    Output ``top_k`` is a recall target: how many docs (for BM25)
    or sentences (for TOKEN / NGRAM / EMBED) to keep. Default 20.
    """

    question: str = InputField(description="The user's question, verbatim.")
    corpus_summary: str = InputField(
        description=(
            "One-line summary of the attached corpus, e.g. "
            "'53 docs (3 PDF, 2 DOCX, 48 HTML); avg ~800 words/doc'."
        )
    )

    strategy: str = OutputField(
        description=(
            "Which narrowing strategy to apply. One of: 'none', 'token', 'ngram', 'bm25', 'embed'."
        )
    )
    tokens: list[str] = OutputField(
        description=("Single-word search tokens (3-10). Empty unless strategy='token'."),
        default_factory=list,
    )
    ngrams: list[str] = OutputField(
        description=("Multi-word search phrases (2-5). Empty unless strategy='ngram'."),
        default_factory=list,
    )
    query: str = OutputField(
        description=(
            "Free-text BM25 or embedding query. Required when strategy is 'bm25' or 'embed'."
        ),
        default="",
    )
    top_k: int = OutputField(
        description=(
            "How many docs (for BM25) or sentences (for "
            "TOKEN / NGRAM / EMBED) to keep after narrowing. "
            "Default 20."
        ),
        default=20,
    )
    reasoning: str = OutputField(
        description=("One sentence on why this strategy + these probes. Surfaces in audit logs.")
    )


@runtime_checkable
class RetrievalPlanner(Protocol):
    """Protocol for retrieval-strategy + probe selection.

    Anything that can take a question + corpus summary and emit a
    :class:`RetrievalPlanResult`. Lets callers swap in heuristic,
    fixture-based, or LLM-driven planners without touching the
    applier or the dispatch wire-in.
    """

    async def plan(self, question: str, corpus_summary: str) -> RetrievalPlanResult: ...


def _coerce_strategy(value: str) -> RetrievalStrategy:
    """Best-effort string-to-enum. Falls back to ``NONE`` on miss.

    Model output is occasionally noisy (extra whitespace, alternate
    casing). We trim + lowercase before parsing, and log on miss
    so the operator sees the divergent string before silently
    degrading to ``NONE``.
    """
    if not value:
        return RetrievalStrategy.NONE
    cleaned = value.strip().lower()
    try:
        return RetrievalStrategy(cleaned)
    except ValueError:
        logger.warning(
            "retrieval_planner: unknown strategy %r -- falling back to NONE",
            value,
        )
        return RetrievalStrategy.NONE


class LLMRetrievalPlanner:
    """LLM-driven implementation of the :class:`RetrievalPlanner` Protocol.

    Single Signature call per ``plan()`` invocation; returns the
    coerced :class:`RetrievalPlanResult` with the LLM's ``InvocationUsage``
    in the ``usage`` slot so the caller can plumb it into
    ``TurnSummary.cost_usd``.

    The caller is responsible for the corpus-size skip-floor heuristic
    (see ``BaseAgent._run_findings_dispatch``). Keeping the floor
    outside this class keeps the primitive single-purpose and
    re-usable.
    """

    def __init__(
        self,
        model: str,
        *,
        examples: list[Example] | None = None,
        core_settings: Any = None,
    ) -> None:
        """Construct the planner.

        Args:
            model: Provider:model string for the underlying ``Call``.
            examples: Optional few-shot grounding examples forwarded
                to the inner ``Call``. Default ``None``.
            core_settings: Optional ``KaosLLMCoreSettings`` forwarded
                to the inner ``Call`` so per-request config (MCP
                ``_meta.kaos_config`` overrides, trace flags) reach
                the underlying call. Mirrors the wiring in
                :class:`LLMQueryExpander`.
        """
        self._call = Call(
            PlanRetrieval,
            model=model,
            examples=examples,
            core_settings=core_settings,
        )

    async def plan(self, question: str, corpus_summary: str) -> RetrievalPlanResult:
        """Run the Signature and coerce its output to typed form."""
        invocation = await self._call.invoke(question=question, corpus_summary=corpus_summary)
        # Invocation.output is the structured field bag.
        output = invocation.output
        strategy = _coerce_strategy(str(output.strategy))
        tokens = tuple(str(t).strip() for t in (output.tokens or ()) if str(t).strip())
        ngrams = tuple(str(n).strip() for n in (output.ngrams or ()) if str(n).strip())
        query = str(output.query or "").strip()
        try:
            top_k = max(1, int(output.top_k))
        except (TypeError, ValueError):
            top_k = 20
        reasoning = str(output.reasoning or "")

        # Carry usage through so the caller can plumb cost / tokens.
        usage = getattr(invocation, "usage", None)
        logger.debug(
            "retrieval_planner: strategy=%s top_k=%d tokens=%d ngrams=%d query=%r reasoning=%r",
            strategy.value,
            top_k,
            len(tokens),
            len(ngrams),
            query[:60],
            reasoning[:80],
        )
        return RetrievalPlanResult(
            strategy=strategy,
            tokens=tokens,
            ngrams=ngrams,
            query=query,
            top_k=top_k,
            reasoning=reasoning,
            usage=usage,
        )


__all__ = [
    "LLMRetrievalPlanner",
    "PlanRetrieval",
    "RetrievalPlanner",
]
