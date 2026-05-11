"""FindingsAgent — exhaustive search pattern (K6).

The recall-first cousin of ``ResearchAgent``. Use when missing a
relevant hit costs more than the extra LLM cycles to look at every
candidate.

Three explicit phases:

1. **Enumerate** (no LLM). A cheap selector walks the document and
   emits every candidate sentence/paragraph that *might* be
   relevant. Selectors are pluggable — bundled options include
   keyword-match, typed-entity-match (composes with K2), and
   "every sentence" (when recall must be 1.0).
2. **Parallel filter** (cheap LLM, fan-out). The candidates are
   chunked and an LLM filter call decides which survive. Runs N
   chunks in parallel via ``asyncio.gather``. Each survivor carries
   a relevance score + one-sentence justification.
3. **Synthesize** (one LLM call). A stronger model answers the
   user's question using only the surviving findings, with inline
   ``finding_id`` references.

The pattern is the opposite trade-off from
:class:`~kaos_agents.patterns.research.ResearchAgent` which does
precision-first RAG (BM25 → top-K → answer). FindingsAgent is the
right tool when:

- Recall must be 1.0 ("did this NDA *ever* mention X?")
- The question is a diligence / audit question
- A miss costs more than the extra spend

Cost model:

- Plain RAG: 1 retrieve + 1 answer = 1 LLM call
- FindingsAgent: 0 retrieve + N filter chunks + 1 synthesis call

For a typical NDA at chunk_size=20 the filter pass is 3-10 chunks of
Haiku ≈ $0.003-0.01, plus a Sonnet synthesis call ≈ $0.01-0.05.
Acceptable for high-stakes review; prohibitive for chat.

The wrapper composes with existing patterns: put a ReflexionLoop
around a FindingsAgent for "exhaustive AND review-checked", route to
a FindingsAgent from a RouterAgent for "legal diligence questions
get the recall-first pattern, everything else stays cheap."

Example::

    from kaos_content.views import DocumentView
    from kaos_agents.patterns.findings import (
        FindingsAgent, sentences_with_token_selector,
    )

    view = DocumentView(content_document, sentence_segmenter=...)
    agent = FindingsAgent(
        selector=sentences_with_token_selector("indemnif"),
        filter_model="anthropic:claude-haiku-4-5",
        synthesis_model="anthropic:claude-sonnet-4-6",
    )
    result = await agent.run(
        question="What are all the indemnification carve-outs?",
        view=view,
    )
    print(result.answer)
    for f in result.findings:
        print(f"  {f.relevance:.2f}  {f.candidate.text}")
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_content.views import DocumentView

logger = get_logger(__name__)

_DEFAULT_FILTER_MODEL = "anthropic:claude-haiku-4-5"
_DEFAULT_SYNTHESIS_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_CHUNK_SIZE = 20
_DEFAULT_NUM_PARALLEL = 4


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FindingCandidate:
    """One enumerated candidate before LLM filtering.

    Attributes:
        finding_id: Short stable id used for cross-reference in the
            synthesis output. Generated once at enumeration time.
        text: The candidate's text — typically one sentence.
        block_ref: AST anchor (JSON pointer) for the containing
            paragraph. None when the selector produced text without
            a back-reference (e.g. literal text input).
        section_title: Containing section's heading text, when known.
        page: 1-based page number, when known.
    """

    finding_id: str
    text: str
    block_ref: str | None = None
    section_title: str | None = None
    page: int | None = None


@dataclass(frozen=True, slots=True)
class FilteredFinding:
    """A candidate that survived the LLM filter pass."""

    candidate: FindingCandidate
    relevance: float
    """0..1 — how relevant the filter considered this candidate.
    Clamped to the valid range at construction in
    :func:`_filter_chunk`."""
    reasoning: str
    """One-sentence justification from the filter LLM. Surfaces in
    audit logs and downstream replay diffs."""


@dataclass(frozen=True, slots=True)
class FindingsResult:
    """Outcome of one FindingsAgent run.

    Attributes:
        question: The original user question.
        answer: The synthesis LLM's final answer.
        findings: The surviving candidates with relevance + reasoning.
            Ordered by descending relevance.
        total_enumerated: Number of candidates Phase 1 emitted.
        total_filtered: Number of candidates that survived Phase 2.
        filter_cost_usd: Sum of all filter-call costs.
        synthesis_cost_usd: Cost of the single synthesis call.
        filter_calls: Number of filter calls actually made (= number
            of non-empty chunks).
    """

    question: str
    answer: str
    findings: tuple[FilteredFinding, ...]
    total_enumerated: int
    total_filtered: int
    filter_cost_usd: float
    synthesis_cost_usd: float
    filter_calls: int

    @property
    def total_cost_usd(self) -> float:
        return self.filter_cost_usd + self.synthesis_cost_usd

    @property
    def total_llm_calls(self) -> int:
        return self.filter_calls + 1  # +1 for synthesis


# ---------------------------------------------------------------------------
# Bundled selectors
# ---------------------------------------------------------------------------


Selector = Callable[["DocumentView", str], Iterable[FindingCandidate]]
"""Phase-1 selector contract: ``(view, question) -> Iterable[FindingCandidate]``.

Selectors are pure functions — they receive the document view and the
user's question and emit every candidate that *might* be relevant.
The filter pass (Phase 2) is what decides whether each candidate is
*actually* relevant; the selector's job is to be exhaustive on recall.

Several common selectors are bundled below. Callers can write their
own — anything callable matching the signature works.
"""


def every_sentence_selector(view: DocumentView, _question: str) -> Iterable[FindingCandidate]:
    """Phase-1 selector that emits every non-empty sentence.

    Use when recall must be 1.0 — e.g. "did this NDA *ever* mention
    X?" — and the document is small enough to afford filtering every
    sentence.
    """
    for sentence in view.sentences:
        if sentence.text and sentence.text.strip():
            yield FindingCandidate(
                finding_id=_short_id(),
                text=sentence.text,
                block_ref=sentence.paragraph_ref,
                section_title=_section_title(view, sentence.section_ref),
                page=sentence.page,
            )


def sentences_with_token_selector(token: str, *, case_sensitive: bool = False) -> Selector:
    """Selector factory: every sentence containing ``token``.

    ``token`` matches against the sentence text via Python ``in`` —
    substring, not whole-word. Pass a multi-word phrase to require
    contiguous appearance.
    """
    needle = token if case_sensitive else token.lower()

    def selector(view: DocumentView, _question: str) -> Iterable[FindingCandidate]:
        for sentence in view.sentences:
            haystack = sentence.text if case_sensitive else sentence.text.lower()
            if needle in haystack:
                yield FindingCandidate(
                    finding_id=_short_id(),
                    text=sentence.text,
                    block_ref=sentence.paragraph_ref,
                    section_title=_section_title(view, sentence.section_ref),
                    page=sentence.page,
                )

    return selector


def sentences_with_entity_selector(entity_type: str) -> Selector:
    """Selector factory: every sentence with at least one match of
    ``entity_type`` (composes with K2).

    Composes naturally with the typed-entity filters from
    ``kaos_content.views.entity_filters``: "find all sentences that
    mention money", "find all sentences with dates", etc. The Phase-2
    filter then decides which of those entity-bearing sentences
    actually answer the user's question.
    """

    def selector(view: DocumentView, _question: str) -> Iterable[FindingCandidate]:
        # Lazy import to keep kaos-content optional. Without
        # kaos-content this selector won't be invoked unless the
        # caller wires it explicitly, so the import cost only fires
        # when needed.
        from kaos_content.views.entity_filters import iter_sentences_with_entity

        for hit in iter_sentences_with_entity(view, entity_type):
            yield FindingCandidate(
                finding_id=_short_id(),
                text=hit.sentence.text,
                block_ref=hit.sentence.paragraph_ref,
                section_title=_section_title(view, hit.sentence.section_ref),
                page=hit.sentence.page,
            )

    return selector


def _short_id() -> str:
    """A short stable id for cross-referencing findings."""
    return uuid.uuid4().hex[:8]


def _section_title(view: DocumentView, section_ref: str | None) -> str | None:
    if section_ref is None:
        return None
    sec = view.section_by_ref(section_ref)
    return sec.heading_text if sec is not None else None


# ---------------------------------------------------------------------------
# FindingsAgent
# ---------------------------------------------------------------------------


class FindingsAgent:
    """Three-phase exhaustive-search wrapper.

    Not a :class:`KaosAgent` subclass — composes with any inner agent
    (or none). Pattern mirrors :class:`ReflexionLoop` and
    :class:`RouterAgent`: typed wrapper around an LLM workflow, no
    runtime / pattern enum changes required.

    Args:
        selector: Phase-1 enumeration function. Must be exhaustive on
            recall — anything filtered out by the selector will never
            be considered.
        filter_model: Model for the per-chunk filter calls. Default
            Haiku 4.5 — filtering is light work.
        synthesis_model: Model for the final answer. Default
            Sonnet 4.6 — this is the only call that touches
            answer quality.
        chunk_size: Number of candidates per filter call. Default 20.
            Lower = more filter calls (better isolation), higher =
            fewer filter calls (cheaper, but each call has more
            context). The default is a balance.
        num_parallel: ``asyncio.gather`` concurrency for filter chunks.
            Default 4 — high enough to be fast, low enough to stay
            polite with rate limits.
        relevance_threshold: Survivors must score above this in
            Phase 2 (default 0.5). Lower = more permissive, higher =
            more aggressive cull.
    """

    __slots__ = (
        "chunk_size",
        "filter_model",
        "num_parallel",
        "relevance_threshold",
        "selector",
        "synthesis_model",
    )

    def __init__(
        self,
        *,
        selector: Selector,
        filter_model: str = _DEFAULT_FILTER_MODEL,
        synthesis_model: str = _DEFAULT_SYNTHESIS_MODEL,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        num_parallel: int = _DEFAULT_NUM_PARALLEL,
        relevance_threshold: float = 0.5,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if num_parallel < 1:
            raise ValueError(f"num_parallel must be >= 1, got {num_parallel}")
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError(f"relevance_threshold must be in [0, 1], got {relevance_threshold}")
        self.selector = selector
        self.filter_model = filter_model
        self.synthesis_model = synthesis_model
        self.chunk_size = chunk_size
        self.num_parallel = num_parallel
        self.relevance_threshold = relevance_threshold

    async def run(
        self,
        question: str,
        view: DocumentView,
    ) -> FindingsResult:
        """Execute the three-phase pipeline against ``view``.

        Returns a populated :class:`FindingsResult`. Always emits at
        least the structural fields; ``answer`` may be empty when
        zero candidates survive Phase 2 (the synthesis call is then
        skipped to avoid a wasted LLM round-trip).
        """
        # Phase 1 — enumerate.
        candidates = tuple(self.selector(view, question))
        total_enumerated = len(candidates)
        logger.debug("findings.phase1: enumerated %d candidates", total_enumerated)

        if not candidates:
            return FindingsResult(
                question=question,
                answer="",
                findings=(),
                total_enumerated=0,
                total_filtered=0,
                filter_cost_usd=0.0,
                synthesis_cost_usd=0.0,
                filter_calls=0,
            )

        # Phase 2 — parallel filter.
        chunks = _chunk(candidates, self.chunk_size)
        sem = asyncio.Semaphore(self.num_parallel)

        async def _run_chunk(
            chunk: tuple[FindingCandidate, ...],
        ) -> tuple[tuple[FilteredFinding, ...], float]:
            async with sem:
                return await _filter_chunk(
                    chunk,
                    question=question,
                    model=self.filter_model,
                    threshold=self.relevance_threshold,
                )

        chunk_results = await asyncio.gather(*(_run_chunk(c) for c in chunks))
        survivors: list[FilteredFinding] = []
        filter_cost = 0.0
        for chunk_survivors, chunk_cost in chunk_results:
            survivors.extend(chunk_survivors)
            filter_cost += chunk_cost

        # Sort by relevance descending, ties broken by finding_id for
        # stable ordering across runs.
        survivors.sort(key=lambda f: (-f.relevance, f.candidate.finding_id))
        total_filtered = len(survivors)
        logger.debug(
            "findings.phase2: %d/%d candidates survived (cost=$%.4f, %d chunks)",
            total_filtered,
            total_enumerated,
            filter_cost,
            len(chunks),
        )

        # Phase 3 — synthesize.
        if not survivors:
            return FindingsResult(
                question=question,
                answer="",
                findings=(),
                total_enumerated=total_enumerated,
                total_filtered=0,
                filter_cost_usd=filter_cost,
                synthesis_cost_usd=0.0,
                filter_calls=len(chunks),
            )

        answer, synthesis_cost = await _synthesize(
            question=question,
            findings=tuple(survivors),
            model=self.synthesis_model,
        )
        logger.debug(
            "findings.phase3: synthesised answer (cost=$%.4f, %d chars)",
            synthesis_cost,
            len(answer),
        )

        return FindingsResult(
            question=question,
            answer=answer,
            findings=tuple(survivors),
            total_enumerated=total_enumerated,
            total_filtered=total_filtered,
            filter_cost_usd=filter_cost,
            synthesis_cost_usd=synthesis_cost,
            filter_calls=len(chunks),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chunk(
    items: tuple[FindingCandidate, ...], size: int
) -> tuple[tuple[FindingCandidate, ...], ...]:
    """Split ``items`` into chunks of length ``size``."""
    return tuple(items[i : i + size] for i in range(0, len(items), size))


async def _filter_chunk(
    chunk: tuple[FindingCandidate, ...],
    *,
    question: str,
    model: str,
    threshold: float,
) -> tuple[tuple[FilteredFinding, ...], float]:
    """Run one Phase-2 LLM filter call over ``chunk``.

    Returns ``(survivors, cost_usd)``. Survivors are those whose
    relevance score >= ``threshold``. The LLM is given just the
    candidate text + finding_id; metadata (block_ref, section,
    page) is preserved by the caller via the original
    :class:`FindingCandidate` lookup.
    """
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core.programs.call import Call
    from kaos_llm_core.signatures.fields import InputField, OutputField
    from kaos_llm_core.signatures.signature import Signature

    class _FilterSignature(Signature):
        """Decide which candidates from the chunk are relevant to the question."""

        question: str = InputField(description="The original user question.")
        candidates: str = InputField(
            description=(
                "JSON-style list, one per line. Each item is "
                "``<finding_id>: <text>``. Decide which are relevant. "
                "Be inclusive — anything that could plausibly inform "
                "the answer should survive."
            ),
        )
        survivors: list[dict] = OutputField(
            description=(
                "List of relevant candidates. Each item is a JSON "
                "object with keys: 'finding_id' (str — match the "
                "input id exactly), 'relevance' (float 0..1, where "
                "1.0 means 'directly answers the question' and 0.0 "
                "means 'completely irrelevant'), 'reasoning' (str — "
                "one sentence why it's relevant). Omit candidates "
                "you consider irrelevant."
            ),
        )

    rendered = "\n".join(f"{c.finding_id}: {c.text}" for c in chunk)
    call = Call(_FilterSignature, model=model)
    invocation = await call.invoke(question=question, candidates=rendered)
    result = invocation.output
    cost = float(getattr(invocation.usage, "cost_usd", 0.0) or 0.0)

    # Build a finding_id → FindingCandidate index for the chunk.
    by_id = {c.finding_id: c for c in chunk}
    survivors: list[FilteredFinding] = []
    for raw in result.survivors:
        if not isinstance(raw, dict):
            continue
        fid = str(raw.get("finding_id") or "").strip()
        cand = by_id.get(fid)
        if cand is None:
            # LLM hallucinated a finding_id; skip.
            continue
        try:
            relevance = float(raw.get("relevance", 0.0))
        except (TypeError, ValueError):
            continue
        relevance = max(0.0, min(1.0, relevance))
        if relevance < threshold:
            continue
        reasoning = str(raw.get("reasoning") or "").strip()
        survivors.append(FilteredFinding(candidate=cand, relevance=relevance, reasoning=reasoning))
    return tuple(survivors), cost


async def _synthesize(
    *,
    question: str,
    findings: tuple[FilteredFinding, ...],
    model: str,
) -> tuple[str, float]:
    """Run the Phase-3 synthesis call.

    Returns ``(answer, cost_usd)``. The synthesis LLM gets the
    surviving findings (ordered by descending relevance) plus the
    original question, and is asked to answer the question with
    inline ``[finding_id]`` references.
    """
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core.programs.call import Call
    from kaos_llm_core.signatures.fields import InputField, OutputField
    from kaos_llm_core.signatures.signature import Signature

    class _SynthesizeSignature(Signature):
        """Answer the question using only the provided findings."""

        question: str = InputField(description="The original user question.")
        findings: str = InputField(
            description=(
                "Surviving findings from the recall-first filter pass. "
                "Each line is ``[finding_id] (relevance=X.XX): text``. "
                "Use these as the *only* evidence for the answer. Do "
                "not draw on outside knowledge."
            ),
        )
        answer: str = OutputField(
            description=(
                "Concise answer to the question, citing finding_ids "
                "inline with the syntax ``[abc12345]``. When the "
                "findings don't actually answer the question, say so "
                "explicitly rather than guessing."
            ),
        )

    rendered = "\n".join(
        f"[{f.candidate.finding_id}] (relevance={f.relevance:.2f}): {f.candidate.text}"
        for f in findings
    )
    call = Call(_SynthesizeSignature, model=model)
    invocation = await call.invoke(question=question, findings=rendered)
    answer = str(invocation.output.answer)
    cost = float(getattr(invocation.usage, "cost_usd", 0.0) or 0.0)
    return answer, cost


# ---------------------------------------------------------------------------
# Public helper: parse finding_ids out of a synthesized answer
# ---------------------------------------------------------------------------


_FINDING_ID_PATTERN = re.compile(r"\[([0-9a-f]{8})\]")
"""Matches ``[abcdef12]`` — the synthesis prompt's citation syntax."""


def extract_finding_id_citations(answer: str) -> tuple[str, ...]:
    """Return the finding_ids cited inline in ``answer``.

    Convenience for downstream consumers (UI, audit logs) that want
    to render the source of each claim. Order preserved from first
    appearance in the answer; duplicates dropped.
    """
    seen: dict[str, None] = {}
    for match in _FINDING_ID_PATTERN.finditer(answer):
        seen[match.group(1)] = None
    return tuple(seen)


__all__ = [
    "FilteredFinding",
    "FindingCandidate",
    "FindingsAgent",
    "FindingsResult",
    "Selector",
    "every_sentence_selector",
    "extract_finding_id_citations",
    "sentences_with_entity_selector",
    "sentences_with_token_selector",
]
