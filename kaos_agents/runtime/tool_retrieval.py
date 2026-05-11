"""Tool retrieval — top-K BM25 selection over a KaosRuntime's tool catalog.

The agent runtime exposes 200+ tools across all KAOS modules. Bridging
every tool into every ReAct call has three problems:

1. **Context tax**: each tool's metadata adds ~150 tokens to the prompt.
   200 tools = ~30K prompt tokens just for the tool list.
2. **Selection noise**: large tool inventories degrade the model's
   ability to pick the right one — recent benchmarks show accuracy
   drops materially past ~30 tools even on frontier models.
3. **Cost**: every additional irrelevant tool in the prompt costs
   real money on every turn.

The fix is to *retrieve* the relevant tools for the current step,
not to bridge them all. This module provides :class:`ToolRetrieval`
— a BM25 index over (tool name + description + group + data types)
that ranks the catalog by query relevance.

Usage::

    retrieval = ToolRetrieval.from_runtime(runtime)
    relevant = retrieval.select("Find PFAS rules in the Federal Register", top_k=10)
    # `relevant` is a list[KaosTool] ordered by descending score.

Or via :func:`~kaos_agents.actions.tool_bridge.bridge_runtime_tools`
which accepts an optional ``relevance_query`` argument that triggers
this code path automatically.

The current band-aid (``max_tools=100`` capping the bridge at iteration
order) silently drops late-iteration tools regardless of relevance.
Tool retrieval replaces it with query-aware selection.

Cost: BM25 index construction is O(n_tools); search is O(log n_tools)
per query. No LLM involvement on the hot path — purely a Rust-backed
``kaos_nlp_core.search.Searcher`` lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_core.base.tool import KaosTool
    from kaos_core.registry.container import KaosRuntime

logger = get_logger(__name__)


# Default top_k mirrors the empirical "tool-selection accuracy drops
# past 30 tools" threshold. Override per-call when the situation demands.
DEFAULT_TOP_K: int = 15


@dataclass(frozen=True, slots=True)
class ToolHit:
    """One retrieval result — a tool plus its BM25 relevance score.

    Returned by :meth:`ToolRetrieval.search`. Use :meth:`ToolRetrieval.select`
    when you only need the tools themselves.
    """

    tool: KaosTool
    score: float


class ToolRetrieval:
    """BM25 index over a KaosRuntime's tool catalog.

    Builds a :class:`kaos_nlp_core.search.Searcher` over each tool's
    indexable text: ``name``, ``description``, and group / data-type
    metadata when available. Subsequent ``select`` / ``search`` calls
    rank the catalog by BM25 against the query.

    The index is **frozen at construction time**. If tools are added
    to the runtime after construction, rebuild via
    :meth:`from_runtime`. The runtime's tool list is small and rarely
    changes mid-run, so cache invalidation is left to the caller.

    Args:
        tools: The catalog to index. Pass a runtime's
            ``list_tool_objects()`` result.
        tokenizer: Optional ``kaos_nlp_core.tokenizer.Tokenizer``.
            Default uses the lowercase tokenizer (matches kaos-content
            and kaos-agents memory search).
        lexicon: Optional ``kaos_nlp_core.lexicon.Lexicon`` for
            synonym / hypernym expansion at search time. Default
            ``None`` (plain BM25). The platform-wide BEIR cross-domain
            finding (BM25 beat lexicon expansion by ~20% NDCG@10 on
            NFCorpus) drove "default-off" everywhere else — but tool
            retrieval is a special case where the indexed docs are
            short (tool name + description, ~50 tokens each) and
            synonym-light, so lexicon expansion typically *helps*
            here. Pass ``Lexicon()`` (a populated OpenGloss instance)
            to opt in; the caller decides which lexicon to use.

            Practical guidance: for a 200+ tool catalog where queries
            use everyday language ("looking for Federal Register
            notices") but tools use acronyms ("kaos-source-fr-search"),
            a lexicon turns near-misses into hits. The cost is purely
            search-time CPU — no LLM call.
    """

    __slots__ = ("_searcher", "_tool_by_id")

    def __init__(
        self,
        tools: list[KaosTool],
        *,
        tokenizer: Any = None,
        lexicon: Any = None,
    ) -> None:
        from kaos_nlp_core.search import Searcher

        if not tools:
            # Empty catalog — store empty maps; select() returns [].
            self._searcher: Any = None
            self._tool_by_id: dict[int, KaosTool] = {}
            return

        records: list[dict[str, Any]] = []
        self._tool_by_id = {}
        for i, tool in enumerate(tools):
            text = _tool_searchable_text(tool)
            records.append({"id": i, "text": text})
            self._tool_by_id[i] = tool

        # field_map=None — single 'text' field. tokenizer default OK.
        kwargs: dict[str, Any] = {}
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        if lexicon is not None:
            kwargs["lexicon"] = lexicon
        self._searcher = Searcher.from_documents(records, **kwargs)

    @classmethod
    def from_runtime(cls, runtime: KaosRuntime, **kwargs: Any) -> ToolRetrieval:
        """Build a retrieval index over every tool registered on the runtime.

        Accepts the same kwargs as ``__init__`` — pass ``lexicon=...``
        for synonym expansion.
        """
        tools = list(runtime.tools.list_tool_objects())
        logger.debug(
            "ToolRetrieval.from_runtime: indexing %d tools",
            len(tools),
        )
        return cls(tools, **kwargs)

    @property
    def size(self) -> int:
        """Number of tools in the index."""
        return len(self._tool_by_id)

    def search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[ToolHit]:
        """Return the top-K tools ranked by BM25 score against ``query``.

        Args:
            query: Free-form search query — typically the user's
                current message or step description.
            top_k: Max results. Empty query or empty index returns [].

        Returns:
            List of :class:`ToolHit` in descending score order. Length
            ≤ ``top_k``. Score 0 hits are filtered out (the BM25
            implementation returns them, but they signal "no token
            overlap" which is rarely useful for tool selection).
        """
        if not query or not query.strip():
            return []
        if self._searcher is None:
            return []

        raw_hits = self._searcher.search(query, top_k=top_k)
        results: list[ToolHit] = []
        for hit in raw_hits:
            tool = self._tool_by_id.get(int(hit.doc_id))
            if tool is None:
                # Defensive: index drift would cause this; skip silently.
                continue
            score = float(getattr(hit, "score", 0.0))
            if score <= 0.0:
                continue
            results.append(ToolHit(tool=tool, score=score))
        return results

    def select(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[KaosTool]:
        """Convenience: like :meth:`search` but returns bare tools, no scores.

        Use when you just want the tool list and don't need the
        relevance signal (e.g. passing into ``bridge_runtime_tools``).
        """
        return [hit.tool for hit in self.search(query, top_k=top_k)]


# ---------------------------------------------------------------------------
# Text extraction — what gets indexed for each tool
# ---------------------------------------------------------------------------


def _tool_searchable_text(tool: KaosTool) -> str:
    """Build the BM25-indexed text for one tool.

    Concatenates the most salient discoverability fields:

    - Tool name (split on hyphens so ``kaos-pdf-extract-page`` indexes
      ``kaos``, ``pdf``, ``extract``, ``page`` as separate tokens)
    - Display name
    - Description (the main signal)
    - Module name + category + capability — coarse semantic tags
    - Input parameter names — often hint at the tool's purpose
      (``term`` / ``agency`` for a search tool, ``path`` for a file tool)

    Order doesn't matter for BM25, but having the name + description
    appear first improves the human-readable index dump for debugging.
    """
    meta = tool.metadata
    pieces: list[str] = []

    # Name, with hyphens replaced by spaces so each segment is its own token.
    name = getattr(meta, "name", "") or ""
    pieces.append(name.replace("-", " "))

    display = getattr(meta, "display_name", "") or ""
    if display:
        pieces.append(display)

    description = getattr(meta, "description", "") or ""
    if description:
        pieces.append(description)

    # Coarse semantic tags
    module = getattr(meta, "module_name", "") or ""
    if module:
        pieces.append(module.replace("-", " "))
    category = getattr(meta, "category", None)
    if category is not None:
        pieces.append(str(getattr(category, "value", category)))
    capability = getattr(meta, "capability", None)
    if capability is not None:
        pieces.append(str(getattr(capability, "value", capability)))

    # Input parameter names — defensive: some tools may not have a schema
    input_schema = getattr(meta, "input_schema", None) or []
    for param in input_schema:
        param_name = getattr(param, "name", None)
        if param_name:
            pieces.append(str(param_name).replace("_", " "))

    return " ".join(pieces)


__all__ = [
    "DEFAULT_TOP_K",
    "ToolHit",
    "ToolRetrieval",
]
