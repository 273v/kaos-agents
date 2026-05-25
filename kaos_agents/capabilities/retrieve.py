"""Unified ``retrieve`` capability — federates SEARCH / READ tools.

Plan §1 of
``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``:

> One tool with ``(query, source_filter, mode={discover|read|grep},
> max_n)``. Internally federates over kaos-web-search +
> kaos-source-*-search + kaos-content-search-document. Returns ranked
> excerpts as a uniform ``RetrievalHit(text, provenance, score,
> source_id)``.

This module ships the **agent-side facade**. It does NOT modify the
per-module tool implementations (``kaos-web``, ``kaos-source``,
``kaos-content``); those keep their existing tool surface. The
federator walks the :class:`CapabilityRegistry`, finds every
capability of the right :class:`CapabilityKind`, looks up the backing
``KaosTool`` instances on the active :class:`KaosRuntime`, invokes them
dynamically with a best-effort args contract, and aggregates the
results into a uniform :class:`RetrievalHit` value.

Step 2 of the lateral redesign — kaos-agents only. Future steps will
register richer per-source ``Capability`` declarations (jurisdiction
tags, body adapters, etc.) so the federator's source-filter contract
gets sharper. For now the contract is intentionally lenient:

* Tools that don't accept ``query`` get skipped silently.
* Tools whose ``execute`` raises get skipped silently (with a debug log).
* Empty / unstructured results contribute zero hits.

This keeps the facade additive — adding a new SEARCH tool to a runtime
just means it joins the federated pool, no agent-side code change
needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from kaos_core.logging import get_logger
from kaos_core.types import Capability, CapabilityKind, CostClass

from kaos_agents.registry.capability_registry import (
    default_capability_registry,
)

if TYPE_CHECKING:
    from kaos_core import KaosRuntime
    from kaos_core.base.tool import KaosTool
    from kaos_core.types.results import ToolResult

    from kaos_agents.registry.capability_registry import CapabilityRegistry


_logger = get_logger(__name__)


RETRIEVE_CAPABILITY_NAME = "retrieve"
"""Canonical capability name for the unified federated retriever."""


RetrieveMode = Literal["discover", "read", "grep"]
"""Mode discriminator the planner picks per call.

* ``discover`` — coarse search; returns pointers (titles + URLs +
  snippets) across every SEARCH-kind capability. Cheap. Default.
* ``read`` — fetch a body given a pointer; returns the body's text via
  every READ-kind capability whose tool name looks like a fetcher
  (``fetch``, ``get-page``, ``get-markdown``).
* ``grep`` — search within an already-ingested document via tools
  whose name contains ``search-document`` (``kaos-content-search-document``
  is the canonical implementer).
"""


# Per-mode tool-name substrings used to pick which backing tools the
# federator actually invokes. We deliberately keep this as a simple
# substring filter — the substring vocabulary mirrors the kaos-* naming
# convention (``*-search``, ``*-fetch-*``, ``*-get-markdown``,
# ``*-search-document``) so adding a new tool that follows the
# convention auto-joins the right mode.
_MODE_NAME_HINTS: dict[RetrieveMode, tuple[str, ...]] = {
    "discover": ("search",),
    "read": ("fetch", "get-page", "get-markdown"),
    "grep": ("search-document",),
}


# Per-mode :class:`CapabilityKind` filter. ``grep`` is a subset of
# SEARCH (search within a stored doc), but reusing SEARCH lets the
# kaos-content-search-document tool participate without needing a new
# kind.
_MODE_KINDS: dict[RetrieveMode, tuple[CapabilityKind, ...]] = {
    "discover": (CapabilityKind.SEARCH,),
    "read": (CapabilityKind.READ,),
    "grep": (CapabilityKind.SEARCH,),
}


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One uniform retrieval result across every federated backend.

    Fields:
        text: The user-visible payload — a snippet, a title-and-URL
            line, a paragraph from a fetched body, or a search-document
            excerpt. The planner reads this directly; no per-source
            decoding required.
        provenance: A stable, human-readable origin string. For a web
            search hit this is the URL; for a federal-register search
            hit it's the document identifier; for a search-document
            hit it's the ``artifact_id#block_ref`` pair. Empty string
            allowed when the tool returned no provenance.
        score: Relevance score in the source's own scale (BM25,
            search-engine rank inverse, etc.). Higher is better.
            Federation does not re-normalise; callers that want
            cross-source ranking can normalise themselves.
        source_id: The capability or tool ``name`` the hit came from —
            ``"kaos-web-search"``, ``"kaos-source-fr-search"``,
            ``"kaos-content-search-document"``, etc. Lets the planner
            attribute provenance back to a specific federated source.
    """

    text: str
    provenance: str
    score: float
    source_id: str


def _matches_source_filter(capability: Capability, source_filter: tuple[str, ...]) -> bool:
    """``True`` when the capability's tags satisfy ``source_filter``.

    Empty ``source_filter`` means "no filter — every capability passes".
    Otherwise the capability must declare at least one matching tag.
    """
    if not source_filter:
        return True
    tags = set(capability.tags)
    return any(tag in tags for tag in source_filter)


def _matches_mode(tool_name: str, mode: RetrieveMode) -> bool:
    """``True`` when ``tool_name`` looks like a backing tool for ``mode``."""
    hints = _MODE_NAME_HINTS[mode]
    return any(hint in tool_name for hint in hints)


def _coerce_hits(result: ToolResult, source_id: str, max_n: int) -> list[RetrievalHit]:
    """Best-effort conversion from a heterogeneous ToolResult to hits.

    Real kaos-web / kaos-source / kaos-content tools return different
    shapes today (markdown text, structured dicts, etc.) — Step 2's
    contract is intentionally lenient so we can land the facade
    without first migrating every tool to a uniform return type
    (that's the Step 2.2 "ContentDocument everywhere" follow-up).

    Strategy:

    1. Look for ``structuredContent["results"]`` (list of dicts) — the
       documented kaos-* search shape. Map common field names
       (``title`` / ``url`` / ``snippet`` / ``score``) to RetrievalHit.
    2. Fall back to ``result.text`` (first :class:`TextContent`) as a
       single hit with score 0.0.
    3. Anything else → empty list (caller logs).
    """
    hits: list[RetrievalHit] = []
    structured = result.structuredContent or {}
    raw_results = structured.get("results")
    if isinstance(raw_results, list):
        for item in raw_results[:max_n]:
            if not isinstance(item, dict):
                continue
            text = _extract_text(item)
            if not text:
                continue
            hits.append(
                RetrievalHit(
                    text=text,
                    provenance=_extract_provenance(item),
                    score=_extract_score(item),
                    source_id=source_id,
                )
            )
        if hits:
            return hits

    text = result.text
    if text:
        hits.append(
            RetrievalHit(
                text=text,
                provenance="",
                score=0.0,
                source_id=source_id,
            )
        )
    return hits[:max_n]


def _extract_text(item: dict[str, Any]) -> str:
    """Pick the most useful display string from a result dict."""
    for key in ("snippet", "text", "title", "preview", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_provenance(item: dict[str, Any]) -> str:
    """Pick a stable origin string from a result dict."""
    for key in ("url", "uri", "provenance", "block_ref", "id", "source"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_score(item: dict[str, Any]) -> float:
    """Pick a numeric score; default to 0.0 when unscored."""
    value = item.get("score")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _build_tool_args(tool: KaosTool, query: str, max_n: int) -> dict[str, Any]:
    """Map (query, max_n) onto the tool's declared input schema.

    Tools accept different parameter names for "how many results"
    (``max_results``, ``top_k``, ``limit``) — this builder reads the
    tool's metadata input schema and fills in the values it knows about.
    Unknown / mismatched fields are skipped so we don't fail
    :meth:`KaosTool.validate_inputs`.
    """
    args: dict[str, Any] = {}
    declared = {p.name for p in tool.metadata.input_schema}
    if "query" in declared:
        args["query"] = query
    for max_key in ("max_results", "top_k", "limit", "n", "max_n"):
        if max_key in declared:
            args[max_key] = max_n
            break
    return args


async def _invoke_tool(tool: KaosTool, query: str, max_n: int) -> ToolResult | None:
    """Invoke ``tool.execute`` defensively; ``None`` on any failure.

    The federated-retrieval contract intentionally swallows per-tool
    failures so a single broken backend can't sink the whole retrieve
    call (Plan §1 — federated SEARCH / READ over many tools, return
    whatever survived). Pre-Theme-A this was a *silent* skip: only a
    debug log fired, so a misconfigured tool that crashed on every
    call looked indistinguishable from "tool returned no hits" in
    every downstream consumer (UI, audit log, OTel traces).

    Theme A (2026-05-25): in addition to the existing debug log, emit
    a typed :class:`~kaos_agents.events.lifecycle.RunError` via
    :func:`~kaos_agents.events.active_emitter` when an emitter is in
    scope. Return contract is unchanged (``None`` on failure), so
    every existing caller continues to work — the only behavioral
    delta is that the skip is now visible in the event stream.
    """
    from kaos_agents.events import RunError, active_emitter

    args = _build_tool_args(tool, query, max_n)
    if "query" not in args:
        # Tool doesn't accept ``query`` — not a real retriever.
        _logger.debug(
            "capabilities.retrieve.skip_tool_no_query",
            extra={"tool": tool.metadata.name},
        )
        return None
    try:
        return await tool.execute(args)
    except Exception as exc:
        _logger.debug(
            "capabilities.retrieve.tool_error",
            extra={"tool": tool.metadata.name, "error": str(exc)},
        )
        # Theme A: make the silent skip observable. Gracefully degrades
        # to a no-op when no emitter is in scope (most non-agent
        # callers — e.g. direct unit tests of ``retrieve()`` — don't
        # open a ``use_emitter`` scope).
        emitter = active_emitter()
        if emitter is not None:
            emitter.emit(
                RunError,
                error_type="tool_execution_failure",
                message=(
                    f"Federated retrieve: tool {tool.metadata.name!r} raised "
                    f"{type(exc).__name__}: {exc}. Skipping this tool's "
                    "contribution to the federated result set."
                ),
                recovery_hint=(
                    "Check the tool's registration in this runtime and the "
                    "tool-specific logs for the underlying error. Other "
                    "retrieval tools continue to contribute hits — the "
                    "federated contract is partial-success."
                ),
            )
        return None


async def retrieve(
    query: str,
    runtime: KaosRuntime,
    source_filter: tuple[str, ...] = (),
    mode: RetrieveMode = "discover",
    max_n: int = 5,
    *,
    registry: CapabilityRegistry | None = None,
) -> tuple[RetrievalHit, ...]:
    """Federate over every SEARCH / READ capability and return hits.

    The facade reads the :class:`CapabilityRegistry`, picks capabilities
    whose :class:`CapabilityKind` matches the requested ``mode`` and
    whose tags satisfy ``source_filter``, then invokes each capability's
    backing tools on the active ``runtime``.

    Args:
        query: Natural-language query passed to every backing tool.
        runtime: Active :class:`KaosRuntime`; the tool registry on it
            is the source of truth for which backing tools actually
            exist in this deployment.
        source_filter: Optional capability-tag filter. Empty tuple
            means "every capability passes". A non-empty tuple narrows
            to capabilities that declare at least one matching tag
            (e.g. ``("legal-source",)`` keeps just the legal-domain
            search tools).
        mode: ``discover`` (default), ``read``, or ``grep``. Determines
            both the :class:`CapabilityKind` filter and the per-tool
            name hint used to pick which backing tools participate.
        max_n: Soft per-tool cap on results. The federated total can
            exceed ``max_n`` when multiple tools contribute hits; the
            caller does the final ranking.
        registry: Optional :class:`CapabilityRegistry`; defaults to the
            module-level :data:`default_capability_registry`. Lets
            tests inject a fresh registry without mutating the
            singleton.

    Returns:
        A tuple of :class:`RetrievalHit`. May be empty when no
        backing tools match (empty registry, source_filter excludes
        everything, no tools support the mode, every invocation
        errored).

    Notes:
        * The facade is **read-only**. It never mutates the registry
          or the runtime.
        * Errors are swallowed; the facade prefers "fewer hits" over
          "raise to the caller". Inspect debug logs under the
          ``kaos.kaos_agents.capabilities.retrieve`` logger for
          per-tool diagnostics.
        * Ranking is **per-source order-preserving**. Callers that
          want a single normalised ranking should re-rank the returned
          tuple (e.g. via :class:`ToolFitnessSignature` or a dedicated
          rerank capability).
    """
    if registry is None:
        registry = default_capability_registry
    kinds = _MODE_KINDS[mode]

    # 1. Pick the capability set (by kind + source_filter).
    matched_capabilities: list[Capability] = []
    seen_capability_names: set[str] = set()
    for kind in kinds:
        for capability in registry.list_by_kind(kind):
            if capability.name in seen_capability_names:
                continue
            if not _matches_source_filter(capability, source_filter):
                continue
            matched_capabilities.append(capability)
            seen_capability_names.add(capability.name)

    if not matched_capabilities:
        _logger.debug(
            "capabilities.retrieve.no_capabilities",
            extra={
                "mode": mode,
                "source_filter": list(source_filter),
                "kinds": [k.value for k in kinds],
            },
        )
        return ()

    # 2. Resolve to backing tool instances, filtered by mode name hints,
    # de-duped across capabilities.
    seen_tool_names: set[str] = set()
    invocations: list[tuple[str, KaosTool]] = []
    for capability in matched_capabilities:
        for tool_name in capability.backing_tool_names:
            if tool_name in seen_tool_names:
                continue
            if not _matches_mode(tool_name, mode):
                continue
            tool = runtime.tools.get_tool(tool_name)
            if tool is None:
                continue
            seen_tool_names.add(tool_name)
            invocations.append((tool_name, tool))

    if not invocations:
        _logger.debug(
            "capabilities.retrieve.no_invocations",
            extra={
                "mode": mode,
                "candidates": [c.name for c in matched_capabilities],
            },
        )
        return ()

    # 3. Invoke each tool; aggregate hits.
    aggregated: list[RetrievalHit] = []
    for tool_name, tool in invocations:
        result = await _invoke_tool(tool, query, max_n)
        if result is None:
            continue
        aggregated.extend(_coerce_hits(result, tool_name, max_n))

    return tuple(aggregated)


# ---------------------------------------------------------------------------
# Register the sample ``retrieve`` capability at import time.
# ---------------------------------------------------------------------------
#
# Step 2 ships the capability *declaration* — the federator above is
# the implementation. ``backing_tool_names`` is intentionally left
# empty because the facade resolves tools dynamically off the active
# :class:`KaosRuntime` (the same tool may or may not be registered in
# every deployment). The registry entry exists so:
#
#   * The M1 fitness ranker / planner sees "retrieve" as a top-level
#     capability name to pick.
#   * The capability catalogue rendering (T4-4
#     :func:`render_tool_catalog`) shows a unified retrieval surface
#     instead of three per-module families.
#
# The declaration is idempotent: re-importing this module does NOT
# raise ``RegistryError`` because we check equality before
# re-registering.

_RETRIEVE_CAPABILITY = Capability(
    name=RETRIEVE_CAPABILITY_NAME,
    kind=CapabilityKind.SEARCH,
    description=(
        "Unified federated retriever. Picks the right backing search / "
        "fetch / search-document tool for the requested mode and "
        "source-filter, invokes it, and returns uniform RetrievalHit "
        "tuples. Use this in place of choosing between kaos-web-search, "
        "kaos-source-fr-search, kaos-source-edgar-search, "
        "kaos-content-search-document and friends. "
        "Modes: 'discover' (default, returns pointers / snippets), "
        "'read' (fetches bodies given a pointer), "
        "'grep' (searches within an already-ingested document). "
        "source_filter is a tuple of capability tags ('legal-source', "
        "'web', 'sec-filings', ...); empty means 'any source'."
    ),
    inputs=("query", "source_filter", "mode", "max_n"),
    outputs=("RetrievalHit[]",),
    cost_class=CostClass.CHEAP,
    backing_tool_names=(),
)


def _register_default_retrieve_capability() -> None:
    """Idempotent module-import-time registration.

    Guarded so the default registry never grows duplicate or
    conflicting entries when this module is re-imported (e.g. by
    pytest collection or test isolation).
    """
    existing = default_capability_registry.get(RETRIEVE_CAPABILITY_NAME)
    if existing == _RETRIEVE_CAPABILITY:
        return
    default_capability_registry.register(_RETRIEVE_CAPABILITY, force=True)


_register_default_retrieve_capability()


__all__ = [
    "RETRIEVE_CAPABILITY_NAME",
    "RetrievalHit",
    "RetrieveMode",
    "retrieve",
]
