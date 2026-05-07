"""Graph-aware context augmentation — Track 3 chunk B4.

After BM25 picks the most relevant FINDINGS for a turn, walk the
session knowledge graph 1 hop from each finding to surface its
provenance: cited documents, producing tool calls, and any
``cito:supports`` links to other findings. The discovered facts are
returned as synthetic :class:`MemoryItem` instances under
:attr:`MemoryType.GRAPH`, so the existing context-assembly /
sections-to-prompt machinery can render them without further changes.

This is the bridge from the chunk B1+B2 graph stack into the agent's
prompt-time context. Without it the graph fills up but never reaches
the LLM. With it, an answer about a verified claim arrives with the
edges that justify it (which doc, which tool call, which step).

The B3 MCP tools (walk / sparql / projection) expose the graph for
agent-driven querying; B4 is the *automatic* surfacing path — every
turn enriches its findings without the agent having to ask.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

from kaos_agents.memory.triples import (
    CITO_CITES,
    CITO_SUPPORTS,
    PROV_DERIVED_FROM,
    PROV_GENERATED_BY,
    finding_iri,
)
from kaos_agents.types.memory import MemoryItem, MemoryType, estimate_tokens

if TYPE_CHECKING:
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)


# Predicates whose inbound/outbound edges we follow during expansion.
# v1 set is conservative: cito:cites (finding → doc), cito:supports
# (finding → finding), and prov:wasGeneratedBy / prov:wasDerivedFrom
# (finding → producing tool call / source). Adding more predicates is
# additive and shouldn't require schema changes.
_EXPANSION_PREDICATES: frozenset[str] = frozenset(
    {
        CITO_CITES,
        CITO_SUPPORTS,
        PROV_DERIVED_FROM,
        PROV_GENERATED_BY,
    }
)


def _finding_iri_for_item(item: MemoryItem) -> str | None:
    """Recover the graph IRI for a FINDINGS memory item.

    The chunk B2 emitter derives finding IRIs from
    ``sha256(claim_text)[:12]``. The same claim text lives on the
    FINDINGS item's ``metadata['statement']`` (set by the research
    pattern when it stores a verified Claim). When metadata is
    missing — older items, hand-built fixtures — fall back to the
    item's content; the hash will still be deterministic and the
    test suite covers both shapes.
    """
    statement = item.metadata.get("statement") or item.content
    if not statement:
        return None
    finding_id = hashlib.sha256(statement.encode()).hexdigest()[:12]
    return finding_iri(finding_id)


def _format_neighbors(
    finding_label: str,
    neighbors: list[tuple[str, str]],
) -> str:
    """Render a finding + its 1-hop neighbors as a single text block.

    Shape:
      "FINDING: {finding_label}"
      "  - cito:cites <doc-iri>"
      "  - prov:wasInformedBy <step-iri>"
      ...

    The output is plain text — it lands in a MemoryItem.content slot
    and goes through the same FIFO assembly + Signature binding as
    every other section. No markdown, no JSON, no special handling.
    """
    lines = [f"FINDING: {finding_label[:160]}"]
    for predicate, target in neighbors:
        # Predicate IRIs are long; shorten for prompt clarity.
        short_pred = predicate.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        lines.append(f"  - {short_pred}: {target}")
    return "\n".join(lines)


def expand_with_graph(
    memory: SessionMemory,
    base_items: dict[MemoryType, list[MemoryItem]],
    *,
    max_neighbors_per_finding: int = 5,
    max_findings: int = 10,
) -> list[MemoryItem]:
    """Walk 1 hop from each retrieved finding and return synthetic
    :attr:`MemoryType.GRAPH` items describing the provenance facts.

    Args:
        memory: SessionMemory whose ``.graph`` to walk.
        base_items: The output of :func:`assemble_context`, keyed by
            section. We read the FINDINGS list and produce one
            synthetic GRAPH item per finding that has graph neighbors.
        max_neighbors_per_finding: Cap on edges followed per finding.
            Higher values produce richer context at higher token cost.
        max_findings: Cap on findings to expand. Above this, only the
            first N (matching FIFO order from BM25 retrieval) are
            expanded — keeps the expansion bounded for very-broad
            BM25 hits.

    Returns:
        A list of synthetic ``MemoryItem`` instances with
        ``section=MemoryType.GRAPH``. Empty list when there are no
        findings, no graph, or no graph neighbors. The items are not
        added to ``memory`` — caller decides whether to merge them
        into context_items.
    """
    findings = base_items.get(MemoryType.FINDINGS, [])
    if not findings:
        return []
    # Lazy-trigger graph access. SessionMemory.graph is None until
    # first read; if we've never had any triples this is harmless
    # (the empty graph just has zero neighbors for everything).
    graph = memory.graph
    if graph.n_edges == 0:
        return []

    expanded: list[MemoryItem] = []
    now = time.time()

    for item in findings[:max_findings]:
        iri = _finding_iri_for_item(item)
        if iri is None or not graph.has_node(iri):
            continue

        neighbors: list[tuple[str, str]] = []
        # Outbound edges: finding → cited doc / supporting finding /
        # source (whatever the predicate). graph.edges() is global —
        # filter by source IRI.
        for edge in graph.edges():
            if edge.source != iri:
                continue
            predicate = str(edge.properties.get("predicate", ""))
            if predicate not in _EXPANSION_PREDICATES:
                continue
            neighbors.append((predicate, edge.target))
            if len(neighbors) >= max_neighbors_per_finding:
                break

        if not neighbors:
            continue

        # Pull the original label off the finding for legibility
        statement = str(item.metadata.get("statement") or item.content or "")
        rendered = _format_neighbors(statement, neighbors)

        expanded.append(
            MemoryItem(
                id=MemoryItem.make_id(rendered, MemoryType.GRAPH, now),
                section=MemoryType.GRAPH,
                content=rendered,
                token_count=estimate_tokens(rendered),
                added_at=now,
                metadata={
                    "source_finding_id": item.id,
                    "source_finding_iri": iri,
                    "neighbor_count": len(neighbors),
                },
            )
        )

    logger.debug(
        "graph_expand: %d findings → %d graph items",
        len(findings),
        len(expanded),
    )
    return expanded


__all__ = ["expand_with_graph"]
