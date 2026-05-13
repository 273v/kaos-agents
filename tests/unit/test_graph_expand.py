"""Tests for graph-aware context expansion (Track 3 chunk B4).

Confirms:

- ``expand_with_graph`` walks 1 hop from each retrieved FINDING and
  produces synthetic ``MemoryType.GRAPH`` items
- IRI lookup uses ``metadata['statement']`` first, falls back to
  ``content``
- Empty / no-finding inputs return ``[]`` without raising
- ``assemble_context`` calls the expander when ``graph_context_enabled=True``
  and injects the GRAPH items under that section
- The expander honours ``max_neighbors_per_finding`` and ``max_findings``
- The expansion never raises, even on out-of-vocab graph state
"""

from __future__ import annotations

import hashlib

import pytest

from kaos_agents.context.assemble import assemble_context
from kaos_agents.context.graph_expand import expand_with_graph
from kaos_agents.events import CitationFound, EventEmitter
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.triples import (
    CITO_CITES,
    PROV_INFORMED_BY,
    emit_from_event,
    finding_iri,
)
from kaos_agents.types.memory import MemoryItem, MemoryType, create_item


def _claim_iri(claim: str) -> str:
    fid = hashlib.sha256(claim.encode()).hexdigest()[:12]
    return finding_iri(fid)


def _populate_graph(mem: SessionMemory, claim: str, source_uri: str) -> None:
    """Fire a CitationFound event so the emitter writes triples."""
    em = EventEmitter(session_id=mem._session_id, run_id="r1")
    emit_from_event(
        em.emit(
            CitationFound,
            claim=claim,
            source_uri=source_uri,
            confidence=0.9,
            verified=True,
        ),
        mem,
    )


def _add_finding_item(mem: SessionMemory, claim: str) -> MemoryItem:
    """Add a FINDINGS item whose statement matches a graph finding."""
    item = create_item(
        section=MemoryType.FINDINGS,
        content=f"[claim] {claim}",
        metadata={"statement": claim, "verified": True},
    )
    mem.add(MemoryType.FINDINGS, item.content, metadata={"statement": claim})
    return item


# ---------------------------------------------------------------------------
# expand_with_graph — pure function
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExpandWithGraph:
    def test_no_findings_returns_empty(self) -> None:
        mem = SessionMemory("s1")
        result = expand_with_graph(mem, base_items={})
        assert result == []

    def test_empty_findings_list_returns_empty(self) -> None:
        mem = SessionMemory("s1")
        result = expand_with_graph(mem, base_items={MemoryType.FINDINGS: []})
        assert result == []

    def test_findings_without_graph_returns_empty(self) -> None:
        """If the graph has no triples, expansion is a no-op."""
        mem = SessionMemory("s1")
        _add_finding_item(mem, "Some claim")
        # Don't fire any CitationFound — graph stays empty
        result = expand_with_graph(
            mem, base_items={MemoryType.FINDINGS: list(mem.get(MemoryType.FINDINGS))}
        )
        assert result == []

    def test_finding_with_graph_neighbor_produces_item(self) -> None:
        mem = SessionMemory("s1")
        claim = "EPA filed 3 enforcement actions"
        _populate_graph(mem, claim, "https://www.epa.gov/abc")
        _add_finding_item(mem, claim)

        finds = list(mem.get(MemoryType.FINDINGS))
        result = expand_with_graph(mem, base_items={MemoryType.FINDINGS: finds})

        assert len(result) == 1
        graph_item = result[0]
        assert graph_item.section == MemoryType.GRAPH
        assert "EPA filed" in graph_item.content
        # Edge is "cito:cites <doc>" — short pred name appears in body
        assert "cites" in graph_item.content
        assert "https://www.epa.gov/abc" in graph_item.content
        # Metadata back-link to source finding
        assert graph_item.metadata["source_finding_iri"] == _claim_iri(claim)
        assert graph_item.metadata["neighbor_count"] >= 1

    def test_max_neighbors_per_finding_caps_edges(self) -> None:
        mem = SessionMemory("s1")
        claim = "Multi-cite claim"
        # Fire 5 separate citations — each adds a cito:cites edge from the
        # same finding (same claim → same finding_iri)
        em = EventEmitter(session_id="s1", run_id="r1")
        for i in range(5):
            emit_from_event(
                em.emit(
                    CitationFound,
                    claim=claim,
                    source_uri=f"https://x.com/doc{i}",
                    confidence=0.9,
                    verified=True,
                ),
                mem,
            )
        _add_finding_item(mem, claim)

        finds = list(mem.get(MemoryType.FINDINGS))
        result = expand_with_graph(
            mem,
            base_items={MemoryType.FINDINGS: finds},
            max_neighbors_per_finding=2,
        )

        assert len(result) == 1
        # Should be capped at 2 neighbors regardless of 5 cito:cites edges
        assert result[0].metadata["neighbor_count"] == 2

    def test_max_findings_caps_expansion_count(self) -> None:
        mem = SessionMemory("s1")
        em = EventEmitter(session_id="s1", run_id="r1")
        # 3 distinct findings, each with one citation
        for i in range(3):
            claim = f"Claim number {i}"
            emit_from_event(
                em.emit(
                    CitationFound,
                    claim=claim,
                    source_uri=f"https://x.com/d{i}",
                    confidence=0.9,
                    verified=True,
                ),
                mem,
            )
            _add_finding_item(mem, claim)

        finds = list(mem.get(MemoryType.FINDINGS))
        result = expand_with_graph(
            mem,
            base_items={MemoryType.FINDINGS: finds},
            max_findings=2,
        )
        assert len(result) == 2

    def test_falls_back_to_content_when_no_statement_metadata(self) -> None:
        """Older items without metadata['statement'] hash the content directly."""
        mem = SessionMemory("s1")
        claim_text = "Bare-content claim"
        _populate_graph(mem, claim_text, "https://x.com/bare")
        # Item with no metadata['statement'] — content IS the statement
        mem.add(MemoryType.FINDINGS, claim_text, metadata={})

        finds = list(mem.get(MemoryType.FINDINGS))
        result = expand_with_graph(mem, base_items={MemoryType.FINDINGS: finds})

        assert len(result) == 1
        assert "https://x.com/bare" in result[0].content


# ---------------------------------------------------------------------------
# assemble_context integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssembleContextIntegration:
    def test_assemble_with_graph_context_enabled_injects_graph_section(self) -> None:
        mem = SessionMemory("s1")
        claim = "Verified claim with graph"
        _populate_graph(mem, claim, "https://x.com/abc")
        _add_finding_item(mem, claim)

        result = assemble_context(
            mem,
            query="claim",
            sections=[MemoryType.FINDINGS],
            total_budget_tokens=10_000,
            graph_context_enabled=True,
        )

        # Both FINDINGS and GRAPH sections populated
        assert MemoryType.FINDINGS in result
        assert MemoryType.GRAPH in result
        assert len(result[MemoryType.GRAPH]) == 1

    def test_assemble_with_graph_context_disabled_skips_graph_section(self) -> None:
        mem = SessionMemory("s1")
        claim = "Verified claim"
        _populate_graph(mem, claim, "https://x.com/abc")
        _add_finding_item(mem, claim)

        result = assemble_context(
            mem,
            query="claim",
            sections=[MemoryType.FINDINGS],
            total_budget_tokens=10_000,
            graph_context_enabled=False,
        )

        # FINDINGS stays; GRAPH is NOT injected
        assert MemoryType.FINDINGS in result
        assert MemoryType.GRAPH not in result

    def test_assemble_no_findings_no_graph_section(self) -> None:
        """Even with graph_context_enabled, no findings → no graph items."""
        mem = SessionMemory("s1")
        # Add some MESSAGES but no FINDINGS
        mem.add(MemoryType.MESSAGES, "user: hello")

        result = assemble_context(
            mem,
            query="hello",
            sections=[MemoryType.MESSAGES],
            total_budget_tokens=10_000,
            graph_context_enabled=True,
        )

        assert MemoryType.GRAPH not in result

    def test_predicate_filtering_only_emits_known_predicates(self) -> None:
        """Edges with unknown predicates are skipped during expansion."""
        mem = SessionMemory("s1")
        claim = "Known-pred claim"
        _populate_graph(mem, claim, "https://x.com/known")
        _add_finding_item(mem, claim)

        # Add a custom edge with an unknown predicate from the finding
        mem.graph.add_node("https://other.example/x")
        mem.graph.add_edge(
            _claim_iri(claim),
            "https://other.example/x",
            predicate="https://example.com/unknownPredicate",
        )

        finds = list(mem.get(MemoryType.FINDINGS))
        result = expand_with_graph(mem, base_items={MemoryType.FINDINGS: finds})

        # The unknown-predicate target should NOT appear in the rendered text
        assert "unknownPredicate" not in result[0].content
        assert "other.example" not in result[0].content
        # Known cito:cites IS still in the rendering
        assert "https://x.com/known" in result[0].content


@pytest.mark.unit
class TestPredicateConstants:
    """Sanity check: graph_expand imports the canonical predicate IRIs."""

    def test_uses_canonical_iris(self) -> None:
        from kaos_agents.context.graph_expand import _EXPANSION_PREDICATES

        assert CITO_CITES in _EXPANSION_PREDICATES
        # PROV_INFORMED_BY is NOT expanded (it's internal step linkage,
        # not finding-relevant). Keep the predicate set tight.
        assert PROV_INFORMED_BY not in _EXPANSION_PREDICATES


# ---------------------------------------------------------------------------
# Settings-default behavior in BaseAgent.run() context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSettingsDefaults:
    def test_graph_context_default_is_enabled(self) -> None:
        """Verify the default flag value matches the settings field."""
        from kaos_agents.settings import KaosAgentSettings

        s = KaosAgentSettings()
        assert s.graph_context_enabled is True
