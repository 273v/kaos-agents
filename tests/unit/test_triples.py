"""Tests for the KaosEvent → RDF triple emitter (Track 3 chunk B2).

Confirms:

- Namespace constants follow W3C / community-standard URIs
- IRI builders return absolute IRIs (kaos_graph requires them for
  Turtle export)
- ``emit_from_event`` translates each handled event into the right
  set of triples in ``memory.graph``
- Events outside the v1 vocabulary are no-ops (return 0)
- Emission is idempotent — re-emitting the same event doesn't
  multiply nodes
- Emission never raises — graph plumbing failures are swallowed
- Exported turtle round-trips: emitted triples survive a Turtle
  serialize → parse cycle via kaos_graph.rdf
"""

from __future__ import annotations

import pytest

from kaos_agents.events import (
    CitationFound,
    EventEmitter,
    IntentClassified,
    SpanSubject,
    TextDelta,
)
from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.triples import (
    CITO,
    CITO_CITES,
    KAOS,
    KAOS_AGENT_RUNTIME,
    KAOS_DOCUMENT,
    KAOS_FINDING,
    KAOS_STEP,
    KAOS_TOOL_CALL,
    PROV,
    PROV_AGENT,
    PROV_ASSOCIATED_WITH,
    RDF,
    RDFS,
    agent_iri,
    doc_iri,
    emit_from_event,
    finding_iri,
    step_iri,
    tool_call_iri,
)


def _emitter() -> EventEmitter:
    return EventEmitter(session_id="test-session", run_id="r1")


# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNamespaceConstants:
    def test_w3c_prov_uri(self) -> None:
        assert PROV == "http://www.w3.org/ns/prov#"

    def test_prov_agent_uri(self) -> None:
        # arxiv 2508.02866 (Aug 2025) PROV-AGENT
        assert PROV_AGENT == "https://w3id.org/prov-agent#"

    def test_cito_uri(self) -> None:
        assert CITO == "http://purl.org/spar/cito/"

    def test_rdf_rdfs_uris(self) -> None:
        assert RDF == "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        assert RDFS == "http://www.w3.org/2000/01/rdf-schema#"

    def test_kaos_namespace(self) -> None:
        assert KAOS.startswith("https://")
        assert KAOS.endswith("/")  # CURIE-friendly trailing slash

    def test_kaos_class_iris_under_namespace(self) -> None:
        for iri in (KAOS_FINDING, KAOS_TOOL_CALL, KAOS_STEP, KAOS_DOCUMENT):
            assert iri.startswith(KAOS)


# ---------------------------------------------------------------------------
# IRI builders
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIRIBuilders:
    def test_finding_iri_absolute(self) -> None:
        iri = finding_iri("f-3c8d")
        assert iri.startswith("https://")
        assert "finding/f-3c8d" in iri

    def test_tool_call_iri_absolute(self) -> None:
        iri = tool_call_iri("tc-7af2")
        assert iri.startswith("https://")
        assert "call/tc-7af2" in iri

    def test_step_iri_absolute(self) -> None:
        iri = step_iri("s001")
        assert iri.startswith("https://")
        assert "step/s001" in iri

    def test_doc_iri_passthrough_for_http(self) -> None:
        # Real URLs pass through unchanged
        assert doc_iri("https://example.com/doc.pdf") == "https://example.com/doc.pdf"

    def test_doc_iri_passthrough_for_urn(self) -> None:
        assert doc_iri("urn:isbn:1234567890") == "urn:isbn:1234567890"

    def test_doc_iri_wraps_bare_string(self) -> None:
        iri = doc_iri("contract-2024.pdf")
        assert iri.startswith(KAOS)
        assert "doc/contract-2024.pdf" in iri

    def test_doc_iri_percent_encodes_spaces(self) -> None:
        # 0.1.27: source_uri is now the user-facing filename, which can
        # contain spaces ("MNDA - Acme.docx"). The wrapped IRI must
        # percent-encode them or kaos_graph rejects the subject
        # ("Invalid IRI code point ' '"). The "/" "#" separators of the
        # composite ``filename#/body/...`` provenance must survive.
        iri = doc_iri("MNDA - Acme.docx#/body/7/children/8/children/0")
        assert " " not in iri
        assert "MNDA%20-%20Acme.docx" in iri
        assert "#/body/7/children/8/children/0" in iri
        # Sanity: it is a valid absolute IRI kaos_graph will accept.
        assert iri.startswith("https://")

    def test_agent_iri_canonical(self) -> None:
        # v1 uses a single canonical agent IRI for runtime attribution
        assert agent_iri() == KAOS_AGENT_RUNTIME


# ---------------------------------------------------------------------------
# emit_from_event — happy-path per event type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmitToolCall:
    def test_tool_call_start_emits_typed_node(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        event = emitter.span_start(
            SpanSubject.TOOL_CALL,
            attributes={"tool_name": "fr-search", "call_id": "tc-1"},
        )

        n = emit_from_event(event, mem)

        assert n >= 3  # type, label, startedAt at minimum
        assert mem.graph.has_node(tool_call_iri("tc-1"))
        assert mem.graph.has_node(KAOS_AGENT_RUNTIME)
        # prov:wasAssociatedWith edge from call → agent
        assert mem.graph.has_edge(tool_call_iri("tc-1"), KAOS_AGENT_RUNTIME)

    def test_tool_call_start_with_step_emits_informed_by(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        event = emitter.span_start(
            SpanSubject.TOOL_CALL,
            attributes={
                "tool_name": "fr-search",
                "call_id": "tc-1",
                "step_id": "s001",
            },
        )

        emit_from_event(event, mem)

        assert mem.graph.has_node(step_iri("s001"))
        # prov:wasInformedBy edge: call → step
        assert mem.graph.has_edge(tool_call_iri("tc-1"), step_iri("s001"))

    def test_tool_call_complete_marks_end_time(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        start = emitter.span_start(
            SpanSubject.TOOL_CALL,
            attributes={"tool_name": "fr-search", "call_id": "tc-2"},
        )
        emit_from_event(start, mem)

        complete = emitter.span_complete(
            SpanSubject.TOOL_CALL,
            span_id=start.span_id,
            duration_ms=120.5,
            attributes={
                "tool_name": "fr-search",
                "call_id": "tc-2",
                "is_error": False,
            },
        )
        n = emit_from_event(complete, mem)
        assert n >= 1

        # COMPLETE is in-place property update on the existing call
        # node — no second call node is created
        call_nodes = [nid for nid in mem.graph.node_ids() if nid.startswith(f"{KAOS}call/")]
        assert call_nodes == [tool_call_iri("tc-2")]

    def test_tool_call_complete_without_call_id_is_noop(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        complete = emitter.span_complete(
            SpanSubject.TOOL_CALL,
            span_id="span-X",
            duration_ms=10.0,
            attributes={},  # no call_id
        )
        n = emit_from_event(complete, mem)
        assert n == 0
        assert mem.graph.n_edges == 0


@pytest.mark.unit
class TestEmitStep:
    def test_step_start_emits_typed_node(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        event = emitter.span_start(
            SpanSubject.STEP,
            attributes={"step_id": "s042", "description": "Search EPA filings"},
        )

        n = emit_from_event(event, mem)

        assert n >= 3
        assert mem.graph.has_node(step_iri("s042"))

    def test_step_complete_records_duration(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        start = emitter.span_start(
            SpanSubject.STEP, attributes={"step_id": "s042", "description": "Search"}
        )
        emit_from_event(start, mem)

        complete = emitter.span_complete(
            SpanSubject.STEP,
            span_id=start.span_id,
            duration_ms=850.0,
            attributes={"step_id": "s042", "is_error": False},
        )
        n = emit_from_event(complete, mem)

        assert n >= 1
        # No duplicate step node
        step_nodes = [nid for nid in mem.graph.node_ids() if nid.startswith(f"{KAOS}step/")]
        assert step_nodes == [step_iri("s042")]


@pytest.mark.unit
class TestEmitCitation:
    def test_citation_emits_finding_doc_cites(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        event = emitter.emit(
            CitationFound,
            claim="EPA filed 3 enforcement actions against ABC Corp.",
            source_uri="https://www.epa.gov/enforcement/abc",
            confidence=0.92,
            verified=True,
        )

        n = emit_from_event(event, mem)

        assert n >= 3
        # finding node exists
        finding_nodes = [nid for nid in mem.graph.node_ids() if nid.startswith(f"{KAOS}finding/")]
        assert len(finding_nodes) == 1

        # doc node exists (passed through as-is for an https: URL)
        assert mem.graph.has_node("https://www.epa.gov/enforcement/abc")

        # cito:cites edge: finding → doc
        assert mem.graph.has_edge(finding_nodes[0], "https://www.epa.gov/enforcement/abc")

    def test_citation_id_is_deterministic(self) -> None:
        """Same claim text → same finding IRI (sha256-derived)."""
        emitter = _emitter()
        ev1 = emitter.emit(
            CitationFound,
            claim="The sky is blue.",
            source_uri="https://x.com/a",
            confidence=1.0,
            verified=True,
        )
        ev2 = emitter.emit(
            CitationFound,
            claim="The sky is blue.",
            source_uri="https://x.com/a",
            confidence=1.0,
            verified=True,
        )

        mem = SessionMemory("s1")
        emit_from_event(ev1, mem)
        emit_from_event(ev2, mem)

        # Same content → same finding IRI → no duplicate node
        finding_nodes = [nid for nid in mem.graph.node_ids() if nid.startswith(f"{KAOS}finding/")]
        assert len(finding_nodes) == 1


# ---------------------------------------------------------------------------
# Out-of-vocab events + safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmitNoOps:
    def test_text_delta_is_noop(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        event = emitter.emit(TextDelta, content="hello", role="assistant")

        n = emit_from_event(event, mem)

        assert n == 0
        # Graph was lazy-init'd by .graph access in emit, but stays empty
        assert mem.graph.n_edges == 0

    def test_intent_classified_is_noop_in_v1(self) -> None:
        """IntentClassified is reserved for a future emission pass."""
        mem = SessionMemory("s1")
        emitter = _emitter()
        event = emitter.emit(IntentClassified, intent="respond", confidence=0.9, reasoning="")

        n = emit_from_event(event, mem)

        assert n == 0


@pytest.mark.unit
class TestEmitIdempotency:
    def test_re_emit_does_not_multiply_nodes(self) -> None:
        mem = SessionMemory("s1")
        emitter = _emitter()
        event = emitter.span_start(
            SpanSubject.TOOL_CALL,
            attributes={"tool_name": "fr-search", "call_id": "tc-X"},
        )

        emit_from_event(event, mem)
        first_n_nodes = mem.graph.n_nodes
        emit_from_event(event, mem)
        emit_from_event(event, mem)

        # Same event re-emitted: node count unchanged
        assert mem.graph.n_nodes == first_n_nodes


@pytest.mark.unit
class TestEmitSafety:
    def test_emit_never_raises_on_corrupt_event(self) -> None:
        """Events with weird/missing attrs should be no-op, not raise."""
        mem = SessionMemory("s1")
        emitter = _emitter()
        # A tool-call START with empty attrs (no tool_name, no call_id)
        event = emitter.span_start(SpanSubject.TOOL_CALL, attributes={})

        # Must not raise
        n = emit_from_event(event, mem)
        assert n == 0


# ---------------------------------------------------------------------------
# Round-trip: emit → Turtle → re-parse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseAgentEmissionWiring:
    """Confirms triples are written to memory.graph during agent.run().

    The dispatch loop in BaseAgent.run() calls emit_from_event for every
    event it yields (Track 3 chunk B2 wiring). This integration test
    drives a full turn with a stubbed dispatch that yields a tool-call
    pair + a citation, then verifies the graph has the expected nodes.
    """

    @pytest.mark.asyncio
    async def test_run_emits_triples_for_dispatched_events(self) -> None:
        from unittest.mock import AsyncMock, patch

        from kaos_core.types.enums import StorageBackend
        from kaos_core.vfs.core import VirtualFileSystem
        from kaos_core.vfs.models import VFSConfig

        from kaos_agents.events import KaosEvent
        from kaos_agents.runtime.agent import BaseAgent
        from kaos_agents.types import IntentResult, IntentType

        vfs = VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))
        agent = BaseAgent(vfs)

        # Stub _dispatch_streaming to yield a tool-call lifecycle + a
        # citation. We can't reach into the run loop's emitter to
        # construct events with matching span_ids, so we yield raw Span
        # / CitationFound instances directly — the emit_from_event
        # function only reads attributes, not internal emitter state.
        async def stub_dispatch(intent, message, memory, context_items, emitter):
            yield emitter.span_start(
                SpanSubject.TOOL_CALL,
                attributes={
                    "tool_name": "fr-search",
                    "call_id": "tc-runtime-1",
                    "step_id": "s-001",
                },
            )
            yield emitter.span_complete(
                SpanSubject.TOOL_CALL,
                span_id="span-X",
                duration_ms=42.0,
                attributes={
                    "tool_name": "fr-search",
                    "call_id": "tc-runtime-1",
                    "is_error": False,
                },
            )
            yield emitter.emit(
                CitationFound,
                claim="Runtime claim",
                source_uri="https://example.com/runtime",
                confidence=0.9,
                verified=True,
            )

        mock_intent = IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="test")
        with (
            patch.object(agent, "_classify", new_callable=AsyncMock, return_value=mock_intent),
            patch.object(agent, "_dispatch_streaming", side_effect=stub_dispatch),
        ):
            events: list[KaosEvent] = []
            async for event in agent.run("Hello", "session-emit"):
                events.append(event)

        # The session memory was created internally; reach into the
        # store to load it and check the graph
        memory = await agent._store.load_or_create("session-emit")

        # Tool call node was emitted
        assert memory.graph.has_node(tool_call_iri("tc-runtime-1"))
        # Citation finding was emitted
        finding_nodes = [
            nid for nid in memory.graph.node_ids() if nid.startswith(f"{KAOS}finding/")
        ]
        assert len(finding_nodes) >= 1
        # Doc node was emitted (https URL passes through verbatim)
        assert memory.graph.has_node("https://example.com/runtime")


@pytest.mark.unit
class TestTurtleExport:
    def test_emitted_triples_serialize_to_turtle(self) -> None:
        """Verify emitted triples appear in the Turtle dump.

        kaos_graph.rdf only ships ``to_turtle`` today (no ``from_turtle``
        yet — see kaos-graph backlog). The contract this test enforces is
        that everything we emit is *expressible* as Turtle: subjects /
        objects / predicates all appear as absolute IRIs in the export.
        Re-parse round-trip is deferred until kaos-graph adds a Turtle
        parser.
        """
        from kaos_graph.rdf import to_turtle

        mem = SessionMemory("s1")
        emitter = _emitter()
        # Build a small graph: tool-call + citation
        emit_from_event(
            emitter.span_start(
                SpanSubject.TOOL_CALL,
                attributes={"tool_name": "fr-search", "call_id": "tc-Z"},
            ),
            mem,
        )
        emit_from_event(
            emitter.emit(
                CitationFound,
                claim="Sample claim",
                source_uri="https://example.com/d1",
                confidence=0.8,
                verified=True,
            ),
            mem,
        )

        # Export to Turtle
        ttl = to_turtle(mem.graph)

        # The Turtle string must contain our IRIs and predicates
        assert tool_call_iri("tc-Z") in ttl
        assert "https://example.com/d1" in ttl
        # cito:cites + prov:wasAssociatedWith predicate IRIs are in the dump
        assert CITO_CITES in ttl
        assert PROV_ASSOCIATED_WITH in ttl

    def test_friendly_filename_source_uri_serializes_to_turtle(self) -> None:
        """0.1.27: ``source_uri`` is now the user-facing filename, which
        can contain spaces ("MNDA - Acme.docx#/body/7"). The document node
        IRI must be percent-encoded so ``to_turtle`` does not raise
        "Invalid IRI code point" — the 2026-05-30 graph-persistence crash
        that wedged the turn after the attribution fix surfaced friendly
        names. This pins the full emit -> serialize path, not just the
        ``doc_iri`` unit.
        """
        from kaos_graph.rdf import to_turtle

        mem = SessionMemory("s1")
        emitter = _emitter()
        composite = "MNDA - Acme.docx#/body/7/children/0"
        emit_from_event(
            emitter.emit(
                CitationFound,
                claim="TERM. This Agreement shall be effective from the Effective Date.",
                source_uri=composite,
                confidence=0.9,
                verified=True,
            ),
            mem,
        )

        # Must not raise ValueError("Invalid IRI code point ' '").
        ttl = to_turtle(mem.graph)

        # The doc node serialized as a valid, percent-encoded IRI.
        assert doc_iri(composite) in ttl
        assert "MNDA%20-%20Acme.docx" in ttl
