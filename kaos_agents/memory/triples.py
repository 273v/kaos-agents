"""KaosEvent → RDF triple emitter for the session knowledge graph.

Track 3 chunk B2 — translates :class:`KaosEvent` instances into typed
RDF triples written to :attr:`SessionMemory.graph`. The vocabulary is
W3C-standard:

- **RDF / RDFS** (typing + labels)
- **PROV-O** (W3C Recommendation) — provenance core
- **PROV-AGENT** (arxiv 2508.02866, Aug 2025) — LLM agent extensions
- **CiTO** — Citation Typing Ontology
- **kaos:** — our domain namespace (kaos:Finding, kaos:ToolCall, ...)

Each event the emitter handles produces 1-N triples in the session
graph. The graph is stored as a single :class:`kaos_graph.Graph`
instance per :class:`SessionMemory`; named-graph layering (the
TrustGraph 3-layer pattern) is deferred — kaos-graph's current
Turtle export is triples-only.

v1 emission set (~12 predicates):
  rdf:type, rdfs:label,
  prov:startedAtTime, prov:endedAtTime, prov:wasAssociatedWith,
  prov:wasGeneratedBy, prov:wasInformedBy, prov:used,
  prov:wasDerivedFrom,
  cito:cites, cito:supports,
  + kaos: classes (kaos:Finding rdfs:subClassOf prov:Entity, etc.)

Future phases add SKOS for entities (cito:mentions / skos:Concept /
skos:broader) and OWL sameAs for entity resolution.

Integration: chunk B1 added :attr:`SessionMemory.graph`; this module
is the layer that *populates* it. :func:`emit_from_event` is called
from :meth:`BaseAgent.run` (chunk B2 wiring) on every yielded event.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_agents.base.event import KaosEvent
    from kaos_agents.memory.session import SessionMemory

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Namespace constants (W3C standards + kaos:)
# ---------------------------------------------------------------------------

KAOS = "https://kaos.273ventures.com/ns/"
PROV = "http://www.w3.org/ns/prov#"
PROV_AGENT = "https://w3id.org/prov-agent#"
CITO = "http://purl.org/spar/cito/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"


# Class IRIs
KAOS_FINDING = f"{KAOS}Finding"
KAOS_TOOL_CALL = f"{KAOS}ToolCall"
KAOS_STEP = f"{KAOS}Step"
KAOS_DOCUMENT = f"{KAOS}Document"
KAOS_AGENT_RUNTIME = f"{KAOS}agent-runtime"  # canonical agent IRI for v1

# Property IRIs
RDF_TYPE = f"{RDF}type"
RDFS_LABEL = f"{RDFS}label"
PROV_STARTED_AT = f"{PROV}startedAtTime"
PROV_ENDED_AT = f"{PROV}endedAtTime"
PROV_GENERATED_BY = f"{PROV}wasGeneratedBy"
PROV_INFORMED_BY = f"{PROV}wasInformedBy"
PROV_USED = f"{PROV}used"
PROV_ASSOCIATED_WITH = f"{PROV}wasAssociatedWith"
PROV_DERIVED_FROM = f"{PROV}wasDerivedFrom"
CITO_CITES = f"{CITO}cites"
CITO_SUPPORTS = f"{CITO}supports"


# ---------------------------------------------------------------------------
# IRI builders
# ---------------------------------------------------------------------------

# kaos_graph.Graph requires absolute IRIs as node IDs (the Rust RDF
# parser refuses bare strings like "tc-1"). Every emitted triple's
# subject + object goes through one of these helpers.


def finding_iri(finding_id: str) -> str:
    """IRI for a :class:`kaos:Finding` node — the agent's verified claim.

    ``finding_id`` is opaque (the agent assigns it). Common shape is a
    short content-derived hash.
    """
    return f"{KAOS}finding/{finding_id}"


def tool_call_iri(call_id: str) -> str:
    """IRI for a :class:`kaos:ToolCall` node — one tool invocation.

    ``call_id`` is the call_id field on the tool-call event/Span.
    """
    return f"{KAOS}call/{call_id}"


def step_iri(step_id: str) -> str:
    """IRI for a :class:`kaos:Step` node — one plan-execute step."""
    return f"{KAOS}step/{step_id}"


def doc_iri(source_uri: str) -> str:
    """IRI for a :class:`kaos:Document` node.

    Pass-through if the input already looks like an IRI; otherwise
    wraps under the kaos: namespace.
    """
    if source_uri.startswith(("http://", "https://", "urn:", "file:", "doi:")):
        return source_uri
    return f"{KAOS}doc/{source_uri}"


def agent_iri() -> str:
    """IRI for the agent runtime itself.

    Single canonical IRI for v1 — every tool call's ``prov:wasAssociatedWith``
    points here. Future phases can per-session-instance a distinct
    agent IRI for multi-agent attribution.
    """
    return KAOS_AGENT_RUNTIME


# ---------------------------------------------------------------------------
# Triple emission helpers
# ---------------------------------------------------------------------------


def _ensure_node(graph, iri: str, **properties) -> None:
    """Add a node to the graph, or update its properties if it exists.

    kaos_graph.Graph.add_node raises if the node already exists; we want
    idempotent emission (an event firing twice shouldn't break). Use
    update_node when the node is already there.
    """
    if graph.has_node(iri):
        if properties:
            graph.update_node(iri, **properties)
    else:
        graph.add_node(iri, **properties)


def _now_iso() -> str:
    """ISO-8601 UTC timestamp for prov:startedAtTime / prov:endedAtTime."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _emit_tool_call_start(event, memory: SessionMemory) -> int:
    """Emit prov: triples for a tool call's START phase.

    Triples emitted:
      <call:X> rdf:type kaos:ToolCall .
      <call:X> rdfs:label "tool_name" .
      <call:X> prov:startedAtTime "..."^^xsd:dateTime .
      <call:X> prov:wasAssociatedWith <kaos:agent-runtime> .
      <call:X> prov:wasInformedBy <step:Y>   (when step_id is present)
    """
    attrs = event.attributes
    call_id = str(attrs.get("call_id", ""))
    if not call_id:
        return 0
    tool_name = str(attrs.get("tool_name", ""))
    step_id = str(attrs.get("step_id", "") or "")

    graph = memory.graph
    call_node = tool_call_iri(call_id)
    agent_node = agent_iri()

    _ensure_node(
        graph,
        call_node,
        rdf_type=KAOS_TOOL_CALL,
        rdfs_label=tool_name,
    )
    _ensure_node(graph, agent_node, rdfs_label="kaos-agents runtime")

    # Properties on the call node — duration etc. arrive on COMPLETE.
    graph.update_node(call_node, prov_startedAtTime=_now_iso())

    # Cross-node provenance edges
    graph.add_edge(call_node, agent_node, predicate=PROV_ASSOCIATED_WITH)

    triples_added = 4  # type, label, startedAt, wasAssociatedWith

    if step_id:
        step_node = step_iri(step_id)
        _ensure_node(graph, step_node, rdf_type=KAOS_STEP)
        graph.add_edge(call_node, step_node, predicate=PROV_INFORMED_BY)
        triples_added += 1

    return triples_added


def _emit_tool_call_complete(event, memory: SessionMemory) -> int:
    """Emit the prov:endedAtTime + duration triples for a completed tool call."""
    attrs = event.attributes
    call_id = str(attrs.get("call_id", ""))
    if not call_id:
        return 0

    graph = memory.graph
    call_node = tool_call_iri(call_id)
    if not graph.has_node(call_node):
        # COMPLETE without a START — emit a stub node so the triple has a subject
        _ensure_node(graph, call_node, rdf_type=KAOS_TOOL_CALL)

    duration_ms = float(event.duration_ms or 0.0)
    is_error = bool(attrs.get("is_error", False))

    graph.update_node(
        call_node,
        prov_endedAtTime=_now_iso(),
        kaos_duration_ms=duration_ms,
        kaos_is_error=is_error,
    )
    return 3  # endedAt, duration_ms, is_error


def _emit_step_start(event, memory: SessionMemory) -> int:
    """Emit kaos:Step typed node + label."""
    attrs = event.attributes
    step_id = str(attrs.get("step_id", ""))
    if not step_id:
        return 0
    description = str(attrs.get("description", ""))

    graph = memory.graph
    step_node = step_iri(step_id)
    _ensure_node(
        graph,
        step_node,
        rdf_type=KAOS_STEP,
        rdfs_label=description or step_id,
        prov_startedAtTime=_now_iso(),
    )
    return 3


def _emit_step_complete(event, memory: SessionMemory) -> int:
    """Emit prov:endedAtTime for a completed step."""
    attrs = event.attributes
    step_id = str(attrs.get("step_id", ""))
    if not step_id:
        return 0

    graph = memory.graph
    step_node = step_iri(step_id)
    if not graph.has_node(step_node):
        _ensure_node(graph, step_node, rdf_type=KAOS_STEP)

    graph.update_node(
        step_node,
        prov_endedAtTime=_now_iso(),
        kaos_duration_ms=float(event.duration_ms or 0.0),
        kaos_is_error=bool(attrs.get("is_error", False)),
    )
    return 3


def _emit_citation_found(event, memory: SessionMemory) -> int:
    """Emit cito:cites triple for a verified citation.

    Triples:
      <finding:X> rdf:type kaos:Finding .
      <finding:X> rdfs:label "claim text" .
      <finding:X> cito:cites <doc:source_uri> .
      <doc:source_uri> rdf:type kaos:Document .
    """
    claim = event.claim
    source_uri = event.source_uri
    if not claim or not source_uri:
        return 0

    # Use a content-derived id for the finding node.
    # ``kaos_graph.Graph`` requires absolute IRIs; build from a
    # short hash of the claim text (deterministic across emit calls
    # so duplicate citations don't multiply nodes).
    import hashlib

    finding_id = hashlib.sha256(claim.encode()).hexdigest()[:12]
    finding_node = finding_iri(finding_id)
    doc_node = doc_iri(source_uri)

    graph = memory.graph
    _ensure_node(
        graph,
        finding_node,
        rdf_type=KAOS_FINDING,
        rdfs_label=claim[:200],  # cap to keep label compact
        kaos_confidence=float(event.confidence),
        kaos_verified=bool(event.verified),
    )
    _ensure_node(graph, doc_node, rdf_type=KAOS_DOCUMENT)
    graph.add_edge(finding_node, doc_node, predicate=CITO_CITES)

    return 4  # finding type+label+confidence (3 props on one node) + cites edge


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def emit_from_event(event: KaosEvent, memory: SessionMemory) -> int:
    """Translate one event into RDF triples on ``memory.graph``.

    Returns the number of triples added (rough count — node properties
    count as triples; counted by hand per emission helper).

    Events not in the v1 vocabulary are no-ops (returns 0). The
    function never raises — graph emission is best-effort instrumentation
    that must not break the agent loop.

    Args:
        event: Any :class:`KaosEvent` instance.
        memory: Target :class:`SessionMemory`. Triples land on
            ``memory.graph`` (lazy-constructed on first access).

    Returns:
        Number of triples / property edges emitted.
    """
    # Lazy imports to avoid cycle on package load.
    from kaos_agents.events import (
        CitationFound,
        Span,
        SpanPhase,
        SpanSubject,
    )

    try:
        if isinstance(event, Span):
            if event.subject == SpanSubject.TOOL_CALL:
                if event.phase == SpanPhase.START:
                    return _emit_tool_call_start(event, memory)
                if event.phase == SpanPhase.COMPLETE:
                    return _emit_tool_call_complete(event, memory)
            elif event.subject == SpanSubject.STEP:
                if event.phase == SpanPhase.START:
                    return _emit_step_start(event, memory)
                if event.phase == SpanPhase.COMPLETE:
                    return _emit_step_complete(event, memory)
            return 0
        if isinstance(event, CitationFound):
            return _emit_citation_found(event, memory)
    except Exception as exc:
        logger.warning(
            "triples.emit_from_event: failed for %s (swallowed): %s",
            type(event).__name__,
            exc,
        )
        return 0

    return 0


__all__ = [
    "CITO",
    "CITO_CITES",
    "CITO_SUPPORTS",
    "KAOS",
    "KAOS_AGENT_RUNTIME",
    "KAOS_DOCUMENT",
    "KAOS_FINDING",
    "KAOS_STEP",
    "KAOS_TOOL_CALL",
    "PROV",
    "PROV_AGENT",
    "PROV_ASSOCIATED_WITH",
    "PROV_DERIVED_FROM",
    "PROV_ENDED_AT",
    "PROV_GENERATED_BY",
    "PROV_INFORMED_BY",
    "PROV_STARTED_AT",
    "PROV_USED",
    "RDF",
    "RDFS",
    "RDFS_LABEL",
    "RDF_TYPE",
    "agent_iri",
    "doc_iri",
    "emit_from_event",
    "finding_iri",
    "step_iri",
    "tool_call_iri",
]
