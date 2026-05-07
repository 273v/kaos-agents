"""Graph-aware MCP tools — Track 3 chunk B3.

Three tools exposing the per-session knowledge graph (B1) populated by
the triple emitter (B2):

- ``kaos-agent-graph-walk`` — N-hop ego subgraph from a starting IRI
- ``kaos-agent-graph-sparql`` — SPARQL SELECT/ASK against the session graph
- ``kaos-agent-graph-projection`` — pre-built typed views (findings,
  tool-call timeline, step timeline)

These tools let agents reason over their own provenance: "what did I
cite for this finding?", "what tool calls happened in step X?", "show
me the support chain for claim Y."

Read-only by design — the graph is populated by emit_from_event and
should not be mutated through MCP. Use the agent's run loop to add
events; the emitter does the rest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.base.tool import KaosTool
from kaos_core.logging import get_logger
from kaos_core.types.metadata import ToolAnnotations, ToolCapability, ToolCategory, ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext

logger = get_logger(__name__)

_MODULE = "kaos-agents"
_VERSION = "0.1.0"

_AGENT_READ_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


def _get_vfs(runtime: Any) -> Any:
    """Resolve the VFS from the runtime, falling back to in-memory.

    Mirror of ``kaos_agents.tools._get_vfs`` to avoid a cross-import
    that would pull the rest of the tools.py surface.
    """
    from kaos_core.types.enums import StorageBackend
    from kaos_core.vfs.core import VirtualFileSystem
    from kaos_core.vfs.models import VFSConfig

    if runtime is not None and hasattr(runtime, "vfs") and runtime.vfs is not None:
        return runtime.vfs
    return VirtualFileSystem(config=VFSConfig(default_backend=StorageBackend.MEMORY))


async def _load_session_memory(session_id: str, context: KaosContext | None) -> Any:
    """Load a SessionMemory by id, or raise with an agent-friendly error."""
    from kaos_agents.memory.store import SessionStore

    runtime = context.runtime if context else None
    vfs = _get_vfs(runtime)
    store = SessionStore(vfs)
    return await store.load_or_create(session_id)


# ---------------------------------------------------------------------------
# Tool: graph-walk
# ---------------------------------------------------------------------------


class AgentGraphWalkTool(KaosTool):
    """N-hop walk from a node in the session knowledge graph.

    Returns the ego subgraph centered at ``start_iri`` within ``max_hops``
    hops — the set of nodes (and edges between them) reachable within
    that radius. Use this to follow provenance edges: from a finding,
    walk to the tool calls that produced it, the steps those calls
    were part of, and the documents they cited.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-graph-walk",
            display_name="Walk Session Graph",
            description=(
                "Walk N hops from a starting IRI in the session knowledge graph "
                "and return the reached subgraph. The graph is populated automatically "
                "during agent runs — every tool call, step, and verified citation "
                "becomes a typed RDF node (kaos:ToolCall / kaos:Step / kaos:Finding / "
                "kaos:Document) connected by PROV-O / CiTO edges. "
                "\n\n"
                "Use this to follow provenance: 'what tool calls produced finding X?', "
                "'what citations support claim Y?'. For SPARQL queries instead, "
                "call kaos-agent-graph-sparql. For pre-built typed views, call "
                "kaos-agent-graph-projection."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description="Session identifier (same one passed to kaos-agent-chat).",
                ),
                ParameterSchema(
                    name="start_iri",
                    type="string",
                    description=(
                        "Starting node IRI. Build with kaos_agents.memory.triples helpers: "
                        "finding_iri(id), tool_call_iri(call_id), step_iri(step_id), or "
                        "doc_iri(uri). Must be an absolute IRI."
                    ),
                ),
                ParameterSchema(
                    name="max_hops",
                    type="integer",
                    description="Maximum walk radius (1-5). Default: 2.",
                    required=False,
                    constraints={"min": 1, "max": 5},
                ),
                ParameterSchema(
                    name="max_nodes",
                    type="integer",
                    description="Result cap to avoid huge subgraphs. Default: 100.",
                    required=False,
                    constraints={"min": 1, "max": 1000},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        session_id = inputs.get("session_id", "")
        start_iri = inputs.get("start_iri", "")
        max_hops = int(inputs.get("max_hops", 2))
        max_nodes = int(inputs.get("max_nodes", 100))

        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                "Provide the session ID whose graph you want to walk. "
                'Example: {"session_id": "research-epa", "start_iri": "https://kaos.273ventures.com/ns/finding/abc123"}'
            )
        if not start_iri:
            return ToolResult.create_error(
                "Missing 'start_iri' parameter. "
                "Provide an absolute IRI to start the walk from. "
                "Build IRIs with kaos_agents.memory.triples helpers "
                "(finding_iri, tool_call_iri, step_iri, doc_iri). "
                'Example: {"start_iri": "https://kaos.273ventures.com/ns/call/tc-1"}'
            )

        try:
            memory = await _load_session_memory(session_id, context)
        except Exception as exc:
            logger.debug("graph-walk: session load failed: %s", exc, exc_info=True)
            return ToolResult.create_error(
                f"Session '{session_id}' not found: {exc}. "
                "Check the session_id matches a previous agent call. "
                "If the session never had any graph activity (no tool calls / citations), "
                "the graph will be empty — call kaos-agent-chat first to produce events."
            )

        graph = memory.graph
        if not graph.has_node(start_iri):
            return ToolResult.create_error(
                f"Node '{start_iri}' is not in the session graph. "
                "List available nodes via kaos-agent-graph-projection "
                "(projection_name='all_nodes') or query with kaos-agent-graph-sparql. "
                "IRIs must match exactly — including the kaos: namespace prefix."
            )

        try:
            ego = graph.ego_graph(start_iri, radius=max_hops)
        except Exception as exc:
            logger.warning("graph-walk: ego_graph failed: %s", exc)
            return ToolResult.create_error(
                f"Walk failed: {exc}. "
                "Try a smaller max_hops, or check that start_iri is a node in the graph."
            )

        node_ids = list(ego.node_ids())[:max_nodes]
        truncated = ego.n_nodes > max_nodes

        edges_out: list[dict[str, str]] = []
        for edge in ego.edges():
            if edge.source in node_ids and edge.target in node_ids:
                edges_out.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "predicate": str(edge.properties.get("predicate", "")),
                    }
                )

        result_data = {
            "session_id": session_id,
            "start_iri": start_iri,
            "max_hops": max_hops,
            "node_count": len(node_ids),
            "edge_count": len(edges_out),
            "truncated": truncated,
            "nodes": node_ids,
            "edges": edges_out,
        }
        summary = (
            f"Walked {max_hops} hops from {start_iri[:60]}: "
            f"{len(node_ids)} nodes, {len(edges_out)} edges"
            f"{' (truncated)' if truncated else ''}"
        )
        return ToolResult.create_success(output=result_data, summary=summary)


# ---------------------------------------------------------------------------
# Tool: graph-sparql
# ---------------------------------------------------------------------------


class AgentGraphSparqlTool(KaosTool):
    """Run a SPARQL SELECT or ASK query against the session graph.

    Requires the ``kaos-graph[rdf]`` extra (pyoxigraph). Returns a clear
    error pointing to the install command if missing.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-graph-sparql",
            display_name="SPARQL Query Session Graph",
            description=(
                "Execute a SPARQL SELECT or ASK query against the session knowledge graph. "
                "Use SPARQL when you need set-oriented retrieval, joins across multiple "
                "predicates, or filtering by literal values. For simple traversal, "
                "kaos-agent-graph-walk is faster. For pre-built views, use "
                "kaos-agent-graph-projection. "
                "\n\n"
                "Vocabulary: rdf:type / rdfs:label / prov:* (PROV-O) / cito:cites / "
                "kaos:Finding / kaos:ToolCall / kaos:Step / kaos:Document. "
                "\n\n"
                "Requires the kaos-graph[rdf] extra to be installed (pyoxigraph)."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description="Session identifier.",
                ),
                ParameterSchema(
                    name="query",
                    type="string",
                    description=(
                        "SPARQL SELECT or ASK query string. "
                        "Example: 'SELECT ?finding ?doc WHERE { ?finding "
                        "<http://purl.org/spar/cito/cites> ?doc }'"
                    ),
                ),
                ParameterSchema(
                    name="query_type",
                    type="string",
                    description="Either 'select' (default) or 'ask'.",
                    required=False,
                    constraints={"choices": ["select", "ask"]},
                ),
                ParameterSchema(
                    name="max_rows",
                    type="integer",
                    description="Result row cap. Default: 200.",
                    required=False,
                    constraints={"min": 1, "max": 10_000},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        session_id = inputs.get("session_id", "")
        query = inputs.get("query", "")
        query_type = str(inputs.get("query_type", "select")).lower()
        max_rows = int(inputs.get("max_rows", 200))

        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                "Provide the session ID whose graph you want to query. "
                'Example: {"session_id": "research-epa", '
                '"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10"}'
            )
        if not query:
            return ToolResult.create_error(
                "Missing 'query' parameter. "
                "Provide a SPARQL SELECT or ASK query. "
                "For simple node lookups, kaos-agent-graph-walk may be easier."
            )
        if query_type not in ("select", "ask"):
            return ToolResult.create_error(
                f"Invalid query_type '{query_type}'. "
                "Must be 'select' (default) or 'ask'. "
                "For graph traversal queries, use kaos-agent-graph-walk."
            )

        try:
            memory = await _load_session_memory(session_id, context)
        except Exception as exc:
            return ToolResult.create_error(
                f"Session '{session_id}' not found: {exc}. "
                "Check the session_id matches a previous agent call."
            )

        try:
            from kaos_graph.rdf.sparql import query_sparql, query_sparql_ask

            if query_type == "ask":
                ask_result = query_sparql_ask(memory.graph, query)
                return ToolResult.create_success(
                    output={
                        "session_id": session_id,
                        "query_type": "ask",
                        "result": bool(ask_result),
                    },
                    summary=f"ASK → {bool(ask_result)}",
                )

            select_result = query_sparql(memory.graph, query)
            rows = select_result.rows[:max_rows]
            truncated = len(select_result.rows) > max_rows

            result_data = {
                "session_id": session_id,
                "query_type": "select",
                "variables": select_result.variables,
                "row_count": len(rows),
                "truncated": truncated,
                "rows": rows,
            }
            summary = f"SELECT → {len(rows)} row(s){' (truncated)' if truncated else ''}"
            return ToolResult.create_success(output=result_data, summary=summary)

        except ImportError as exc:
            return ToolResult.create_error(
                f"SPARQL support is unavailable: {exc}. "
                "Install the kaos-graph[rdf] extra: 'uv add kaos-graph[rdf]'. "
                "For provenance-only walks, kaos-agent-graph-walk works without pyoxigraph."
            )
        except ValueError as exc:
            return ToolResult.create_error(
                f"SPARQL query failed: {exc}. "
                "Check query syntax. SELECT and ASK are supported; CONSTRUCT and DESCRIBE are not. "
                "For literal IRIs, wrap in <angle brackets>. "
                "For variables, use ?name. "
                "Common predicates: <http://www.w3.org/ns/prov#wasGeneratedBy>, "
                "<http://purl.org/spar/cito/cites>, <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>."
            )
        except Exception as exc:
            logger.warning("graph-sparql: query failed: %s", exc)
            return ToolResult.create_error(
                f"SPARQL execution failed: {exc}. "
                "If the graph is empty, run kaos-agent-chat to populate it first."
            )


# ---------------------------------------------------------------------------
# Tool: graph-projection
# ---------------------------------------------------------------------------


# Pre-built SPARQL templates — keyed by projection name. These are the
# typed views agents reach for most often. New projections can be added
# without changing the tool surface.
_PROJECTIONS: dict[str, dict[str, str]] = {
    "findings_with_citations": {
        "description": "All kaos:Finding nodes with their cited documents (cito:cites).",
        "query": (
            "SELECT ?finding ?label ?doc WHERE { "
            "?finding <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<https://kaos.273ventures.com/ns/Finding> . "
            "OPTIONAL { ?finding <http://www.w3.org/2000/01/rdf-schema#label> ?label } "
            "OPTIONAL { ?finding <http://purl.org/spar/cito/cites> ?doc } "
            "}"
        ),
    },
    "tool_calls_by_step": {
        "description": (
            "Tool calls grouped by their parent step (prov:wasInformedBy). "
            "Returns ?call ?tool_name ?step pairs."
        ),
        "query": (
            "SELECT ?call ?tool_name ?step WHERE { "
            "?call <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<https://kaos.273ventures.com/ns/ToolCall> . "
            "OPTIONAL { ?call <http://www.w3.org/2000/01/rdf-schema#label> ?tool_name } "
            "OPTIONAL { ?call <http://www.w3.org/ns/prov#wasInformedBy> ?step } "
            "}"
        ),
    },
    "step_timeline": {
        "description": "All kaos:Step nodes — for plan-execute provenance traversal.",
        "query": (
            "SELECT ?step ?label WHERE { "
            "?step <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<https://kaos.273ventures.com/ns/Step> . "
            "OPTIONAL { ?step <http://www.w3.org/2000/01/rdf-schema#label> ?label } "
            "}"
        ),
    },
    "all_nodes": {
        "description": "Inventory: every node IRI in the graph (no SPARQL needed).",
        "query": "",  # special-case — handled in execute()
    },
}


class AgentGraphProjectionTool(KaosTool):
    """Pre-built typed views over the session knowledge graph.

    Convenience layer above kaos-agent-graph-sparql. Each projection
    encapsulates a common query pattern (findings + citations, tool
    calls by step, step timeline). Use this when you don't want to
    write SPARQL by hand.
    """

    @property
    def metadata(self) -> ToolMetadata:
        names = sorted(_PROJECTIONS.keys())
        return ToolMetadata(
            name="kaos-agent-graph-projection",
            display_name="Project Session Graph",
            description=(
                "Run a pre-built typed view over the session knowledge graph. "
                "Available projections: " + ", ".join(names) + ". "
                "Each projection is a curated SPARQL query that returns rows shaped "
                "for a common provenance question. For ad-hoc queries, use "
                "kaos-agent-graph-sparql directly. For one-node ego subgraphs, "
                "use kaos-agent-graph-walk. "
                "\n\n"
                "Most projections require the kaos-graph[rdf] extra (pyoxigraph). "
                "The 'all_nodes' projection is the exception — it walks the "
                "graph directly without SPARQL."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_AGENT_READ_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description="Session identifier.",
                ),
                ParameterSchema(
                    name="projection_name",
                    type="string",
                    description=("Which projection to run. Available: " + ", ".join(names) + "."),
                    constraints={"choices": names},
                ),
                ParameterSchema(
                    name="max_rows",
                    type="integer",
                    description="Result row cap. Default: 200.",
                    required=False,
                    constraints={"min": 1, "max": 10_000},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        session_id = inputs.get("session_id", "")
        projection_name = inputs.get("projection_name", "")
        max_rows = int(inputs.get("max_rows", 200))

        if not session_id:
            return ToolResult.create_error(
                "Missing 'session_id' parameter. "
                'Example: {"session_id": "research-epa", '
                '"projection_name": "findings_with_citations"}'
            )
        if not projection_name:
            available = ", ".join(sorted(_PROJECTIONS.keys()))
            return ToolResult.create_error(
                f"Missing 'projection_name' parameter. Available: {available}. "
                "Use 'all_nodes' to inventory the graph if you don't know what's in it."
            )
        if projection_name not in _PROJECTIONS:
            available = ", ".join(sorted(_PROJECTIONS.keys()))
            return ToolResult.create_error(
                f"Unknown projection '{projection_name}'. Available: {available}. "
                "For ad-hoc queries, use kaos-agent-graph-sparql instead."
            )

        try:
            memory = await _load_session_memory(session_id, context)
        except Exception as exc:
            return ToolResult.create_error(
                f"Session '{session_id}' not found: {exc}. "
                "Check the session_id matches a previous agent call."
            )

        projection = _PROJECTIONS[projection_name]

        # Special case: all_nodes is a graph walk, not SPARQL.
        if projection_name == "all_nodes":
            node_ids = list(memory.graph.node_ids())[:max_rows]
            truncated = memory.graph.n_nodes > max_rows
            return ToolResult.create_success(
                output={
                    "session_id": session_id,
                    "projection_name": projection_name,
                    "description": projection["description"],
                    "row_count": len(node_ids),
                    "truncated": truncated,
                    "nodes": node_ids,
                },
                summary=(
                    f"all_nodes → {len(node_ids)} node(s){' (truncated)' if truncated else ''}"
                ),
            )

        try:
            from kaos_graph.rdf.sparql import query_sparql

            result = query_sparql(memory.graph, projection["query"])
        except ImportError as exc:
            return ToolResult.create_error(
                f"This projection requires the kaos-graph[rdf] extra (pyoxigraph): {exc}. "
                "Install with 'uv add kaos-graph[rdf]'. "
                "The 'all_nodes' projection works without pyoxigraph."
            )
        except ValueError as exc:
            return ToolResult.create_error(
                f"Projection '{projection_name}' query failed: {exc}. "
                "This is a bug in the projection template — file an issue."
            )
        except Exception as exc:
            logger.warning("graph-projection: %s failed: %s", projection_name, exc)
            return ToolResult.create_error(
                f"Projection '{projection_name}' failed: {exc}. "
                "If the graph is empty, run kaos-agent-chat to populate it first."
            )

        rows = result.rows[:max_rows]
        truncated = len(result.rows) > max_rows
        result_data = {
            "session_id": session_id,
            "projection_name": projection_name,
            "description": projection["description"],
            "variables": result.variables,
            "row_count": len(rows),
            "truncated": truncated,
            "rows": rows,
        }
        summary = f"{projection_name} → {len(rows)} row(s){' (truncated)' if truncated else ''}"
        return ToolResult.create_success(output=result_data, summary=summary)


__all__ = [
    "AgentGraphProjectionTool",
    "AgentGraphSparqlTool",
    "AgentGraphWalkTool",
]
