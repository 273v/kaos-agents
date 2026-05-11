"""K7 — kaos-agent-findings MCP tool wrapping FindingsAgent.

Exposes the K6 :class:`~kaos_agents.patterns.findings.FindingsAgent`
to MCP callers. Takes a stored ContentDocument artifact, a question,
and the selector to use, runs the three-phase pipeline, returns the
synthesized answer + surviving findings.

Single tool — three selector modes packed into one ``select_by``
parameter — keeps the agent's tool surface tidy (one tool to learn,
three modes to choose from) vs. three near-identical tools to
discover and remember.

Annotations: readOnlyHint=False (this tool makes paid LLM calls;
``readOnlyHint=True`` would auto-approve in Claude Code which is
not what we want for a tool that spends), idempotentHint=False
(LLM-driven, non-deterministic), destructiveHint=False (writes
nothing), openWorldHint=False (operates only on the named
artifact + LLM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.base.tool import KaosTool
from kaos_core.logging import get_logger
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.metadata import ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext

logger = get_logger(__name__)

_MODULE = "kaos-agents"
_VERSION = "0.1.0"

# Tool is not read-only because it spends money on LLM calls.
# Auto-approval in agents like Claude Code is gated on readOnlyHint;
# we want the user to explicitly approve each findings run since the
# cost can reach ~$0.10+ on a real document.
_FINDINGS_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

_SELECT_BY_OPTIONS = ("every_sentence", "token", "entity")
"""Selector modes supported by the MCP tool.

- ``every_sentence``: every non-empty sentence in the doc. Use when
  recall must be 1.0 and the doc fits within the filter-pass budget.
- ``token``: every sentence containing a literal substring (param
  ``selector_arg``). Cheapest and most precise when you know the
  vocabulary.
- ``entity``: every sentence containing at least one match of a
  typed entity (``selector_arg`` in
  ``{"dates", "money", "percents", "durations", "numbers"}``).
  Composes with K2."""


class AgentFindingsTool(KaosTool):
    """MCP wrapper around :class:`FindingsAgent`.

    Returns a JSON payload with the synthesized answer, surviving
    findings (text + relevance + reasoning + AST refs), and cost
    breakdown. Errors fail with an actionable message rather than
    raising.
    """

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-findings",
            display_name="Findings (Exhaustive Search)",
            description=(
                "Recall-first exhaustive-search agent over a stored "
                "ContentDocument. Three phases: enumerate every "
                "candidate sentence via a Phase-1 selector "
                "(every_sentence / token / entity), filter survivors "
                "via parallel LLM calls (chunked), synthesize a final "
                "answer that cites finding_ids inline. Use for "
                "diligence / audit questions where missing a relevant "
                "sentence costs more than the extra LLM cycles. Plain "
                "RAG (kaos-agent-chat with retrieval) is cheaper for "
                "ordinary Q&A; this tool is the right reach when "
                "recall must be 1.0. Cost typically $0.05-0.10 per "
                "real NDA. Pair with kaos-content-corpus-narrow to "
                "first triage a large corpus to one or a few "
                "artifacts, then run findings against each. "
                "REFUSAL CONTRACT: when the agent cannot answer "
                "from the document (Phase 1 produced no candidates, "
                "or Phase 2 filtered them all out) the response is "
                "still a structured SUCCESS (isError=false) but "
                "carries ``refusal_reason`` (one of "
                "``no_candidates_enumerated`` / "
                "``no_relevant_candidates``) and ``refusal_message``. "
                "Always check ``refusal_reason`` before treating an "
                "empty ``answer`` as a failure — an empty answer with "
                "a populated refusal is the agent honestly reporting "
                "'this document does not contain the answer.'"
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_FINDINGS_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="artifact_id",
                    type="string",
                    description=(
                        "Artifact ID of the stored ContentDocument to "
                        "search over. Get one from kaos-content-* "
                        "parsing/reader tools."
                    ),
                ),
                ParameterSchema(
                    name="question",
                    type="string",
                    description=(
                        "The question to answer. The synthesis step "
                        "consumes this verbatim; phrase it as you "
                        "would ask a human reviewer."
                    ),
                ),
                ParameterSchema(
                    name="select_by",
                    type="string",
                    description=(
                        "Phase-1 selector mode. 'every_sentence' "
                        "(recall=1.0 across the whole doc), 'token' "
                        "(substring match; provide selector_arg), or "
                        "'entity' (typed-entity match; provide "
                        "selector_arg in {dates, money, percents, "
                        "durations, numbers})."
                    ),
                    required=False,
                    default="every_sentence",
                    constraints={"enum": list(_SELECT_BY_OPTIONS)},
                ),
                ParameterSchema(
                    name="selector_arg",
                    type="string",
                    description=(
                        "Argument for the selector. Required when "
                        "select_by='token' (the substring) or "
                        "'entity' (the entity type name)."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="filter_model",
                    type="string",
                    description=(
                        "Model for the per-chunk filter calls. "
                        "Default 'anthropic:claude-haiku-4-5'."
                    ),
                    required=False,
                    default="anthropic:claude-haiku-4-5",
                ),
                ParameterSchema(
                    name="synthesis_model",
                    type="string",
                    description=(
                        "Model for the final synthesis call. Default 'anthropic:claude-sonnet-4-6'."
                    ),
                    required=False,
                    default="anthropic:claude-sonnet-4-6",
                ),
                ParameterSchema(
                    name="chunk_size",
                    type="integer",
                    description=(
                        "Candidates per filter call. Lower = better "
                        "isolation, higher = cheaper. Default 20."
                    ),
                    required=False,
                    default=20,
                ),
                ParameterSchema(
                    name="num_parallel",
                    type="integer",
                    description=("Parallel filter-call concurrency. Default 4."),
                    required=False,
                    default=4,
                ),
                ParameterSchema(
                    name="relevance_threshold",
                    type="number",
                    description=("Filter survivors below this score are dropped. Default 0.5."),
                    required=False,
                    default=0.5,
                ),
                ParameterSchema(
                    name="temperature",
                    type="number",
                    description=(
                        "Sampling temperature for filter + synthesis "
                        "Calls. Default 0.0 (deterministic) — Sprint-2 "
                        "#5 quality gate: two associates running the "
                        "tool on the same NDA see the same findings. "
                        "Set non-zero (e.g. 0.7) only for "
                        "experimentation / optimizer search."
                    ),
                    required=False,
                    default=0.0,
                ),
                ParameterSchema(
                    name="runs",
                    type="integer",
                    description=(
                        "Number of independent filter passes to UNION. "
                        "Default 1 (single pass). Setting runs>1 issues "
                        "N filter pipelines, unions surviving findings "
                        "by deterministic finding_id, and synthesizes "
                        "once over the union. Cost = runs*filter_cost "
                        "+ synthesis_cost. Use 2-3 for diligence "
                        "where missing a clause is unacceptable."
                    ),
                    required=False,
                    default=1,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error("No runtime context. Register tools with a KaosRuntime.")

        artifact_id = inputs.get("artifact_id")
        if not artifact_id:
            return ToolResult.create_error(
                "Missing 'artifact_id'. Provide the ID of a stored "
                "ContentDocument (from kaos-content-parse-markdown or "
                "any reader tool)."
            )
        question = inputs.get("question")
        if not question or not str(question).strip():
            return ToolResult.create_error("Missing 'question'. Provide the question to answer.")

        select_by = str(inputs.get("select_by") or "every_sentence")
        if select_by not in _SELECT_BY_OPTIONS:
            return ToolResult.create_error(
                f"Invalid select_by={select_by!r}. Must be one of {list(_SELECT_BY_OPTIONS)}."
            )
        selector_arg = inputs.get("selector_arg")

        chunk_size_raw = inputs.get("chunk_size")
        chunk_size = 20 if chunk_size_raw is None else int(chunk_size_raw)
        num_parallel_raw = inputs.get("num_parallel")
        num_parallel = 4 if num_parallel_raw is None else int(num_parallel_raw)
        relevance_threshold_raw = inputs.get("relevance_threshold")
        relevance_threshold = (
            0.5 if relevance_threshold_raw is None else float(relevance_threshold_raw)
        )
        temperature_raw = inputs.get("temperature")
        temperature = 0.0 if temperature_raw is None else float(temperature_raw)
        runs_raw = inputs.get("runs")
        runs = 1 if runs_raw is None else int(runs_raw)
        filter_model = str(inputs.get("filter_model") or "anthropic:claude-haiku-4-5")
        synthesis_model = str(inputs.get("synthesis_model") or "anthropic:claude-sonnet-4-6")

        # Validate selector_arg / build selector.
        try:
            selector = _build_selector(select_by, selector_arg)
        except ValueError as exc:
            return ToolResult.create_error(str(exc))

        # Load document and build view.
        try:
            from kaos_content.artifacts import load_document
            from kaos_content.views import DocumentView
            from kaos_nlp_core._defaults import get_default_punkt_tokenizer

            doc = await load_document(str(artifact_id), context.runtime)
            view = DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to load artifact {artifact_id!r}: {exc}. "
                "Verify the artifact exists in the runtime's VFS."
            )

        # Run the pipeline.
        from kaos_agents.patterns.findings import FindingsAgent

        try:
            agent = FindingsAgent(
                selector=selector,
                filter_model=filter_model,
                synthesis_model=synthesis_model,
                chunk_size=chunk_size,
                num_parallel=num_parallel,
                relevance_threshold=relevance_threshold,
                temperature=temperature,
                runs=runs,
            )
        except ValueError as exc:
            return ToolResult.create_error(f"FindingsAgent rejected the configuration: {exc}")

        try:
            result = await agent.run(str(question), view)
        except Exception as exc:
            logger.exception("findings tool failed")
            return ToolResult.create_error(
                f"FindingsAgent run failed: {exc}. Check the recorded "
                "JSONL trace under tests/integration/runs/ for the "
                "partial state when this failed."
            )

        # Build JSON-friendly payload.
        findings_payload = [
            {
                "finding_id": f.candidate.finding_id,
                "text": f.candidate.text,
                "relevance": f.relevance,
                "reasoning": f.reasoning,
                "block_ref": f.candidate.block_ref,
                "char_span": (
                    list(f.candidate.char_span) if f.candidate.char_span is not None else None
                ),
                "section_title": f.candidate.section_title,
                "page": f.candidate.page,
            }
            for f in result.findings
        ]
        output: dict[str, Any] = {
            "artifact_id": artifact_id,
            "question": question,
            "select_by": select_by,
            "selector_arg": selector_arg,
            "temperature": temperature,
            "runs": runs,
            "answer": result.answer,
            "findings": findings_payload,
            "total_enumerated": result.total_enumerated,
            "total_filtered": result.total_filtered,
            "filter_calls": result.filter_calls,
            "filter_cost_usd": result.filter_cost_usd,
            "synthesis_cost_usd": result.synthesis_cost_usd,
            "total_cost_usd": result.total_cost_usd,
            "total_llm_calls": result.total_llm_calls,
            # Refusal contract — see metadata.description. ``None``
            # when synthesis produced a real answer; populated when
            # the agent honestly cannot answer from this document.
            # Downstream consumers (UI, audit, agent caller) must
            # branch on ``refusal_reason`` before treating
            # ``answer == ""`` as a failure.
            "refusal_reason": (result.refusal.reason if result.refusal is not None else None),
            "refusal_message": (result.refusal.message if result.refusal is not None else None),
        }
        if result.refusal is not None:
            summary = (
                f"Findings refusal ({result.refusal.reason}): "
                f"enumerated={result.total_enumerated} "
                f"filtered={result.total_filtered} "
                f"cost=${result.total_cost_usd:.4f}"
            )
        else:
            summary = (
                f"Findings: enumerated={result.total_enumerated} "
                f"filtered={result.total_filtered} "
                f"calls={result.total_llm_calls} "
                f"cost=${result.total_cost_usd:.4f}"
            )
        return ToolResult.create_success(output=output, summary=summary)


# ---------------------------------------------------------------------------
# Selector construction
# ---------------------------------------------------------------------------


def _build_selector(select_by: str, selector_arg: Any) -> Any:
    """Translate the MCP-level (select_by, selector_arg) into a callable
    selector for the FindingsAgent.

    Raises ``ValueError`` with an actionable message when the arguments
    don't match the chosen mode. The execute() body catches and
    forwards the message verbatim to ``ToolResult.create_error``.
    """
    from kaos_agents.patterns.findings import (
        every_sentence_selector,
        sentences_with_entity_selector,
        sentences_with_token_selector,
    )

    if select_by == "every_sentence":
        return every_sentence_selector

    if select_by == "token":
        if not selector_arg or not str(selector_arg).strip():
            raise ValueError("select_by='token' requires 'selector_arg' (the substring to match).")
        return sentences_with_token_selector(str(selector_arg))

    if select_by == "entity":
        if not selector_arg or not str(selector_arg).strip():
            raise ValueError(
                "select_by='entity' requires 'selector_arg' (the entity "
                "type). Valid types: dates, money, percents, "
                "durations, numbers."
            )
        return sentences_with_entity_selector(str(selector_arg))

    raise ValueError(f"Unknown select_by={select_by!r}")


__all__ = ["AgentFindingsTool"]
