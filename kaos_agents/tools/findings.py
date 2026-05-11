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

_SELECT_BY_OPTIONS = ("every_sentence", "token", "semantic", "entity")
"""Selector modes supported by the MCP tool.

- ``every_sentence``: every non-empty sentence in the doc. Recall
  =1.0 across the whole doc; most expensive (the filter pass sees
  every sentence). Use when missing a clause is unacceptable and
  the doc fits within the filter budget.
- ``token``: every sentence containing a literal substring of
  ``selector_arg`` (case-insensitive). Cheap and most precise when
  you know the document's vocabulary. **Silently low-recall when
  your term isn't in the doc verbatim** — Sprint-2 #6 surfaces a
  structured warning in this case so the recall failure doesn't
  look like an LLM failure. Pair with ``select_by='semantic'``
  when the user phrases the question abstractly.
- ``semantic``: one cheap LLM pre-call rewrites the user's intent
  into vocabulary likely to appear in the document, then runs the
  union of literal-substring matches across those terms. Buys
  vocabulary expansion for ~$0.001 — the right reach when the
  question uses high-level language ("cyber risk mitigation")
  that doesn't literally appear in the document body
  ("multi-factor authentication and quarterly penetration
  testing").
- ``entity``: every sentence with at least one match of a typed
  entity (``selector_arg`` in
  ``{"dates", "money", "percents", "durations", "numbers"}``).
  Composes with K2.

The literal-substring semantic on ``token`` is intentional and
load-bearing: the agent's recall here is a function of the
*caller's* vocabulary choice, not the model's understanding. If
the document and the question use different words, ``token``
will quietly miss; ``semantic`` is the bridge."""


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
                "(every_sentence / token / semantic / entity), "
                "filter survivors via parallel LLM calls (chunked), "
                "synthesize a final answer that cites finding_ids "
                "inline. Use for diligence / audit questions where "
                "missing a relevant sentence costs more than the "
                "extra LLM cycles. Plain RAG (kaos-agent-chat with "
                "retrieval) is cheaper for ordinary Q&A; this tool "
                "is the right reach when recall must be 1.0. Cost "
                "typically $0.05-0.10 per real NDA. Pair with "
                "kaos-content-corpus-narrow to first triage a large "
                "corpus to one or a few artifacts, then run findings "
                "against each. "
                "SELECTOR MODES: 'every_sentence' recall=1.0 across "
                "the whole doc (most expensive); 'token' literal "
                "substring on selector_arg (cheap but SILENTLY "
                "LOW-RECALL when your term isn't in the doc "
                "verbatim — surfaces a warning in "
                "structuredContent['warnings']); 'semantic' LLM "
                "rewrites your intent into literal-tokens then "
                "token-matches the union (one cheap pre-call buys "
                "vocabulary expansion — use when the question is "
                "abstract); 'entity' typed-entity match (dates / "
                "money / percents / durations / numbers). "
                "REFUSAL CONTRACT: when the agent cannot answer "
                "from the document (Phase 1 produced no candidates, "
                "Phase 2 filtered them all out, or max_cost_usd "
                "fired before completion) the response is still a "
                "structured SUCCESS (isError=false) but carries "
                "``refusal_reason`` (one of "
                "``no_candidates_enumerated`` / "
                "``no_relevant_candidates`` / ``budget_exceeded``) "
                "and ``refusal_message``. Always check "
                "``refusal_reason`` before treating an empty "
                "``answer`` as a failure — an empty answer with a "
                "populated refusal is the agent honestly reporting "
                "'this document does not contain the answer' (the "
                "first two reasons) or 'the budget ran out before I "
                "finished looking' (budget_exceeded). "
                "BUDGET CONTRACT: when ``max_cost_usd`` is set, the "
                "tool aborts BEFORE dispatching the next filter "
                "wave once the cap would be breached and skips "
                "synthesis if there's no headroom. Returns "
                "``budget_exceeded=true`` in structuredContent and "
                "``cost_usd`` reports actual spend. Worst-case "
                "overshoot is one wave's in-flight cost — typically "
                "within 5% of the cap. "
                "WARNINGS CONTRACT: structuredContent['warnings'] "
                "may include a 'low_recall_token_selector' entry "
                "when select_by='token' enumerated < 5 candidates "
                "for a >= 6-word question; the agent's behavior is "
                "unchanged, the warning surfaces the recall risk so "
                "a missed clause isn't blamed on synthesis. "
                "Consider switching to select_by='semantic' when "
                "this fires."
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
                        "Phase-1 selector mode. "
                        "'every_sentence' — recall=1.0 across the "
                        "whole doc, most expensive. "
                        "'token' — literal substring match on "
                        "selector_arg; CHEAP BUT SILENTLY LOW-RECALL "
                        "when your term isn't in the doc verbatim "
                        "(a structured warning surfaces in "
                        "structuredContent['warnings'] when this "
                        "happens). "
                        "'semantic' — LLM rewrites your intent into "
                        "literal-tokens, then token-matches; one "
                        "cheap pre-call (~$0.001) buys vocabulary "
                        "expansion. Use when the user phrases the "
                        "question abstractly. "
                        "'entity' — typed-entity match (dates / "
                        "money / percents / durations / numbers) on "
                        "selector_arg."
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
                        "'entity' (the entity type name). Ignored "
                        "for 'every_sentence' and 'semantic' — "
                        "semantic mode derives its terms from the "
                        "question via the rewrite LLM."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="semantic_rewrite_model",
                    type="string",
                    description=(
                        "Model for the semantic-rewrite pre-call "
                        "(only used when select_by='semantic'). "
                        "Default 'anthropic:claude-haiku-4-5' — the "
                        "rewrite is a thin classifier; the spend per "
                        "call is ~$0.001."
                    ),
                    required=False,
                    default="anthropic:claude-haiku-4-5",
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
                ParameterSchema(
                    name="max_cost_usd",
                    type="number",
                    description=(
                        "Strict cost ceiling for the entire findings "
                        "run (semantic rewrite + filter chunks + "
                        "synthesis combined). When set, the tool "
                        "ABORTS BEFORE dispatching the next Phase-2 "
                        "filter wave once accumulated cost would "
                        "exceed this cap, and SKIPS Phase-3 synthesis "
                        "when there is no headroom for a synthesis "
                        "call. Returns structuredContent["
                        "'budget_exceeded']=true and "
                        "refusal_reason='budget_exceeded' when the "
                        "cap fires. Worst-case overshoot is one "
                        "filter wave's worth of in-flight cost "
                        "(typically <= num_parallel * per-chunk-cost) "
                        "— the cap is enforced at wave boundaries, "
                        "not mid-call. Returns a PARTIAL result "
                        "(surviving findings observed before the "
                        "cap fired); the response is still a "
                        "structured SUCCESS, not isError=true. "
                        "Set None (or omit) for no cap. The "
                        "headline 'cost_usd' field on the response "
                        "is the source of truth — compare against "
                        "this value, not the per-stage breakdown."
                    ),
                    required=False,
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
        max_cost_usd_raw = inputs.get("max_cost_usd")
        max_cost_usd: float | None = None if max_cost_usd_raw is None else float(max_cost_usd_raw)
        if max_cost_usd is not None and max_cost_usd <= 0.0:
            return ToolResult.create_error(
                f"max_cost_usd must be > 0 when set, got {max_cost_usd}. "
                "Pass null (or omit) to disable the cap."
            )
        filter_model = str(inputs.get("filter_model") or "anthropic:claude-haiku-4-5")
        synthesis_model = str(inputs.get("synthesis_model") or "anthropic:claude-sonnet-4-6")
        semantic_rewrite_model = str(
            inputs.get("semantic_rewrite_model") or "anthropic:claude-haiku-4-5"
        )

        # Sprint-2 #6 — semantic mode handles its own selector
        # construction because the term list comes from an async LLM
        # call. For every other mode the sync ``_build_selector``
        # path is unchanged.
        semantic_terms: tuple[str, ...] = ()
        semantic_cost: float = 0.0
        low_recall_arg: str | None = None
        if select_by == "semantic":
            from kaos_agents.patterns.findings import (
                expand_question_to_terms,
                sentences_with_any_token_selector,
            )

            try:
                semantic_terms, semantic_cost = await expand_question_to_terms(
                    str(question),
                    model=semantic_rewrite_model,
                )
            except Exception as exc:
                logger.exception("semantic rewrite failed")
                return ToolResult.create_error(
                    f"Semantic-rewrite LLM call failed: {exc}. "
                    "Verify ANTHROPIC_API_KEY is set, or fall back "
                    "to select_by='token' / 'every_sentence'."
                )
            selector = sentences_with_any_token_selector(semantic_terms)
            # Surface the expanded terms in the low-recall hint
            # context so a downstream zero-candidate refusal still
            # carries the term list for the audit trail.
            low_recall_arg = ",".join(semantic_terms) if semantic_terms else None
        else:
            # Validate selector_arg / build selector.
            try:
                selector = _build_selector(select_by, selector_arg)
            except ValueError as exc:
                return ToolResult.create_error(str(exc))
            if select_by == "token" and selector_arg:
                low_recall_arg = str(selector_arg)

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

        # Sprint-3 #9 — the agent-level cap covers filter + synthesis,
        # but the K7 tool may have ALREADY spent ``semantic_cost`` on
        # the rewrite pre-call before constructing the agent. Subtract
        # that out so the agent's accumulator is comparing apples to
        # apples (the agent only knows about filter + synthesis cost;
        # the tool surface adds the rewrite). If the rewrite already
        # blew the cap, refuse before launching any further work.
        agent_cap: float | None = None
        if max_cost_usd is not None:
            remaining = max_cost_usd - semantic_cost
            if remaining <= 0.0:
                return ToolResult.create_success(
                    output={
                        "artifact_id": artifact_id,
                        "question": question,
                        "select_by": select_by,
                        "selector_arg": selector_arg,
                        "answer": "",
                        "findings": [],
                        "total_enumerated": 0,
                        "total_filtered": 0,
                        "filter_calls": 0,
                        "filter_cost_usd": 0.0,
                        "synthesis_cost_usd": 0.0,
                        "semantic_rewrite_cost_usd": semantic_cost,
                        "total_cost_usd": semantic_cost,
                        "total_llm_calls": 1 if select_by == "semantic" else 0,
                        "semantic_terms": list(semantic_terms),
                        "warnings": [],
                        "budget_exceeded": True,
                        "refusal_reason": "budget_exceeded",
                        "refusal_message": (
                            f"Semantic rewrite already spent "
                            f"${semantic_cost:.4f} which meets or "
                            f"exceeds max_cost_usd=${max_cost_usd:.4f}. "
                            "No filter / synthesis budget remained. "
                            "Raise the cap or use a non-semantic mode."
                        ),
                    },
                    summary=(
                        f"Budget exceeded by semantic rewrite alone "
                        f"(${semantic_cost:.4f} >= cap=${max_cost_usd:.4f})"
                    ),
                )
            agent_cap = remaining

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
                low_recall_selector_arg=low_recall_arg,
                max_cost_usd=agent_cap,
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
        # Sprint-2 #6 — semantic mode adds the rewrite call's cost
        # to the headline ``total_cost_usd`` so the figure the caller
        # sees in summary == real spend. Per-stage costs stay
        # separated for accounting purposes.
        total_cost_with_rewrite = result.total_cost_usd + semantic_cost
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
            "semantic_rewrite_cost_usd": semantic_cost,
            "total_cost_usd": total_cost_with_rewrite,
            "total_llm_calls": (result.total_llm_calls + (1 if select_by == "semantic" else 0)),
            # Sprint-2 #6 — semantic-mode terms for audit / debug.
            # Empty list for non-semantic modes so the wire shape
            # stays consistent. Consumers can compare against the
            # selector_arg the user sent vs. the LLM's expansion.
            "semantic_terms": list(semantic_terms),
            # Sprint-2 #6 — structured warnings. Always present
            # (default empty list); the K7 agent's caller (a
            # higher-level agent or a UI) iterates this to surface
            # recall risk before treating an empty answer as a
            # synthesis failure. Each warning has stable ``kind``
            # string, human ``message``, and structured ``details``.
            "warnings": [
                {
                    "kind": w.kind,
                    "message": w.message,
                    "details": dict(w.details),
                }
                for w in result.warnings
            ],
            # Refusal contract — see metadata.description. ``None``
            # when synthesis produced a real answer; populated when
            # the agent honestly cannot answer from this document.
            # Downstream consumers (UI, audit, agent caller) must
            # branch on ``refusal_reason`` before treating
            # ``answer == ""`` as a failure. Three reasons total:
            # ``no_candidates_enumerated`` (Phase 1 emitted nothing),
            # ``no_relevant_candidates`` (Phase 2 culled everything),
            # ``budget_exceeded`` (Sprint-3 #9 — the cap fired).
            "refusal_reason": (result.refusal.reason if result.refusal is not None else None),
            "refusal_message": (result.refusal.message if result.refusal is not None else None),
            # Sprint-3 #9 — explicit boolean reflecting whether
            # max_cost_usd was hit. True when the cap fired in
            # Phase-2 (partial filter coverage) OR Phase-3 was
            # skipped (no synthesis headroom). False on the happy
            # path, including when no cap was configured. The
            # ``cost_usd`` headline field is the source of truth for
            # what was actually spent — clients verifying the
            # contract assert ``cost_usd <= max_cost_usd * (1 +
            # tolerance)``.
            "budget_exceeded": result.budget_exceeded,
            # Headline cost field — same value as ``total_cost_usd``
            # but with the canonical name the rest of the platform
            # uses. Sprint-3 #9 makes this the contract-checked
            # number; per-stage costs (filter / synthesis / rewrite)
            # are kept above for accounting transparency.
            "cost_usd": total_cost_with_rewrite,
        }
        if result.refusal is not None:
            summary = (
                f"Findings refusal ({result.refusal.reason}): "
                f"enumerated={result.total_enumerated} "
                f"filtered={result.total_filtered} "
                f"cost=${total_cost_with_rewrite:.4f}"
            )
            if result.budget_exceeded and max_cost_usd is not None:
                summary = f"{summary} cap=${max_cost_usd:.4f}"
        else:
            summary = (
                f"Findings: enumerated={result.total_enumerated} "
                f"filtered={result.total_filtered} "
                f"calls={result.total_llm_calls} "
                f"cost=${total_cost_with_rewrite:.4f}"
            )
            if result.budget_exceeded and max_cost_usd is not None:
                summary = f"[BudgetPartial cap=${max_cost_usd:.4f}] {summary}"
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

    Sprint-2 #6: ``semantic`` mode is intentionally NOT handled here
    because it requires an async LLM call to construct the term list.
    The execute() body builds the semantic selector itself before
    calling this function for the non-semantic modes.
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

    if select_by == "semantic":
        # The K7 execute() body handles semantic mode directly
        # because constructing the selector requires an async LLM
        # call. This branch exists only to give a clear error if
        # something ever invokes _build_selector with this mode.
        raise ValueError(
            "select_by='semantic' is handled by the execute() body, "
            "not _build_selector. This is a programming error — "
            "open a bug."
        )

    raise ValueError(f"Unknown select_by={select_by!r}")


__all__ = ["AgentFindingsTool"]
