"""kaos-agent-interpret-extraction — iterative Extract↔Synthesize loop.

The user-facing endpoint of the dynamic deliverable-schema architecture.
Composes :class:`AgentDesignExtractionTool` (which produces typed
grounded rows) with
:class:`~kaos_llm_core.programs.interpret_extraction.InterpretExtractionSignature`
(which reformulates those rows into the deliverable shape the user
asked for), and loops the two:

    loop iter in 1..max_iters:
        extracted = design_extraction(schema)            # typed fan-out
        memo, needs_more, requested_cols = synthesize(...)
        if not needs_more or budget_exhausted: break
        schema += requested_cols                         # augment
    return memo + cumulative typed rows + iteration trace

The synthesizer is bounded by the typed extraction — it cannot
fabricate facts that aren't in the rows. When the rows are
insufficient, it requests specific column ids and the loop augments
the schema for the next extraction round. This is the ReAct pattern
applied to dynamic-schema extraction.

Returns the final memo (the user-facing deliverable) plus the
cumulative typed extraction (for the Citations panel / downstream
verification) plus a per-iteration trace (for cost auditing + UX
display of "the agent decided to look for X next").
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from kaos_core.base.tool import KaosTool
from kaos_core.logging import get_logger
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.metadata import ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

from kaos_agents.tools.design_extraction import AgentDesignExtractionTool

if TYPE_CHECKING:
    from kaos_core.base.context import KaosContext

logger = get_logger(__name__)

_MODULE = "kaos-agents"
_VERSION = "0.1.0"
_DEFAULT_SYNTH_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_MAX_ITERS = 3
_DEFAULT_BUDGET_USD = 1.50

# Cost-incurring (extraction + synthesis LLM calls), no writes.
_INTERPRET_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)


class AgentInterpretExtractionTool(KaosTool):
    """Iterative Extract↔Synthesize — the user-shaped deliverable tool."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-interpret-extraction",
            display_name="Interpret Extraction (Iterative)",
            description=(
                "Build a per-document typed table from a corpus, then "
                "render it as the deliverable the user asked for. "
                "Pick this when the question shape is one row per "
                "document with named columns: comparison tables, "
                "side-by-side reviews, batch attribute extraction, "
                "CSV-ready outputs, exec summaries over N agreements. "
                "Cheaper and more reliable than chaining "
                "search-document + context-window N times for the same "
                "shape. Every cell carries the source artifact_id; the "
                "synthesizer cannot fabricate facts outside the "
                "extracted cells."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_INTERPRET_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="question",
                    type="string",
                    description=(
                        "The user's question / objective, phrased as you "
                        "would describe it to a human subject-matter expert."
                    ),
                ),
                ParameterSchema(
                    name="artifact_ids",
                    type="array",
                    description=(
                        "List of stored ContentDocument artifact IDs "
                        "(strings). The designer and per-doc extractor "
                        "operate against this list."
                    ),
                ),
                ParameterSchema(
                    name="domain_hint",
                    type="string",
                    description=(
                        "Optional one-line domain context (e.g. "
                        "'mutual NDAs', 'commercial leases'). "
                        "Empty string is the safe default."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="deliverable_hint",
                    type="string",
                    description=(
                        "Optional one-line hint about the deliverable "
                        "shape, e.g. 'one-page exec summary for non-"
                        "lawyer CEO', 'CSV-ready table', 'compare A "
                        "and B'. Empty string is the safe default; "
                        "the synthesizer will infer shape from the "
                        "question."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="max_iters",
                    type="integer",
                    description=(
                        f"Maximum loop iterations. Default {_DEFAULT_MAX_ITERS}. "
                        "Convergence (needs_more=false) typically "
                        "happens at iter 1 or 2 when the initial "
                        "designer schema is comprehensive; harder "
                        "questions may use the full budget."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="budget_usd",
                    type="number",
                    description=(
                        f"Hard cost cap in USD. Default {_DEFAULT_BUDGET_USD:.2f}. "
                        "The loop breaks before the next iteration "
                        "when cumulative cost reaches this cap."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="synth_model",
                    type="string",
                    description=(
                        f"Model for the synthesizer call. Default {_DEFAULT_SYNTH_MODEL!r}."
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

        question = inputs.get("question")
        if not question or not str(question).strip():
            return ToolResult.create_error(
                "Missing 'question'. Provide the user's review objective."
            )
        artifact_ids = inputs.get("artifact_ids")
        if not artifact_ids or not isinstance(artifact_ids, list):
            return ToolResult.create_error("Missing 'artifact_ids' (list of strings).")

        domain_hint = str(inputs.get("domain_hint") or "")
        deliverable_hint = str(inputs.get("deliverable_hint") or "")
        max_iters = max(1, int(inputs.get("max_iters") or _DEFAULT_MAX_ITERS))
        budget_usd = float(inputs.get("budget_usd") or _DEFAULT_BUDGET_USD)
        synth_model = str(inputs.get("synth_model") or _DEFAULT_SYNTH_MODEL)

        try:
            from kaos_llm_core.programs.call import Call
            from kaos_llm_core.programs.interpret_extraction import (
                InterpretExtractionSignature,
            )
        except ImportError:
            return ToolResult.create_error(
                "kaos-llm-core>=0.1.6 not installed. Install the "
                "[llm] extra: `uv sync --group dev --extra llm`."
            )

        design_tool = AgentDesignExtractionTool()

        cumulative_rows: dict[str, dict[str, Any]] = {}
        cumulative_cols: dict[str, dict[str, Any]] = {}
        extract_cost_total = 0.0
        synth_cost_total = 0.0
        iteration_trace: list[dict[str, Any]] = []
        last_memo = ""
        last_score = 0
        last_needs_more = False
        last_requested: tuple[str, ...] = ()
        converged_at: int | None = None
        budget_exhausted = False

        for it in range(1, max_iters + 1):
            # Augment the question for iter > 1 with the synthesizer's
            # requested column proposals.
            augmenting_hint = ""
            if it > 1 and last_requested:
                augmenting_hint = "; ".join(last_requested)
            extract_question = str(question)
            if augmenting_hint:
                extract_question = (
                    f"{question}\n\n(Iteration {it} follow-up — focus the "
                    f"schema on these specific column proposals: "
                    f"{augmenting_hint})"
                )

            # 1) Extract
            extract_inputs: dict[str, Any] = {
                "question": extract_question,
                "artifact_ids": artifact_ids,
                "domain_hint": domain_hint,
            }
            extract_result = await design_tool.execute(extract_inputs, context=context)
            if extract_result.isError:
                detail = _result_text(extract_result)
                if it == 1:
                    return ToolResult.create_error(
                        f"design_extraction failed on first iteration: {detail}"
                    )
                # iter > 1: prior memo still useful; break with what we have
                iteration_trace.append(
                    {
                        "iter": it,
                        "stage": "extract",
                        "error": detail,
                    }
                )
                break
            extract_sc = extract_result.structuredContent or {}
            extract_cost = float(extract_sc.get("cost_usd") or 0.0)
            extract_cost_total += extract_cost

            # Merge into cumulative state
            for col in extract_sc.get("columns") or []:
                cumulative_cols[col["id"]] = col
            for row in extract_sc.get("rows") or []:
                aid = row.get("artifact_id")
                if not aid:
                    continue
                prior = cumulative_rows.get(aid) or {"artifact_id": aid, "cells": {}}
                prior_cells = prior.get("cells") or {}
                new_cells = row.get("cells") or {}
                prior["cells"] = {**prior_cells, **new_cells}
                cumulative_rows[aid] = prior

            # 2) Synthesize
            rows_json = _project_rows_for_synth(cumulative_rows, cumulative_cols)
            try:
                inv = await Call(InterpretExtractionSignature, model=synth_model).invoke(
                    user_question=str(question),
                    extracted_rows=rows_json,
                    deliverable_hint=deliverable_hint,
                    iteration=it,
                )
            except Exception as exc:
                logger.exception("interpret_extraction: synthesizer Call failed")
                return ToolResult.create_error(
                    f"InterpretExtraction synthesizer Call failed at iter {it}: {exc}"
                )
            synth_cost = float(inv.usage.cost_usd) if inv.usage else 0.0
            synth_cost_total += synth_cost
            out = inv.output
            # Use getattr so ty doesn't complain about the dynamically-
            # built output model; field defaults guarantee presence.
            last_memo = str(getattr(out, "memo", "") or "")
            last_score = int(getattr(out, "score", 7) or 7)
            last_needs_more = bool(getattr(out, "needs_more_extraction", False))
            last_requested = tuple(getattr(out, "requested_columns", ()) or ())

            iteration_trace.append(
                {
                    "iter": it,
                    "extract_cost_usd": extract_cost,
                    "synth_cost_usd": synth_cost,
                    "score": last_score,
                    "needs_more_extraction": last_needs_more,
                    "requested_columns": list(last_requested),
                    "cumulative_cols": len(cumulative_cols),
                    "cumulative_rows": len(cumulative_rows),
                }
            )

            if not last_needs_more:
                converged_at = it
                break
            total_so_far = extract_cost_total + synth_cost_total
            if total_so_far >= budget_usd:
                budget_exhausted = True
                logger.debug(
                    "interpret_extraction: budget cap $%.2f reached at iter %d "
                    "(spent $%.4f); stopping loop with current memo.",
                    budget_usd,
                    it,
                    total_so_far,
                )
                break

        total_cost = extract_cost_total + synth_cost_total
        merged_extract = {
            "columns": list(cumulative_cols.values()),
            "rows": list(cumulative_rows.values()),
            "row_count": len(cumulative_rows),
            "col_count": len(cumulative_cols),
        }
        loop_status = (
            "converged"
            if converged_at is not None
            else ("budget_exhausted" if budget_exhausted else "max_iters_reached")
        )

        return ToolResult.create_success(
            output={
                "memo": last_memo,
                "score": last_score,
                "loop_status": loop_status,
                "converged_at_iter": converged_at,
                "iterations_run": len(iteration_trace),
                "iteration_trace": iteration_trace,
                "extract_cost_usd": extract_cost_total,
                "synth_cost_usd": synth_cost_total,
                "cost_usd": total_cost,
                "total_tokens": 0,  # populated when extractor + synth surface tokens
                "extracted": merged_extract,
            },
            summary=last_memo or "_(empty memo — see iteration_trace)_",
        )


def _result_text(result: ToolResult) -> str:
    """Lift first content item's text payload off a ToolResult."""
    if not result.content:
        return ""
    return str(getattr(result.content[0], "text", "") or "")


def _project_rows_for_synth(
    cumulative_rows: dict[str, dict[str, Any]],
    cumulative_cols: dict[str, dict[str, Any]],
) -> str:
    """Compact JSON projection of cumulative rows for the synthesizer.

    Drops span quote text — the synthesizer only needs values + document
    identity. Citation provenance (with full span data) stays in the
    tool's structuredContent.extracted field for downstream renderers.
    """
    rows_out: list[dict[str, Any]] = []
    for row in cumulative_rows.values():
        cells_out: dict[str, Any] = {}
        for col_id, cell in (row.get("cells") or {}).items():
            v = cell.get("value") if isinstance(cell, dict) else cell
            cells_out[col_id] = v
        rows_out.append(
            {
                "document": row.get("artifact_id"),
                "cells": cells_out,
            }
        )
    return json.dumps(
        {
            "columns": [
                {"id": c["id"], "description": c.get("description", "")}
                for c in cumulative_cols.values()
            ],
            "rows": rows_out,
        },
        ensure_ascii=False,
    )
