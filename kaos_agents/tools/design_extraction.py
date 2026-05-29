"""kaos-agent-design-extraction — LLM-driven extraction schema + fan-out.

The agent calls this tool when the user's question fits a structured
extraction shape (table, list, per-doc-attribute, comparison). The
tool runs the full design-and-execute pipeline:

1. Invoke :func:`kaos_llm_core.programs.designers.design_schema` to
   ask the LLM to propose a typed
   :class:`kaos_llm_core.signatures.extraction.ExtractionSchema` for
   the deliverable (which columns, what types, which carry citations).
2. Compile a runtime ``Extract_<schema_id>`` Signature via
   :meth:`ExtractionSchema.to_signature` with ``provenance="cited"``.
3. Fan out per-document ``Call(Extract_<schema_id>).invoke(...)``
   in parallel under a concurrency cap.
4. Apply :func:`kaos_llm_core.signatures.stamp_source_uri` to every
   non-null cell so the dispatcher (this tool body) — NOT the LLM —
   owns the ``source_uri`` field on each :class:`Cited` value.
5. Return typed rows + the schema + aggregated cost in
   ``structuredContent`` for the agent to render.

PR-1a shipped steps 1 only; PR-1b adds 2-5 — closing the loop from
"user question" to "typed cited rows" via the dynamic deliverable
schema architecture
(``kaos-modules/docs/plans/2026-05-28-dynamic-deliverable-schema-architecture.md``).

`source_uri` is dispatcher-owned per the plan's pre-mortem §3.6 +
cited Instructor/Pydantic literature: never trust LLM-emitted
citation identifiers. The ``stamp_source_uri`` helper closes the
fabrication attack surface by overriding every span post-extraction
with the authoritative ``artifact_id`` the dispatcher passed to the
underlying Call.
"""

from __future__ import annotations

import asyncio
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
_DEFAULT_DESIGNER_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_EXTRACT_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_CORPUS_SAMPLE_CHARS_PER_DOC = 500
_DEFAULT_MAX_DOC_CHARS = 200_000  # safety cap on per-doc source_text
_DEFAULT_FAN_OUT_CONCURRENCY = 5

# This tool spends money on a single LLM Call (the designer). Mirror
# the corpus-filter convention: readOnlyHint=False (LLM spend) +
# destructiveHint=False (no writes) so Claude Code asks before each
# invocation.
_DESIGN_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,  # cost-incurring; designers re-run cost real money
    openWorldHint=False,
)


class AgentDesignExtractionTool(KaosTool):
    """LLM-driven extraction-schema design + per-document fan-out."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-design-extraction",
            display_name="Design + Execute Extraction Schema (LLM)",
            description=(
                "Extract typed rows from N stored documents in "
                "parallel — one row per document, columns proposed by "
                "an LLM from the question + corpus. Returns the typed "
                "schema + cited rows (no synthesis on top); use "
                "kaos-agent-interpret-extraction when you also want "
                "the user-facing memo rendered from the rows. Pick "
                "this when you need just the structured data (CSV "
                "export, programmatic downstream consumer, schema "
                "inspection). Every cell carries the source "
                "artifact_id — the LLM cannot fabricate citation "
                "identities."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_DESIGN_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="question",
                    type="string",
                    description=(
                        "The user's review question or objective, "
                        "phrased as you would describe it to a human "
                        "subject-matter expert."
                    ),
                ),
                ParameterSchema(
                    name="artifact_ids",
                    type="array",
                    description=(
                        "List of stored ContentDocument artifact IDs "
                        "(strings). Both the designer (which sees a "
                        "head sample) and the per-document extractor "
                        "(which sees full text up to "
                        f"{_DEFAULT_MAX_DOC_CHARS} chars) operate "
                        "against this list. Typically the IDs of all "
                        "documents in the session."
                    ),
                ),
                ParameterSchema(
                    name="domain_hint",
                    type="string",
                    description=(
                        "Optional one-line domain context, e.g. "
                        "'mutual NDAs', 'commercial real estate "
                        "leases'. Empty string is the safe default."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="designer_model",
                    type="string",
                    description=(
                        "Model for the schema designer call. Defaults "
                        f"to {_DEFAULT_DESIGNER_MODEL!r}. Use a "
                        "research-tier model — cheap models "
                        "under-design columns."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="extract_model",
                    type="string",
                    description=(
                        "Model for the per-document extract calls. "
                        f"Defaults to {_DEFAULT_EXTRACT_MODEL!r}. "
                        "Same research-tier minimum as the designer."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="design_only",
                    type="boolean",
                    description=(
                        "When true, only the designer runs and the "
                        "fan-out is skipped. Useful for inspecting "
                        "what schema the LLM would propose without "
                        "paying for per-document extraction. Default "
                        "false (full design + fan-out)."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="max_concurrency",
                    type="integer",
                    description=(
                        "Cap on simultaneous per-document extract "
                        f"calls. Default {_DEFAULT_FAN_OUT_CONCURRENCY}. "
                        "Tune up for small corpora on rate-limit-"
                        "tolerant providers."
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

        artifact_ids_raw = inputs.get("artifact_ids")
        if not artifact_ids_raw or not isinstance(artifact_ids_raw, list):
            return ToolResult.create_error(
                "Missing 'artifact_ids' (list of strings). Pass the IDs "
                "of the documents the schema should target."
            )

        domain_hint = str(inputs.get("domain_hint") or "")
        # Accept both ``designer_model`` (PR-1b new) and ``model`` (PR-1a
        # back-compat) so a caller upgrading from PR-1a keeps working.
        designer_model = str(
            inputs.get("designer_model") or inputs.get("model") or _DEFAULT_DESIGNER_MODEL
        )
        extract_model = str(inputs.get("extract_model") or _DEFAULT_EXTRACT_MODEL)
        design_only = bool(inputs.get("design_only") or False)
        max_concurrency_raw = inputs.get("max_concurrency")
        max_concurrency = (
            _DEFAULT_FAN_OUT_CONCURRENCY
            if max_concurrency_raw is None
            else max(1, int(max_concurrency_raw))
        )

        # Phase 1 — load corpus, build head sample for the designer AND
        # keep a per-doc full-text map for the fan-out extractor.
        from kaos_content.artifacts import load_document

        sample_chunks: list[str] = []
        per_doc_full_text: list[tuple[str, str]] = []  # (artifact_id, source_text)
        for aid in artifact_ids_raw:
            aid_str = str(aid)
            try:
                doc = await load_document(aid_str, context.runtime)
            except Exception as exc:
                logger.debug("design_extraction: failed to load %s: %s", aid_str, exc)
                continue
            head = _doc_head_text(doc, _DEFAULT_CORPUS_SAMPLE_CHARS_PER_DOC)
            sample_chunks.append(f"=== {aid_str} ===\n{head}")
            full_text = _doc_head_text(doc, _DEFAULT_MAX_DOC_CHARS)
            per_doc_full_text.append((aid_str, full_text))

        if not sample_chunks:
            return ToolResult.create_error(
                "No artifacts loadable. Verify the artifact_ids exist in the runtime's VFS."
            )

        corpus_sample = "\n\n".join(sample_chunks)
        loaded_count = len(per_doc_full_text)

        # Phase 2 — designer call. schema_id auto-derives per kaos-llm-core PR-A.
        try:
            from kaos_llm_core.programs.designers import design_schema
        except ImportError:
            return ToolResult.create_error(
                "kaos-llm-core is not installed. Install the "
                "[llm] extra: `uv sync --group dev --extra llm`."
            )

        try:
            schema = await design_schema(
                question=str(question),
                corpus_sample=corpus_sample,
                domain_hint=domain_hint,
                model=designer_model,
            )
        except Exception as exc:
            logger.exception("design_extraction: design_schema call failed")
            return ToolResult.create_error(
                f"SchemaDesigner LLM call failed: {exc}. Verify "
                "ANTHROPIC_API_KEY is set and the model identifier is "
                "valid."
            )

        # Sprint-3 #10 — designer cost. `design_schema` currently does
        # not surface its `InvocationUsage`; track as zero for now and
        # promote to a real number when a small kaos-llm-core follow-up
        # exposes the usage. The per-doc extract cost (below) IS
        # tracked and dominates total spend.
        designer_cost_usd = 0.0
        designer_tokens = 0

        common_schema_payload: dict[str, Any] = {
            "schema_id": schema.id,
            "schema_version": schema.version,
            "columns": [
                {
                    "id": col.id,
                    "label": col.label,
                    "column_type": col.column_type,
                    "description": col.description,
                    "required": col.required,
                }
                for col in schema.columns
            ],
            "artifacts_sampled": loaded_count,
            "artifacts_requested": len(artifact_ids_raw),
            "designer_cost_usd": designer_cost_usd,
        }

        if design_only:
            # PR-1a parity for callers that just want to inspect the
            # designer's proposal without paying for extraction.
            return ToolResult.create_success(
                output={
                    **common_schema_payload,
                    "rows": [],
                    "execution_mode": "design_only",
                    "extraction_cost_usd": 0.0,
                    "cost_usd": designer_cost_usd,
                    "total_tokens": designer_tokens,
                },
                summary=(
                    f"Designed extraction schema {schema.id!r} with "
                    f"{len(schema.columns)} columns "
                    f"(design_only=true; fan-out skipped)."
                ),
            )

        # Phase 3 — compile runtime Extract Signature.
        try:
            extract_sig = schema.to_signature(provenance="cited")
        except Exception as exc:
            logger.exception("design_extraction: schema.to_signature failed")
            return ToolResult.create_error(
                f"Failed to compile runtime Signature from schema: {exc}. "
                "The designer produced a schema kaos-llm-core could not "
                "compile; report this with the proposed columns for "
                "investigation."
            )

        # Phase 4 — fan-out per document under a concurrency cap.
        try:
            from kaos_llm_core.programs.call import Call
            from kaos_llm_core.signatures.grounding import Cited, stamp_source_uri
        except ImportError:
            return ToolResult.create_error(
                "kaos-llm-core is not installed. Install the [llm] extra."
            )

        extract_call = Call(extract_sig, model=extract_model)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_one(aid: str, source_text: str) -> tuple[str, Any]:
            async with semaphore:
                try:
                    invocation = await extract_call.invoke(source_text=source_text)
                    return aid, invocation
                except Exception as exc:
                    logger.warning(
                        "design_extraction: extract failed for %s: %s",
                        aid,
                        exc,
                    )
                    return aid, exc

        extract_results = await asyncio.gather(
            *(_run_one(aid, text) for aid, text in per_doc_full_text),
            return_exceptions=False,
        )

        # Phase 5 — stamp source_uri on every Cited cell (dispatcher
        # owns the citation identity; LLM-emitted source_uri is
        # ignored). Build serializable row dicts for structuredContent.
        rows: list[dict[str, Any]] = []
        extraction_cost_usd = 0.0
        extraction_tokens = 0
        null_cell_count = 0
        failed_doc_count = 0

        for aid, result in extract_results:
            if isinstance(result, Exception):
                failed_doc_count += 1
                null_cell_count += len(schema.columns)
                rows.append(
                    {
                        "artifact_id": aid,
                        "error": str(result)[:240],
                        "cells": {col.id: None for col in schema.columns},
                    }
                )
                continue

            usage = getattr(result, "usage", None)
            if usage is not None:
                extraction_cost_usd += float(getattr(usage, "cost_usd", 0.0) or 0.0)
                extraction_tokens += int(getattr(usage, "total_tokens", 0) or 0)

            output = getattr(result, "output", None)
            cells: dict[str, Any] = {}
            for col in schema.columns:
                raw_value = getattr(output, col.id, None) if output is not None else None
                if raw_value is None:
                    cells[col.id] = None
                    null_cell_count += 1
                    continue
                if isinstance(raw_value, Cited):
                    stamped = stamp_source_uri(raw_value, source_uri=aid)
                    cells[col.id] = stamped.model_dump(mode="json")
                else:
                    # Non-Cited path (provenance != "cited" somehow, or
                    # the runtime Signature emitted something else).
                    # Surface as-is via model_dump if possible.
                    dump = getattr(raw_value, "model_dump", None)
                    cells[col.id] = dump(mode="json") if callable(dump) else str(raw_value)
            rows.append({"artifact_id": aid, "cells": cells})

        total_cost_usd = designer_cost_usd + extraction_cost_usd
        total_tokens = designer_tokens + extraction_tokens

        return ToolResult.create_success(
            output={
                **common_schema_payload,
                "rows": rows,
                "execution_mode": "full",
                "row_count": len(rows),
                "failed_doc_count": failed_doc_count,
                "null_cell_count": null_cell_count,
                "extraction_cost_usd": extraction_cost_usd,
                "cost_usd": total_cost_usd,
                "total_tokens": total_tokens,
            },
            summary=(
                f"Designed schema {schema.id!r} ({len(schema.columns)} "
                f"columns); extracted {len(rows)} rows from "
                f"{loaded_count} artifacts "
                f"({null_cell_count} null cells, "
                f"{failed_doc_count} failed docs). Total spend "
                f"${total_cost_usd:.4f}."
            ),
        )


def _doc_head_text(doc: Any, max_chars: int) -> str:
    """Return up to ``max_chars`` characters of the document's head text.

    Walks the document body in order, concatenating each block's plain-
    text rendering, and truncates at ``max_chars``. Used as the per-doc
    sample fed to the SchemaDesigner. Plain-text fallback rather than
    AST-aware sampling — the latter is deferred per plan §6.1.6 until
    measurement shows first-N-chars bites on persona prompts.
    """
    parts: list[str] = []
    remaining = max_chars
    body = getattr(doc, "body", ()) or ()
    for block in body:
        text = _block_text(block)
        if not text:
            continue
        if len(text) >= remaining:
            parts.append(text[:remaining])
            break
        parts.append(text)
        remaining -= len(text) + 1  # +1 for the joining newline
        if remaining <= 0:
            break
    return "\n".join(parts)


def _block_text(block: Any) -> str:
    """Best-effort plain-text rendering of one block.

    Prepends ``numbering_label`` when present so the LLM sees the
    visible section numeral (e.g. ``"12. GOVERNING LAW…"``) instead of
    just the heading text. kaos-office's default text serialization
    strips the numeral; for typed-extraction prompts that ask for
    "EXACT section number", this prefix is the difference between
    "GOVERNING LAW" (heading) and "12." (number).

    Defers to ``block.text`` when the block exposes it; else walks
    ``children`` looking for ``Text.value``. Returns empty string when
    no text content is recoverable (e.g. image-only blocks).
    """
    label = getattr(block, "numbering_label", None)
    prefix = f"{label} " if isinstance(label, str) and label else ""
    direct = getattr(block, "text", None)
    if isinstance(direct, str):
        return prefix + direct
    children = getattr(block, "children", None)
    if not children:
        return prefix
    pieces: list[str] = []
    for child in children:
        value = getattr(child, "value", None)
        if isinstance(value, str):
            pieces.append(value)
        else:
            nested = _block_text(child)
            if nested:
                pieces.append(nested)
    rendered = "".join(pieces)
    if rendered or prefix:
        return prefix + rendered
    return ""
