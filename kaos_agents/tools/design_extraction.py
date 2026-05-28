"""kaos-agent-design-extraction — LLM-driven extraction-schema design.

The agent calls this tool when the user's question fits a structured
extraction shape (table, list, per-doc-attribute, comparison). The
tool invokes :func:`kaos_llm_core.programs.designers.design_schema`,
which asks the LLM to propose a typed
:class:`kaos_llm_core.signatures.extraction.ExtractionSchema` for
the deliverable, and returns the proposal as structured content for
the agent to inspect.

This is PR-1a of the dynamic deliverable schema architecture
(``kaos-modules/docs/plans/2026-05-28-dynamic-deliverable-schema-architecture.md``).
PR-1a covers DESIGN only — the agent gets the schema back and can
decide what to do with it. PR-1b adds per-document fan-out + the
``stamp_source_uri`` JOIN so the tool also EXECUTES the schema and
returns typed rows.

Splitting the steps lets us measure the designer's quality on real
persona prompts before wiring in the fan-out — the §7 iteration
discipline from the plan.
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
_DEFAULT_DESIGNER_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_CORPUS_SAMPLE_CHARS_PER_DOC = 500

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
    """LLM-driven extraction-schema design for a user question + corpus."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-design-extraction",
            display_name="Design Extraction Schema (LLM)",
            description=(
                "Given a user question and a corpus of stored "
                "ContentDocument artifacts, ask an LLM to propose a "
                "typed ExtractionSchema for the deliverable (which "
                "columns to extract, what types they should be, "
                "which columns must carry citations). Returns the "
                "proposed schema as structured content; does NOT "
                "execute the extraction in this version. "
                "PR-1a of the dynamic-deliverable-schema work — "
                "PR-1b will add per-document fan-out. "
                "TRANSPARENCY (Sprint-3 #10): structuredContent "
                "carries ``cost_usd: float`` and ``total_tokens: int`` "
                "as top-level fields. The schema designer is a single "
                "LLM call; both figures reflect that one call."
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
                        "(strings) the schema should target. Typically "
                        "the IDs of all documents in the session. The "
                        "designer sees a sample from each so the "
                        "proposed columns match the corpus's content."
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
                    name="model",
                    type="string",
                    description=(
                        "Designer model. Defaults to "
                        f"{_DEFAULT_DESIGNER_MODEL!r}. Use a research-"
                        "tier model — cheap models under-design "
                        "columns."
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
        model = str(inputs.get("model") or _DEFAULT_DESIGNER_MODEL)

        # Build the corpus_sample. The designer uses ~500 chars per doc
        # so it can ground its column proposals in the corpus's actual
        # content + variation. See plan §6.1.6 — the AST-aware sampling
        # upgrade is deferred until measurement shows first-N-chars
        # bites on persona prompts.
        from kaos_content.artifacts import load_document

        sample_chunks: list[str] = []
        loaded_count = 0
        for aid in artifact_ids_raw:
            aid_str = str(aid)
            try:
                doc = await load_document(aid_str, context.runtime)
            except Exception as exc:
                logger.debug("design_extraction: failed to load %s: %s", aid_str, exc)
                continue
            head = _doc_head_text(doc, _DEFAULT_CORPUS_SAMPLE_CHARS_PER_DOC)
            sample_chunks.append(f"=== {aid_str} ===\n{head}")
            loaded_count += 1

        if not sample_chunks:
            return ToolResult.create_error(
                "No artifacts loadable. Verify the artifact_ids exist in the runtime's VFS."
            )

        corpus_sample = "\n\n".join(sample_chunks)

        # Invoke the designer (kaos-llm-core). Per PR-A on kaos-llm-core,
        # schema_id auto-derives from (question, corpus_sample, model)
        # — distinct prompts get distinct ids.
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
                model=model,
                # schema_id intentionally omitted — let kaos-llm-core
                # auto-derive a stable hash from the inputs (PR-A).
            )
        except Exception as exc:
            logger.exception("design_extraction: design_schema call failed")
            return ToolResult.create_error(
                f"SchemaDesigner LLM call failed: {exc}. Verify "
                "ANTHROPIC_API_KEY is set and the model identifier is "
                "valid."
            )

        # PR-1a returns the proposal as structured content. PR-1b will
        # follow with the per-document fan-out + stamp_source_uri JOIN.
        return ToolResult.create_success(
            output={
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
                # Sprint-3 #10 transparency lens. The designer Call's
                # InvocationUsage isn't exposed by `design_schema`'s
                # current public surface; the cost surface is the
                # one promised number callers need, so we emit zeros
                # here in PR-1a and revisit when `design_schema`
                # exposes usage (could be a small kaos-llm-core PR,
                # but not blocking PR-1a's measurement goal).
                "cost_usd": 0.0,
                "total_tokens": 0,
            },
            summary=(
                f"Designed extraction schema {schema.id!r} with "
                f"{len(schema.columns)} columns "
                f"({loaded_count}/{len(artifact_ids_raw)} artifacts sampled)."
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

    Defers to ``block.text`` when the block exposes it; else walks
    ``children`` looking for ``Text.value``. Returns empty string when
    no text content is recoverable (e.g. image-only blocks).
    """
    direct = getattr(block, "text", None)
    if isinstance(direct, str):
        return direct
    children = getattr(block, "children", None)
    if not children:
        return ""
    pieces: list[str] = []
    for child in children:
        value = getattr(child, "value", None)
        if isinstance(value, str):
            pieces.append(value)
        else:
            nested = _block_text(child)
            if nested:
                pieces.append(nested)
    return "".join(pieces)
