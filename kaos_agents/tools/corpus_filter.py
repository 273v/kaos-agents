"""K8 — kaos-agent-corpus-filter LLM-aided scope tightener.

Implements bonus pattern #1 from
``docs/design/findings-entities-summary.md``. Composes with K4's
``kaos-content-corpus-narrow`` (BM25 over summaries) as a two-stage
funnel:

- Stage 1 (cheap, deterministic): ``kaos-content-corpus-narrow`` ranks
  10,000 artifacts down to ~100 via BM25 over their summaries.
- Stage 2 (this tool, single LLM call): an LLM classifier prunes
  100 → 10-20 by reading the same summaries and reasoning about
  relevance to the user's intent.

One LLM call total — not one per artifact. The call sees the
concatenated artifact summaries (head_tokens + top_ngrams +
entity_counts), not the full bodies, so the prompt stays under a few
thousand tokens even at hundreds of artifacts.

This tool lives in kaos-agents because:
- It depends on kaos-llm-core (which kaos-content does not).
- It's an agent capability, not a document model primitive.
- The kaos-content tools register surface is read-only;
  kaos-agents/tools/registry already has the LLM-spending tools
  (chat, plan, findings, ...).
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

_FILTER_ANNOTATIONS = ToolAnnotations(
    # This tool spends money on an LLM call. readOnlyHint=False
    # prevents Claude Code from auto-approving — the user must
    # explicitly approve each filter run.
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)


class AgentCorpusFilterTool(KaosTool):
    """LLM-aided precision filter for a triaged corpus."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-agent-corpus-filter",
            display_name="Filter Corpus by Intent (LLM-aided)",
            description=(
                "Given a user intent and a candidate corpus of stored "
                "ContentDocument artifacts, return the subset most "
                "relevant to the intent via a single LLM "
                "classification call. Two-stage workflow: use "
                "kaos-content-corpus-narrow first (cheap BM25 over "
                "summaries to triage 10,000 → 100), then this tool "
                "(precision LLM filter, 100 → 10-20). The LLM "
                "receives only each artifact's summary "
                "(head_tokens + top_ngrams + entity_counts), not the "
                "full body — keeps the prompt under a few thousand "
                "tokens even for hundreds of artifacts."
            ),
            category=ToolCategory.DOCUMENT,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_FILTER_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="intent",
                    type="string",
                    description=(
                        "The user's intent / question / goal. Phrased "
                        "as you would describe to a human triage "
                        "reviewer."
                    ),
                ),
                ParameterSchema(
                    name="artifact_ids",
                    type="array",
                    description=(
                        "Candidate artifact IDs to filter. Typically "
                        "the output of kaos-content-corpus-narrow."
                    ),
                ),
                ParameterSchema(
                    name="max_keep",
                    type="integer",
                    description=(
                        "Soft cap on number of artifacts to keep. The "
                        "LLM is asked to prioritise but may keep "
                        "fewer. Default 20."
                    ),
                    required=False,
                    default=20,
                ),
                ParameterSchema(
                    name="model",
                    type="string",
                    description=(
                        "Model for the filter call. Default 'anthropic:claude-haiku-4-5'."
                    ),
                    required=False,
                    default="anthropic:claude-haiku-4-5",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        if context is None or context.runtime is None:
            return ToolResult.create_error("No runtime context. Register tools with a KaosRuntime.")

        intent = inputs.get("intent")
        if not intent or not str(intent).strip():
            return ToolResult.create_error("Missing 'intent'. Provide a non-empty intent string.")

        raw_ids = inputs.get("artifact_ids")
        if not raw_ids or not isinstance(raw_ids, list):
            return ToolResult.create_error("Missing 'artifact_ids' (list of strings) to filter.")

        max_keep_raw = inputs.get("max_keep")
        max_keep = 20 if max_keep_raw is None else int(max_keep_raw)
        if max_keep < 1:
            return ToolResult.create_error("'max_keep' must be >= 1.")

        model = str(inputs.get("model") or "anthropic:claude-haiku-4-5")

        # Load each artifact, build (or reuse) its summary, render a
        # compact description for the LLM.
        from kaos_content.artifacts import load_document
        from kaos_content.summarize import build_document_summary

        rendered_artifacts: list[dict[str, Any]] = []
        for aid in raw_ids:
            aid_str = str(aid)
            try:
                doc = await load_document(aid_str, context.runtime)
            except Exception:
                continue
            summary = doc.summary
            if summary is None:
                try:
                    summary = build_document_summary(doc)
                except Exception:
                    continue
            rendered_artifacts.append(
                {
                    "id": aid_str,
                    "head": summary.head_tokens[:300],
                    "top_ngrams": [ng.ngram for ng in summary.top_ngrams[:10]],
                    "entities": dict(summary.entity_counts),
                }
            )

        if not rendered_artifacts:
            return ToolResult.create_success(
                output={
                    "intent": intent,
                    "kept": [],
                    "dropped": [],
                    "total_input": len(raw_ids),
                    "total_loadable": 0,
                    "cost_usd": 0.0,
                },
                summary="No artifacts loadable; nothing to filter.",
            )

        try:
            kept, dropped, cost = await _run_corpus_filter_llm(
                intent=str(intent),
                artifacts=rendered_artifacts,
                max_keep=max_keep,
                model=model,
            )
        except Exception as exc:
            logger.exception("corpus filter LLM call failed")
            return ToolResult.create_error(
                f"Filter LLM call failed: {exc}. Fall back to "
                "kaos-content-corpus-narrow which is BM25-only."
            )

        output = {
            "intent": intent,
            "kept": kept,
            "dropped": dropped,
            "total_input": len(raw_ids),
            "total_loadable": len(rendered_artifacts),
            "cost_usd": cost,
        }
        summary_text = (
            f"Kept {len(kept)} of {len(rendered_artifacts)} loadable "
            f"artifacts (max_keep={max_keep}), cost=${cost:.4f}"
        )
        return ToolResult.create_success(output=output, summary=summary_text)


async def _run_corpus_filter_llm(
    *,
    intent: str,
    artifacts: list[dict[str, Any]],
    max_keep: int,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Run the single LLM filter call. Returns (kept, dropped, cost_usd).

    Defensive about LLM output: validates artifact_id round-trip,
    clamps relevance to [0, 1], enforces max_keep as a soft cap
    server-side, drops hallucinated entries silently.
    """
    from kaos_agents._llm_imports import require_llm_core

    require_llm_core()
    from kaos_llm_core.programs.call import Call
    from kaos_llm_core.signatures.fields import InputField, OutputField
    from kaos_llm_core.signatures.signature import Signature

    class _CorpusFilterSig(Signature):
        """Decide which artifacts are relevant to the intent.

        For each artifact, decide whether it is relevant to the
        user's intent. Be specific about why (or why not) — the
        reasoning surfaces to the user.
        """

        intent: str = InputField(description="The user's intent / goal.")
        max_keep: int = InputField(
            description="Soft cap on number to keep. May return fewer.",
        )
        artifacts: str = InputField(
            description=(
                "Newline-separated artifact summaries. Each line is "
                "``<artifact_id>: head=<head_snippet> "
                "ngrams=<top_ngrams> entities=<entity_counts>``."
            ),
        )
        kept: list[dict] = OutputField(
            description=(
                "Artifacts kept as relevant. Each item: "
                "``{'artifact_id': str, 'relevance': float 0..1, "
                "'reasoning': str}``."
            ),
        )
        dropped: list[dict] = OutputField(
            description=(
                "Artifacts dropped as irrelevant. Each item: "
                "``{'artifact_id': str, 'reason': str}``."
            ),
        )

    rendered = "\n".join(
        f"{a['id']}: head={a['head']!r} ngrams={a['top_ngrams']} entities={a['entities']}"
        for a in artifacts
    )
    call = Call(_CorpusFilterSig, model=model)
    invocation = await call.invoke(intent=intent, max_keep=max_keep, artifacts=rendered)
    output = invocation.output
    cost = float(getattr(invocation.usage, "cost_usd", 0.0) or 0.0)

    valid_ids = {a["id"] for a in artifacts}
    kept_clean: list[dict[str, Any]] = []
    for entry in output.kept:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("artifact_id") or "").strip()
        if aid not in valid_ids:
            continue
        try:
            relevance = float(entry.get("relevance", 0.0))
        except (TypeError, ValueError):
            continue
        relevance = max(0.0, min(1.0, relevance))
        kept_clean.append(
            {
                "artifact_id": aid,
                "relevance": relevance,
                "reasoning": str(entry.get("reasoning") or "").strip(),
            }
        )

    dropped_clean: list[dict[str, Any]] = []
    for entry in output.dropped:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("artifact_id") or "").strip()
        if aid not in valid_ids:
            continue
        dropped_clean.append(
            {
                "artifact_id": aid,
                "reason": str(entry.get("reason") or "").strip(),
            }
        )

    # Enforce the soft cap by moving lowest-relevance keepers to dropped.
    kept_clean.sort(key=lambda k: -k["relevance"])
    if len(kept_clean) > max_keep:
        overflow = kept_clean[max_keep:]
        kept_clean = kept_clean[:max_keep]
        for o in overflow:
            dropped_clean.append(
                {
                    "artifact_id": o["artifact_id"],
                    "reason": f"truncated by max_keep={max_keep}",
                }
            )

    return kept_clean, dropped_clean, cost


__all__ = ["AgentCorpusFilterTool"]
