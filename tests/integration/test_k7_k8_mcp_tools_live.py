"""KC5 — Live integration tests for K7 + K8 MCP tool wrappers.

The unit tests in ``tests/unit/test_findings_tool.py`` +
``test_corpus_filter_tool.py`` validate the MCP wrapper plumbing
(argument validation, error paths, ToolResult shape) but stub the
underlying ``FindingsAgent`` / corpus-filter LLM call. KC5 closes the
"does the wrapper actually run end-to-end against a real LLM" gap.

Each test:
1. Stores real NDA docx files as VFS artifacts (via store_document).
2. Builds a real KaosContext + KaosRuntime.
3. Calls ``tool.execute(...)`` on the MCP wrapper with realistic
   inputs — same shape an MCP client would pass over the wire.
4. Asserts the returned ToolResult is a structured success carrying
   the expected fields (answer, findings, cost, etc.) AND that those
   fields hold real values (non-empty answer, surviving findings,
   non-zero cost).

Runtime spend per test is bounded by the underlying agent's
budgets (Findings caps cost ~$0.50; corpus_filter is one LLM call
~$0.01) so the full suite costs under $1 on Anthropic at current
rates. Skipped without ``ANTHROPIC_API_KEY`` or NDA fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from kaos_agents.tools.corpus_filter import AgentCorpusFilterTool
from kaos_agents.tools.findings import AgentFindingsTool

NDA_DIR = Path.home() / "projects" / "273v" / "kelvin-app" / "samples" / "docx"


def _nda_paths() -> list[Path]:
    if not NDA_DIR.exists():
        return []
    return sorted(NDA_DIR.glob("MNDA*.docx"))


requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)

requires_nda_fixtures = pytest.mark.skipif(
    not _nda_paths(),
    reason=f"NDA fixtures missing at {NDA_DIR}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _store_nda_artifact(runtime: Any, context: Any, path: Path) -> str:
    """Parse one NDA + store as a VFS artifact. Returns the artifact_id."""
    from kaos_content.artifacts import store_document
    from kaos_office import parse_docx

    doc = parse_docx(str(path))
    manifest = await store_document(doc, runtime, context, name=path.stem)
    return manifest.artifact_id


@pytest.fixture
def runtime() -> Any:
    from kaos_core.registry.container import KaosRuntime

    return KaosRuntime()


@pytest.fixture
def context(runtime: Any) -> Any:
    from kaos_core.base.context import KaosContext

    return KaosContext.create(session_id="kc5-mcp-live", runtime=runtime)


# ---------------------------------------------------------------------------
# K7 — kaos-agent-findings live
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
@requires_nda_fixtures
class TestAgentFindingsToolLive:
    async def test_token_selector_runs_end_to_end(self, runtime: Any, context: Any) -> None:
        """select_by='token' over a real NDA produces a structured payload
        with non-trivial findings + non-zero cost."""
        artifact_id = await _store_nda_artifact(runtime, context, _nda_paths()[0])

        tool = AgentFindingsTool()
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "question": ("What counts as Confidential Information under this agreement?"),
                "select_by": "token",
                "selector_arg": "confidential",
                "filter_model": "anthropic:claude-haiku-4-5",
                "synthesis_model": "anthropic:claude-haiku-4-5",
                "chunk_size": 20,
                "num_parallel": 3,
                "relevance_threshold": 0.4,
            },
            context,
        )

        assert not result.isError, f"Tool returned error: {result.text}"
        payload = result.structuredContent
        assert payload is not None, "Tool returned no structured payload"

        # Shape check on the JSON payload.
        for key in (
            "artifact_id",
            "question",
            "answer",
            "findings",
            "total_enumerated",
            "total_filtered",
            "filter_calls",
            "filter_cost_usd",
            "synthesis_cost_usd",
            "total_cost_usd",
            "total_llm_calls",
        ):
            assert key in payload, f"missing {key} in payload"

        # Production-shaped assertions on the values.
        assert payload["artifact_id"] == artifact_id
        assert isinstance(payload["answer"], str)
        assert len(payload["answer"]) > 20, "Synthesis answer is suspiciously short"
        assert payload["total_enumerated"] >= 3, (
            f"Phase 1 selector found {payload['total_enumerated']} sentences "
            "containing 'confidential' on an NDA — selector regression?"
        )
        assert payload["total_filtered"] >= 1, (
            "Phase 2 filter dropped everything; filter is too strict or LLM misbehaved"
        )
        assert payload["total_filtered"] <= payload["total_enumerated"]
        assert payload["filter_cost_usd"] > 0
        assert payload["synthesis_cost_usd"] > 0
        assert 0 < payload["total_cost_usd"] < 0.50, (
            f"total_cost_usd={payload['total_cost_usd']} outside sanity gate"
        )

        # Every surviving finding has the AST refs that downstream consumers expect.
        for finding in payload["findings"]:
            assert finding["finding_id"]
            assert isinstance(finding["text"], str)
            assert 0.0 <= finding["relevance"] <= 1.0
            # block_ref points back into the source AST (KC4 / K6 contract).
            assert finding["block_ref"] is not None

    async def test_invalid_select_by_returns_actionable_error(
        self, runtime: Any, context: Any
    ) -> None:
        """The MCP wrapper's enum validation fires before any LLM call.

        No LLM cost — this is the cheap precondition path. We keep it
        in the live suite because the validation lives in the same
        execute() body as the live path; the test guards against a
        future refactor accidentally bypassing the check.
        """
        artifact_id = await _store_nda_artifact(runtime, context, _nda_paths()[0])
        tool = AgentFindingsTool()
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                "question": "any",
                "select_by": "no-such-mode",
            },
            context,
        )
        assert result.isError
        assert "select_by" in (result.text or "")


# ---------------------------------------------------------------------------
# K8 — kaos-agent-corpus-filter live
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
@requires_nda_fixtures
class TestAgentCorpusFilterToolLive:
    async def test_filter_returns_subset_with_real_llm(self, runtime: Any, context: Any) -> None:
        """K8 filters a corpus of real NDAs by an intent string.

        Asserts the wrapper:
        - Successfully loads each artifact, builds DocumentSummary,
          and passes the compact representation to the LLM.
        - Returns a ToolResult with kept + dropped + cost.
        - The kept set is a real subset (not all, not empty) and
          every kept entry references a real artifact_id from the
          input.
        - Cost is bounded.
        """
        paths = _nda_paths()
        # Need at least 3 docs so kept ⊂ corpus is meaningful.
        if len(paths) < 3:
            pytest.skip("Need >=3 NDA fixtures for a meaningful filter")

        artifact_ids: list[str] = []
        for path in paths:
            artifact_ids.append(await _store_nda_artifact(runtime, context, path))

        tool = AgentCorpusFilterTool()
        # Intent that's broadly relevant to NDAs — every doc should
        # plausibly survive, exercising the LLM's ranking rather than
        # a trivial "all relevant" pass.
        result = await tool.execute(
            {
                "intent": (
                    "Find agreements that govern the disclosure of "
                    "confidential financial information."
                ),
                "artifact_ids": artifact_ids,
                "max_keep": 2,
                "model": "anthropic:claude-haiku-4-5",
            },
            context,
        )

        assert not result.isError, f"Tool returned error: {result.text}"
        payload = result.structuredContent
        assert payload is not None

        for key in ("intent", "kept", "dropped", "total_input", "total_loadable", "cost_usd"):
            assert key in payload

        assert payload["total_input"] == len(artifact_ids)
        assert payload["total_loadable"] == len(artifact_ids), (
            "Some artifacts failed to load — check store_document compatibility"
        )

        kept = payload["kept"]
        dropped = payload["dropped"]
        assert isinstance(kept, list)
        assert isinstance(dropped, list)
        # max_keep=2 + 3+ docs => meaningful split.
        assert len(kept) <= 2, f"max_keep=2 but LLM returned {len(kept)} kept"
        assert len(kept) >= 1, "Filter dropped everything — LLM misbehaved or prompt is wrong"

        # Round-trip: every kept/dropped entry references a real artifact_id.
        # Tool payload schema:
        #   kept    = [{"artifact_id": ..., "relevance": float, "reasoning": ...}]
        #   dropped = [{"artifact_id": ..., "reason": ...}]
        kept_ids = {entry["artifact_id"] for entry in kept if isinstance(entry, dict)}
        dropped_ids = {entry["artifact_id"] for entry in dropped if isinstance(entry, dict)}
        assert kept_ids <= set(artifact_ids), (
            f"LLM hallucinated artifact_ids not in input: {kept_ids - set(artifact_ids)}"
        )
        # No overlap — kept and dropped partition the loadable set.
        assert not (kept_ids & dropped_ids)
        # Relevance scores are in [0, 1] per the tool's own clamp.
        for entry in kept:
            assert 0.0 <= entry["relevance"] <= 1.0

        # Cost bounded (one LLM call ~$0.01 on Haiku for a handful of summaries).
        assert 0 < payload["cost_usd"] < 0.10, f"cost_usd={payload['cost_usd']} outside sanity gate"

    async def test_empty_artifact_list_returns_actionable_error(
        self, runtime: Any, context: Any
    ) -> None:
        """Wrapper rejects empty corpora before paying for an LLM call."""
        tool = AgentCorpusFilterTool()
        result = await tool.execute(
            {"intent": "anything", "artifact_ids": []},
            context,
        )
        assert result.isError
        assert "artifact_ids" in (result.text or "").lower()
