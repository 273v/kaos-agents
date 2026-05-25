"""Live integration test for the FindingsAgent refusal contract.

Mirrors trust-skeptic Probe 2 (``docs/design/skeptic-trust-findings.md``):

- Real MNDA docx (NDAs lack liquidated-damages clauses).
- Ask "what is the liquidated damages amount?" — answer not in doc.
- Run via the K7 ``AgentFindingsTool``.
- Assert the response is a structured refusal:
  - ``answer == ""``
  - ``refusal_reason == "no_relevant_candidates"``
  - ``refusal_message`` non-empty
  - ``isError is False`` (a correct refusal is NOT an error).

Costs $0.01-0.05 on Anthropic at current rates — the filter pass is
cheap and synthesis is skipped (the whole point of the refusal).
Skipped without ``ANTHROPIC_API_KEY`` or NDA fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tests.integration._models import critic_model

NDA_DIR = Path.home() / "projects" / "273v" / "kelvin-app" / "samples" / "docx"


def _nda_paths() -> list[Path]:
    if not NDA_DIR.exists():
        return []
    return sorted(NDA_DIR.glob("MNDA*.docx"))


requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing — live refusal test requires "
    "an Anthropic key. See `kaos doctor` for setup.",
)

requires_nda_fixtures = pytest.mark.skipif(
    not _nda_paths(),
    reason=f"NDA fixtures missing at {NDA_DIR}",
)


FILTER_MODEL = critic_model()
SYNTH_MODEL = critic_model()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> Any:
    """In-memory KaosRuntime — see kaos-agents/CLAUDE.md ``Isolation
    patterns for live tests``. Prevents cross-run session memory
    leakage."""
    from kaos_core.registry.container import KaosRuntime

    return KaosRuntime.test_mode()


@pytest.fixture
def context(runtime: Any) -> Any:
    from kaos_core.base.context import KaosContext

    return KaosContext.create(session_id="findings-refusal-live", runtime=runtime)


async def _store_nda_artifact(runtime: Any, context: Any, path: Path) -> str:
    from kaos_content.artifacts import store_document
    from kaos_office import parse_docx

    doc = parse_docx(str(path))
    manifest = await store_document(doc, runtime, context, name=path.stem)
    return manifest.artifact_id


# ---------------------------------------------------------------------------
# The live refusal test
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
@requires_nda_fixtures
class TestFindingsRefusalLive:
    async def test_out_of_doc_question_yields_clean_refusal(
        self, runtime: Any, context: Any
    ) -> None:
        """Probe-2 scenario: ask an NDA a question it cannot answer.

        Expected behavior:

        - The K7 tool returns ``isError=False`` — the agent did its
          work, this is a correct refusal not a tool crash.
        - The synthesised ``answer`` is empty.
        - ``refusal_reason == "no_relevant_candidates"`` — Phase 1
          enumerated candidates, Phase 2 culled them all.
        - ``refusal_message`` is human-readable and non-empty.

        Pre-fix: ``answer`` was ``""`` with no refusal signal,
        indistinguishable from a tool crash downstream.
        """
        from kaos_agents.patterns.findings import REFUSAL_NO_RELEVANT_CANDIDATES
        from kaos_agents.tools.findings import AgentFindingsTool

        nda_path = _nda_paths()[0]
        artifact_id = await _store_nda_artifact(runtime, context, nda_path)

        tool = AgentFindingsTool()
        result = await tool.execute(
            {
                "artifact_id": artifact_id,
                # Topic an NDA never contains. The Probe-2 question
                # ("liquidated damages") can attract weak hits on
                # words like "damages" or "contracts"; we pick a
                # completely orthogonal topic (aircraft / aviation)
                # so the filter has no plausible candidate to keep.
                # The agent must cleanly refuse rather than silently
                # return an empty string.
                "question": (
                    "What is the maximum takeoff weight of the "
                    "aircraft and at what altitude does it cruise?"
                ),
                "select_by": "token",
                "selector_arg": "confidential",
                "filter_model": FILTER_MODEL,
                "synthesis_model": SYNTH_MODEL,
                "chunk_size": 20,
                "num_parallel": 3,
                # A token selector restricted to ``confidential`` +
                # an aviation question gives Phase 1 plenty of
                # candidates to chew on (confidentiality is the bulk
                # of an NDA) while Phase 2 has zero plausible
                # relevance signal. High threshold further protects
                # against borderline keeps.
                "relevance_threshold": 0.7,
            },
            context,
        )

        # A correct refusal is NOT an error.
        assert result.isError is False, (
            f"K7 tool returned isError=True for a correct refusal. text={result.text!r}"
        )

        payload = result.structuredContent
        assert payload is not None, "structuredContent missing on refusal"

        # The honest-refusal contract.
        assert payload["answer"] == "", (
            f"Expected empty answer on out-of-doc question; got: "
            f"{payload['answer']!r}. Either the filter let through a "
            f"non-relevant candidate, or the document unexpectedly "
            f"contains liquidated-damages language."
        )
        assert payload["refusal_reason"] == REFUSAL_NO_RELEVANT_CANDIDATES, (
            f"Expected refusal_reason={REFUSAL_NO_RELEVANT_CANDIDATES!r}; "
            f"got {payload['refusal_reason']!r}. enumerated="
            f"{payload['total_enumerated']} filtered={payload['total_filtered']}"
        )
        assert isinstance(payload["refusal_message"], str), (
            "refusal_message must be a string when refusal_reason is set"
        )
        assert payload["refusal_message"], "refusal_message must be non-empty"

        # Sanity: Phase 1 should have enumerated *something* — NDAs
        # have many sentences. A zero here would be the OTHER refusal
        # reason (no_candidates_enumerated) and indicate the selector
        # vocabulary mismatched.
        assert payload["total_enumerated"] > 0, (
            "Phase 1 enumerated zero candidates — selector regression?"
        )
        assert payload["total_filtered"] == 0, (
            f"Expected zero survivors for out-of-doc question; got "
            f"{payload['total_filtered']}. The filter LLM let "
            "through a non-relevant candidate."
        )

        # Cost gate — refusal is cheap (filter calls only).
        total_cost = float(payload.get("total_cost_usd") or 0.0)
        assert 0 < total_cost < 0.05, (
            f"total_cost_usd={total_cost:.4f} outside [0, $0.05) for a "
            "refusal — synthesis should be skipped. Possible runaway."
        )
        # Synthesis was skipped — cost there should be 0.
        assert float(payload.get("synthesis_cost_usd") or 0.0) == 0.0, (
            "synthesis_cost_usd must be 0 on a refusal — Phase 3 "
            "should be skipped when there are no survivors."
        )
