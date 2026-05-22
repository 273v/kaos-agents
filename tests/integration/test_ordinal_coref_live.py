"""Live LLM integration test for ordinal coreference resolution
(plan §Issue 8 / B1.5 acceptance row).

Acceptance bar from the launch-blocker plan:

    Ordinal coref | 12-scenario fixture | ≥90% resolution rate

The 12-scenario fixture itself lives in
``tests/unit/test_ordinal_coref_signature.py::test_12_scenario_fixture_resolution_rate``
where the deterministic resolver hits 12/12 by construction. This
file is the **live** layer that drives the same 12 scripts through
a real LLM call:

1. Build a 5-document corpus with stable filenames (nda-1.pdf …
   nda-5.pdf).
2. For each scenario, compose a minimal worker prompt that includes
   the document list AND the ``<context>`` coref tag emitted by
   :func:`build_coref_context_tag`.
3. Ask the model: "Which document number (1-5) is the user
   asking about? Respond with just the number."
4. Parse the answer, compare against the expected 1-based ordinal,
   tally hits.
5. Plan acceptance: ≥11/12 ≈ 91.67%, comfortably above the ≥90%
   bar.

This closes Issue 8 / B1.5 **end-to-end at the live tier** — the
primitive (resolver) + the tag formatter + the integration helper
+ the live-LLM evidence that an actual model consuming the tag
binds to the right referent.

Requires ``KAOS_LLM_OPENAI_API_KEY``. Default model is
``openai:gpt-5.4-mini`` — the cheapest current-generation model.
The full 12-scenario sweep typically costs < $0.005 in aggregate.
"""

from __future__ import annotations

import os
import re

import pytest

from kaos_agents.context.coreference import build_coref_context_tag
from kaos_agents.memory.session import SessionMemory
from kaos_agents.types.memory import MemoryType

# Plan-acceptance bar.
_PLAN_BAR_HITS: int = 11  # ≥ 11 of 12 ≥ 90%

# Default to OpenAI's cheapest current-gen model. The integration
# tier policy (kaos-modules/CLAUDE.md) explicitly calls these out
# as the integration-test workhorses.
_DEFAULT_MODEL = "openai:gpt-5.4-mini"

# Canonical 5-document corpus. We use filenames the model can
# disambiguate cleanly (nda-1 through nda-5) — the coref tag binds
# the user's ordinal phrase to one of these.
_DOC_FILENAMES: tuple[str, ...] = (
    "nda-1.pdf",
    "nda-2.pdf",
    "nda-3.pdf",
    "nda-4.pdf",
    "nda-5.pdf",
)

# The 12-scenario fixture. Each tuple is (script, expected_1based_index).
# Scripts mirror the deterministic-layer fixture verbatim so we are
# measuring the SAME inputs across both tiers.
_SCENARIOS: tuple[tuple[str, int], ...] = (
    ("Compare the NDAs. What's the third NDA's governing law?", 3),
    ("Show me the first one again.", 1),
    ("Re-read the second document.", 2),
    ("Summarize the fourth file.", 4),
    ("What does the fifth case say about indemnity?", 5),
    ("The last filing — what's its date?", 5),
    ("Pull up the previous document.", 5),
    ("Look at the 2nd one and tell me about jurisdiction.", 2),
    ("What does the 3rd doc say?", 3),
    ("Re-read the 1st upload.", 1),
    ("Summarize the latest filing.", 5),
    ("The most recent one — what's the term length?", 5),
)


def _make_corpus_memory() -> SessionMemory:
    """5-document SessionMemory used as the candidate list."""
    memory = SessionMemory("ordinal-coref-live")
    for fname in _DOC_FILENAMES:
        memory.add(
            MemoryType.DOCUMENTS,
            content=f"[Body of {fname}: a mutual NDA with the usual clauses.]",
            metadata={"filename": fname},
        )
    return memory


def _build_worker_prompt(script: str, tag: str) -> str:
    """Compose the worker prompt the LLM sees.

    Format mirrors what ``assemble_context`` would inject into the
    worker thinking_note: document list + ``<context>`` coref tag +
    user message + a constrained answer instruction.
    """
    doc_lines = "\n".join(f"{i}. {fname}" for i, fname in enumerate(_DOC_FILENAMES, start=1))
    return (
        "You are answering a follow-up question about a list of attached documents.\n"
        "\n"
        f"The user has uploaded these documents (in upload order):\n{doc_lines}\n"
        "\n"
        f"{tag}\n"
        "\n"
        f'User message: "{script}"\n'
        "\n"
        "Question: which document number (1-5) is the user asking about? "
        "Respond with JUST the single digit, nothing else."
    )


def _parse_answer(text: str) -> int | None:
    """Extract the first 1-5 digit from the model's response."""
    if not text:
        return None
    m = re.search(r"\b([1-5])\b", text)
    if m is None:
        return None
    return int(m.group(1))


def _resolve_model() -> str:
    """Allow override via env so a future contributor can re-run
    against Sonnet or Haiku without editing this file."""
    return os.environ.get("KAOS_AGENT_LIVE_MODEL") or _DEFAULT_MODEL


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("KAOS_LLM_OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="Requires KAOS_LLM_OPENAI_API_KEY for live LLM call (plan §Issue 8 / B1.5)",
)
async def test_ordinal_coref_live_12_scenario_acceptance() -> None:
    """Plan §Issue 8 / B1.5 acceptance: ≥90% resolution rate when a
    real LLM consumes the ``<context>`` coref tag.

    For each of the 12 scripts:
      1. Build the coref tag against the 5-document corpus.
      2. Compose the worker prompt with the document list + tag +
         script + answer instruction.
      3. Call the model.
      4. Parse the digit response.
      5. Compare against the expected 1-based ordinal.

    Acceptance: ≥11 hits / 12 (≥91.67%). Plan bar is ≥90%; we pin
    one stricter so a regression that drops to exactly 10/12 fails.
    """
    from kaos_llm_core import InputField, OutputField, Signature
    from kaos_llm_core.programs.call import Call

    class _BindOrdinalSig(Signature):
        """Bind an ordinal coreference to the correct document number."""

        prompt: str = InputField(description="The composed worker prompt.")
        document_number: str = OutputField(
            description="A single digit 1-5 indicating which document the user means."
        )

    model = _resolve_model()
    call = Call(_BindOrdinalSig, model=model)
    memory = _make_corpus_memory()

    hits = 0
    misses: list[tuple[str, int, int | None, str]] = []
    total_cost_usd = 0.0

    for script, expected_ordinal in _SCENARIOS:
        tag = build_coref_context_tag(memory, script)
        assert tag is not None, f"Resolver missed an ordinal in script: {script!r}"

        prompt = _build_worker_prompt(script, tag)
        invocation = await call.invoke(prompt=prompt)
        answer_text = (getattr(invocation.output, "document_number", "") or "").strip()
        parsed = _parse_answer(answer_text)

        usage = getattr(invocation, "usage", None)
        if usage is not None:
            total_cost_usd += float(getattr(usage, "cost_usd", 0.0) or 0.0)

        if parsed == expected_ordinal:
            hits += 1
        else:
            misses.append((script, expected_ordinal, parsed, answer_text))

    # Cost ceiling — if a regression makes the worker prompt 10x
    # larger, this catches it before we wake up to a surprise bill.
    assert total_cost_usd < 0.05, f"12-scenario sweep cost ${total_cost_usd:.4f} > $0.05 ceiling"

    miss_report = "\n".join(
        f"  - script={s!r} expected={e} got={p!r} raw={raw!r}" for s, e, p, raw in misses
    )
    assert hits >= _PLAN_BAR_HITS, (
        f"Live ordinal-coref resolution rate {hits}/12 < plan bar "
        f"{_PLAN_BAR_HITS}/12 on model={model}. "
        f"Cost=${total_cost_usd:.4f}. Misses:\n{miss_report}"
    )


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("KAOS_LLM_OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="Requires KAOS_LLM_OPENAI_API_KEY for live LLM call",
)
async def test_live_tag_without_coref_returns_none_baseline() -> None:
    """Sanity-check the baseline: when no ordinal phrase fires, the
    helper returns None and we do NOT inject anything into the
    worker prompt. This is the "do no harm" invariant — the integration
    must not change behavior for non-ordinal turns."""
    memory = _make_corpus_memory()
    # No ordinal phrase in this message.
    tag = build_coref_context_tag(memory, "What's the governing law typically in an NDA?")
    assert tag is None
