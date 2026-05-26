"""CS-B family experiment 3: end-to-end FindingsAgent dispatch through ChatAgent.

Tests whether a ``ChatAgent`` subclass that overrides
``_handle_research`` to use ``FindingsAgent`` recovers the S05
needle through the real Runner pipeline.

Architecturally: corpus-stress S05 deliberately REDACTS the needle
from ``SessionMemory.DOCUMENTS``, so FindingsAgent operating over
the in-memory section directly cannot find it. The dispatch must:

  1. Read attached files from the session VFS.
  2. Parse each into a ``ContentDocument``.
  3. Concatenate into one ``DocumentView``.
  4. Run ``FindingsAgent``.
  5. Return ``(answer, [], usage)`` matching ``_handle_research``'s contract.

This collapses the LLM's tool-loop responsibility (the part that
CS-B3 "give-up cliff" hits) into deterministic file collection. If
the dispatch works, we have a concrete production blueprint.

Run::

    cd /home/mjbommar/projects/273v/kaos-agents
    KAOS_TEST_RESPOND_MODEL=anthropic:claude-sonnet-4-6 \\
    KAOS_TEST_CRITIC_MODEL=anthropic:claude-sonnet-4-6 \\
        uv run python tests/scratch/findings_experiments/run_dispatch.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Make ``tests.integration._models`` + ``_corpus_fixtures`` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.views.document_view import DocumentView
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime
from kaos_nlp_core._defaults import get_default_punkt_tokenizer

from kaos_agents.memory.store import SessionStore
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.patterns.findings import (
    FindingsAgent,
    every_sentence_selector,
)
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types.intents import IntentResult, IntentType
from kaos_agents.types.memory import MemoryType
from kaos_agents.types.usage import InvocationUsage

from tests.integration._corpus_fixtures import (
    SynthDoc,
    hydrate_corpus_into_memory,
    synth_html,
    synth_json,
    synth_text,
    write_corpus_to_vfs,
)
from tests.integration._models import critic_model, respond_model

FILTER_MODEL = critic_model()
SYNTH_MODEL = respond_model()


# ─── S05 fixture (copy of the corpus-stress shape) ──────────────────


def build_s05_docs() -> tuple[list[SynthDoc], str, str]:
    json_needle = "config_token: KAOS-S05-JSON-OK"
    docs = [
        SynthDoc(
            filename="release-notes.html",
            bytes=synth_html(
                "Release 2.4.1\n\nBug fixes and performance improvements.",
                title="Notes",
            ),
            mime="text/html",
            is_needle=False,
        ),
        SynthDoc(
            filename="readme.txt",
            bytes=synth_text(
                "Run the installer with --quiet for unattended mode.\n"
                "Logs land in /var/log/kaos/.\n"
            ),
            mime="text/plain",
            is_needle=False,
        ),
        SynthDoc(
            filename="config.json",
            bytes=synth_json(
                {
                    "environment": "production",
                    "config_token": "KAOS-S05-JSON-OK",
                    "feature_flags": {"corpus_v2": True},
                }
            ),
            mime="application/json",
            is_needle=True,
            needle_fact=json_needle,
        ),
    ]
    prompt = (
        "Three files are attached (HTML release notes, plain text README, "
        "and a JSON config). What is the value of 'config_token' in the "
        "JSON file? Answer with the exact token string."
    )
    return docs, prompt, "KAOS-S05-JSON-OK"


# ─── Bytes → ContentDocument adapter ────────────────────────────────


def _bytes_to_content_document(filename: str, mime: str, body: bytes) -> ContentDocument:
    """Parse file bytes into a ContentDocument with format dispatch."""
    text = body.decode("utf-8", errors="replace")
    if "html" in mime:
        from kaos_content.parsers.html import parse_html

        return parse_html(text)
    if "json" in mime:
        # Pretty-print so each key/value pair ends on its own line.
        try:
            obj = json.loads(text)
            pretty = json.dumps(obj, indent=2)
        except json.JSONDecodeError:
            pretty = text
        blocks: list[Paragraph] = [
            Paragraph(children=(Text(value=line),))
            for line in pretty.splitlines()
            if line.strip()
        ]
        return ContentDocument(body=tuple(blocks))
    from kaos_content import parse_plain_text

    return parse_plain_text(text)


# ─── Subclassed agent with FindingsAgent dispatch ───────────────────


_FACT_LOOKUP_TOKENS = (
    "file",
    "files",
    "attached",
    "attachment",
    "json",
    "config",
    "value of",
    "what is",
    "what's",
    "quote",
    "verbatim",
    "exact",
    "find",
    "look up",
    "lookup",
)


def _looks_like_fact_lookup(message: str) -> bool:
    """Cheap keyword screen — does the message look like a doc-grounded ask?"""
    lowered = message.lower()
    return any(tok in lowered for tok in _FACT_LOOKUP_TOKENS)


class FindingsResearchAgent(ChatAgent):
    """ChatAgent subclass that routes RESEARCH intent through FindingsAgent.

    The experiment-specific fixture file list is stashed as an attr
    AFTER construction (see ``main()``) since ChatAgent's ctor doesn't
    accept arbitrary kwargs.
    """

    # Set by the experiment driver after construction.
    _experiment_docs: list[SynthDoc] = []
    # When True, _classify() force-promotes RESPOND/TOOL_USE → RESEARCH
    # whenever corpus_attached + fact-lookup keywords both hold.
    _force_research_when_corpus_attached: bool = False

    async def _classify(
        self,
        message,
        memory,
        context_items=None,
    ):
        result = await super()._classify(message, memory, context_items)

        if not self._force_research_when_corpus_attached:
            return result

        # Demote-then-promote logic: keep the classifier's RESEARCH/PLAN
        # verdicts; override only when the classifier picked something
        # that won't reach _handle_research().
        if result.intent in (IntentType.RESEARCH, IntentType.PLAN):
            return result

        # Corpus-attached signal — count DOCUMENTS items in context.
        docs_in_context = 0
        if context_items is not None:
            docs_in_context = len(context_items.get(MemoryType.DOCUMENTS, []) or [])
        # Fall back to memory section count when context_items is empty.
        if docs_in_context == 0 and memory.has_section(MemoryType.DOCUMENTS):
            docs_in_context = memory.section_item_count(MemoryType.DOCUMENTS)

        if docs_in_context == 0:
            return result
        if not _looks_like_fact_lookup(message):
            return result

        # Force RESEARCH. Preserve the original classifier's usage so the
        # turn-end cost aggregate still reflects the LLM call we already
        # paid for.
        return IntentResult(
            intent=IntentType.RESEARCH,
            confidence=1.0,
            reasoning=(
                f"forced: corpus_attached ({docs_in_context} docs) + "
                f"fact-lookup keywords; classifier said {result.intent} "
                f"({result.confidence:.2f}): {result.reasoning[:120]}"
            ),
            usage=result.usage,
        )

    async def _handle_research(self, message, memory, context_items):
        runtime = self._runtime
        ctx = self._context
        if runtime is None or ctx is None or not self._experiment_docs:
            return await super()._handle_research(message, memory, context_items)

        session_id = ctx.session_id
        namespace = ctx.default_vfs_namespace or f"sessions/{session_id}/files/"

        # 1. Read attached files from VFS + parse each into a ContentDocument.
        blocks: list[Paragraph] = []
        for d in self._experiment_docs:
            vfs_path = f"{namespace}{d.filename}"
            body_bytes = await runtime.vfs.read(vfs_path)
            doc = _bytes_to_content_document(d.filename, d.mime, body_bytes)
            blocks.append(
                Paragraph(children=(Text(value=f"=== {d.filename} ==="),))
            )
            blocks.extend(doc.body)

        # 2. Build a single DocumentView.
        view = DocumentView(
            ContentDocument(body=tuple(blocks)),
            sentence_segmenter=get_default_punkt_tokenizer(),
        )

        # 3. Run FindingsAgent.
        agent = FindingsAgent(
            selector=every_sentence_selector,
            filter_model=FILTER_MODEL,
            synthesis_model=SYNTH_MODEL,
            chunk_size=20,
            num_parallel=3,
            relevance_threshold=0.4,
        )
        t0 = time.monotonic()
        result = await agent.run(message, view)
        elapsed = time.monotonic() - t0

        total_cost = result.filter_cost_usd + result.synthesis_cost_usd
        usage = InvocationUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=total_cost,
        )

        answer = result.answer or (
            "[FindingsResearchAgent refused: no findings survived filter]"
        )
        print(
            f"    [FindingsAgent dispatch: "
            f"survivors={result.total_filtered}/{result.total_enumerated} "
            f"cost=${total_cost:.4f} time={elapsed:.1f}s "
            f"refused={result.refusal is not None}]"
        )
        return answer, [], usage


# ─── Driver ────────────────────────────────────────────────────────


async def _setup_agent_and_memory(*, force_research: bool = False):
    """Shared fixture: hydrate S05 docs + build the FindingsResearchAgent.

    Returns ``(agent, memory, prompt, needle, docs)`` so callers can
    either drive ``agent.turn(prompt)`` (full classifier path) or
    call ``agent._handle_research(...)`` directly (bypass path).
    Pass ``force_research=True`` to flip the classifier patch on.
    """
    runtime = KaosRuntime.test_mode()
    session_id = f"s05-dispatch-{int(time.time())}"

    docs, prompt, needle = build_s05_docs()

    # Hydrate exactly like corpus-stress: VFS bytes + redacted DOCUMENTS section.
    await write_corpus_to_vfs(docs, runtime, session_id=session_id)
    store = SessionStore(runtime.vfs)
    memory = await store.load_or_create(session_id)
    hydrate_corpus_into_memory(docs, memory, session_id=session_id)
    await store.save(memory)

    namespace = f"sessions/{session_id}/files/"
    context = KaosContext(
        session_id=session_id,
        runtime=runtime,
        vfs=runtime.vfs,
        default_vfs_namespace=namespace,
    )

    agent = FindingsResearchAgent(
        runtime.vfs,
        runtime=runtime,
        context=context,
        model=SYNTH_MODEL,
        settings=KaosAgentSettings.resolve(None),
    )
    agent._experiment_docs = docs
    agent._force_research_when_corpus_attached = force_research
    return agent, memory, prompt, needle, docs


async def run_dispatch_via_classifier() -> dict:
    """Run the full ``agent.turn(prompt)`` path — classifier picks intent."""
    print(f"\n{'=' * 70}")
    print("  S05_via_classifier — agent.turn(prompt) → IntentClassifier → handler")
    print(f"{'=' * 70}")
    print(f"  filter model:    {FILTER_MODEL}")
    print(f"  synthesis model: {SYNTH_MODEL}")

    agent, _, prompt, needle, _ = await _setup_agent_and_memory()
    session_id = agent._context.session_id

    t0 = time.monotonic()
    response = await agent.turn(prompt, session_id=session_id)
    elapsed = time.monotonic() - t0

    needle_present = needle in response.text
    intent = response.intent if hasattr(response, "intent") else None

    print(f"\n  → intent chosen:    {intent!r}")
    print(f"  → tool calls fired: {len(response.tool_calls)}")
    print(f"  → needle present:   {needle_present}")
    print(f"  → response cost:    ${response.cost_usd:.4f}")
    print(f"  → elapsed:          {elapsed:.1f}s")
    print(f"  → answer head:      {response.text[:400]!r}")

    return {
        "scenario": "S05_via_classifier",
        "intent_chosen": str(intent) if intent else None,
        "tool_calls_fired": len(response.tool_calls),
        "needle_present": needle_present,
        "answer_head": response.text[:500],
        "cost_usd": round(response.cost_usd, 4),
        "elapsed_s": round(elapsed, 2),
        "models": {"filter": FILTER_MODEL, "synthesis": SYNTH_MODEL},
    }


async def run_dispatch_bypassing_classifier() -> dict:
    """Skip the intent classifier; call ``_handle_research`` directly.

    Isolates "does FindingsAgent dispatch work?" from "does the
    classifier route correctly?" — the latter we already proved
    broken (it picks RESPOND because the redacted DOCUMENTS section
    looks like it already contains the answer). This bypass tells
    us whether fixing the classifier alone would be sufficient.
    """
    print(f"\n{'=' * 70}")
    print("  S05_bypass_classifier — direct call to _handle_research (forced RESEARCH)")
    print(f"{'=' * 70}")
    print(f"  filter model:    {FILTER_MODEL}")
    print(f"  synthesis model: {SYNTH_MODEL}")

    agent, memory, prompt, needle, _ = await _setup_agent_and_memory()

    t0 = time.monotonic()
    # _handle_research signature: (message, memory, context_items).
    # context_items can be empty — our override doesn't read it.
    answer, tool_calls, usage = await agent._handle_research(
        prompt, memory, {}
    )
    elapsed = time.monotonic() - t0

    needle_present = needle in answer

    print(f"\n  → tool calls fired: {len(tool_calls)}")
    print(f"  → needle present:   {needle_present}")
    print(f"  → usage cost:       ${usage.cost_usd:.4f}")
    print(f"  → elapsed:          {elapsed:.1f}s")
    print(f"  → answer head:      {answer[:400]!r}")

    return {
        "scenario": "S05_bypass_classifier",
        "tool_calls_fired": len(tool_calls),
        "needle_present": needle_present,
        "answer_head": answer[:500],
        "cost_usd": round(usage.cost_usd, 4),
        "elapsed_s": round(elapsed, 2),
        "models": {"filter": FILTER_MODEL, "synthesis": SYNTH_MODEL},
    }


async def run_dispatch_via_patched_classifier() -> dict:
    """Full ``agent.turn(prompt)`` path with the corpus+fact-lookup patch ON.

    Validates the classifier override end-to-end: the LLM still gets
    to classify (so its usage hits the cost aggregate), but if it
    picks RESPOND/TOOL_USE while DOCUMENTS are attached and the
    prompt asks for a file-grounded fact, we force RESEARCH and the
    turn dispatches into our ``_handle_research`` override.

    Compare against ``run_dispatch_via_classifier`` (same flow,
    patch OFF) — the only difference between the two should be the
    intent and the needle outcome.
    """
    print(f"\n{'=' * 70}")
    print(
        "  S05_via_patched_classifier — agent.turn(prompt) "
        "with corpus+keywords override"
    )
    print(f"{'=' * 70}")
    print(f"  filter model:    {FILTER_MODEL}")
    print(f"  synthesis model: {SYNTH_MODEL}")

    agent, _, prompt, needle, _ = await _setup_agent_and_memory(
        force_research=True
    )
    session_id = agent._context.session_id

    t0 = time.monotonic()
    response = await agent.turn(prompt, session_id=session_id)
    elapsed = time.monotonic() - t0

    needle_present = needle in response.text
    intent = response.intent if hasattr(response, "intent") else None

    print(f"\n  → intent chosen:    {intent!r}")
    print(f"  → tool calls fired: {len(response.tool_calls)}")
    print(f"  → needle present:   {needle_present}")
    print(f"  → response cost:    ${response.cost_usd:.4f}")
    print(f"  → elapsed:          {elapsed:.1f}s")
    print(f"  → answer head:      {response.text[:400]!r}")

    return {
        "scenario": "S05_via_patched_classifier",
        "intent_chosen": str(intent) if intent else None,
        "tool_calls_fired": len(response.tool_calls),
        "needle_present": needle_present,
        "answer_head": response.text[:500],
        "cost_usd": round(response.cost_usd, 4),
        "elapsed_s": round(elapsed, 2),
        "models": {"filter": FILTER_MODEL, "synthesis": SYNTH_MODEL},
    }


async def main() -> int:
    results: list[dict] = []
    for runner in (
        run_dispatch_bypassing_classifier,
        run_dispatch_via_classifier,
        run_dispatch_via_patched_classifier,
    ):
        try:
            result = await runner()
        except Exception:
            import traceback

            traceback.print_exc()
            return 1
        results.append(result)
        result = results[-1]  # keep `result` defined for the summary block below

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        print(
            f"  {r['scenario']:<32} needle={r['needle_present']!s:<5} "
            f"cost=${r['cost_usd']:.4f} time={r['elapsed_s']:.1f}s"
        )

    out_path = Path(__file__).parent / "results.jsonl"
    with out_path.open("a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  → appended to {out_path}\n")

    return 0 if all(r["needle_present"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
