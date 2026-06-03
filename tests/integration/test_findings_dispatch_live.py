"""Live end-to-end test for the FindingsAgent dispatch default.

Closes CS-B2 (hallucination) + CS-B3 (give-up cliff) at the dispatch
contract: when the SPA uploads files into ``MemoryType.DOCUMENTS``
with VFS-backed bytes, the ChatAgent's intent classifier should
promote a fact-lookup question to RESEARCH, and ``BaseAgent``'s new
FindingsAgent-backed default should re-read raw bytes and return
the planted needle — even when the DOCUMENTS section's in-memory
preview has been redacted.

Mirrors the shape of the scratch experiment at
``tests/scratch/findings_experiments/run_dispatch.py`` but drives
through the production ``ChatAgent.turn()`` path so the classifier
promotion + ``_dispatch_streaming`` routing get exercised
end-to-end.
"""

from __future__ import annotations

import os
import time

import pytest
from kaos_core.base.context import KaosContext
from kaos_core.registry.container import KaosRuntime

from kaos_agents.memory.store import SessionStore
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types import IntentType
from tests.integration._corpus_fixtures import (
    SynthDoc,
    hydrate_corpus_into_memory,
    synth_html,
    synth_json,
    synth_text,
    write_corpus_to_vfs,
)
from tests.integration._models import critic_model, respond_model

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)


def _build_s05_docs() -> tuple[list[SynthDoc], str, str]:
    """Three-file fixture mirroring corpus-stress S05.

    The needle ``"KAOS-S05-JSON-OK"`` is hydrated into VFS bytes but
    redacted from the DOCUMENTS section preview, so any agent path
    that reads ONLY the in-memory preview will hallucinate. The
    FindingsAgent default re-reads VFS bytes and recovers the
    grounded value.
    """
    needle_fact = "config_token: KAOS-S05-JSON-OK"
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
            needle_fact=needle_fact,
        ),
    ]
    prompt = (
        "Three files are attached (HTML release notes, plain text README, "
        "and a JSON config). What is the value of 'config_token' in the "
        "JSON file? Answer with the exact token string."
    )
    return docs, prompt, "KAOS-S05-JSON-OK"


@pytest.mark.live
@requires_anthropic
class TestFindingsDispatchLive:
    """End-to-end: ChatAgent.turn() with an attached corpus recovers the needle."""

    async def test_chat_agent_routes_content_question_to_findings_agent(self) -> None:
        """The full happy path: LLM router sends a document-CONTENT question
        to RESEARCH (grounded on bytes) + FindingsAgent default.

        Routing is now an LLM decision over the attached-document filenames
        (``documents_available``), not the old ``corpus_attached_promotion``
        keyword override. See
        docs/design/2026-06-02-agentic-routing-and-transcript-grounding.md.

        Architectural assertions (verify USER outcome, not symptom text):
        - The needle string is present in the final response.
        - The classifier routed a document-content question to RESEARCH.
        - No tool calls were fired (FindingsAgent dispatch bypasses ReAct).
        - The total cost is bounded (sanity gate $0.50/turn).
        """
        runtime = KaosRuntime.test_mode()
        session_id = f"findings-dispatch-{int(time.time())}"

        docs, prompt, needle = _build_s05_docs()

        # Hydrate corpus exactly like the SPA: VFS bytes + redacted
        # DOCUMENTS section.
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

        agent = ChatAgent(
            runtime.vfs,
            runtime=runtime,
            context=context,
            model=respond_model(),
            settings=KaosAgentSettings.resolve(None),
        )

        response = await agent.turn(prompt, session_id=session_id)

        assert needle in response.text, (
            f"Needle {needle!r} not present in response. Got: {response.text!r}"
        )

        # Confirm the dispatch went through RESEARCH (the LLM router sends a
        # document-content question to research when documents are attached).
        intent = getattr(response, "intent", None)
        if intent is not None:
            # IntentResult shape
            assert intent.intent == IntentType.RESEARCH, (
                f"Expected RESEARCH dispatch, got {intent.intent!r}: {intent.reasoning!r}"
            )

        # Exactly one synthetic tool call — kaos-agent-findings-dispatch —
        # which surfaces the FindingsAgent run for the SPA / Citations
        # panel / Opus-as-judge tool-trace contract. No ReAct tool calls
        # (kaos-content-search-document / kaos-vfs-read) should appear
        # because the FindingsAgent path bypasses ReAct.
        tool_names = [tc.tool_name for tc in response.tool_calls]
        assert "kaos-agent-findings-dispatch" in tool_names, (
            f"FindingsAgent synthetic tool call missing from response. "
            f"Got tool calls: {tool_names!r}"
        )
        assert all(tc.tool_name == "kaos-agent-findings-dispatch" for tc in response.tool_calls), (
            f"Unexpected non-FindingsAgent tool calls fired: {tool_names!r}"
        )

        # Sanity cost gate.
        assert response.cost_usd < 0.50, (
            f"FindingsAgent dispatch cost ${response.cost_usd:.4f} exceeded "
            f"the $0.50 sanity gate — possible runaway."
        )

    async def test_no_corpus_does_not_force_research(self) -> None:
        """With no DOCUMENTS attached, a fact-lookup-shaped message is NOT
        forced to RESEARCH.

        ``documents_available`` is empty, so the LLM router has no corpus to
        send the question to. There is no keyword override anymore — the
        classifier's organic verdict stands.
        """
        runtime = KaosRuntime.test_mode()
        session_id = f"findings-no-corpus-{int(time.time())}"
        store = SessionStore(runtime.vfs)
        memory = await store.load_or_create(session_id)
        await store.save(memory)

        context = KaosContext(
            session_id=session_id,
            runtime=runtime,
            vfs=runtime.vfs,
            default_vfs_namespace=f"sessions/{session_id}/files/",
        )

        agent = ChatAgent(
            runtime.vfs,
            runtime=runtime,
            context=context,
            model=critic_model(),
            settings=KaosAgentSettings.resolve(None),
        )

        # Same fact-lookup-looking message, NO attached docs.
        response = await agent.turn(
            "What is the value of 'config_token' in the JSON file?",
            session_id=session_id,
        )

        # The intent should NOT be a "corpus_attached_promotion" forced
        # RESEARCH — there are no DOCUMENTS to promote against. If
        # intent metadata is present, sanity-check it.
        intent = getattr(response, "intent", None)
        if intent is not None:
            assert "corpus_attached_promotion" not in intent.reasoning, (
                f"Promotion fired without DOCUMENTS attached: {intent.reasoning!r}"
            )

        # The text is whatever the simple-respond path produced — we
        # don't assert content here; the contract is just "don't force
        # RESEARCH on an empty corpus".
        assert isinstance(response.text, str)
