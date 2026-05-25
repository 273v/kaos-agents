"""WU-G.2 / #352 — live multi-turn corpus context retention regression.

The bug this test pins:

  - Turn 1: user attaches a small PDF + asks a question about it →
    agent answers from the attached body.
  - Turn 2: user says ``"summarize that"`` with no fresh attachment.
    Pre-0.1.0a19, ``assemble_context``'s budget-trim phase dropped
    every DOCUMENTS body and the agent answered "I don't have any
    documents to summarize" — the UX-C2 regression from 2026-05-17.
  - Post-0.1.0a19, ``SessionMemory.corpus_ever_attached`` is sticky and
    ``assemble_context`` retains a compact DOCUMENTS handle
    (filename list + ``search_memory`` hint) so the agent knows to
    follow up via ``search_memory(section='documents')``.

The assertion target is operational: the agent's turn-2 response must
either (a) contain content drawn from the attached corpus, or (b)
demonstrate a ``search_memory``-tool call that finds the document. We
accept either outcome — both are the correct behaviour — and fail only
on the prior regression (a confident "no documents attached" refusal).

Live test budget: <$0.01 on Haiku 4.5. Two turns x ~$0.003 each.
"""

from __future__ import annotations

import os

import pytest
from kaos_core.types.enums import IsolationMode, StorageBackend
from kaos_core.vfs.core import VirtualFileSystem
from kaos_core.vfs.models import VFSConfig

from kaos_agents.memory.session import SessionMemory
from kaos_agents.memory.store import SessionStore
from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.settings import KaosAgentSettings
from kaos_agents.types.memory import MemoryType
from tests.integration._models import respond_model

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing — multi-turn corpus retention is live-only",
)


MODEL = respond_model()


# A small synthetic "PDF" body — enough that an LLM can draw a
# coherent summary, small enough that the live cost stays under
# $0.01 across two turns. The numeric anchors (``$89``, ``Delaware``,
# ``2026-05-19``) give the assertion a precise hook: a turn-2
# summary that mentions any of them is grounded; one that mentions
# none of them is likely hallucinated.
_ATTACHED_DOC_FILENAME = "kaos-273v-acme-nda-2026.txt"
_ATTACHED_DOC_BODY = (
    "MUTUAL NON-DISCLOSURE AGREEMENT\n"
    "Parties: 273V LLC (a Delaware limited liability company) and "
    "Acme Holdings Inc.\n"
    "Effective Date: 2026-05-19.\n"
    "Governing Law: Delaware General Corporation Law.\n"
    "Filing Fee: $89 per certificate of incorporation filed with the "
    "Delaware Secretary of State.\n"
    "Term: 3 years from the Effective Date.\n"
    "Confidential Information includes business plans, customer "
    "lists, and pricing models exchanged between the Parties.\n"
    "Survival: Confidentiality obligations survive termination for "
    "an additional 2 years.\n"
)


def _memory_vfs() -> VirtualFileSystem:
    """In-memory VFS so two turns of the same session don't leak to disk."""
    return VirtualFileSystem(
        config=VFSConfig(
            default_backend=StorageBackend.MEMORY,
            isolation_mode=IsolationMode.GLOBAL,
        )
    )


def _settings_for_test() -> KaosAgentSettings:
    """Generous settings — the test wants the agent to succeed, not to
    fail on budget or iteration caps."""
    return KaosAgentSettings(
        default_llm_model=MODEL,
        planning_llm_model=MODEL,
        max_react_iterations=4,
        tool_timeout_seconds=30.0,
    )


@pytest.mark.live
@requires_anthropic
class TestMultiTurnCorpusRetention:
    """The canonical WU-G.2 / #352 regression."""

    async def test_followup_finds_document_via_corpus_handle(self) -> None:
        """Two turns; second turn must find the doc via the
        retained-context handle (or via ``search_memory``).
        """
        vfs = _memory_vfs()
        session_id = "wu-g2-multi-turn-corpus"

        # Seed an empty session, then attach the document into
        # ``MemoryType.DOCUMENTS``. This mirrors the SPA's
        # ``POST /v1/sessions/{id}/documents`` ingest path.
        store = SessionStore(vfs)
        memory = SessionMemory(session_id)
        memory.add(
            MemoryType.DOCUMENTS,
            _ATTACHED_DOC_BODY,
            metadata={
                "uri": f"file:{_ATTACHED_DOC_FILENAME}",
                "source": _ATTACHED_DOC_FILENAME,
                "filename": _ATTACHED_DOC_FILENAME,
            },
        )
        await store.save(memory)

        agent = ChatAgent(
            vfs,
            model=MODEL,
            settings=_settings_for_test(),
        )

        # Turn 1 — explicit reference to the attached document.
        response_t1 = await agent.turn(
            "What is the term length stated in the attached NDA?",
            session_id=session_id,
        )
        assert response_t1.text, "turn 1 produced empty text"
        # Sanity: the doc body says 3 years. We don't fail the test on
        # this — it's a precondition; the regression we care about is
        # turn 2.
        if "3" not in response_t1.text and "three" not in response_t1.text.lower():
            pytest.skip(
                "Turn 1 didn't surface the term length — the live model "
                "behaviour shifted upstream. The WU-G.2 regression hook "
                "only fires when turn 1 worked, so skip rather than "
                "false-fail this test."
            )

        # Sanity check: the sticky flag must have flipped True after
        # turn 1's classification, and the docs section is still
        # populated (no eviction).
        reloaded_after_t1 = await store.load(session_id)
        assert reloaded_after_t1.corpus_ever_attached is True, (
            "memory.corpus_ever_attached should fire on turn 1 — this is "
            "the WU-G.2 hook that the rest of the test depends on"
        )
        assert reloaded_after_t1.section_item_count(MemoryType.DOCUMENTS) >= 1

        # Turn 2 — the canonical anaphoric follow-up. Pre-0.1.0a19, the
        # agent confidently refused "I don't have any documents to
        # summarize"; post-0.1.0a19 the handle keeps the corpus
        # reachable.
        response_t2 = await agent.turn(
            "Summarize that.",
            session_id=session_id,
        )
        assert response_t2.text, "turn 2 produced empty text"

        text_lower = response_t2.text.lower()

        # The regression fingerprint: a confident refusal that the
        # document does not exist. Any of these phrases is the bug.
        regression_phrases = (
            "no document",
            "no attached document",
            "i don't have any documents",
            "i do not have any documents",
            "no documents attached",
            "no documents to summarize",
            "no documents in session",
            "i don't have access to any documents",
            "i do not have access to any documents",
        )
        for phrase in regression_phrases:
            assert phrase not in text_lower, (
                f"turn 2 reproduced the WU-G.2 / #352 regression — the "
                f"agent confidently said {phrase!r} despite an attached "
                f"NDA still in MemoryType.DOCUMENTS. Full reply: "
                f"{response_t2.text!r}"
            )

        # The positive evidence: the response should either cite a
        # specific anchor from the attached doc OR mention searching /
        # finding it via tools. Either is grounded; both prove the
        # handle did its job.
        document_anchors = (
            "delaware",
            "$89",
            "89 ",
            "2026-05-19",
            "3 year",
            "three year",
            "273v",
            "acme",
            "confidential",
            "non-disclosure",
            "nda",
        )
        tool_call_anchors = (
            "search_memory",
            "search-memory",
            "kaos-retrieval",
            "retrieved",
            "retrieving",
            "found in",
            "located in",
            "looking up",
            "checking the document",
        )
        grounded = any(a in text_lower for a in document_anchors)
        searched = any(a in text_lower for a in tool_call_anchors)
        assert grounded or searched, (
            f"turn 2 neither grounded in the attached NDA nor demonstrated "
            f"a search-memory call. The corpus handle is failing to signal "
            f"reachability to the LLM. Reply: {response_t2.text!r}"
        )
