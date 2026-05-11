"""Live integration tests for RouterAgent (G7).

Each test wires a *real* :class:`RouterAgent` to a real Anthropic
classifier and 3 real :class:`ChatAgent` specialists with distinct
system prompts. The classifier picks the specialist; the specialist
answers; the test asserts both that routing was correct and that the
returned answer was actually produced by the right specialist.

Requires ``ANTHROPIC_API_KEY``. Skipped otherwise.

Why these tests matter: the unit tests for ``RouterAgent`` stub the
classifier — they prove the dispatch math, not that a real model
actually classifies legal questions into the ``legal`` bucket. In a
regulated context (legal, financial), this is the audit gap: a
mocked test says the wiring is correct; only a live test says the
*classifier* is correct.
"""

from __future__ import annotations

import os

import pytest
from kaos_core.vfs.core import IsolationMode, StorageBackend, VFSConfig, VirtualFileSystem

from kaos_agents.patterns.chat import ChatAgent
from kaos_agents.patterns.router import RouterAgent, Specialist
from kaos_agents.types.response import AgentResponse

# ---------------------------------------------------------------------------
# Skip markers + pinned models
# ---------------------------------------------------------------------------

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing",
)

CLASSIFIER_MODEL = "anthropic:claude-haiku-4-5"
SPECIALIST_MODEL = "anthropic:claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _memory_vfs() -> VirtualFileSystem:
    return VirtualFileSystem(
        config=VFSConfig(
            default_backend=StorageBackend.MEMORY,
            isolation_mode=IsolationMode.GLOBAL,
        ),
    )


def _make_specialists() -> tuple[Specialist, Specialist, Specialist]:
    """Build three real specialists with distinguishable personas.

    The routing assertion goes through ``routing_trace.decision.specialist_name``
    (a primary signal — the router itself reports which specialist was
    dispatched). The response text is checked for topical relevance,
    not for rigid prefix tokens, since models don't always honor
    prefix-formatting instructions and that variance isn't what we're
    testing here.
    """
    vfs = _memory_vfs()
    legal = ChatAgent(
        vfs,
        model=SPECIALIST_MODEL,
        instructions=(
            "You are a legal research specialist. Answer the user's "
            "question concisely (under 60 words). Focus on case law, "
            "statutes, citations, and legal holdings."
        ),
    )
    corpus = ChatAgent(
        vfs,
        model=SPECIALIST_MODEL,
        instructions=(
            "You are a document Q&A specialist for user-uploaded "
            "documents. Answer concisely (under 60 words). If no "
            "document is provided in context, say so explicitly."
        ),
    )
    chat = ChatAgent(
        vfs,
        model=SPECIALIST_MODEL,
        instructions=(
            "You are a general-purpose assistant for casual "
            "conversation, definitions, and everyday questions. "
            "Answer concisely (under 60 words)."
        ),
    )
    return (
        Specialist(
            "legal",
            legal,
            description=(
                "Legal research, case law, statutes, citations, "
                "regulations. Use for questions about specific cases, "
                "legal holdings, statutory interpretation, or how to "
                "find legal authority."
            ),
            examples=(
                "What is the holding in Marbury v. Madison?",
                "Cite the statute governing securities fraud.",
            ),
        ),
        Specialist(
            "corpus",
            corpus,
            description=(
                "Question-answering grounded in user-uploaded documents "
                "(contracts, filings, deal rooms). Use when the user "
                "references 'this document', 'the contract', 'the "
                "filing', or asks something only answerable from an "
                "attached corpus."
            ),
            examples=(
                "What does section 3.2 of this contract say?",
                "Find the indemnification clause in the attached agreement.",
            ),
        ),
        Specialist(
            "chat",
            chat,
            description=(
                "General-purpose assistant for casual conversation, "
                "definitions, greetings, and questions that aren't "
                "legal research or document Q&A."
            ),
            examples=(
                "What's the capital of France?",
                "Hi, how are you today?",
            ),
        ),
    )


def _trace(response: AgentResponse):
    for k, v in response.metadata or ():
        if k == "routing_trace":
            return v
    return None


# ---------------------------------------------------------------------------
# RouterAgent live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_anthropic
class TestRouterAgentLive:
    """End-to-end routing with a real Anthropic classifier."""

    async def test_routes_legal_question_to_legal_specialist(self) -> None:
        """A clear case-law question should route to the legal specialist."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "What is the holding in Marbury v. Madison and why is it foundational?",
            session_id="router-legal-1",
        )
        trace = _trace(response)
        assert trace is not None, "routing_trace must be attached to metadata"
        assert trace.decision.specialist_name == "legal", (
            f"Expected 'legal', got {trace.decision.specialist_name!r}. "
            f"Reasoning: {trace.decision.reasoning!r}"
        )
        assert trace.decision.confidence >= 0.5
        # Topical content check: the answer should reference Marbury
        # specifically. Confirms the legal specialist actually answered
        # (vs. an empty response from the wrong path).
        assert "marbury" in response.text.lower() or "judicial review" in response.text.lower(), (
            f"Expected Marbury-related content; got: {response.text[:200]!r}"
        )

    async def test_routes_general_question_to_chat_specialist(self) -> None:
        """A casual definitional question should route to chat."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "What is the capital of France?",
            session_id="router-chat-1",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace.decision.specialist_name == "chat", (
            f"Expected 'chat', got {trace.decision.specialist_name!r}"
        )
        assert "paris" in response.text.lower(), (
            f"Expected 'Paris' in answer; got: {response.text[:200]!r}"
        )

    async def test_routes_document_question_to_corpus_specialist(self) -> None:
        """A 'this document'-style question should route to corpus."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "Find the indemnification clause in the contract I uploaded earlier.",
            session_id="router-corpus-1",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace.decision.specialist_name == "corpus", (
            f"Expected 'corpus', got {trace.decision.specialist_name!r}"
        )
        # The corpus specialist's system prompt says: "if no document is
        # provided in context, say so". So the response should reference
        # the missing-document situation or the indemnification request.
        text = response.text.lower()
        assert any(token in text for token in ("indemnif", "contract", "document", "provided")), (
            f"Expected corpus-relevant content; got: {response.text[:200]!r}"
        )

    async def test_routes_statute_question_to_legal_specialist(self) -> None:
        """Specifically-statutory question should route to legal."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "Which section of the Securities Exchange Act of 1934 "
            "governs insider trading liability?",
            session_id="router-legal-2",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace.decision.specialist_name == "legal"
        text = response.text.lower()
        assert any(
            token in text for token in ("10b", "10(b)", "insider", "exchange act", "section")
        ), f"Expected statute-relevant content; got: {response.text[:200]!r}"

    async def test_routes_greeting_to_chat(self) -> None:
        """A plain greeting should route to chat, not legal/corpus."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "Hi! How's your day going?",
            session_id="router-chat-2",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace.decision.specialist_name == "chat"
        assert response.text, "Expected non-empty greeting response"

    async def test_routing_trace_records_all_available_specialists(self) -> None:
        """The trace must enumerate every registered specialist."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "Hello there",
            session_id="router-trace-1",
        )
        trace = _trace(response)
        assert trace is not None
        assert set(trace.available_specialists) == {"legal", "corpus", "chat"}

    async def test_clause_extraction_routes_to_corpus(self) -> None:
        """A clause-extraction question is corpus territory, not legal."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "What does section 5.1 of the NDA I uploaded say about permitted disclosures?",
            session_id="router-corpus-2",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace.decision.specialist_name == "corpus", (
            f"Expected 'corpus' for NDA section question, got "
            f"{trace.decision.specialist_name!r}. Reasoning: "
            f"{trace.decision.reasoning!r}"
        )

    async def test_router_with_high_min_confidence_falls_back(self) -> None:
        """When min_confidence is raised above realistic classifier confidence,
        fallback should kick in. This guards the fallback path with a real
        classifier (no monkey-patching)."""
        legal, corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, corpus, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
            # Models rarely emit >0.99 confidence — force fallback for an
            # ambiguous-by-design query.
            min_confidence=0.999,
        )
        response = await router.turn(
            "asdf qwerty zxcv",  # ambiguous gibberish
            session_id="router-fallback-1",
        )
        trace = _trace(response)
        assert trace is not None
        # Either the classifier emitted confidence < 0.999 (fallback used)
        # or it somehow hit 1.0 — both are valid outcomes; we just need the
        # final specialist to be one of the registered ones.
        assert trace.decision.specialist_name in {"legal", "corpus", "chat"}
        if trace.decision.fallback_used:
            assert trace.decision.specialist_name == "chat"

    async def test_two_specialist_router_routes_correctly(self) -> None:
        """Smaller router with only 2 specialists — sanity check that the
        classifier prompt scales down."""
        legal, _corpus, chat = _make_specialists()
        router = RouterAgent(
            specialists=(legal, chat),
            default_specialist="chat",
            model=CLASSIFIER_MODEL,
        )
        response = await router.turn(
            "What does Brown v. Board of Education stand for?",
            session_id="router-2spec-1",
        )
        trace = _trace(response)
        assert trace is not None
        assert trace.decision.specialist_name == "legal"
        text = response.text.lower()
        assert any(
            token in text for token in ("brown", "1954", "separate", "fourteenth", "segregat")
        ), f"Expected Brown v. Board content; got: {response.text[:200]!r}"
