"""Unit tests for kaos_agents.patterns.router.

Deterministic — exercises the dispatch + fallback control flow with
a stub classifier (monkeypatched ``_invoke_classifier``) and stub
specialist agents. Live LLM integration is tested separately.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from kaos_agents.base.agent import KaosAgent
from kaos_agents.patterns.router import (
    RouterAgent,
    RoutingDecision,
    RoutingTrace,
    Specialist,
)
from kaos_agents.types import IntentResult, IntentType
from kaos_agents.types.response import AgentResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str) -> AgentResponse:
    return AgentResponse.create(
        text=text,
        intent=IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="stub"),
        tool_calls=(),
        turn_number=1,
        tokens_used=0,
    )


class _StubSpecialist:
    """Records turn() calls and returns a canned response."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[str, str]] = []

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        self.calls.append((message, session_id))
        return _make_response(f"[{self.label}] {message}")


def _trace_from_metadata(response: AgentResponse) -> RoutingTrace | None:
    for k, v in response.metadata or ():
        if k == "routing_trace":
            return cast("RoutingTrace", v)
    return None


def _make_router(
    *,
    classifier_result: tuple[str, float, str],
    default_specialist: str | None = None,
    min_confidence: float = 0.3,
) -> tuple[RouterAgent, dict[str, _StubSpecialist]]:
    """Build a RouterAgent with stub specialists and a monkeypatched classifier."""
    legal = _StubSpecialist("legal")
    corpus = _StubSpecialist("corpus")
    chat = _StubSpecialist("chat")

    router = RouterAgent(
        specialists=(
            Specialist("legal", cast("KaosAgent", legal), "Legal research"),
            Specialist("corpus", cast("KaosAgent", corpus), "Corpus Q&A"),
            Specialist("chat", cast("KaosAgent", chat), "General chat"),
        ),
        default_specialist=default_specialist,
        min_confidence=min_confidence,
    )

    async def fake_classify(_message: str) -> tuple[str, float, str]:
        return classifier_result

    # Bypass the LLM call.
    router._invoke_classifier = fake_classify  # ty: ignore[invalid-assignment]
    return router, {"legal": legal, "corpus": corpus, "chat": chat}


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestRouterConstruction:
    def test_empty_specialists_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RouterAgent(specialists=())

    def test_duplicate_specialist_name_rejected(self) -> None:
        stub = cast("KaosAgent", _StubSpecialist("x"))
        with pytest.raises(ValueError, match="Duplicate"):
            RouterAgent(
                specialists=(
                    Specialist("foo", stub, "first"),
                    Specialist("foo", stub, "second"),
                ),
            )

    def test_blank_specialist_name_rejected(self) -> None:
        stub = cast("KaosAgent", _StubSpecialist("x"))
        with pytest.raises(ValueError, match="non-empty"):
            RouterAgent(specialists=(Specialist("", stub, "desc"),))

    def test_blank_description_rejected(self) -> None:
        stub = cast("KaosAgent", _StubSpecialist("x"))
        with pytest.raises(ValueError, match="description"):
            RouterAgent(specialists=(Specialist("foo", stub, ""),))

    def test_unknown_default_specialist_rejected(self) -> None:
        stub = cast("KaosAgent", _StubSpecialist("x"))
        with pytest.raises(ValueError, match="not a registered"):
            RouterAgent(
                specialists=(Specialist("foo", stub, "desc"),),
                default_specialist="bar",
            )

    def test_min_confidence_out_of_range(self) -> None:
        stub = cast("KaosAgent", _StubSpecialist("x"))
        with pytest.raises(ValueError, match="min_confidence"):
            RouterAgent(
                specialists=(Specialist("foo", stub, "desc"),),
                min_confidence=1.5,
            )


# ---------------------------------------------------------------------------
# RoutingDecision + RoutingTrace value types
# ---------------------------------------------------------------------------


class TestRoutingValueTypes:
    def test_routing_decision_frozen(self) -> None:
        d = RoutingDecision(specialist_name="x", confidence=0.9, reasoning="ok")
        with pytest.raises((AttributeError, TypeError)):
            d.confidence = 0.1  # ty: ignore[invalid-assignment]

    def test_routing_trace_frozen(self) -> None:
        d = RoutingDecision(specialist_name="x", confidence=0.9, reasoning="ok")
        t = RoutingTrace(decision=d, available_specialists=("x", "y"))
        with pytest.raises((AttributeError, TypeError)):
            t.decision = d  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Dispatch — happy path
# ---------------------------------------------------------------------------


class TestRouterDispatch:
    def test_routes_to_classified_specialist(self) -> None:
        router, stubs = _make_router(
            classifier_result=("legal", 0.92, "case-law question"),
        )
        response = asyncio.run(router.turn("Marbury v. Madison?", "s1"))
        # Specialist saw the message
        assert stubs["legal"].calls == [("Marbury v. Madison?", "s1")]
        assert stubs["corpus"].calls == []
        assert stubs["chat"].calls == []
        # Specialist's response is what bubbles up
        assert response.text == "[legal] Marbury v. Madison?"
        # Routing trace attached
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace.decision.specialist_name == "legal"
        assert trace.decision.confidence == pytest.approx(0.92)
        assert trace.decision.fallback_used is False
        assert trace.available_specialists == ("legal", "corpus", "chat")

    def test_routes_a_second_specialist(self) -> None:
        router, stubs = _make_router(
            classifier_result=("corpus", 0.78, "doc Q&A"),
        )
        response = asyncio.run(router.turn("what does section 3.2 say?", "s2"))
        assert stubs["corpus"].calls == [("what does section 3.2 say?", "s2")]
        assert response.text == "[corpus] what does section 3.2 say?"

    def test_confidence_clamped_to_unit_interval(self) -> None:
        # Classifier emits an out-of-range confidence — should clamp.
        router, _ = _make_router(classifier_result=("legal", 1.7, "very sure"))
        response = asyncio.run(router.turn("q", "s"))
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace.decision.confidence == 1.0


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestRouterFallback:
    def test_unknown_name_with_default_falls_back(self) -> None:
        router, stubs = _make_router(
            classifier_result=("plumbing", 0.95, "not sure"),
            default_specialist="chat",
        )
        response = asyncio.run(router.turn("hello", "s"))
        # Routed to default
        assert stubs["chat"].calls == [("hello", "s")]
        assert stubs["legal"].calls == []
        # Trace flags the fallback
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace.decision.specialist_name == "chat"
        assert trace.decision.fallback_used is True
        assert "plumbing" in trace.decision.reasoning

    def test_low_confidence_with_default_falls_back(self) -> None:
        # Classifier picks a valid name but confidence is below threshold.
        router, stubs = _make_router(
            classifier_result=("legal", 0.1, "guessing"),
            default_specialist="chat",
            min_confidence=0.3,
        )
        response = asyncio.run(router.turn("hmm", "s"))
        assert stubs["chat"].calls == [("hmm", "s")]
        assert stubs["legal"].calls == []
        trace = _trace_from_metadata(response)
        assert trace is not None
        assert trace.decision.fallback_used is True

    def test_unknown_name_without_default_raises(self) -> None:
        router, _ = _make_router(
            classifier_result=("nonsense", 0.9, "shrug"),
            default_specialist=None,
        )
        with pytest.raises(RuntimeError, match="not a registered specialist"):
            asyncio.run(router.turn("q", "s"))

    def test_valid_name_at_threshold_is_not_fallback(self) -> None:
        # Exactly at the threshold counts as confident.
        router, stubs = _make_router(
            classifier_result=("legal", 0.3, "okay"),
            default_specialist="chat",
            min_confidence=0.3,
        )
        asyncio.run(router.turn("q", "s"))
        assert stubs["legal"].calls == [("q", "s")]
        assert stubs["chat"].calls == []


# ---------------------------------------------------------------------------
# Catalog rendering
# ---------------------------------------------------------------------------


class TestCatalogRendering:
    def test_simple_catalog(self) -> None:
        stub = cast("KaosAgent", _StubSpecialist("x"))
        router = RouterAgent(
            specialists=(
                Specialist("legal", stub, "Legal research"),
                Specialist("chat", stub, "General chat"),
            ),
        )
        rendered = router._format_specialist_catalog()
        assert "legal: Legal research" in rendered
        assert "chat: General chat" in rendered

    def test_catalog_with_examples(self) -> None:
        stub = cast("KaosAgent", _StubSpecialist("x"))
        router = RouterAgent(
            specialists=(
                Specialist(
                    "legal",
                    stub,
                    "Legal research",
                    examples=("What's the holding in X?", "Cite the relevant rule."),
                ),
            ),
        )
        rendered = router._format_specialist_catalog()
        assert "What's the holding in X?" in rendered
        assert "Cite the relevant rule." in rendered
        assert "example:" in rendered
