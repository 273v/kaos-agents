"""Unit tests for :class:`kaos_agents.patterns.router.RouterAgent`.

These tests pin the classifier-contract refactor that swapped the
function-local ``_RoutingSignature`` for
:class:`kaos_llm_core.FewShotClassify`. The contract under test:

1. Picked specialist with confidence >= ``min_confidence`` is routed
   to that specialist's ``.agent.turn(...)``.
2. Picked specialist with confidence < ``min_confidence`` falls back
   to ``default_specialist`` (when configured) or raises.
3. ``ABSTAIN_LABEL`` from the classifier triggers the same fallback.
4. The router's ``classify()`` (cost-less public API) returns the
   same ``RoutingDecision`` as ``_classify_with_cost`` minus the
   cost component.
5. ``RoutingTrace`` is attached to the routed response's metadata
   with ``classifier_cost_usd`` populated from the Program trace.

The classifier Program is replaced via the
``_classifier_program`` cache attribute (set before any classify
call) so these tests run without ``[llm]`` extras or live network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kaos_agents.patterns.router import RouterAgent, Specialist
from kaos_agents.types.intents import IntentResult, IntentType
from kaos_agents.types.response import AgentResponse


def _make_response(text: str) -> AgentResponse:
    return AgentResponse(
        text=text,
        intent=IntentResult(intent=IntentType.RESPOND, confidence=1.0, reasoning="stub"),
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _StubClassification:
    """Minimal stand-in for :class:`kaos_llm_core.results.Classification`."""

    labels: list[Any]
    scores: dict[str, float]
    abstained: bool = False
    rationale: str | None = None


@dataclass
class _StubLabel:
    name: str


@dataclass
class _StubTrace:
    cost_usd: float = 0.0


class _StubClassifier:
    """Stand-in for ``FewShotClassify`` / ``ZeroShotClassify``.

    Configurable via the ``verdicts`` mapping: maps an incoming
    ``text`` to the ``Classification`` we want the router to see.
    Records the cost on ``last_trace`` so :meth:`_invoke_classifier`'s
    trace rollup has something to read.
    """

    def __init__(
        self,
        *,
        default: _StubClassification,
        verdicts: dict[str, _StubClassification] | None = None,
        cost_usd: float = 0.0001,
    ) -> None:
        self._default = default
        self._verdicts = verdicts or {}
        self._cost = cost_usd
        self.last_trace = _StubTrace(cost_usd=cost_usd)
        self.calls: list[str] = []

    async def __call__(self, *, text: str) -> _StubClassification:
        self.calls.append(text)
        return self._verdicts.get(text, self._default)


class _StubAgent:
    """Minimal :class:`kaos_agents.base.agent.KaosAgent` stand-in."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.turns: list[tuple[str, str]] = []

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        self.turns.append((message, session_id))
        return _make_response(f"[{self._name}] {message}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def legal_agent() -> _StubAgent:
    return _StubAgent("legal")


@pytest.fixture
def chat_agent() -> _StubAgent:
    return _StubAgent("chat")


@pytest.fixture
def specialists(legal_agent: _StubAgent, chat_agent: _StubAgent) -> tuple[Specialist, ...]:
    return (
        Specialist(
            name="legal",
            agent=legal_agent,  # ty: ignore[invalid-argument-type]
            description="Legal research, citations, case law.",
            examples=("Find the holding in Marbury v. Madison.",),
        ),
        Specialist(
            name="chat",
            agent=chat_agent,  # ty: ignore[invalid-argument-type]
            description="General conversation.",
            examples=("Hello, how are you?",),
        ),
    )


def _install_classifier(router: RouterAgent, classifier: _StubClassifier) -> None:
    """Pre-populate the lazy-built classifier cache so the router
    never tries to import kaos-llm-core in the test runtime.
    """
    router._classifier_program = classifier


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRouterAgentClassifier:
    """The classifier contract: high-confidence → route, low → fallback / raise."""

    async def test_high_confidence_routes_to_named_specialist(
        self,
        specialists: tuple[Specialist, ...],
        legal_agent: _StubAgent,
    ) -> None:
        router = RouterAgent(specialists=specialists, default_specialist="chat")
        _install_classifier(
            router,
            _StubClassifier(
                default=_StubClassification(
                    labels=[_StubLabel(name="legal")],
                    scores={"legal": 0.92},
                    rationale="Asks about a case holding.",
                ),
            ),
        )

        response = await router.turn("What is the holding in Marbury v. Madison?", session_id="s1")

        # Routed to the legal specialist
        assert legal_agent.turns == [
            ("What is the holding in Marbury v. Madison?", "s1"),
        ]
        # Response carries the routing trace
        trace_entry = dict(response.metadata).get("routing_trace")
        assert trace_entry is not None, f"no routing_trace in metadata: {response.metadata!r}"
        assert trace_entry.decision.specialist_name == "legal"
        assert trace_entry.decision.confidence == pytest.approx(0.92)
        assert trace_entry.decision.fallback_used is False
        assert trace_entry.classifier_cost_usd == pytest.approx(0.0001)

    async def test_low_confidence_falls_back_to_default(
        self,
        specialists: tuple[Specialist, ...],
        chat_agent: _StubAgent,
    ) -> None:
        router = RouterAgent(
            specialists=specialists,
            default_specialist="chat",
            min_confidence=0.5,
        )
        _install_classifier(
            router,
            _StubClassifier(
                default=_StubClassification(
                    labels=[_StubLabel(name="legal")],
                    scores={"legal": 0.18},
                    rationale="Maybe legal? unclear.",
                ),
            ),
        )

        response = await router.turn("hi", session_id="s2")

        assert chat_agent.turns == [("hi", "s2")]
        trace_entry = dict(response.metadata).get("routing_trace")
        assert trace_entry is not None
        assert trace_entry.decision.specialist_name == "chat"
        assert trace_entry.decision.fallback_used is True

    async def test_abstain_label_triggers_fallback(
        self,
        specialists: tuple[Specialist, ...],
        chat_agent: _StubAgent,
    ) -> None:
        from kaos_llm_core.labels import ABSTAIN_LABEL

        router = RouterAgent(specialists=specialists, default_specialist="chat")
        _install_classifier(
            router,
            _StubClassifier(
                default=_StubClassification(
                    labels=[],
                    scores={ABSTAIN_LABEL: 0.0},
                    abstained=True,
                    rationale="Could not decide.",
                ),
            ),
        )

        response = await router.turn("...", session_id="s3")

        assert chat_agent.turns == [("...", "s3")]
        trace_entry = dict(response.metadata).get("routing_trace")
        assert trace_entry is not None
        assert trace_entry.decision.specialist_name == "chat"
        assert trace_entry.decision.fallback_used is True

    async def test_no_fallback_raises_on_low_confidence(
        self,
        specialists: tuple[Specialist, ...],
    ) -> None:
        router = RouterAgent(specialists=specialists, default_specialist=None, min_confidence=0.5)
        _install_classifier(
            router,
            _StubClassifier(
                default=_StubClassification(
                    labels=[_StubLabel(name="legal")],
                    scores={"legal": 0.1},
                ),
            ),
        )

        with pytest.raises(RuntimeError, match="default_specialist"):
            await router.turn("foo", session_id="s4")

    async def test_classify_returns_same_decision_as_internal(
        self,
        specialists: tuple[Specialist, ...],
    ) -> None:
        router = RouterAgent(specialists=specialists, default_specialist="chat")
        _install_classifier(
            router,
            _StubClassifier(
                default=_StubClassification(
                    labels=[_StubLabel(name="legal")],
                    scores={"legal": 0.88},
                    rationale="legal-flavored question.",
                ),
            ),
        )

        decision = await router.classify("Find the case I mentioned.")
        assert decision.specialist_name == "legal"
        assert decision.confidence == pytest.approx(0.88)
        assert decision.fallback_used is False

    async def test_unknown_specialist_falls_back(
        self,
        specialists: tuple[Specialist, ...],
        chat_agent: _StubAgent,
    ) -> None:
        """Defense-in-depth: if the classifier emits a label name that
        isn't in the registered specialists (should be prevented by the
        LabelSet, but guard anyway), we still fall back cleanly.
        """
        router = RouterAgent(specialists=specialists, default_specialist="chat")
        _install_classifier(
            router,
            _StubClassifier(
                default=_StubClassification(
                    labels=[_StubLabel(name="medical")],  # not registered
                    scores={"medical": 0.95},
                ),
            ),
        )

        response = await router.turn("...", session_id="s5")

        assert chat_agent.turns == [("...", "s5")]
        trace_entry = dict(response.metadata).get("routing_trace")
        assert trace_entry is not None
        assert trace_entry.decision.specialist_name == "chat"
        assert trace_entry.decision.fallback_used is True


class TestRouterAgentConstruction:
    """Validation contract on __init__ — unchanged by the refactor."""

    def test_rejects_empty_specialists(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RouterAgent(specialists=())

    def test_rejects_duplicate_specialist_names(self, chat_agent: _StubAgent) -> None:
        dup = (
            Specialist(name="x", agent=chat_agent, description="d"),  # ty: ignore[invalid-argument-type]
            Specialist(name="x", agent=chat_agent, description="d"),  # ty: ignore[invalid-argument-type]
        )
        with pytest.raises(ValueError, match="Duplicate"):
            RouterAgent(specialists=dup)

    def test_rejects_unknown_default_specialist(
        self,
        specialists: tuple[Specialist, ...],
    ) -> None:
        with pytest.raises(ValueError, match="not"):
            RouterAgent(specialists=specialists, default_specialist="not-real")

    def test_rejects_out_of_range_min_confidence(
        self,
        specialists: tuple[Specialist, ...],
    ) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            RouterAgent(specialists=specialists, min_confidence=1.5)
