"""RouterAgent — specialist routing wrapper.

A :class:`RouterAgent` wraps N specialist :class:`KaosAgent` instances
and a thin LLM classifier. Each incoming message is classified into
exactly one specialist; the router then delegates the actual work to
that specialist's ``turn()`` and returns its response with a
``routing_trace`` entry attached to ``metadata``.

This is the generic / portable analogue of the "triage agent" pattern
that ships in OpenAI's Agents SDK and Anthropic's multi-agent
cookbooks. Distinct from ``Agent.handoffs``:

- ``handoffs`` are an option a working agent's LLM may emit. The
  primary agent is doing real work and *may* transfer control.
- ``RouterAgent`` does *only* dispatch. It has no domain instructions
  of its own. Use it when the right move is "pick a specialist, then
  step out of the way".

Architectural shape: ``RouterAgent`` is a typed wrapper, not a
pattern enum value. It composes with any :class:`KaosAgent`. No
runtime changes are required to use it — construct around your
specialists and call ``turn()`` / ``run()`` as usual.

Example::

    legal = ChatAgent(instructions="...legal research...", ...)
    corpus = ResearchAgent(instructions="...RAG over user docs...", ...)
    chat = ChatAgent(instructions="...general assistant...", ...)

    router = RouterAgent(
        specialists=(
            Specialist("legal", legal, "Legal research, citations, case law."),
            Specialist("corpus", corpus, "Q&A grounded in user-uploaded documents."),
            Specialist("chat", chat, "General conversation and anything else."),
        ),
        default_specialist="chat",
    )
    response = await router.turn("What's the holding in Marbury v. Madison?", "s1")
    # → classifier picks "legal" → routes there → returns its answer +
    #   ``("routing_trace", RoutingTrace(...))`` in metadata.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

if TYPE_CHECKING:
    from kaos_agents.base.agent import KaosAgent
    from kaos_agents.types.response import AgentResponse

logger = get_logger(__name__)


_DEFAULT_ROUTER_MODEL: str = "anthropic:claude-haiku-4-5"
"""Routing is a cheap classification task — default to a fast/cheap model."""

_MIN_CONFIDENCE: float = 0.3
"""Below this the classifier is treated as having given up — fall back."""


@dataclass(frozen=True, slots=True)
class Specialist:
    """One routable specialist agent.

    Attributes:
        name: Stable identifier used by the classifier (1-3 lowercase
            words, no spaces). Surfaces in the routing trace and audit
            logs. Must be unique within a router.
        agent: The wrapped :class:`KaosAgent` that actually answers the
            message when this specialist is selected.
        description: One- or two-sentence summary the classifier reads
            to decide whether this specialist should handle the query.
            Write it for an LLM, not a human — be specific about scope.
        examples: Optional example queries that this specialist should
            handle. Few-shot signal for the classifier; can be empty.
    """

    name: str
    agent: KaosAgent
    description: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The classifier's output for one incoming message.

    Attributes:
        specialist_name: Which specialist was picked. Always one of the
            registered :class:`Specialist` names, or — when the
            classifier abstained and a default was configured — the
            ``default_specialist``.
        confidence: 0..1 confidence the classifier reported. Clamped at
            construction.
        reasoning: One-sentence justification from the classifier.
        fallback_used: True when the classifier emitted an unknown
            specialist name or low confidence and the router fell back
            to ``default_specialist``.
    """

    specialist_name: str
    confidence: float
    reasoning: str
    fallback_used: bool = False


@dataclass(frozen=True, slots=True)
class RoutingTrace:
    """Trace attached to the response's ``metadata`` as ``routing_trace``."""

    decision: RoutingDecision
    available_specialists: tuple[str, ...]


class RouterAgent:
    """Wraps N specialists with classification-based dispatch.

    Each ``turn(message, session_id)`` call runs a cheap LLM
    classification, picks the best specialist, and delegates. The
    routing decision is appended to the response's ``metadata`` as a
    ``("routing_trace", RoutingTrace)`` entry.

    Args:
        specialists: One or more :class:`Specialist` definitions.
            Order is preserved in the classifier prompt. Names must
            be unique.
        default_specialist: Name of the specialist to fall back to
            when the classifier fails or returns low confidence. None
            means raise on routing failure instead of falling back.
        model: Model identifier for the classification call.
        min_confidence: Below this the classifier is treated as
            having abstained and ``default_specialist`` is used.
            Defaults to 0.3.

    The router itself never invokes tools — it only classifies and
    delegates. Cost is one cheap classifier call per turn plus
    whatever the chosen specialist costs.
    """

    def __init__(
        self,
        specialists: tuple[Specialist, ...],
        *,
        default_specialist: str | None = None,
        model: str = _DEFAULT_ROUTER_MODEL,
        min_confidence: float = _MIN_CONFIDENCE,
    ) -> None:
        if not specialists:
            raise ValueError(
                "RouterAgent.specialists must be non-empty. Provide one or "
                "more Specialist(name=, agent=, description=) entries, or "
                "use the wrapped agent directly instead of routing."
            )
        seen: set[str] = set()
        for s in specialists:
            if not s.name or not s.name.strip():
                raise ValueError("Specialist.name must be non-empty.")
            if s.name in seen:
                raise ValueError(
                    f"Duplicate Specialist.name {s.name!r}. Each specialist "
                    f"must have a unique name within a RouterAgent."
                )
            if not s.description or not s.description.strip():
                raise ValueError(
                    f"Specialist({s.name!r}).description must be non-empty. "
                    "The classifier reads this to decide whether to route "
                    "to this specialist."
                )
            seen.add(s.name)

        if default_specialist is not None and default_specialist not in seen:
            raise ValueError(
                f"RouterAgent.default_specialist={default_specialist!r} is not "
                f"a registered Specialist name. Known: {sorted(seen)}."
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"RouterAgent.min_confidence must be in [0, 1], got {min_confidence!r}."
            )

        self.specialists = specialists
        self.default_specialist = default_specialist
        self.model = model
        self.min_confidence = min_confidence
        self._by_name: dict[str, Specialist] = {s.name: s for s in specialists}

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        """Classify ``message``, then delegate to the chosen specialist."""
        decision = await self.classify(message)
        specialist = self._by_name[decision.specialist_name]

        logger.debug(
            "router: routed message=%r to specialist=%s confidence=%.3f fallback=%s",
            (message[:60] + "...") if len(message) > 60 else message,
            decision.specialist_name,
            decision.confidence,
            decision.fallback_used,
        )

        response = await specialist.agent.turn(message, session_id)
        trace = RoutingTrace(
            decision=decision,
            available_specialists=tuple(s.name for s in self.specialists),
        )
        return _attach_trace(response, trace)

    async def classify(self, message: str) -> RoutingDecision:
        """Run the routing classifier and resolve fallbacks.

        Returns a :class:`RoutingDecision` whose ``specialist_name`` is
        always a registered specialist. If the classifier emits an
        unknown name or confidence below ``min_confidence`` and a
        ``default_specialist`` is configured, the decision's
        ``fallback_used`` field is True.
        """
        raw_name, raw_confidence, raw_reasoning = await self._invoke_classifier(message)

        confidence = max(0.0, min(1.0, float(raw_confidence)))
        if raw_name in self._by_name and confidence >= self.min_confidence:
            return RoutingDecision(
                specialist_name=raw_name,
                confidence=confidence,
                reasoning=raw_reasoning,
            )

        # Classifier failed — pick the fallback.
        if self.default_specialist is None:
            raise RuntimeError(
                f"Router classifier returned specialist_name={raw_name!r} "
                f"(confidence={confidence:.2f}), which is not a registered "
                f"specialist (known: {sorted(self._by_name)}). "
                f"Set default_specialist=... on the RouterAgent to fall back "
                f"instead of raising."
            )
        return RoutingDecision(
            specialist_name=self.default_specialist,
            confidence=confidence,
            reasoning=(
                f"Fell back to {self.default_specialist!r}: classifier emitted "
                f"{raw_name!r} (confidence={confidence:.2f}, threshold="
                f"{self.min_confidence}). Original reasoning: {raw_reasoning}"
            ),
            fallback_used=True,
        )

    async def _invoke_classifier(self, message: str) -> tuple[str, float, str]:
        """Run the LLM classification call. Returns ``(name, confidence, reasoning)``.

        Lazy-imports kaos-llm-core so the agent module stays importable
        without the ``[llm]`` extra.
        """
        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core.programs.call import Call
        from kaos_llm_core.signatures.fields import InputField, OutputField
        from kaos_llm_core.signatures.signature import Signature

        class _RoutingSignature(Signature):
            """Classify a user message into exactly one specialist."""

            message: str = InputField(description="The user's message to route.")
            specialists: str = InputField(
                description=(
                    "Available specialists, formatted as 'name: description'. "
                    "Each line is one option."
                ),
            )
            specialist_name: str = OutputField(
                description=(
                    "Exactly one of the specialist names from the input. Lowercase, no spaces."
                ),
            )
            confidence: float = OutputField(
                description=(
                    "Confidence in the routing decision, 0.0..1.0. Use 0.5 "
                    "when the message could plausibly fit multiple "
                    "specialists; near 1.0 when one is clearly correct."
                ),
            )
            reasoning: str = OutputField(
                description="One-sentence justification for the choice.",
            )

        call = Call(_RoutingSignature, model=self.model)
        result = await call(
            message=message,
            specialists=self._format_specialist_catalog(),
        )
        return (
            str(result.specialist_name).strip(),
            float(result.confidence),
            str(result.reasoning),
        )

    def _format_specialist_catalog(self) -> str:
        """Render the specialist list as the classifier prompt input."""
        lines: list[str] = []
        for s in self.specialists:
            lines.append(f"{s.name}: {s.description}")
            if s.examples:
                for ex in s.examples:
                    lines.append(f"  example: {ex}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attach_trace(response: AgentResponse, trace: RoutingTrace) -> AgentResponse:
    """Return a new response with the routing trace appended to metadata.

    ``AgentResponse`` is a frozen dataclass with
    ``metadata: tuple[tuple[str, Any], ...]``. We use
    ``dataclasses.replace`` to build a new instance with the trace
    payload appended as a new ``("routing_trace", ...)`` entry.
    """
    existing: tuple[tuple[str, Any], ...] = response.metadata or ()
    new_metadata: tuple[tuple[str, Any], ...] = (*existing, ("routing_trace", trace))
    return dataclasses.replace(response, metadata=new_metadata)


__all__ = [
    "RouterAgent",
    "RoutingDecision",
    "RoutingTrace",
    "Specialist",
]
