"""RouterAgent — specialist routing wrapper.

A :class:`RouterAgent` wraps N specialist :class:`KaosAgent` instances
and a thin LLM classifier. Each incoming message is classified into
exactly one specialist; the router then delegates the actual work to
that specialist's ``turn()`` and returns its response with a
``routing_trace`` entry attached to ``metadata``.

Design: the classifier is :class:`kaos_llm_core.FewShotClassify` over
a :class:`~kaos_llm_core.labels.LabelSet` built from the registered
:class:`Specialist` definitions. The specialist names become labels,
descriptions become label descriptions, and per-specialist
``examples`` become labelled :class:`~kaos_llm_core.Example`
demonstrations attached to the underlying :class:`~kaos_llm_core.Call`.

This replaces a hand-maintained function-local ``_RoutingSignature``
+ raw ``Call`` from a previous iteration. The benefit is the
classification surface now composes with the rest of kaos-llm-core:
optimizers (MIPRO / BootstrapOptimizer) can tune the example pool
without touching this file, ``LabelSet`` supports an explicit
``ABSTAIN_LABEL`` fall-through, and trace/cost rollups go through
the same :class:`~kaos_llm_core.observability.ExecutionTrace`
machinery the rest of the runtime uses.

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
    """Trace attached to the response's ``metadata`` as ``routing_trace``.

    ``classifier_cost_usd`` is the USD cost of the routing classifier
    call. Populated when the classifier is an LLM call routed through
    ``Call.invoke()`` so ``Invocation.usage.cost_usd`` is reachable;
    0.0 for deterministic / cached / non-LLM routers.
    """

    decision: RoutingDecision
    available_specialists: tuple[str, ...]
    classifier_cost_usd: float = 0.0


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

    # Lazy-built classifier Program (None until first invocation). Cached
    # on the instance because building it imports kaos-llm-core, which is
    # the optional ``[llm]`` extra. See :meth:`_get_classifier`.
    _classifier_program: Any | None

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
        self._classifier_program = None

    async def turn(self, message: str, session_id: str) -> AgentResponse:
        """Classify ``message``, then delegate to the chosen specialist."""
        decision, classifier_cost = await self._classify_with_cost(message)
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
            classifier_cost_usd=classifier_cost,
        )
        return _attach_trace(response, trace)

    async def classify(self, message: str) -> RoutingDecision:
        """Run the routing classifier and resolve fallbacks.

        Returns a :class:`RoutingDecision` whose ``specialist_name`` is
        always a registered specialist. If the classifier emits an
        unknown name or confidence below ``min_confidence`` and a
        ``default_specialist`` is configured, the decision's
        ``fallback_used`` field is True.

        Public API. For routing-with-cost (used internally by
        :meth:`turn`) see :meth:`_classify_with_cost`.
        """
        decision, _cost = await self._classify_with_cost(message)
        return decision

    async def _classify_with_cost(self, message: str) -> tuple[RoutingDecision, float]:
        """Run the classifier and return ``(decision, classifier_cost_usd)``.

        Internal counterpart to :meth:`classify` — :meth:`turn` uses
        this to populate ``RoutingTrace.classifier_cost_usd`` without
        widening the public :meth:`classify` return signature.
        """
        raw_name, raw_confidence, raw_reasoning, cost = await self._invoke_classifier(message)

        confidence = max(0.0, min(1.0, float(raw_confidence)))
        if raw_name in self._by_name and confidence >= self.min_confidence:
            return (
                RoutingDecision(
                    specialist_name=raw_name,
                    confidence=confidence,
                    reasoning=raw_reasoning,
                ),
                cost,
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
        return (
            RoutingDecision(
                specialist_name=self.default_specialist,
                confidence=confidence,
                reasoning=(
                    f"Fell back to {self.default_specialist!r}: classifier emitted "
                    f"{raw_name!r} (confidence={confidence:.2f}, threshold="
                    f"{self.min_confidence}). Original reasoning: {raw_reasoning}"
                ),
                fallback_used=True,
            ),
            cost,
        )

    def _get_classifier(self) -> Any:
        """Lazy-build and cache the :class:`FewShotClassify` /
        :class:`ZeroShotClassify` Program for this router.

        Construction is deferred to first invocation because
        ``kaos_llm_core`` is the optional ``[llm]`` extra. The Program
        captures the :class:`LabelSet` (one Label per Specialist) and
        the :class:`Example` set (one Example per
        ``Specialist.examples`` entry) for the lifetime of this
        :class:`RouterAgent`.
        """
        if self._classifier_program is not None:
            return self._classifier_program

        from kaos_agents._llm_imports import require_llm_core

        require_llm_core()
        from kaos_llm_core import Example, FewShotClassify, ZeroShotClassify
        from kaos_llm_core.labels import Label, LabelSet

        labels = LabelSet(
            labels=[
                Label(
                    name=s.name,
                    description=s.description,
                    examples=list(s.examples),
                )
                for s in self.specialists
            ],
            exclusive=True,
            allow_abstain=self.default_specialist is not None,
        )
        examples: list[Example] = []
        for s in self.specialists:
            for ex in s.examples:
                examples.append(
                    Example(
                        inputs={"text": ex},
                        outputs={"label": s.name, "confidence": 0.95, "rationale": ""},
                    )
                )

        if examples:
            program = FewShotClassify(labels=labels, examples=examples, model=self.model)
        else:
            program = ZeroShotClassify(labels=labels, model=self.model)

        self._classifier_program = program
        return program

    async def _invoke_classifier(self, message: str) -> tuple[str, float, str, float]:
        """Run the :class:`FewShotClassify` Program against ``message``.

        Returns ``(name, confidence, reasoning, cost_usd)``. The
        Program emits a :class:`~kaos_llm_core.results.Classification`;
        we unpack the picked label, its confidence score, and the
        classifier's rationale. When the LabelSet allows abstention
        and the model abstains, we return ``ABSTAIN_LABEL`` and let
        :meth:`_classify_with_cost` apply the configured fallback.

        Cost is read from the per-call invocation's
        :class:`~kaos_llm_core.programs.Invocation` via the trace tree
        the Program records — the Classifier's ``last_trace`` carries
        cost_usd populated by the kaos-llm-core Call layer.
        """
        from kaos_llm_core.labels import ABSTAIN_LABEL

        program = self._get_classifier()
        classification = await program(text=message)

        # Pull rationale + picked label + confidence.
        rationale = str(classification.rationale or "")
        if classification.abstained:
            name = ABSTAIN_LABEL
            confidence = float(classification.scores.get(ABSTAIN_LABEL, 0.0))
        else:
            picked = classification.labels[0]
            name = picked.name
            confidence = float(classification.scores.get(picked.name, 0.0))

        # Cost rollup from the Program's last trace tree.
        cost = 0.0
        last_trace = getattr(program, "last_trace", None)
        if last_trace is not None:
            cost = float(getattr(last_trace, "cost_usd", 0.0) or 0.0)

        return (name, confidence, rationale, cost)


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
