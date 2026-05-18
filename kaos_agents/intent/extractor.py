"""IntentExtractor — Program that produces IntentResult from a message.

A small composing :class:`~kaos_llm_core.programs.base.Program` that
wraps an :class:`~kaos_llm_core.programs.call.Call` (or
:class:`~kaos_llm_core.programs.chain_of_thought.ChainOfThought`) over
:class:`~kaos_agents.intent.signature.IntentSignature` and projects the
returned Pydantic output into a :class:`~kaos_agents.intent.types.IntentResult`.

The extractor knows nothing about memory, triggers, or the agent loop.
It operates purely on ``(message, recent_messages, domain_examples)``
and returns a frozen :class:`IntentResult`. Phase 2's ``AgentLoop``
decides what to do with the result — escalate when
``requires_clarification=True``, dispatch to a planner otherwise.

Mirrors the kaos-llm-core :class:`Grounded` / :class:`Judge` composition
pattern: hold one Call as a child attribute (auto-registered into
``_child_registry`` for trace collection / optimization) and project
its output to a typed result type in ``forward()``.

Usage::

    extractor = IntentExtractor(model="anthropic:claude-haiku-4-5")
    invocation = await extractor.invoke(message="...", recent_messages="...")
    intent: IntentResult = invocation.output
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger
from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.call import Call
from kaos_llm_core.programs.chain_of_thought import ChainOfThought

from kaos_agents.config import AgentPattern
from kaos_agents.intent.signature import IntentSignature
from kaos_agents.intent.types import (
    Ambiguity,
    Constraint,
    Goal,
    IntentResult,
)
from kaos_agents.settings import DEFAULT_MODEL

logger = get_logger("kaos.agents.intent.extractor")


class IntentExtractor(Program):
    """Compose :class:`IntentSignature` into a typed :class:`IntentResult`.

    The underlying Call (or ChainOfThought wrapper) drives the LLM with
    :class:`IntentSignature`. ``forward()`` invokes it, then projects the
    Pydantic output onto a frozen :class:`IntentResult`. Trace collection,
    cost accounting, and validation retry come from kaos-llm-core's
    standard pipeline — this class adds nothing to those layers.

    Args:
        model: Provider-prefixed model id. Defaults to
            ``KAOS_AGENT_DEFAULT_LLM_MODEL`` (haiku-tier — intent
            extraction is the kind of cheap, structured task that does
            not need a frontier model).
        max_retries: Validation-retry attempts inside the inner Call.
            Defaults to 1; the LLM rarely fails to honor the schema.
        chain_of_thought: When True, wrap the Call in
            :class:`ChainOfThought` so the model emits explicit
            reasoning before the typed fields. Defaults to False since
            the IntentSignature's docstring already encodes the decision
            rules and the additional CoT field roughly doubles the
            output token count.
        instructions_override: Replace the IntentSignature docstring's
            instructions for this extractor instance only. Useful for
            optimizers (InstructionOptimizer / MiproV2) that learn an
            improved instruction string.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_retries: int = 1,
        chain_of_thought: bool = False,
        instructions_override: str | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._max_retries = max_retries
        self._chain_of_thought = chain_of_thought
        self._instructions_override = instructions_override

        # Construct the inner Call. ChainOfThought is itself a Call
        # subclass, so the assignment auto-registers as a child of this
        # Program either way (see Program.__setattr__).
        call_kwargs: dict[str, Any] = {
            "model": model,
            "max_retries": max_retries,
        }
        if instructions_override is not None:
            call_kwargs["instructions"] = instructions_override

        if chain_of_thought:
            self._call: Call = ChainOfThought(IntentSignature, **call_kwargs)
        else:
            self._call = Call(IntentSignature, **call_kwargs)

    @property
    def model(self) -> str:
        """The provider-prefixed model id this extractor uses."""
        return self._model

    @property
    def chain_of_thought(self) -> bool:
        """Whether this extractor wraps the inner Call in ChainOfThought."""
        return self._chain_of_thought

    async def forward(self, **kwargs: Any) -> IntentResult:
        """Run the inner Call and project its output to :class:`IntentResult`.

        ``Program.invoke`` opens a trace collector and calls this method,
        so every successful inner-Call execution flows into the
        extractor's parent trace automatically.

        Inputs (keyword-only):

        * ``message`` — required: the user's natural-language request.
        * ``recent_messages`` — optional: flattened conversation context.
        * ``domain_examples`` — optional: calibration examples (paper §3.5).
        * ``corpus_attached`` — optional, default ``False``: whether the
          session has documents in memory (SessionMemory.DOCUMENTS).
          When ``True`` the classifier prefers RESEARCH for indirect
          document references (IntentSignature rule 6).
        * ``corpus_size`` — optional, default ``0``: document count in
          the corpus, calibration signal for rule 6.
        * ``corpus_headlines`` — optional, default ``""``: newline-
          separated filenames of attached corpus items. Used by rule 7
          to populate the structured ``targets`` output. The extractor
          validates each emitted target against this list and drops
          unknown filenames before constructing the IntentResult.

        The projection is non-destructive: the IntentSignature output's
        fields land verbatim on the IntentResult, plus ``raw_input`` is
        populated from ``message`` so downstream consumers can
        re-anchor :class:`Ambiguity` spans.

        ``**kwargs: Any`` matches :meth:`Program.forward` (LSP). Unknown
        kwargs propagate to the inner Call which rejects them via its
        Signature input validator.
        """
        if "message" not in kwargs:
            raise TypeError("IntentExtractor.forward() missing required keyword: 'message'")
        message = kwargs["message"]
        recent_messages = kwargs.get("recent_messages", "")
        domain_examples = kwargs.get("domain_examples", "")
        corpus_attached = bool(kwargs.get("corpus_attached", False))
        corpus_size = int(kwargs.get("corpus_size", 0))
        corpus_headlines = str(kwargs.get("corpus_headlines", ""))
        invocation = await self._call.invoke(
            message=message,
            recent_messages=recent_messages,
            domain_examples=domain_examples,
            corpus_attached=corpus_attached,
            corpus_size=corpus_size,
            corpus_headlines=corpus_headlines,
        )
        return _project_signature_output(
            invocation.output,
            raw_input=message,
            corpus_headlines=corpus_headlines,
        )


def _project_signature_output(
    output: Any,
    *,
    raw_input: str,
    corpus_headlines: str = "",
) -> IntentResult:
    """Project an :class:`IntentSignature` output onto :class:`IntentResult`.

    Single source of truth for the Signature → IntentResult mapping;
    isolated here so test stubs can call it directly without round-tripping
    through a Call.

    Pulls every IntentResult field from the output by name. Coerces
    bare BaseModel goal/constraint/ambiguity values via ``model_dump``
    / ``model_validate`` so the resulting IntentResult is a *frozen*
    pydantic model regardless of what the inner Call's output_model
    looked like.

    ``corpus_headlines`` (the same input the signature received) is the
    allow-list used to validate the structured ``targets`` output. The
    first whitespace-token of each non-empty line is the canonical
    filename — that's the same anchor the IntentSignature docstring
    instructs the model to emit. Targets that don't match any allowed
    filename are dropped (the extractor's contract per
    :class:`IntentSignature` rule 7) and logged so we can spot model
    drift.
    """
    goal = _coerce_goal(output.goal)
    constraints = tuple(_coerce_constraint(c) for c in getattr(output, "constraints", ()) or ())
    ambiguities = tuple(_coerce_ambiguity(a) for a in getattr(output, "ambiguities", ()) or ())
    requires_clarification = bool(getattr(output, "requires_clarification", False))
    pattern_value = getattr(output, "pattern", AgentPattern.CHAT)
    pattern = (
        pattern_value
        if isinstance(pattern_value, AgentPattern)
        else AgentPattern(str(pattern_value))
    )
    confidence = float(getattr(output, "confidence", 1.0))
    # IntentResult.model_validate enforces the 0.0-1.0 bound, but a
    # mis-tuned LLM might emit slightly out-of-range values; clamp here
    # so an off-by-epsilon doesn't propagate as a ValidationError.
    confidence = max(0.0, min(1.0, confidence))

    targets = _coerce_targets(
        getattr(output, "targets", ()) or (),
        corpus_headlines=corpus_headlines,
    )

    return IntentResult(
        goal=goal,
        constraints=constraints,
        ambiguities=ambiguities,
        requires_clarification=requires_clarification,
        pattern=pattern,
        confidence=confidence,
        targets=targets,
        raw_input=raw_input,
    )


def _parse_headline_filenames(corpus_headlines: str) -> frozenset[str]:
    """Extract the canonical filename anchor from each non-empty headline line.

    Each headline line follows the convention ``"<filename> — <metadata>"``
    (see ``IntentSignature.corpus_headlines`` docstring). We take the text
    before the first occurrence of an em dash (``—``) or hyphen-with-spaces
    (``" - "``) as the canonical filename; if neither separator is
    present, the full stripped line is the filename. Empty headlines
    (no corpus attached) yield an empty allow-list — every emitted
    target gets dropped, which is the correct behavior for non-corpus
    sessions.
    """
    allowed: set[str] = set()
    for line in corpus_headlines.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Three distinct Unicode separators (em dash / en dash /
        # hyphen-minus). The intentional non-ASCII characters trip
        # ruff's ambiguous-Unicode check; suppressed on the
        # exact line.
        for sep in (" — ", " – ", " - "):  # noqa: RUF001
            if sep in stripped:
                stripped = stripped.split(sep, 1)[0].strip()
                break
        if stripped:
            allowed.add(stripped)
    return frozenset(allowed)


def _coerce_targets(
    raw_targets: Any,
    *,
    corpus_headlines: str,
) -> tuple[str, ...]:
    """Validate model-emitted targets against the corpus allow-list.

    Every accepted target must appear verbatim in ``corpus_headlines``.
    Empty ``corpus_headlines`` (no corpus attached) drops all emitted
    targets — there is no allow-list to match against, so the
    "MUST appear verbatim" contract from IntentSignature rule 7 rejects
    everything. The caller can distinguish "no corpus, targets dropped"
    from "corpus present, all rejected" by inspecting
    ``corpus_attached`` on the IntentResult.

    Order of accepted targets is preserved; a single deduplication pass
    keeps the tuple stable for downstream consumers that key on it.
    """
    if not raw_targets:
        return ()
    allowed = _parse_headline_filenames(corpus_headlines)
    accepted: list[str] = []
    seen: set[str] = set()
    for raw in raw_targets:
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if not name or name in seen:
            continue
        if name not in allowed:
            logger.debug("intent: dropping unknown target %r (not in corpus_headlines)", name)
            continue
        seen.add(name)
        accepted.append(name)
    return tuple(accepted)


def _coerce_goal(value: Any) -> Goal:
    """Return ``value`` as a frozen :class:`Goal` instance."""
    if isinstance(value, Goal):
        return value
    if hasattr(value, "model_dump"):
        return Goal.model_validate(value.model_dump())
    return Goal.model_validate(value)


def _coerce_constraint(value: Any) -> Constraint:
    """Return ``value`` as a frozen :class:`Constraint` instance."""
    if isinstance(value, Constraint):
        return value
    if hasattr(value, "model_dump"):
        return Constraint.model_validate(value.model_dump())
    return Constraint.model_validate(value)


def _coerce_ambiguity(value: Any) -> Ambiguity:
    """Return ``value`` as a frozen :class:`Ambiguity` instance."""
    if isinstance(value, Ambiguity):
        return value
    if hasattr(value, "model_dump"):
        return Ambiguity.model_validate(value.model_dump())
    return Ambiguity.model_validate(value)
