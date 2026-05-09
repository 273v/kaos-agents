"""ClarificationLoop — handles ``requires_clarification=True`` intents.

When an :class:`~kaos_agents.intent.types.IntentResult` comes back with
``requires_clarification=True``, the agent loop has two options:

(a) **Escalate to the user** — emit a ``ClarificationRequired`` event
    and pause the run for human input. This is the Phase 4 path
    (see ``kaos_agents.escalation``); it is not implemented in Phase 1.A.
(b) **Re-prompt the LLM** — feed the ambiguity list back into the
    extractor with explicit clarifying instructions and try again.
    Phase 4 will add this LLM round-trip via this same class.

Phase 1.A ships a deterministic version of (a)'s composition step: this
class produces the *clarifying question string* from
:attr:`IntentResult.ambiguities` without any LLM call. The caller (the
agent loop) decides whether to surface that string to the user or feed
it back into a follow-up extractor invocation.

The deterministic question is built from the most-blocking
:class:`~kaos_agents.intent.types.Ambiguity`'s
``preferred_clarification`` text. When no preferred_clarification is
populated, the class composes a generic question from the kind +
excerpt.
"""

from __future__ import annotations

from typing import Any

from kaos_llm_core.programs.base import Program

from kaos_agents.intent.types import (
    Ambiguity,
    AmbiguityKind,
    IntentResult,
)

# Blocking kinds in priority order — the first kind to appear in
# ``intent.ambiguities`` from this list wins. Mirrors
# :meth:`IntentResult.has_blocking_ambiguity`.
_BLOCKING_KINDS: tuple[AmbiguityKind, ...] = (
    AmbiguityKind.MISSING_CONTEXT,
    AmbiguityKind.OUT_OF_DOMAIN,
)


class ClarificationLoop(Program):
    """Compose a clarifying question for a partial :class:`IntentResult`.

    Phase 1.A behavior is deterministic — no LLM round-trip. Phase 4
    will extend ``forward()`` to call back into the extractor with the
    user's reply and produce an updated IntentResult that consumes
    resolved ambiguities. Until then, the second tuple element of the
    returned pair is the unchanged input intent.
    """

    async def forward(self, **kwargs: Any) -> tuple[str, IntentResult]:
        """Return ``(clarifying_question, intent_after_clarification)``.

        Inputs (keyword-only):

        * ``original_message`` — required: the trigger payload that
          produced the intent. Reserved for the Phase 4 LLM round-trip;
          unused in Phase 1.A.
        * ``intent`` — required: the :class:`IntentResult` to clarify.

        When ``intent.requires_clarification`` is False, returns
        ``("", intent)`` unchanged — there is nothing to clarify.

        When True, walks ``intent.ambiguities`` and picks the most
        blocking one (priority: ``MISSING_CONTEXT`` → ``OUT_OF_DOMAIN``
        → first non-blocking). The returned question is that ambiguity's
        ``preferred_clarification`` if populated, otherwise a generic
        question composed from the kind + excerpt.

        ``**kwargs: Any`` matches :meth:`Program.forward` (LSP).
        """
        if "intent" not in kwargs:
            raise TypeError("ClarificationLoop.forward() missing required keyword: 'intent'")
        if "original_message" not in kwargs:
            raise TypeError(
                "ClarificationLoop.forward() missing required keyword: 'original_message'"
            )
        intent: IntentResult = kwargs["intent"]
        # ``original_message`` is reserved for the Phase 4 LLM round-trip.
        if not intent.requires_clarification:
            return "", intent

        chosen = _select_ambiguity(intent.ambiguities)
        if chosen is None:
            # ``requires_clarification=True`` with no ambiguities is a
            # degenerate case (the LLM said "I don't know" without
            # naming what it doesn't know). Fall back to a generic
            # question that tells the user what we have and what we need.
            return (
                "I'm not sure how to proceed with this request. "
                "Could you clarify what you'd like me to do?"
            ), intent

        return _compose_question(chosen), intent


def _select_ambiguity(ambiguities: tuple[Ambiguity, ...]) -> Ambiguity | None:
    """Pick the most blocking ambiguity from the list.

    Priority order:

    1. First ``MISSING_CONTEXT`` ambiguity.
    2. First ``OUT_OF_DOMAIN`` ambiguity.
    3. First ambiguity of any other kind.
    4. ``None`` when the tuple is empty.
    """
    if not ambiguities:
        return None
    by_kind: dict[AmbiguityKind, Ambiguity] = {}
    for amb in ambiguities:
        by_kind.setdefault(amb.kind, amb)
    for kind in _BLOCKING_KINDS:
        if kind in by_kind:
            return by_kind[kind]
    return ambiguities[0]


def _compose_question(amb: Ambiguity) -> str:
    """Build a clarifying question from one :class:`Ambiguity`.

    Prefers the LLM-provided ``preferred_clarification`` string. Falls
    back to a kind-specific template when that field is empty.
    """
    preferred = amb.preferred_clarification.strip()
    if preferred:
        return preferred

    excerpt = amb.excerpt.strip()
    if amb.kind is AmbiguityKind.UNKNOWN_REFERENCE:
        return f'Which "{excerpt}" do you mean?' if excerpt else "Which one do you mean?"
    if amb.kind is AmbiguityKind.AMBIGUOUS_PRONOUN:
        return f'What does "{excerpt}" refer to?' if excerpt else "What does that refer to?"
    if amb.kind is AmbiguityKind.CONFLICTING_REQUIREMENTS:
        return (
            f'Your requirements appear to conflict ("{excerpt}"). Which should I prioritize?'
            if excerpt
            else "Your requirements appear to conflict. Which should I prioritize?"
        )
    if amb.kind is AmbiguityKind.MISSING_CONTEXT:
        return (
            f"I need more context before I can proceed: {excerpt}"
            if excerpt
            else "I need more context before I can proceed. What should I work with?"
        )
    if amb.kind is AmbiguityKind.OUT_OF_DOMAIN:
        return (
            f'This appears outside what I can help with ("{excerpt}"). '
            "Could you reframe it or point me to a more specific aspect?"
            if excerpt
            else "This appears outside what I can help with. Could you reframe it?"
        )
    return f'Could you clarify "{excerpt}"?' if excerpt else "Could you clarify your request?"
