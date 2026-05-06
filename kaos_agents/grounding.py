"""Grounding policy enforcement (FUND-8).

Applies the ``Agent.refusal_policy`` to ``GroundedAnswer`` results
produced by ResearchAgent's RAG calls or any other Program that
returns ``Answer[T] | InsufficientEvidence``.

The key invariant: **if the policy is set and the answer's confidence
is below threshold, the answer is collapsed to InsufficientEvidence.**
This prevents the agent from bullshitting — a low-confidence answer
that would otherwise be passed through to the user as if it were
reliable is replaced with an honest refusal and a visibility event.

Usage inside pattern handlers::

    from kaos_agents.grounding import apply_refusal_policy

    grounded, event = apply_refusal_policy(
        grounded_answer, agent.refusal_policy, sequence_no
    )
    if event is not None:
        yield event  # GroundingRefusalTriggered — visible to hooks + UI
    # ``grounded`` is now either the original Answer or InsufficientEvidence

The caller decides what to do with the event: logging hooks display
it, plan-execute patterns trigger replan, chat patterns include the
refusal in the response.
"""

from __future__ import annotations

from typing import Any

from kaos_agents.events import (
    EvidenceInsufficient,
    GroundingRefusalTriggered,
)


def apply_refusal_policy(
    grounded_answer: Any,
    refusal_policy: Any | None,
    sequence: int = 0,
    *,
    timestamp: float = 0.0,
    session_id: str = "",
    run_id: str = "",
) -> tuple[Any, GroundingRefusalTriggered | None]:
    """Apply a refusal policy to a GroundedAnswer.

    Args:
        grounded_answer: A ``GroundedAnswer[T]`` discriminated union
            instance — either ``Answer[T]`` or ``InsufficientEvidence``.
        refusal_policy: A ``RefusalPolicy`` from kaos-llm-core, or
            ``None`` (no enforcement).
        sequence: Event sequence number.

    Returns:
        ``(result, event)`` where ``result`` is the original answer
        (if it passes the policy) or a new ``InsufficientEvidence``
        (if it was collapsed), and ``event`` is ``None`` when no
        collapse happened or a ``GroundingRefusalTriggered`` when it
        did.
    """
    if refusal_policy is None:
        return grounded_answer, None

    # Use isinstance against the typed Answer / InsufficientEvidence
    # discriminated union from kaos-llm-core, not duck-typed
    # ``getattr("kind")`` checks. Duck-typing here was a real safety
    # bug: an unexpected type with a ``kind`` attribute but no
    # ``confidence`` field would default to confidence=1.0 via the
    # second ``getattr`` and silently pass the policy threshold —
    # exactly the failure mode the policy exists to prevent.
    try:
        from kaos_llm_core.signatures.grounding import (
            Answer,
            InsufficientEvidence,
        )
    except ImportError:
        return grounded_answer, None

    if not isinstance(grounded_answer, Answer):
        # Already InsufficientEvidence (no policy work to do) or an
        # unexpected type (cannot safely apply the policy — pass
        # through unchanged so downstream code handles the
        # type mismatch explicitly rather than via a silent default).
        return grounded_answer, None

    confidence = float(grounded_answer.confidence)
    min_conf = getattr(refusal_policy, "min_confidence", 0.7)

    if confidence >= min_conf:
        return grounded_answer, None

    reason = (
        f"Answer confidence {confidence:.2f} below policy threshold "
        f"{min_conf:.2f}. Refusing to present an uncertain answer."
    )
    collapsed = InsufficientEvidence(
        reason=reason,
        attempted_claims=list(grounded_answer.claims),
    )
    import time as _time

    event = GroundingRefusalTriggered(
        timestamp=timestamp or _time.monotonic(),
        sequence=sequence,
        session_id=session_id,
        run_id=run_id,
        original_confidence=confidence,
        min_confidence=min_conf,
        reason=reason,
    )
    return collapsed, event


def make_evidence_insufficient_event(
    collapsed: Any,
    sequence: int = 0,
    *,
    timestamp: float = 0.0,
    session_id: str = "",
    run_id: str = "",
) -> EvidenceInsufficient:
    """Build an EvidenceInsufficient event from an InsufficientEvidence model."""
    return EvidenceInsufficient(
        timestamp=timestamp,
        sequence=sequence,
        session_id=session_id,
        run_id=run_id,
        reason=getattr(collapsed, "reason", ""),
        what_would_resolve=getattr(collapsed, "what_would_resolve", "") or "",
    )


__all__ = ["apply_refusal_policy", "make_evidence_insufficient_event"]
