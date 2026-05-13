"""Citation extraction + verification helper.

Bug #5 of the workflow telemetry audit: explain records reported
``citations: 0`` even when assistant answers contained URLs and FR
citations like ``2025-19882``. The agent platform had ``kaos-citations``
as an optional registered tool set, but the agent loop never auto-ran
citation extraction over its own output. So the citation pipeline that
markets this stack as "Harvey-class" was unreachable from the default
turn loop.

This module provides :func:`emit_citations_for_text` — a small helper
that runs ``kaos_citations.extract_citations`` over a piece of assistant
text and emits one :class:`CitationFound` event per detected citation
into the active :class:`EventEmitter`. The emitter routes those events
through the active :class:`EventCollector`, so they land in:

  * the streamed event iterator a consumer is reading
  * the chat CLI's JSONL log (``--log``)
  * the chat CLI's per-turn explain record (``--explain``)
  * any :class:`KaosHook` listening for ``CitationFound``

The helper is silent when ``kaos-citations`` is not installed (it's an
optional dep) and silent when no citations are found. No false-positive
events.

Usage from a pattern / AgentLoop:

.. code-block:: python

    # After producing assistant text, before the TurnSummary fires:
    emit_citations_for_text(emitter, response_text)

The helper is idempotent and side-effect-free other than emitting
events into the active collector.
"""

from __future__ import annotations

import logging

from kaos_agents.events import CitationFound, EventEmitter

logger = logging.getLogger(__name__)


def emit_citations_for_text(
    emitter: EventEmitter,
    text: str,
    *,
    source_uri: str = "",
) -> list[CitationFound]:
    """Extract citations from ``text`` and emit one :class:`CitationFound` per hit.

    Returns the emitted events as a list so callers in async-generator
    contexts (v1 ``BaseAgent.run``) can ``yield`` them downstream.
    Callers inside an active :class:`EventCollector` scope (v2
    ``AgentLoop._run_8_step_turn``) can ignore the return value — the
    emit already routed the events through the collector + the
    consumer queue.

    Args:
        emitter: The active EventEmitter for the current turn.
        text: Assistant text to scan for citations.
        source_uri: Optional URI to attach to citations that don't
            carry their own ``source_uri``. Defaults to empty string.

    Returns:
        List of CitationFound events emitted (empty when kaos-citations
        is not installed, empty when text is empty, empty when no
        citations were found).

    Never raises — extraction failures are logged and treated as zero
    citations to keep the agent loop robust.
    """
    if not text:
        return []

    try:
        from kaos_citations import extract_citations
    except ImportError:
        # Optional dep — keep silent. Operators who want citation
        # verification install kaos-agents[citations].
        return []

    try:
        citations = extract_citations(text)
    except Exception as exc:
        logger.warning("citation_verifier: extract_citations raised: %s", exc)
        return []

    emitted: list[CitationFound] = []
    for cit in citations:
        normalized = str(getattr(cit, "normalized", "") or "")
        raw = str(getattr(cit, "raw", "") or "")
        kind = type(cit).__name__
        cit_source_uri = str(getattr(cit, "source_uri", "") or source_uri)

        # CitationFound today carries (claim, source_uri, confidence,
        # verified). Pre-resolution we set verified=False and stamp
        # the kind into the claim text so consumers can split on it
        # when projecting; once kaos-source / kaos-llm-core's Cited[T]
        # verifier is wired in (parallel work to the audit's bug #5),
        # `verified=True` + a non-empty source_uri become the gate.
        claim_text = f"[{kind}] {normalized or raw}".strip()
        event = emitter.emit(
            CitationFound,
            claim=claim_text,
            verified=False,
            confidence=1.0 if normalized else 0.5,
            source_uri=cit_source_uri,
        )
        assert isinstance(event, CitationFound)
        emitted.append(event)

    if emitted:
        logger.debug("citation_verifier: emitted %d CitationFound event(s)", len(emitted))
    return emitted


__all__ = ["emit_citations_for_text"]
