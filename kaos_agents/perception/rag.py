"""Thin wrapper over kaos-llm-core RAG that emits CitationFound events.

The plan says the Perceiver emits CitationFound events when RAG
returns a grounded answer with verified spans. Phase 1.B ships the
wrapper; Phase 2 wires the emission into the agent's event stream.

Usage::

    rag = PerceptionRAG(model="anthropic:claude-haiku-4-5")
    output = await rag(question="...", documents={"doc:a": "..."})

When called inside a :func:`kaos_agents.events.collector.collect_events`
block, every Span carried by the output produces one ``CitationFound``
event in that collector. When no collector is active, the wrapper is
silent — preserves backward-compat for direct Python callers.

The CitationFound event class lives in
:mod:`kaos_agents.events.research`. Its ``LifecycleEvent`` parent
requires ``timestamp`` / ``sequence`` / ``session_id`` / ``run_id``
for full agent-loop emission; the perception wrapper fills these
with placeholder values appropriate for a Phase 1.B "captured" event
that has not yet been threaded through the loop. Phase 2 wiring will
replace this with the canonical :class:`EventEmitter`.
"""

from __future__ import annotations

import time
from typing import Any

from kaos_llm_core.programs.base import Program
from kaos_llm_core.programs.rag import RAG
from kaos_llm_core.signatures.grounding import Span

from kaos_agents.events.collector import push_event
from kaos_agents.events.research import CitationFound


class PerceptionRAG(Program):
    """RAG wrapper that pushes CitationFound events to the active collector.

    Composes a kaos-llm-core :class:`~kaos_llm_core.programs.rag.RAG`
    Program. ``forward()`` runs it, walks the output for
    :class:`~kaos_llm_core.signatures.grounding.Span` citations, and
    pushes one :class:`~kaos_agents.events.research.CitationFound`
    event per span.

    Args:
        rag_program: An already-constructed RAG Program. Mutually
            exclusive with ``rag_kwargs`` — pass one or the other.
        **rag_kwargs: Forwarded to :class:`RAG.__init__` when
            ``rag_program`` is None. ``model`` is required.
    """

    def __init__(
        self,
        *,
        rag_program: Any | None = None,
        **rag_kwargs: Any,
    ) -> None:
        # Pre-Program init: build (or accept) the inner RAG. We assign it
        # to a private attribute (leading underscore) so the Program
        # auto-registration in __setattr__ does NOT pick it up — Phase 1.B
        # is purely additive and we don't want surprising graph entries.
        # The RAG itself still registers its own .call sub-Program when
        # accessed via .invoke / forward, which is the desired behavior.
        super().__init__()
        if rag_program is None:
            self._rag: Any = RAG(**rag_kwargs)
        else:
            if rag_kwargs:
                raise ValueError(
                    "PerceptionRAG: pass either rag_program OR rag_kwargs, not both. "
                    "Drop the kwargs and configure the RAG directly when you "
                    "build it, or omit rag_program and let PerceptionRAG build "
                    "it from the kwargs."
                )
            self._rag = rag_program

    @property
    def rag(self) -> Any:
        """The underlying RAG program (typed ``Any`` to admit test stubs)."""
        return self._rag

    async def forward(self, **kwargs: Any) -> Any:
        """Run RAG, emit CitationFound events for every Span, return output."""
        invocation = await self._rag.invoke(**kwargs)
        output = invocation.output
        for span in _iter_spans(output):
            push_event(_build_citation_found(span))
        return output


def _iter_spans(value: Any) -> list[Span]:
    """Walk *value* and return every embedded :class:`Span`.

    Handles the shapes kaos-llm-core RAG can produce:

    - ``RAGResult`` — has ``grounded_answer`` (Answer or
      InsufficientEvidence). The ``Answer.spans`` convenience iterates
      every span across every claim.
    - ``Answer[T]`` directly — same shape via ``.spans``.
    - ``InsufficientEvidence`` — yields nothing (no citations).
    - Anything else — best-effort recursive walk over pydantic
      models / lists / tuples / dicts looking for Span instances.

    Defensive: never raises on unexpected shapes — returns ``[]``.
    """
    spans: list[Span] = []
    _walk_for_spans(value, spans, depth=0)
    return spans


def _walk_for_spans(value: Any, out: list[Span], depth: int) -> None:
    """Best-effort recursive walk collecting Span instances.

    Bounded depth (max 8) so a pathological cyclic shape can't hang.
    """
    if depth > 8 or value is None:
        return
    if isinstance(value, Span):
        out.append(value)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _walk_for_spans(item, out, depth + 1)
        return
    if isinstance(value, dict):
        for item in value.values():
            _walk_for_spans(item, out, depth + 1)
        return
    # Pydantic v2 model — iterate declared fields.
    fields = getattr(type(value), "model_fields", None)
    if fields is not None:
        for name in fields:
            try:
                child = getattr(value, name)
            except AttributeError:
                continue
            _walk_for_spans(child, out, depth + 1)
        return
    # Dataclass / arbitrary object with __dict__.
    obj_dict = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        for child in obj_dict.values():
            _walk_for_spans(child, out, depth + 1)


def _build_citation_found(span: Span) -> CitationFound:
    """Construct a CitationFound event from a verified Span.

    Phase 1.B placeholder fields:

    - ``session_id`` / ``run_id`` are empty strings (the perception
      subsystem is not yet wired to the runner; the runner's
      :class:`EventEmitter` will replace these in Phase 2).
    - ``sequence`` is 0 (the active collector uses arrival order).
    - ``timestamp`` uses :func:`time.monotonic` to match
      :class:`KaosEvent`'s convention.
    - ``confidence`` defaults to 1.0 because the Span has already
      been verified (kaos-llm-core RAG only embeds verified spans
      in its output by the time forward() returns).
    - ``verified=True`` for the same reason.
    """
    return CitationFound(
        timestamp=time.monotonic(),
        sequence=0,
        session_id="",
        run_id="",
        claim=span.quote,
        source_uri=span.source_uri,
        confidence=1.0,
        verified=True,
    )


__all__ = ["PerceptionRAG"]
