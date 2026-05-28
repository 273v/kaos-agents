"""Pin the block-identity contract of ``_narrow_document_from_search_results``.

The ``BaseAgent._run_findings_dispatch`` source_uri JOIN (NDA-matrix P3
regression fix, 2026-05-27) keys its block-uri lookup by ``id(block)``.
This is load-bearing because the narrowing step in
:mod:`kaos_agents.patterns.retrieval.apply` rebuilds the merged
``ContentDocument`` with a SUBSET of the original blocks at NEW
positional indices — but it currently does so by reusing the same
immutable block objects by reference, so ``id()`` stays stable.

If a future refactor switches to ``block.model_copy()`` (or any other
clone-on-narrow operation), the ``id()`` lookup will silently degrade
to ``None`` and citations will revert to bare ``block_ref`` hashes
without filename — re-introducing the P3 class-1 confidently-wrong
file→fact swap.

This test pins the contract: after narrowing, every block in the
returned document must be the *same Python object* as one in the
input document.
"""

from __future__ import annotations

from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text

from kaos_agents.patterns.retrieval.apply import _narrow_document_from_search_results


def _doc_with_n_blocks(n: int) -> ContentDocument:
    blocks = tuple(Paragraph(children=(Text(value=f"block-{i}"),)) for i in range(n))
    return ContentDocument(body=blocks)


def test_narrow_returns_subset_of_original_blocks_by_reference() -> None:
    """Narrowing must reuse the original block objects, not clone them."""
    document = _doc_with_n_blocks(10)
    original_ids = {id(block) for block in document.body}

    # Keep blocks at positions 1, 3, 5, 7 — non-contiguous subset
    kept_refs = {f"#/body/{i}" for i in (1, 3, 5, 7)}
    narrowed = _narrow_document_from_search_results(document, kept_refs)

    assert len(narrowed.body) == 4
    for narrowed_block in narrowed.body:
        assert id(narrowed_block) in original_ids, (
            f"narrowed block {narrowed_block!r} is not the same Python object "
            f"as any block in the original document — the source_uri JOIN "
            f"in BaseAgent._run_findings_dispatch will silently degrade."
        )


def test_narrow_preserves_block_order_from_input() -> None:
    """When the original blocks at positions 1, 3, 5, 7 are kept, the
    narrowed body must contain them in input order — not in some
    refs-set or hash-iteration order."""
    document = _doc_with_n_blocks(10)
    kept_refs = {"#/body/7", "#/body/1", "#/body/5", "#/body/3"}  # unordered

    narrowed = _narrow_document_from_search_results(document, kept_refs)

    expected = [document.body[i] for i in (1, 3, 5, 7)]
    assert [id(b) for b in narrowed.body] == [id(b) for b in expected]


def test_narrow_empty_refs_returns_original_document() -> None:
    """The empty-refs guard returns the original document object —
    callers (the source_uri JOIN's id-map) rely on this so a zero-match
    narrowing step does not silently strip provenance."""
    document = _doc_with_n_blocks(3)
    narrowed = _narrow_document_from_search_results(document, set())
    assert narrowed is document


def test_narrow_zero_survivors_falls_back_to_original_document() -> None:
    """When all refs miss (e.g. ``#/body/99`` against a 3-block doc),
    the fallback returns the original document object — same id-map
    preservation guarantee."""
    document = _doc_with_n_blocks(3)
    narrowed = _narrow_document_from_search_results(document, {"#/body/99"})
    assert narrowed is document


def test_narrow_block_id_lookup_round_trip() -> None:
    """End-to-end the contract the NDA-matrix P3 fix depends on:
    build an id→str map keyed by the input blocks, narrow, then
    look up each narrowed block's id in the map. Every lookup
    must hit."""
    document = _doc_with_n_blocks(20)
    id_to_label: dict[int, str] = {id(block): f"label-{i}" for i, block in enumerate(document.body)}

    kept_refs = {f"#/body/{i}" for i in (2, 4, 6, 8, 10, 12, 14, 16, 18)}
    narrowed = _narrow_document_from_search_results(document, kept_refs)

    for narrowed_block in narrowed.body:
        assert id(narrowed_block) in id_to_label, (
            "block-identity preservation broken — source_uri lookup "
            "would degrade to None for this candidate"
        )
