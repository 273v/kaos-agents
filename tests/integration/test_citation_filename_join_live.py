"""Live regression test for the NDA-matrix P3 confidently-wrong file swap.

On 2026-05-27 the NDA persona matrix (10 cases on a 5-NDA corpus over the
SPA on kaos-agents 0.1.24) surfaced a class-1 confidently-wrong attribution:
asked to compare EMNA Mutual NDA and MNDA - Acme, the agent quoted the
correct clause text but SWAPPED which file each quote came from — said
"the EMNA Mutual NDA references Michigan" when ground truth is
EMNA=Delaware, Acme=Michigan.

Root cause: ``FindingCandidate`` had no ``source_uri`` field. Citations
rendered as bare content-hashes like ``[72e43288d19d]`` with no filename,
so the synthesis LLM had to *infer* which file each quoted clause came
from and got it backwards.

Fix (0.1.25, this regression test asserts it stays fixed):

* ``FindingCandidate.source_uri: str | None``
* ``BaseAgent._resolve_corpus_view_with_document`` builds an
  ``id(block) → source_uri`` map keyed by block identity (survives
  ``apply_retrieval_plan`` narrowing)
* ``_selector_with_source_uri`` adapter wraps the pure selector with
  the per-candidate source_uri JOIN
* ``CitationFound.source_uri`` emits ``"{filename}#{block_ref}"``
* ``_wrap_untrusted_text`` / ``_render_synthesis_findings`` emit
  ``source_uri="..."`` as an XML attribute (escaped via
  ``quoteattr``, not ``escape``, to close a ``"``-injection vector)

This test reproduces the original P3 prompt against the real NDA
fixtures using the same merge path the production ``BaseAgent`` uses,
runs the live synthesis on the legal-research test floor, and asserts
file→fact attribution is correct in both directions.

Cost: ~$0.05-0.15 per run on Sonnet-4-6 / gpt-5.4-mini. Cap set to
$0.30 to catch regressions in chunk_size / model selection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from kaos_agents.patterns.findings import (
    FindingCandidate,
    FindingsAgent,
    every_sentence_selector,
)
from tests.integration._models import critic_model, requires_provider_for, respond_model
from tests.integration.ladder.fixtures.nda_ground_truth import GROUND_TRUTH

NDA_DIR = Path(__file__).parent / "ladder" / "fixtures" / "nda"

EMNA = NDA_DIR / "EMNA Mutual NDA.docx"
ACME = NDA_DIR / "MNDA - Acme.docx"

requires_emna = pytest.mark.skipif(
    not EMNA.exists(),
    reason=f"EMNA fixture missing at {EMNA}",
)
requires_acme = pytest.mark.skipif(
    not ACME.exists(),
    reason=f"Acme fixture missing at {ACME}",
)


_BODY_IDX_RE = re.compile(r"^#/body/(\d+)")


def _build_merged_view_and_uri_map() -> tuple[Any, dict[int, str]]:
    """Mirror ``BaseAgent._resolve_corpus_view_with_document``'s merge.

    The production builder is async and consumes ``DOCUMENTS`` memory
    items; we build the same 2-doc merged view directly from the
    parsed ``ContentDocument`` instances so this regression test
    pins the in-pattern behavior without standing up a full session.
    """
    from kaos_content.model.blocks import Paragraph
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Text
    from kaos_content.views.document_view import DocumentView
    from kaos_nlp_core._defaults import get_default_punkt_tokenizer
    from kaos_office import parse_docx

    blocks: list[Any] = []
    id_to_uri: dict[int, str] = {}

    for path in (EMNA, ACME):
        parsed = parse_docx(str(path))
        # Filename is the human-readable identifier — same precedence
        # the production builder uses (filename > parsed.metadata.title).
        uri = path.name

        header = Paragraph(children=(Text(value=f"=== {path.name} ==="),))
        blocks.append(header)
        id_to_uri[id(header)] = uri

        for block in parsed.body:
            blocks.append(block)
            id_to_uri[id(block)] = uri

    document = ContentDocument(body=tuple(blocks))
    view = DocumentView(document, sentence_segmenter=get_default_punkt_tokenizer())
    return view, id_to_uri


def _wrapped_selector(id_to_uri: dict[int, str]):
    """Mirror ``BaseAgent._selector_with_source_uri``.

    Wraps the pure ``every_sentence_selector`` so each emitted
    candidate carries the originating-doc source_uri resolved
    via the block-identity map.
    """

    def _selector(view: Any, question: str):
        for cand in every_sentence_selector(view, question):
            uri: str | None = None
            if cand.block_ref:
                m = _BODY_IDX_RE.match(cand.block_ref)
                if m is not None:
                    idx = int(m.group(1))
                    body = view.document.body
                    if 0 <= idx < len(body):
                        uri = id_to_uri.get(id(body[idx]))
            if uri is None:
                yield cand
            else:
                yield FindingCandidate(
                    finding_id=cand.finding_id,
                    text=cand.text,
                    block_ref=cand.block_ref,
                    char_span=cand.char_span,
                    section_title=cand.section_title,
                    page=cand.page,
                    injection_suspected=cand.injection_suspected,
                    source_uri=uri,
                )

    return _selector


MODEL = respond_model()
FILTER_MODEL = critic_model()
SYNTH_MODEL = respond_model()

requires_provider = requires_provider_for(MODEL)


@pytest.mark.live
@requires_provider
@requires_emna
@requires_acme
async def test_citation_source_uri_eliminates_p3_attribution_swap() -> None:
    """Regression test for the 2026-05-27 NDA-matrix P3 class-1 bug.

    Asks the agent the same comparative governing-law question that
    swapped EMNA↔Acme on 0.1.24. Asserts:

      1. Filtered findings carry ``source_uri`` populated from the
         per-block JOIN (proves the selector adapter ran).
      2. The synthesized answer mentions BOTH ground-truth pairings
         (EMNA + Delaware, Acme + Michigan).
      3. The answer does NOT contain the inverse "swap" pairings
         (EMNA + Michigan, Acme + Delaware) in close proximity.
    """
    view, id_to_uri = _build_merged_view_and_uri_map()

    agent = FindingsAgent(
        selector=_wrapped_selector(id_to_uri),
        filter_model=FILTER_MODEL,
        synthesis_model=SYNTH_MODEL,
        chunk_size=20,
        num_parallel=3,
        relevance_threshold=0.4,
    )

    question = (
        "Which agreement is governed by Delaware law, and which by "
        "Michigan law? Cite the filename of each."
    )
    result = await agent.run(question, view)

    # Cost cap — guards against chunk_size / model regressions.
    assert result.total_cost_usd < 0.30, (
        f"unexpected cost spike: ${result.total_cost_usd:.4f} — "
        "investigate chunk_size or model selection"
    )

    # Refusal would mean filter dropped everything; the test stops here
    # with the diagnostic instead of a confusing assertion error below.
    assert result.refusal is None, f"FindingsAgent refused: {result.refusal!r}"

    # (1) source_uri propagated through the adapter into the filtered set.
    survivors_with_uri = [f for f in result.findings if f.candidate.source_uri]
    assert len(survivors_with_uri) >= 2, (
        f"only {len(survivors_with_uri)} of {len(result.findings)} survivors "
        "carry source_uri — the selector adapter is not wrapping candidates"
    )

    surfaced_uris = {f.candidate.source_uri for f in survivors_with_uri}
    assert "EMNA Mutual NDA.docx" in surfaced_uris, (
        f"EMNA filename missing from survivor source_uri set: {surfaced_uris!r}"
    )
    assert "MNDA - Acme.docx" in surfaced_uris, (
        f"Acme filename missing from survivor source_uri set: {surfaced_uris!r}"
    )

    # (2) + (3) Ground-truth attribution in the synthesized answer.
    answer = result.answer.lower()
    emna_truth = next(g for g in GROUND_TRUTH if g.filename == "EMNA Mutual NDA.docx")
    acme_truth = next(g for g in GROUND_TRUTH if g.filename == "MNDA - Acme.docx")
    assert emna_truth.governing_law.lower() == "delaware"
    assert acme_truth.governing_law.lower() == "michigan"

    assert "emna" in answer and "delaware" in answer, (
        f"answer does not mention EMNA + Delaware:\n{result.answer}"
    )
    assert "acme" in answer and "michigan" in answer, (
        f"answer does not mention Acme + Michigan:\n{result.answer}"
    )

    # The P3 confidently-wrong inverse: "EMNA ... Michigan" within
    # ~160 chars (one sentence). Same for "Acme ... Delaware".
    swap_emna_michigan = re.search(r"emna[^.]{0,160}michigan", answer, flags=re.S)
    swap_acme_delaware = re.search(r"acme[^.]{0,160}delaware", answer, flags=re.S)

    assert swap_emna_michigan is None, (
        f"P3 regression: answer associates EMNA with Michigan within one "
        f"sentence — match: {swap_emna_michigan.group(0)!r}\n\n"
        f"Full answer:\n{result.answer}"
    )
    assert swap_acme_delaware is None, (
        f"P3 regression: answer associates Acme with Delaware within one "
        f"sentence — match: {swap_acme_delaware.group(0)!r}\n\n"
        f"Full answer:\n{result.answer}"
    )
