"""Unit tests for the scanned-PDF OCR fallback in the corpus pre-parse path.

Covers ``BaseAgent._parse_binary_bytes_to_content_document`` and the
``BaseAgent._ocr_pdf_bytes_to_content_document`` helper it dispatches
to when ``parse_pdf_bytes`` returns an empty ContentDocument (the
scanned-PDF case that surfaced corpus-stress S03 in the SPA acceptance
matrix).

The tests skip cleanly when pytesseract / system tesseract aren't
available — same gate the integration suite uses (``_has_ocr``).
"""

from __future__ import annotations

import io
import shutil

import pytest

from kaos_agents.runtime.agent import BaseAgent


def _has_ocr() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def _synth_scanned_pdf(text: str) -> bytes:
    """Build a single-page raster-only PDF embedding ``text``.

    Same shape as ``tests/integration/_corpus_fixtures.synth_pdf_image``
    but inlined here so the unit suite doesn't depend on the integration
    fixture module.
    """
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    width_px, height_px = 1700, 2200
    img = Image.new("RGB", (width_px, height_px), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    margin = 60
    line_height = 28
    y = margin
    for raw_line in text.split("\n"):
        chunks = [raw_line[i : i + 80] for i in range(0, max(1, len(raw_line)), 80)]
        for chunk in chunks:
            draw.text((margin, y), chunk, fill="black", font=font)
            y += line_height
    png_buf = io.BytesIO()
    img.save(png_buf, format="PNG")
    png_buf.seek(0)

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=LETTER)
    page_w, page_h = LETTER
    c.drawImage(ImageReader(png_buf), 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()
    return pdf_buf.getvalue()


def _block_text(block: object) -> str:
    return "".join(getattr(c, "value", "") for c in getattr(block, "children", ()))


@pytest.mark.skipif(not _has_ocr(), reason="tesseract OCR not available")
def test_ocr_fallback_recovers_text_from_scanned_pdf() -> None:
    """Scanned PDF with no text layer → OCR fallback returns the text.

    Regression for S03: pre-0.1.21 the dispatch path called
    ``parse_pdf_bytes`` which returned an empty body for scanned
    PDFs, so FindingsAgent enumerated 0 candidates and synthesis
    emitted ``(empty)``. The OCR fallback recovers the planted
    token (``FALCON-2026``) so enumeration has something to ground on.
    """
    needle = "FALCON-2026"
    body = _synth_scanned_pdf(
        "INTERNAL MEMO\nCodename: FALCON-2026.\nDistribution: leadership only."
    )

    doc = BaseAgent._parse_binary_bytes_to_content_document(
        filename="memo-scanned.pdf",
        mime="application/pdf",
        body=body,
    )

    assert doc is not None, "expected OCR fallback to return a ContentDocument"
    assert len(doc.body) > 0, "expected OCR fallback to emit at least one block"
    combined = "\n".join(_block_text(b) for b in doc.body)
    assert needle in combined, f"expected OCR text to contain {needle!r}; got: {combined!r}"


@pytest.mark.skipif(not _has_ocr(), reason="tesseract OCR not available")
def test_ocr_fallback_emits_page_markers() -> None:
    """OCR fallback emits ``[page N]`` markers so synthesis can cite per-page."""
    body = _synth_scanned_pdf("CONFIDENTIAL\nProject KESTREL is active.")

    doc = BaseAgent._parse_binary_bytes_to_content_document(
        filename="kestrel.pdf",
        mime="application/pdf",
        body=body,
    )

    assert doc is not None
    page_markers = [_block_text(b) for b in doc.body if _block_text(b).startswith("[page ")]
    assert page_markers, "expected at least one [page N] marker"
    assert "[page 1]" in page_markers


def test_text_layer_pdf_bypasses_ocr_fallback() -> None:
    """Text-layer PDFs should not invoke the OCR helper.

    ``parse_pdf_bytes`` returns non-empty body for a regular PDF, so
    the OCR branch is never reached. This guards against the
    obvious regression where the empty-body check fires on every PDF.
    """
    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    buf = io.BytesIO()
    pdf = SimpleDocTemplate(buf, pagesize=LETTER)
    styles = getSampleStyleSheet()
    pdf.build([Paragraph("Hello text-layer PDF.", styles["BodyText"])])
    body = buf.getvalue()

    doc = BaseAgent._parse_binary_bytes_to_content_document(
        filename="hello.pdf",
        mime="application/pdf",
        body=body,
    )

    assert doc is not None
    combined = " ".join(_block_text(b) for b in doc.body)
    assert "Hello text-layer PDF" in combined
    # The text-layer path emits no synthetic [page N] markers — those
    # are an OCR-only artifact.
    assert not any(_block_text(b).startswith("[page ") for b in doc.body)


def test_empty_pdf_with_no_ocr_returns_empty_doc() -> None:
    """If the PDF is empty and the OCR helper returns None, fall through.

    The caller still gets the (empty) ContentDocument from
    ``parse_pdf_bytes`` rather than a hard failure — the downstream
    dispatch correctly handles "0 candidates" as a refusal signal.
    """
    # Build a minimal PDF with no content
    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.showPage()
    c.save()
    body = buf.getvalue()

    doc = BaseAgent._parse_binary_bytes_to_content_document(
        filename="blank.pdf",
        mime="application/pdf",
        body=body,
    )

    # Either OCR yields nothing (empty page → no text) and we get the
    # empty parse result back, or we get None — both are acceptable.
    # The key invariant is that we don't raise.
    if doc is not None:
        combined = " ".join(_block_text(b) for b in doc.body)
        # An empty page produces no usable text.
        assert "FALCON" not in combined  # sanity
