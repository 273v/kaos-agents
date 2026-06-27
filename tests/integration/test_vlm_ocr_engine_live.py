"""Live test for VlmOcrEngine — real Anthropic vision OCR over a rendered page.

Renders a known sentence to an image, wraps it as a KaosImage, and runs the
VLM OCR engine. Asserts the recovered text contains the rendered words.

Gated on:
  - ANTHROPIC_API_KEY present (the engine makes a real vision call), and
  - kaos-llm-core[vision] + kaos-content[images] importable.

Without those the test skips — per the repo policy we never silently pass.
"""

from __future__ import annotations

import importlib.util
import io
import os

import pytest

requires_anthropic = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ,
    reason="ANTHROPIC_API_KEY missing — VLM OCR needs a live provider call",
)

requires_vision = pytest.mark.skipif(
    importlib.util.find_spec("kaos_llm_core.vision") is None
    or importlib.util.find_spec("kaos_content.images") is None
    or importlib.util.find_spec("PIL") is None,
    reason="kaos-llm-core[vision] / kaos-content[images] not installed",
)

_SENTENCE = "The quick brown fox jumps over the lazy dog."


def _render_sentence_to_kaos_image():  # type: ignore[no-untyped-def]
    """Render ``_SENTENCE`` onto a white canvas and wrap it as a KaosImage."""
    from kaos_content.images.model import KaosImage
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 80), _SENTENCE, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return KaosImage.from_bytes(buffer.getvalue())


@pytest.mark.live
@requires_anthropic
@requires_vision
class TestVlmOcrEngineLive:
    async def test_async_extract_recovers_text(self) -> None:
        from kaos_agents.runtime.ocr_engines import VlmOcrEngine

        engine = VlmOcrEngine()
        result = await engine.extract(_render_sentence_to_kaos_image())
        recovered = result.text.lower()
        # The vision model reliably recovers these content words.
        for word in ("quick", "brown", "fox", "lazy", "dog"):
            assert word in recovered, f"{word!r} missing from {recovered!r}"
        assert result.engine_name.startswith("vlm:")

    def test_sync_extract_in_thread(self) -> None:
        # extract_sync from a plain sync context (no running loop).
        from kaos_agents.runtime.ocr_engines import VlmOcrEngine

        engine = VlmOcrEngine()
        result = engine.extract_sync(_render_sentence_to_kaos_image())
        recovered = result.text.lower()
        assert "fox" in recovered and "dog" in recovered
