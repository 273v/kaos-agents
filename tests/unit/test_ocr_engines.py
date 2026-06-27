"""Unit tests for VLM OCR engines (Tesseract → VLM escalation).

No network: the vision model is faked by injecting a stand-in
``kaos_llm_core.vision`` module into ``sys.modules``, and the tiered engine is
driven by canned ``OCREngine`` stand-ins. The live path is covered separately
in ``tests/integration`` (gated on an API key).
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import pytest
from kaos_pdf.ocr.base import OCREngine, OCRLine, OCRResult

from kaos_agents.errors import VisionOcrUnavailableError
from kaos_agents.runtime.ocr_engines import (
    DEFAULT_VISION_MODEL,
    TieredOCREngine,
    VlmOcrEngine,
)

if TYPE_CHECKING:
    from kaos_content.images.model import KaosImage

_IMAGE = cast("KaosImage", object())  # sentinel; fakes never inspect it

# Real garbled native layer (staten_v_united_states.pdf p0) and its truth.
_GARBLED = "0RlGlt IAt lJn tbe @nitp! btutts ourt of trs lsims"
_CLEAN = "In the United States Court of Federal Claims filed today"


@dataclass(frozen=True)
class _FakePageOCRResult:
    text: str
    model: str


def _install_fake_vision(monkeypatch: pytest.MonkeyPatch, *, text: str) -> dict[str, object]:
    """Inject a fake ``kaos_llm_core.vision`` whose ocr_page returns ``text``."""
    captured: dict[str, object] = {}

    async def ocr_page(image: object, *, model: str) -> _FakePageOCRResult:
        captured["image"] = image
        captured["model"] = model
        return _FakePageOCRResult(text=text, model=model)

    module = types.ModuleType("kaos_llm_core.vision")
    setattr(module, "ocr_page", ocr_page)  # noqa: B010
    monkeypatch.setitem(sys.modules, "kaos_llm_core.vision", module)
    return captured


# ---------------------------------------------------------------------------
# VlmOcrEngine
# ---------------------------------------------------------------------------


class TestVlmOcrEngine:
    def test_text_maps_to_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_vision(monkeypatch, text="First line\n\n  Second line  \n")
        engine = VlmOcrEngine()
        result = engine.extract_sync(_IMAGE)
        texts = [line.text for line in result.lines]
        assert texts == ["First line", "  Second line  "]  # blank line dropped
        assert all(line.bbox is None and line.confidence == 1.0 for line in result.lines)
        assert result.engine_name == f"vlm:{DEFAULT_VISION_MODEL}"

    def test_model_override_is_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _install_fake_vision(monkeypatch, text="x y z")
        engine = VlmOcrEngine(model="anthropic:claude-sonnet-4-6")
        result = engine.extract_sync(_IMAGE)
        assert captured["model"] == "anthropic:claude-sonnet-4-6"
        assert result.engine_name == "vlm:anthropic:claude-sonnet-4-6"

    def test_extract_sync_inside_running_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The agent runtime is async; extract_sync must not raise
        # "cannot be called from a running event loop".
        _install_fake_vision(monkeypatch, text="inside loop")

        async def driver() -> OCRResult:
            engine = VlmOcrEngine()
            return engine.extract_sync(_IMAGE)  # called within a live loop

        result = asyncio.run(driver())
        assert [line.text for line in result.lines] == ["inside loop"]

    async def test_async_extract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_vision(monkeypatch, text="async path")
        engine = VlmOcrEngine()
        result = await engine.extract(_IMAGE)
        assert [line.text for line in result.lines] == ["async path"]

    def test_missing_vision_extra_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # None in sys.modules makes the lazy import raise ImportError.
        monkeypatch.setitem(sys.modules, "kaos_llm_core.vision", None)
        engine = VlmOcrEngine()
        with pytest.raises(VisionOcrUnavailableError) as exc:
            engine.extract_sync(_IMAGE)
        # Actionable message: names the fix and the fallback.
        assert "kaos-agents[vision]" in str(exc.value)
        assert "Tesseract" in str(exc.value)


# ---------------------------------------------------------------------------
# TieredOCREngine
# ---------------------------------------------------------------------------


class _CannedEngine(OCREngine):
    """Returns a fixed OCRResult and counts invocations."""

    name: ClassVar[str] = "canned"

    def __init__(self, text: str, confidence: float, *, engine_name: str = "canned") -> None:
        self._text = text
        self._confidence = confidence
        self._engine_name = engine_name
        self.calls = 0

    def extract_sync(self, image: KaosImage) -> OCRResult:
        self.calls += 1
        lines = [
            OCRLine(text=line, bbox=None, confidence=self._confidence)
            for line in self._text.splitlines()
            if line.strip()
        ]
        return OCRResult(lines=lines, engine_name=self._engine_name)


def _tiered(
    primary: OCREngine,
    escalation: OCREngine,
    *,
    min_confidence: float = 0.6,
    quality_threshold: float = 0.35,
    max_escalations: int | None = None,
) -> TieredOCREngine:
    return TieredOCREngine(
        primary,
        escalation,
        min_confidence=min_confidence,
        quality_threshold=quality_threshold,
        max_escalations=max_escalations,
    )


class TestTieredOCREngine:
    def test_escalates_on_garbled_layer(self) -> None:
        primary = _CannedEngine(_GARBLED, confidence=0.95)  # high conf, garbage text
        escalation = _CannedEngine(_CLEAN, confidence=1.0, engine_name="vlm")
        tiered = _tiered(primary, escalation)
        result = tiered.extract_sync(_IMAGE)
        assert result.engine_name == "vlm"  # escalated despite high confidence
        assert escalation.calls == 1
        assert tiered.escalation_count == 1

    def test_no_escalation_on_clean_high_confidence(self) -> None:
        primary = _CannedEngine(_CLEAN, confidence=0.95)
        escalation = _CannedEngine(_CLEAN, confidence=1.0, engine_name="vlm")
        tiered = _tiered(primary, escalation)
        result = tiered.extract_sync(_IMAGE)
        assert result.engine_name == "canned"
        assert escalation.calls == 0
        assert tiered.escalation_count == 0

    def test_escalates_on_low_confidence(self) -> None:
        primary = _CannedEngine(_CLEAN, confidence=0.2)  # clean text but low conf
        escalation = _CannedEngine(_CLEAN, confidence=1.0, engine_name="vlm")
        tiered = _tiered(primary, escalation, min_confidence=0.6)
        result = tiered.extract_sync(_IMAGE)
        assert result.engine_name == "vlm"
        assert escalation.calls == 1

    def test_max_escalations_budget(self) -> None:
        primary = _CannedEngine(_GARBLED, confidence=0.95)
        escalation = _CannedEngine(_CLEAN, confidence=1.0, engine_name="vlm")
        tiered = _tiered(primary, escalation, max_escalations=1)
        first = tiered.extract_sync(_IMAGE)
        second = tiered.extract_sync(_IMAGE)
        assert first.engine_name == "vlm"  # first escalates
        assert second.engine_name == "canned"  # budget exhausted → keep primary
        assert escalation.calls == 1
        assert tiered.escalation_count == 1

    def test_reset_refreshes_budget(self) -> None:
        primary = _CannedEngine(_GARBLED, confidence=0.95)
        escalation = _CannedEngine(_CLEAN, confidence=1.0, engine_name="vlm")
        tiered = _tiered(primary, escalation, max_escalations=1)
        tiered.extract_sync(_IMAGE)
        tiered.reset()
        result = tiered.extract_sync(_IMAGE)
        assert result.engine_name == "vlm"
        assert escalation.calls == 2

    def test_escalation_failure_falls_back_to_primary(self) -> None:
        class _BoomEngine(OCREngine):
            name: ClassVar[str] = "boom"

            def extract_sync(self, image: KaosImage) -> OCRResult:
                raise RuntimeError("synthetic vlm failure")

        primary = _CannedEngine(_GARBLED, confidence=0.95)
        tiered = _tiered(primary, _BoomEngine())
        result = tiered.extract_sync(_IMAGE)
        assert result.engine_name == "canned"  # escalation crashed → primary

    def test_confidence_only_when_quality_helper_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate an older kaos-pdf without is_low_quality_layer: the garbled
        # high-confidence page must NOT escalate (no legibility signal).
        import kaos_pdf

        monkeypatch.delattr(kaos_pdf, "is_low_quality_layer", raising=False)
        primary = _CannedEngine(_GARBLED, confidence=0.95)
        escalation = _CannedEngine(_CLEAN, confidence=1.0, engine_name="vlm")
        tiered = _tiered(primary, escalation)
        result = tiered.extract_sync(_IMAGE)
        assert result.engine_name == "canned"
        assert escalation.calls == 0
