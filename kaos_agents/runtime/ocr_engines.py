"""VLM OCR engines for scanned-PDF extraction (Tesseract -> VLM escalation).

These engines implement kaos-pdf's :class:`~kaos_pdf.ocr.base.OCREngine` ABC so
they plug directly into ``kaos_pdf.parse_pdf(ocr_engine=...)`` and into
:meth:`BaseAgent._ocr_pdf_bytes_to_content_document`.

They live in kaos-agents — not kaos-pdf — because the VLM path depends on
``kaos_llm_core.vision`` and kaos-pdf must not depend on the LLM stack
(extraction -> LLM is one-directional; see kaos-pdf audit PDF-001). kaos-agents
already depends on both kaos-pdf and kaos-llm-core, so it is the correct
integration home.

Two engines:

- :class:`VlmOcrEngine` — runs ``kaos_llm_core.vision.ocr_page`` (a vision
  model, default Claude Haiku) over a rendered page image. Higher accuracy on
  degraded / handwritten / multi-column scans, but ~10x the per-page cost of
  Tesseract and a live API call. The VLM returns plain text with no per-line
  geometry, so emitted :class:`~kaos_pdf.ocr.base.OCRLine` values carry
  ``bbox=None`` and ``confidence=1.0``.
- :class:`TieredOCREngine` — runs a cheap primary engine (Tesseract) first and
  escalates a page to the VLM only when the primary output looks bad: low mean
  confidence OR a garbled native-text-style layer (``is_low_quality_layer``).
  Empirically Tesseract is over-confident on hard scans, so the legibility
  signal is the stronger gate. A per-instance ``max_escalations`` budget bounds
  spend across a document.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import TYPE_CHECKING, ClassVar

from kaos_core.logging import get_logger
from kaos_pdf.ocr.base import OCREngine, OCRLine, OCRResult

from kaos_agents.errors import VisionOcrUnavailableError

if TYPE_CHECKING:
    from kaos_content.images.model import KaosImage

logger = get_logger(__name__)

#: Default vision model — mirrors ``kaos_llm_core.vision.DEFAULT_VISION_MODEL``.
#: Duplicated as a string so importing this module does not require the
#: ``[vision]`` extra (the real default is read lazily at call time).
DEFAULT_VISION_MODEL = "anthropic:claude-haiku-4-5"


class VlmOcrEngine(OCREngine):
    """OCR engine backed by ``kaos_llm_core.vision.ocr_page`` (a vision model).

    Construct with an optional ``model`` override (``provider:model`` string);
    ``None`` uses the kaos-llm-core default (Claude Haiku). The vision
    dependencies (``kaos-llm-core[vision]``) and a provider API key are required
    at call time, not import time — a missing extra raises
    :class:`~kaos_agents.errors.VisionOcrUnavailableError` with recovery
    guidance.
    """

    name: ClassVar[str] = "vlm"

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    async def extract(self, image: KaosImage) -> OCRResult:
        """Async path — call the vision model directly (no executor hop)."""
        try:
            from kaos_llm_core.vision import ocr_page
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            msg = (
                "VLM OCR requested but kaos-llm-core[vision] is not installed. "
                "Install it with `pip install 'kaos-agents[vision]'` (or "
                "`pip install 'kaos-llm-core[vision]'`) and set a provider API "
                "key. Alternative: disable VLM escalation "
                "(KAOS_AGENT_OCR_VLM_ESCALATION=0) to use Tesseract-only OCR."
            )
            raise VisionOcrUnavailableError(msg) from exc

        model = self._model or DEFAULT_VISION_MODEL
        result = await ocr_page(image, model=model)
        lines = [
            OCRLine(text=line, bbox=None, confidence=1.0)
            for line in result.text.splitlines()
            if line.strip()
        ]
        return OCRResult(lines=lines, engine_name=f"vlm:{result.model}")

    def extract_sync(self, image: KaosImage) -> OCRResult:
        """Sync path — safe whether or not an event loop is already running.

        The agent runtime is async, so a bare ``asyncio.run`` here would raise
        "cannot be called from a running event loop". When a loop is already
        running we offload to a worker thread that owns its own loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.extract(image))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(self.extract(image)))
            return future.result()


class TieredOCREngine(OCREngine):
    """Run a cheap primary OCR engine, escalate bad pages to a costlier one.

    A page is escalated when the primary result's mean confidence is below
    ``min_confidence`` OR its text reads like a garbled layer per
    ``kaos_pdf.is_low_quality_layer`` (the stronger signal, since Tesseract is
    over-confident on hard scans). ``max_escalations`` bounds the number of
    escalations over this engine's lifetime — useful as a per-document spend
    cap; ``None`` means unbounded.

    .. note::

        Unlike the stateless-engine guidance on :class:`OCREngine`, this engine
        is **deliberately stateful**: it keeps an escalation counter (guarded by
        a lock) so the budget survives across per-page calls. Call
        :meth:`reset` between documents to refresh the budget.
    """

    name: ClassVar[str] = "tiered"

    def __init__(
        self,
        primary: OCREngine,
        escalation: OCREngine,
        *,
        min_confidence: float = 0.6,
        quality_threshold: float = 0.35,
        max_escalations: int | None = None,
    ) -> None:
        self._primary = primary
        self._escalation = escalation
        self._min_confidence = min_confidence
        self._quality_threshold = quality_threshold
        self._max_escalations = max_escalations
        self._escalation_count = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Reset the escalation budget counter (call between documents)."""
        with self._lock:
            self._escalation_count = 0

    @property
    def escalation_count(self) -> int:
        """Number of pages escalated to the secondary engine so far."""
        with self._lock:
            return self._escalation_count

    def _should_escalate(self, result: OCRResult) -> bool:
        if result.mean_confidence < self._min_confidence:
            return True
        try:
            from kaos_pdf import is_low_quality_layer
        except ImportError:
            # Older kaos-pdf without the legibility helper — confidence only.
            return False
        return is_low_quality_layer(result.text, threshold=self._quality_threshold)

    def extract_sync(self, image: KaosImage) -> OCRResult:
        primary = self._primary.extract_sync(image)
        if not self._should_escalate(primary):
            return primary

        with self._lock:
            if (
                self._max_escalations is not None
                and self._escalation_count >= self._max_escalations
            ):
                logger.info(
                    "tiered_ocr: escalation budget (%d) reached; keeping primary "
                    "result (engine=%s, mean_confidence=%.3f)",
                    self._max_escalations,
                    self._primary.name,
                    primary.mean_confidence,
                )
                return primary
            self._escalation_count += 1
            current = self._escalation_count

        logger.info(
            "tiered_ocr: escalating page to %s (escalation #%d, primary engine=%s "
            "mean_confidence=%.3f)",
            self._escalation.name,
            current,
            self._primary.name,
            primary.mean_confidence,
        )
        try:
            return self._escalation.extract_sync(image)
        except Exception:
            logger.exception(
                "tiered_ocr: escalation engine %s failed; falling back to primary",
                self._escalation.name,
            )
            return primary


__all__ = [
    "DEFAULT_VISION_MODEL",
    "TieredOCREngine",
    "VlmOcrEngine",
]
