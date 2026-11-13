"""Gemini-based OCR for scanned document pages.

There is intentionally no local OCR dependency in this module. A scanned page
is sent to Gemini as an image and Gemini returns the visible text. Gemini does
not expose word-level OCR confidence, so the result is assigned a conservative
page confidence rather than inventing precise per-word scores.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, Union

from PIL import Image

from vendorflow.config import Settings, load_settings
from vendorflow.llm.base import LlmError
from vendorflow.llm.gemini_client import GeminiClient
from vendorflow.utils.text_normalize import alnum_lower


@dataclass
class OcrResult:
    text: str
    words: List[Tuple[str, float]] = field(default_factory=list)
    mean_conf: float = 0.0
    engine: str = ""

    def confidence_for(self, value: str) -> float:
        """Return matching word confidence, or a conservative page confidence."""
        tokens = [alnum_lower(token) for token in value.split() if alnum_lower(token)]
        if not tokens:
            return self.mean_conf
        confidences = []
        for token in tokens:
            hits = [confidence for word, confidence in self.words if alnum_lower(word) == token]
            if hits:
                confidences.append(sum(hits) / len(hits))
        if len(confidences) < len(tokens):
            confidences.append(self.mean_conf * 0.75)
        return sum(confidences) / len(confidences)


class OcrEngine(Protocol):
    name: str

    def recognize(self, img: Image.Image) -> OcrResult: ...


class OcrUnavailable(RuntimeError):
    """Raised when Gemini OCR cannot be used."""


class GeminiOcrEngine:
    """Read a document image using Gemini's multimodal ``generateContent`` API."""

    name = "gemini"
    # Gemini does not return calibrated word-level OCR confidence. Keep this
    # below text-layer certainty but above the default 0.60 quality threshold.
    page_confidence = 0.75
    _PROMPT = (
        "You are an OCR engine. Transcribe all visible text in this document image exactly as written. "
        "Preserve labels, values, line breaks, punctuation, and identifier characters. "
        "Do not summarize, explain, correct, infer missing characters, or add markdown."
    )

    def __init__(self, settings: Settings, client: Optional[GeminiClient] = None):
        if not settings.provider_configured("gemini"):
            raise OcrUnavailable("Gemini OCR requires GEMINI_API_KEY and GEMINI_MODEL")
        self.settings = settings
        self.client = client or GeminiClient(settings.gemini_api_key, settings.gemini_model)

    def recognize(self, img: Image.Image) -> OcrResult:
        payload = io.BytesIO()
        img.convert("RGB").save(payload, format="PNG", optimize=True)
        try:
            response = self.client.generate_text_from_image(
                self._PROMPT, payload.getvalue(), "image/png", self.settings.llm_timeout_s
            )
        except LlmError as exc:
            raise OcrUnavailable(f"Gemini OCR failed: {exc}") from exc
        text = response.text.strip()
        if not text:
            raise OcrUnavailable("Gemini OCR returned no readable text")
        words = [(word, self.page_confidence) for word in text.split()]
        return OcrResult(text, words, self.page_confidence, self.name)


def get_ocr_engine(config: Union[Settings, str, None] = None) -> Optional[OcrEngine]:
    """Return Gemini OCR when configured; never fall back to a local OCR engine.

    ``str`` is still accepted for older callers. Only ``gemini`` and ``auto``
    can create an engine; ``none`` and retired local engine names return None.
    """
    if isinstance(config, Settings):
        settings, setting = config, config.ocr_engine
    else:
        settings, setting = load_settings(), (config or "gemini")
    if setting.lower() not in {"auto", "gemini"}:
        return None
    try:
        return GeminiOcrEngine(settings)
    except OcrUnavailable:
        return None
