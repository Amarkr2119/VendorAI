"""Google Gemini over REST (generateContent). API key is sent in a header, never in the URL or logs.

NOT live-tested in the build sandbox (no network). Verify locally with:  python scripts/check_llm.py --ping gemini
"""
from __future__ import annotations

import base64
import time
from typing import Any, List

import requests

from vendorflow.llm.base import LlmError, LlmErrorKind, LlmResponse, classify_http, retry_after_seconds

BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, session: Any = None):
        self.api_key, self.model = api_key, model
        self.session = session or requests

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def generate_json(self, system: str, user: str, timeout: float) -> LlmResponse:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        t0 = time.monotonic()
        try:
            r = self.session.post(f"{BASE}/models/{self.model}:generateContent", headers=self._headers(), json=body, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise LlmError(LlmErrorKind.TRANSIENT, f"{type(e).__name__}: {e}") from e
        latency = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            raise LlmError(classify_http(r.status_code, r.text), f"HTTP {r.status_code}: {r.text[:300]}", r.status_code,
                           retry_after_seconds(getattr(r, "headers", None)))
        return self._response(r, latency)

    def generate_text_from_image(self, prompt: str, image: bytes, mime_type: str, timeout: float) -> LlmResponse:
        """Ask Gemini to transcribe an image without exposing the API key in URLs or logs."""
        body = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image).decode("ascii")}},
            ]}],
            "generationConfig": {"temperature": 0},
        }
        t0 = time.monotonic()
        try:
            r = self.session.post(f"{BASE}/models/{self.model}:generateContent", headers=self._headers(), json=body, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise LlmError(LlmErrorKind.TRANSIENT, f"{type(e).__name__}: {e}") from e
        latency = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            raise LlmError(classify_http(r.status_code, r.text), f"HTTP {r.status_code}: {r.text[:300]}", r.status_code,
                           retry_after_seconds(getattr(r, "headers", None)))
        return self._response(r, latency)

    @staticmethod
    def _response(response: Any, latency: int) -> LlmResponse:
        try:
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
                raise LlmError(LlmErrorKind.BAD_OUTPUT, f"Gemini returned no candidates ({reason})")
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts)
        except ValueError as exc:
            raise LlmError(LlmErrorKind.BAD_OUTPUT, f"Response was not JSON: {exc}") from exc
        return LlmResponse(text, latency, data.get("usageMetadata", {}))

    def list_models(self, timeout: float = 20.0) -> List[str]:
        try:
            r = self.session.get(f"{BASE}/models", headers=self._headers(), params={"pageSize": 200}, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise LlmError(LlmErrorKind.TRANSIENT, str(e)) from e
        if r.status_code != 200:
            raise LlmError(classify_http(r.status_code, r.text), f"HTTP {r.status_code}: {r.text[:300]}", r.status_code)
        return sorted(m["name"].split("/", 1)[-1] for m in r.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))
