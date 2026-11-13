"""Groq over its OpenAI-compatible REST API. Used as the text-only fallback provider.

NOT live-tested in the build sandbox (no network). Verify locally with:  python scripts/check_llm.py --ping groq
"""
from __future__ import annotations

import time
from typing import Any, List

import requests

from vendorflow.llm.base import LlmError, LlmErrorKind, LlmResponse, classify_http, retry_after_seconds

BASE = "https://api.groq.com/openai/v1"


class GroqClient:
    name = "groq"

    def __init__(self, api_key: str, model: str, session: Any = None):
        self.api_key, self.model = api_key, model
        self.session = session or requests

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate_json(self, system: str, user: str, timeout: float) -> LlmResponse:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},      # JSON mode requires the word "JSON" in the prompt (it is)
        }
        t0 = time.monotonic()
        try:
            r = self.session.post(f"{BASE}/chat/completions", headers=self._headers(), json=body, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise LlmError(LlmErrorKind.TRANSIENT, f"{type(e).__name__}: {e}") from e
        latency = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            raise LlmError(classify_http(r.status_code, r.text), f"HTTP {r.status_code}: {r.text[:300]}", r.status_code,
                           retry_after_seconds(getattr(r, "headers", None)))
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LlmError(LlmErrorKind.BAD_OUTPUT, f"Unexpected Groq response shape: {e}") from e
        return LlmResponse(text, latency, data.get("usage", {}))

    def list_models(self, timeout: float = 20.0) -> List[str]:
        try:
            r = self.session.get(f"{BASE}/models", headers=self._headers(), timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            raise LlmError(LlmErrorKind.TRANSIENT, str(e)) from e
        if r.status_code != 200:
            raise LlmError(classify_http(r.status_code, r.text), f"HTTP {r.status_code}: {r.text[:300]}", r.status_code)
        return sorted(m["id"] for m in r.json().get("data", []))
