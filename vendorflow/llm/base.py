"""Common LLM types. Clients are thin REST wrappers; policy (retry, fallback, cooldown) lives in the router."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from vendorflow.domain.enums import StrEnum


class LlmErrorKind(StrEnum):
    QUOTA = "QUOTA"              # 429 / rate limit: cool down, move to fallback
    TRANSIENT = "TRANSIENT"      # timeout, connection error, 5xx: retry with backoff
    AUTH = "AUTH"                # bad/missing key: disable provider for the session
    CONFIG = "CONFIG"            # unknown model etc.: disable provider for the session
    BAD_REQUEST = "BAD_REQUEST"  # 4xx the request itself caused: do not retry
    BAD_OUTPUT = "BAD_OUTPUT"    # reply was not usable (invalid JSON, blocked)


class LlmError(Exception):
    def __init__(self, kind: LlmErrorKind, message: str, status_code: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.kind, self.message, self.status_code, self.retry_after = kind, message, status_code, retry_after


class LlmUnavailable(Exception):
    """Every configured provider failed (or none is configured). Callers fall back to deterministic extraction."""

    def __init__(self, attempts: List[Dict[str, Any]]):
        super().__init__("No LLM provider could serve the request")
        self.attempts = attempts


@dataclass
class LlmResponse:
    text: str
    latency_ms: int = 0
    usage: Dict[str, Any] = field(default_factory=dict)


class LlmClient(Protocol):
    name: str
    model: str

    def generate_json(self, system: str, user: str, timeout: float) -> LlmResponse: ...
    def list_models(self, timeout: float = 20.0) -> List[str]: ...


def classify_http(status: int, body: str) -> LlmErrorKind:
    low = (body or "").lower()
    if status == 429:
        return LlmErrorKind.QUOTA
    if status in (401, 403):
        return LlmErrorKind.AUTH
    if status == 404:
        return LlmErrorKind.CONFIG
    if status == 400:
        # Gemini reports an invalid key as HTTP 400 with API_KEY_INVALID / "API key not valid".
        if "api key" in low or "api_key_invalid" in low:
            return LlmErrorKind.AUTH
        return LlmErrorKind.BAD_REQUEST
    if status == 408 or status >= 500:
        return LlmErrorKind.TRANSIENT
    return LlmErrorKind.BAD_REQUEST


def retry_after_seconds(headers: Any) -> Optional[float]:
    try:
        v = headers.get("Retry-After") if headers else None
        return float(v) if v is not None else None
    except (TypeError, ValueError, AttributeError):
        return None
