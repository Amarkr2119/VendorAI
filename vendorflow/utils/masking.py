"""Masking of sensitive values for display, logs and stored API payloads."""
from __future__ import annotations

import re
from typing import Any, Optional

_SENSITIVE_KEYS = ("account_number", "bank_account", "account_no", "iban", "api_key", "authorization")
# Long digit runs (bank account numbers) and common API-key shapes.
# 8-18 contiguous digits, or digits grouped in fours ("1234 5678 9012"). Dates like 2026-11-10 do not match.
_DIGIT_RUN = re.compile(r"(?<![\d])(\d{8,18}|(?:\d{4}[ -]){2,4}\d{1,4})(?![\d])")
_KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),     # Google API key shape
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),        # Groq API key shape
    re.compile(r"(?i)bearer\s+[0-9A-Za-z._\-]{16,}"),
]


def mask_account(value: Optional[str], visible: int = 4) -> str:
    """'123456789012' -> '********9012'. Empty stays empty."""
    if not value:
        return ""
    s = re.sub(r"\s+", "", str(value))
    if len(s) <= visible:
        return "*" * len(s)
    return "*" * (len(s) - visible) + s[-visible:]


def _mask_digit_run(m: "re.Match[str]") -> str:
    raw = m.group(1)
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 8:          # too short to be an account number (dates, pins, sort codes...)
        return raw
    return mask_account(digits)


def mask_text(text: Optional[str]) -> str:
    """Mask account-number-like digit runs and API keys inside free text (logs, snippets)."""
    if not text:
        return ""
    out = str(text)
    for pat in _KEY_PATTERNS:
        out = pat.sub("[REDACTED_KEY]", out)
    return _DIGIT_RUN.sub(_mask_digit_run, out)


def mask_payload(obj: Any) -> Any:
    """Recursively mask a dict/list payload: sensitive keys are masked, strings are scrubbed."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if any(s in lk for s in _SENSITIVE_KEYS) and isinstance(v, (str, int)):
                out[k] = "[REDACTED]" if ("key" in lk or "authorization" in lk) else mask_account(str(v))
            else:
                out[k] = mask_payload(v)
        return out
    if isinstance(obj, list):
        return [mask_payload(v) for v in obj]
    if isinstance(obj, str):
        return mask_text(obj)
    return obj
