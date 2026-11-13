"""Evidence grounding: an extracted value is only trusted if its quoted source snippet really exists in
the document text AND actually contains the value. Catches LLM fabrication and 'helpful corrections'."""
from __future__ import annotations

from typing import Optional, Tuple

from vendorflow.utils.text_normalize import alnum_lower


def is_grounded(value: Optional[str], snippet: Optional[str], source_text: str) -> Tuple[bool, str]:
    v, s, src = alnum_lower(value or ""), alnum_lower(snippet or ""), alnum_lower(source_text or "")
    if not v:
        return False, "no value"
    if not s:
        return False, "no source snippet was provided"
    if s not in src:
        return False, "the quoted snippet does not appear in the document text"
    if v not in s:
        return False, "the quoted snippet does not contain the extracted value"
    return True, "snippet found in document and contains the value"
