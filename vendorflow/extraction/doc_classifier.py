"""Document-type detection from content (title keywords), tolerant of OCR noise.

Used to catch 'vendors attach the wrong documents' (e.g. a bank letter uploaded as the W-9).
A detection is only trusted when its confidence is high (see completeness.WRONG_TYPE_MIN_CONFIDENCE).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

TITLES = {
    "registration_form": ["VENDOR REGISTRATION FORM"],
    "registration_certificate": ["CERTIFICATE OF INCORPORATION", "ARTICLES OF ORGANIZATION", "CERTIFICATE OF FORMATION",
                                 "CERTIFICATE OF ORGANIZATION"],
    "tax_certificate": ["GST REGISTRATION CERTIFICATE", "FORM W-9", "VAT REGISTRATION CERTIFICATE"],
    "tax_id_proof": ["PERMANENT ACCOUNT NUMBER CARD"],
    "bank_proof": ["BANK ACCOUNT CONFIRMATION LETTER", "CANCELLED CHEQUE", "VOIDED CHECK"],
}
HEAD_CHARS = 400          # titles live at the top; later text (footers, body labels) must not influence the type
MIN_SCORE = 0.60
MIN_MARGIN = 0.20


def _compact(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _phrase_score(head_upper: str, head_compact: str, phrase: str) -> float:
    tokens = set(re.findall(r"[A-Z0-9]+", head_upper))
    words = re.findall(r"[A-Z0-9]+", phrase)
    token_frac = sum(1 for w in words if w in tokens) / len(words)
    cp = _compact(phrase)
    if cp in head_compact:
        return 1.0
    m = SequenceMatcher(None, head_compact, cp, autojunk=False).find_longest_match(0, len(head_compact), 0, len(cp))
    return max(token_frac, m.size / len(cp))


def classify(text: str) -> Tuple[Optional[str], float, List[str]]:
    """Return (doc_type | None, confidence 0..1, matched phrases). None = unknown or ambiguous."""
    head = (text or "")[:HEAD_CHARS]
    hu, hc = head.upper(), _compact(head)
    if len(hc) < 10:
        return None, 0.0, []
    scores = []
    for dtype, phrases in TITLES.items():
        best = max(((_phrase_score(hu, hc, p), p) for p in phrases), key=lambda x: x[0])
        scores.append((best[0], dtype, best[1]))
    scores.sort(reverse=True)
    top, second = scores[0], scores[1]
    if top[0] < MIN_SCORE or top[0] - second[0] < MIN_MARGIN:
        return None, round(top[0], 2), []
    return top[1], round(top[0], 2), [top[2]]
