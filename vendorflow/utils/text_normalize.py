"""Text normalization helpers shared by ingestion, grounding and (later) name matching."""
from __future__ import annotations

import re
import unicodedata

_LEGAL_SUFFIX = {
    "pvt": "private", "ltd": "limited", "llc": "limited liability company",
    "l l c": "limited liability company", "inc": "incorporated", "corp": "corporation",
    "co": "company", "plc": "public limited company", "llp": "limited liability partnership",
}


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def alnum_lower(s: str) -> str:
    """Lowercase letters+digits only. Used for tolerant 'is this text inside that text' checks."""
    return re.sub(r"[^a-z0-9]", "", nfkc(s).lower())


def normalize_company_name(name: str) -> str:
    """Light normalization: case, '&'->and, punctuation, common legal-suffix expansion.

    This is deliberately *not* fuzzy. Phase 4's name matcher builds on it, and fuzzy
    similarity is never used to establish legal identity on its own.
    """
    s = nfkc(name).lower().replace("&", " and ")
    s = re.sub(r"[.,'’`\"()/\\-]", " ", s)
    s = collapse_ws(s)
    s = s.replace("l l c", "llc")
    tokens = [_LEGAL_SUFFIX.get(t, t) for t in s.split(" ") if t]
    s = " ".join(tokens)
    return collapse_ws(s)
