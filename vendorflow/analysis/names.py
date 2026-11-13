"""Company-name comparison. Deterministic, explainable, deliberately conservative.

Levels:  MATCH (>= 0.97)   same name after normalization ('Pvt. Ltd.' == 'Private Limited', '&' == 'and', case)
         SIMILAR (>= 0.80) probably related but not identical -> a human decides
         MISMATCH          clearly different names
Similarity alone never establishes fraud or identity: callers turn SIMILAR/MISMATCH into a REVIEW item, nothing more.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, Tuple

from vendorflow.utils.text_normalize import normalize_company_name

MATCH_MIN, SIMILAR_MIN = 0.97, 0.80
_FORM_TOKENS = {"private", "limited", "company", "incorporated", "corporation", "llc", "inc", "ltd", "plc", "llp", "pvt",
                "liability", "partnership", "public", "co", "corp", "the", "and"}


def _core(norm: str) -> str:
    return " ".join(t for t in norm.split() if t not in _FORM_TOKENS)


def name_score(a: str, b: str) -> float:
    na, nb = normalize_company_name(a or ""), normalize_company_name(b or "")
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ca, cb = _core(na), _core(nb)
    if ca and ca == cb:
        return 0.93                                   # same core name, different legal form (e.g. Limited vs Private Limited)
    ratio = SequenceMatcher(None, ca or na, cb or nb).ratio()
    ta, tb = set((ca or na).split()), set((cb or nb).split())
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return round(min(ratio, 0.96) if jac < 1 else min(max(ratio, jac), 0.96), 3)


def classify_names(a: str, b: str) -> Tuple[str, float]:
    s = name_score(a, b)
    return ("MATCH" if s >= MATCH_MIN else "SIMILAR" if s >= SIMILAR_MIN else "MISMATCH"), s


def address_score(a: str, b: str) -> float:
    """Token-set overlap on alphanumerics (addresses are reordered/abbreviated more than names)."""
    import re
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / min(len(ta), len(tb)), 3)


def explain(a: str, b: str) -> Dict[str, object]:
    level, score = classify_names(a, b)
    return {"a": a, "b": b, "level": level, "score": score}
