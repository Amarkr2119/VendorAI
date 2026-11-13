"""Deterministic extractor: labelled lines ("Label: value") and identifier-pattern scans.

Two roles: (1) the whole extraction path when no LLM is available, (2) an independent second opinion
beside the LLM. Agreement between the two raises confidence; disagreement lowers it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from vendorflow.domain.fields import FIELD_LABELS
from vendorflow.utils.text_normalize import alnum_lower

_PATTERNS = {
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "gstin": r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b",
    "cin": r"\b[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b",
    "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "ein": r"\b[0-9]{2}-[0-9]{7}\b",
    "vat_number": r"\bGB\s?[0-9]{3}\s?[0-9]{4}\s?[0-9]{2}\b",
    "contact_email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
}


@dataclass
class Candidate:
    value: str
    snippet: str
    kind: str            # label | pattern


def _clean(v: str) -> str:
    return re.sub(r"\s+", " ", v).strip().strip("\"'").strip()


def labelled_lines(text: str) -> List[Tuple[str, str, str]]:
    """(normalized_label, value, raw_line) for every line shaped like 'Label: value'."""
    out = []
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        if not value.strip() or len(label) > 70:
            continue
        out.append((alnum_lower(label), _clean(value), line.strip()))
    return out


def regex_candidates(text: str, expected: Sequence[Tuple[str, bool]]) -> Dict[str, List[Candidate]]:
    """All candidates per expected field: label matches first (in document order), then pattern matches."""
    label_to_field: Dict[str, str] = {}
    for name, _ in expected:
        for lab in FIELD_LABELS.get(name, []):
            label_to_field.setdefault(alnum_lower(lab), name)
    out: Dict[str, List[Candidate]] = {name: [] for name, _ in expected}
    for nlabel, value, raw in labelled_lines(text):
        field = label_to_field.get(nlabel)
        if field and value:
            out[field].append(Candidate(value, raw, "label"))
    for name, _ in expected:
        pat = _PATTERNS.get(name)
        if not pat:
            continue
        seen = {alnum_lower(c.value) for c in out[name]}
        for line in (text or "").splitlines():
            for m in re.finditer(pat, line):
                if alnum_lower(m.group(0)) not in seen:
                    out[name].append(Candidate(m.group(0), line.strip(), "pattern"))
                    seen.add(alnum_lower(m.group(0)))
    return out
