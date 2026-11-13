"""Labelled-date extraction from document TEXT and conservative date checks.

Only dates that sit next to a recognised label are used. Slash dates are ambiguous (03/09/2019 is 3 Sep or 9 Mar),
so a date is flagged only when it is a problem under EVERY valid interpretation. Scanned images are not parsed here
(OCR date reads are not trusted for a blocking rule); that is a documented limitation.
"""
from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Tuple

_EXPIRY = re.compile(r"(valid\s*(until|upto|up to|till|through)|expiry(\s*date)?|expires?(\s*on)?|date of expiry)", re.I)
_DATE = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})|(\d{4})-(\d{2})-(\d{2})")


def candidates(text: str) -> List[date]:
    """All valid readings of one date token."""
    m = _DATE.search(text)
    if not m:
        return []
    if m.group(4):
        try:
            return [date(int(m.group(4)), int(m.group(5)), int(m.group(6)))]
        except ValueError:
            return []
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    out = []
    for d, mo in ((a, b), (b, a)):
        try:
            out.append(date(y, mo, d))
        except ValueError:
            pass
    return sorted(set(out))


def labelled_dates(text: str) -> List[Tuple[str, str, List[date]]]:
    """[(kind, label, candidate_dates)] kind = EXPIRY | OTHER"""
    out = []
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        label, rest = line.split(":", 1)
        cands = candidates(rest)
        if cands and len(label) < 60:
            out.append(("EXPIRY" if _EXPIRY.search(label) else "OTHER", label.strip(), cands))
    return out


def expired(cands: List[date], today: date) -> bool:
    return bool(cands) and all(c < today for c in cands)


def in_future(cands: List[date], today: date) -> bool:
    return bool(cands) and all(c > today for c in cands)


def after(a: List[date], b: List[date]) -> bool:
    """True only if a is later than b under every pairing of interpretations."""
    return bool(a and b) and all(x > y for x in a for y in b)


def reference_date(items: List[Tuple[str, str, List[date]]], label_part: str) -> Optional[List[date]]:
    for _, label, c in items:
        if label_part in label.lower():
            return c
    return None
