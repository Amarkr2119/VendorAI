"""Pure functions for identifier formats and check digits (India, USA, UK).

IMPORTANT: a valid *format* proves nothing about whether an identifier exists, is active, or
belongs to the submitted business. Existence/ownership needs an external provider (Phase 4).
Phase 3 uses these functions for (a) the extraction format-validity signal and (b) generating
format-valid *synthetic* identifiers. Phase 4 wraps them in country rule packs.
"""
from __future__ import annotations

import re
from typing import Optional

# ----------------------------------------------------------------------------- India
_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_GSTIN_RE = re.compile(r"^(0[1-9]|[12][0-9]|3[0-8]|9[79])[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_CIN_RE = re.compile(r"^[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}$")
_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def is_valid_pan(v: str) -> bool:
    return bool(_PAN_RE.match(v or ""))


def pan_entity_type(pan: str) -> Optional[str]:
    """4th PAN character encodes the holder type (C = company, F = firm, P = person ...)."""
    return pan[3] if is_valid_pan(pan) else None


def gstin_check_char(first14: str) -> str:
    """GSTIN check character: mod-36 weighted sum over the first 14 characters."""
    total = 0
    for i, ch in enumerate(first14):
        val = _GST_CHARS.index(ch)
        prod = val * (1 if i % 2 == 0 else 2)
        total += prod // 36 + prod % 36
    return _GST_CHARS[(36 - total % 36) % 36]


def is_valid_gstin_format(v: str) -> bool:
    return bool(_GSTIN_RE.match(v or ""))


def is_valid_gstin(v: str) -> bool:
    """Pattern AND check character."""
    v = v or ""
    return is_valid_gstin_format(v) and gstin_check_char(v[:14]) == v[14]


def gstin_embedded_pan(gstin: str) -> Optional[str]:
    return gstin[2:12] if is_valid_gstin_format(gstin) else None


def is_valid_cin(v: str) -> bool:
    return bool(_CIN_RE.match(v or ""))


def is_valid_ifsc(v: str) -> bool:
    return bool(_IFSC_RE.match(v or ""))


def is_valid_in_account(v: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{9,18}", v or ""))


# ----------------------------------------------------------------------------- USA
_EIN_RE = re.compile(r"^[0-9]{2}-?[0-9]{7}$")
# Prefixes the IRS has not assigned (commonly published list). Used as a *warning-level* check
# only: verify against current IRS guidance before relying on it in production.
_EIN_UNASSIGNED_PREFIXES = {"07", "08", "09", "17", "18", "19", "28", "29", "49", "69", "70",
                            "78", "79", "89", "96", "97"}


def is_valid_ein(v: str) -> bool:
    return bool(_EIN_RE.match(v or ""))


def ein_prefix_known(v: str) -> bool:
    digits = re.sub(r"\D", "", v or "")
    return len(digits) == 9 and digits[:2] not in _EIN_UNASSIGNED_PREFIXES


def aba_check_digit(first8: str) -> str:
    d = [int(c) for c in first8]
    total = 3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5])
    return str((10 - total % 10) % 10)


def is_valid_aba(v: str) -> bool:
    """9 digits and the 3-7-1 weighted checksum is divisible by 10."""
    if not re.fullmatch(r"[0-9]{9}", v or ""):
        return False
    return aba_check_digit(v[:8]) == v[8]


def is_valid_us_account(v: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{4,17}", v or ""))


# ----------------------------------------------------------------------------- UK
_UK_COMPANY_RE = re.compile(r"^([A-Z]{2}[0-9]{6}|[0-9]{8})$")
_VAT_WEIGHTS = (8, 7, 6, 5, 4, 3, 2)


def _vat_total(first7: str) -> int:
    return sum(int(c) * w for c, w in zip(first7, _VAT_WEIGHTS))


def uk_vat_check_digits(first7: str, new_style: bool = False) -> str:
    total = _vat_total(first7) + (55 if new_style else 0)
    return f"{(-total) % 97:02d}"


def is_valid_uk_vat(v: str) -> bool:
    """Standard 9-digit UK VAT mod-97 check (old and '97-55' variants). Digits only, no GB prefix."""
    if not re.fullmatch(r"[0-9]{9}", v or ""):
        return False
    total, check = _vat_total(v[:7]), int(v[7:])
    return (total + check) % 97 == 0 or (total + 55 + check) % 97 == 0


def is_valid_uk_company_number(v: str) -> bool:
    return bool(_UK_COMPANY_RE.match(v or ""))


def is_valid_sort_code(v: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{6}", v or ""))


def is_valid_uk_account(v: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{8}", v or ""))


# ----------------------------------------------------------------------------- shared
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def is_valid_email(v: str) -> bool:
    return bool(_EMAIL_RE.match(v or ""))


def format_valid(country: str, field: str, normalized_value: str) -> Optional[bool]:
    """Format (and check-digit) validity of a *normalized* value; None = no format rule exists."""
    v = normalized_value or ""
    table = {
        "pan": is_valid_pan, "gstin": is_valid_gstin, "cin": is_valid_cin, "ifsc": is_valid_ifsc,
        "ein": is_valid_ein, "routing_number": is_valid_aba,
        "company_number": is_valid_uk_company_number, "vat_number": is_valid_uk_vat,
        "sort_code": is_valid_sort_code, "contact_email": is_valid_email,
    }
    if field == "ein":
        return is_valid_ein(v)
    if field in table:
        return table[field](v)
    if field == "bank_account_number":
        return {"IN": is_valid_in_account, "US": is_valid_us_account, "GB": is_valid_uk_account}.get(country, lambda x: None)(v)
    return None
