"""Country rule packs: format and check-digit validation of SUBMITTED identifiers.

A valid format proves nothing about existence (that is the verification stage). An invalid format is
a correctable data-entry problem, so it leads to PENDING with a request for correction, never to rejection.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Tuple

from vendorflow.domain.fields import field_label
from vendorflow.domain.models import Finding
from vendorflow.utils.masking import mask_account
from vendorflow.validation import identifiers as ids

# (field, validator, human hint)   -- validators receive the NORMALIZED value
_PACKS: Dict[str, List[Tuple[str, Callable[[str], bool], str]]] = {
    "IN": [("pan", ids.is_valid_pan, "10 characters: 5 letters, 4 digits, 1 letter"),
           ("gstin", ids.is_valid_gstin, "15 characters with a valid state code, embedded PAN and check character"),
           ("cin", ids.is_valid_cin, "21 characters, for example U12345MH2020PTC123456"),
           ("ifsc", ids.is_valid_ifsc, "11 characters: 4 letters, a zero, 6 letters or digits"),
           ("bank_account_number", ids.is_valid_in_account, "9 to 18 digits")],
    "US": [("ein", ids.is_valid_ein, "9 digits, written NN-NNNNNNN"),
           ("routing_number", ids.is_valid_aba, "9 digits that satisfy the ABA checksum"),
           ("bank_account_number", ids.is_valid_us_account, "4 to 17 digits")],
    "GB": [("company_number", ids.is_valid_uk_company_number, "8 characters: 8 digits, or 2 letters and 6 digits"),
           ("vat_number", ids.is_valid_uk_vat, "GB followed by 9 digits with valid check digits"),
           ("sort_code", ids.is_valid_sort_code, "6 digits, written NN-NN-NN"),
           ("bank_account_number", ids.is_valid_uk_account, "8 digits")],
}


def _shown(field: str, value: str) -> str:
    return mask_account(value) if field == "bank_account_number" else value


def validate_formats(country: str, norm: Mapping[str, str]) -> List[Finding]:
    out: List[Finding] = []
    for name, check, hint in _PACKS.get(country, []):
        v = norm.get(name, "")
        if not v:
            continue                                   # absence is the completeness stage's business
        ok = check(v)
        out.append(Finding(f"FORMAT.{name}", "FORMAT", "PASS" if ok else "FAIL", field=name,
                           reason_code=None if ok else "INVALID_FORMAT", expected=None if ok else hint,
                           observed=None if ok else _shown(name, v),
                           message=(f"{field_label(name)} has a valid format (existence is not implied)." if ok else
                                    f"{field_label(name)} '{_shown(name, v)}' is not a valid format. Expected: {hint}.")))
    if country == "IN" and norm.get("gstin") and norm.get("pan") and ids.is_valid_gstin_format(norm["gstin"]) and ids.is_valid_pan(norm["pan"]):
        same = ids.gstin_embedded_pan(norm["gstin"]) == norm["pan"]
        out.append(Finding("FORMAT.gstin_pan_link", "FORMAT", "PASS" if same else "FAIL", field="gstin",
                           reason_code=None if same else "INVALID_FORMAT",
                           expected=None if same else "GSTIN characters 3-12 equal the PAN",
                           observed=None if same else f"GSTIN embeds {ids.gstin_embedded_pan(norm['gstin'])}, PAN is {norm['pan']}",
                           message=("The PAN embedded in the GSTIN equals the submitted PAN." if same else
                                    f"The PAN embedded in the GSTIN ({ids.gstin_embedded_pan(norm['gstin'])}) differs from the submitted "
                                    f"PAN ({norm['pan']}). One of them is wrong, or they belong to different entities.")))
    return out


def other_country_matches(country: str, norm: Mapping[str, str]) -> Optional[str]:
    """If the selected country's primary tax identifier is invalid but valid for ANOTHER country, name that country."""
    primary = {"IN": "gstin", "US": "ein", "GB": "vat_number"}
    v = norm.get(primary[country], "")
    if not v or _PACKS_OK(country, primary[country], v):
        return None
    for other, field in primary.items():
        if other != country and (ids.is_valid_gstin(v) if other == "IN" else ids.is_valid_ein(v) if other == "US" else ids.is_valid_uk_vat(v)):
            return other
    return None


def _PACKS_OK(country: str, field: str, v: str) -> bool:
    for name, check, _ in _PACKS.get(country, []):
        if name == field:
            return check(v)
    return True
