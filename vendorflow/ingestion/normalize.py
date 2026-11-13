"""Turn raw input (form dict, JSON, extracted PDF-form fields) into one canonical payload."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from vendorflow.domain.fields import (COUNTRY_FIELDS, DECLARATIONS, canonical_field_name, display_value, fields_for,
                                      normalize_country, normalize_value)
from vendorflow.domain.models import Submission

_TRUE = {"true", "yes", "y", "1", "on"}


def parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in _TRUE


def canonicalize_fields(raw: Mapping[str, Any]) -> Dict[str, str]:
    """Map arbitrary input keys ('GST No.', 'Account Holder'...) onto canonical field names. Unknown keys are dropped."""
    out: Dict[str, str] = {}
    for key, value in raw.items():
        name = canonical_field_name(str(key))
        if name and name not in out:
            out[name] = display_value(None if value is None else str(value))
    return out


def canonicalize_declarations(country: str, raw: Mapping[str, Any]) -> Dict[str, bool]:
    return {d: parse_bool(raw.get(d, False)) for d in DECLARATIONS.get(country, [])}


def normalize_fields(country: str, fields: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (display, normalized). Only fields that belong to the country's form are kept."""
    allowed = {f.name for f in fields_for(country)}
    display = {k: display_value(v) for k, v in fields.items() if k in allowed}
    norm = {k: normalize_value(country, k, v) for k, v in display.items()}
    return display, norm


def extract_identifiers(country: str, norm: Mapping[str, str], display: Mapping[str, str]) -> List[Dict[str, str]]:
    """Normalized identifiers stored for duplicate detection (Phase 4)."""
    out: List[Dict[str, str]] = []
    for spec in COUNTRY_FIELDS[country]:
        if not spec.identifier_type:
            continue
        v = norm.get(spec.name, "")
        if not v:
            continue
        if spec.identifier_type == "STATE_FILE_NO":
            v = f"{norm.get('state_of_registration', '')}:{v}"
        out.append({"id_type": spec.identifier_type, "value_norm": v, "value_display": display.get(spec.name, "")})
    acct = norm.get("bank_account_number", "")
    bank_key = norm.get({"IN": "ifsc", "US": "routing_number", "GB": "sort_code"}[country], "")
    if acct:
        out.append({"id_type": "BANK_ACCOUNT", "value_norm": f"{bank_key}:{acct}", "value_display": ""})
    return out


def build_payload(sub: Submission, doc_meta: List[Dict[str, str]]) -> Dict[str, Any]:
    display, norm = normalize_fields(sub.country, sub.fields)
    return {
        "country": sub.country,
        "fields": display,
        "normalized": norm,
        "declarations": canonicalize_declarations(sub.country, sub.declarations),
        "documents": doc_meta,
        "source": sub.source,
        "source_ref": sub.source_ref,
    }


def country_or_raise(value: str) -> str:
    c = normalize_country(value)
    if not c:
        raise ValueError(f"Unsupported or missing country: {value!r} (supported: India, USA, United Kingdom)")
    return c
