"""Canonical field model: what a submission contains, what each document should contain,
how fields are labelled in documents, and how raw values are normalized.

One place defines this so ingestion, completeness, extraction and (later) validation agree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vendorflow.utils.text_normalize import alnum_lower, collapse_ws, nfkc, normalize_company_name


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str = "text"                      # text | email | id | bank | select
    identifier_type: Optional[str] = None   # key used in application_identifiers (duplicate detection)


COMMON_FIELDS: List[FieldSpec] = [
    FieldSpec("legal_name", "Legal name of the company"),
    FieldSpec("trade_name", "Trade / business name (if different)"),
    FieldSpec("entity_type", "Entity type"),
    FieldSpec("address", "Registered address"),
    FieldSpec("contact_email", "Contact email", "email"),
    FieldSpec("bank_name", "Bank name", "bank"),
    FieldSpec("bank_holder_name", "Bank account holder name", "bank"),
    FieldSpec("bank_account_number", "Bank account number", "bank"),
]

COUNTRY_FIELDS: Dict[str, List[FieldSpec]] = {
    "IN": [
        FieldSpec("pan", "PAN", "id", "PAN"),
        FieldSpec("gstin", "GSTIN", "id", "GSTIN"),
        FieldSpec("cin", "CIN (Corporate Identity Number)", "id", "CIN"),
        FieldSpec("ifsc", "IFSC code", "bank"),
    ],
    "US": [
        FieldSpec("ein", "EIN", "id", "EIN"),
        FieldSpec("state_of_registration", "State of registration"),
        FieldSpec("state_file_number", "State file number", "id", "STATE_FILE_NO"),
        FieldSpec("routing_number", "ABA routing number", "bank"),
    ],
    "GB": [
        FieldSpec("company_number", "Companies House number", "id", "CRN"),
        FieldSpec("vat_number", "VAT number", "id", "VAT"),
        FieldSpec("sort_code", "Sort code", "bank"),
    ],
}

# Declarations a vendor may make instead of supplying a tax number (reviewed by a human in Phase 4).
DECLARATIONS: Dict[str, List[str]] = {"IN": ["gst_not_registered"], "US": [], "GB": ["vat_not_registered"]}

COUNTRY_NAMES = {"IN": "India", "US": "USA", "GB": "United Kingdom"}
_COUNTRY_ALIASES = {
    "in": "IN", "india": "IN", "ind": "IN", "bharat": "IN",
    "us": "US", "usa": "US", "unitedstates": "US", "unitedstatesofamerica": "US", "america": "US",
    "gb": "GB", "uk": "GB", "unitedkingdom": "GB", "greatbritain": "GB", "england": "GB", "britain": "GB",
}


def normalize_country(value: str) -> Optional[str]:
    return _COUNTRY_ALIASES.get(re.sub(r"[^a-z]", "", (value or "").lower()))


def fields_for(country: str) -> List[FieldSpec]:
    """Ordered fields shown/collected for a country: identity, country-specific, contact, bank."""
    return COMMON_FIELDS[:4] + COUNTRY_FIELDS[country] + COMMON_FIELDS[4:]


def all_field_names(country: str) -> List[str]:
    return [f.name for f in fields_for(country)]


def field_label(name: str) -> str:
    for specs in [COMMON_FIELDS, *COUNTRY_FIELDS.values()]:
        for s in specs:
            if s.name == name:
                return s.label
    return name


# ------------------------------------------------------------------ key aliases for JSON / form input
_ALIASES = {
    "legalname": "legal_name", "companyname": "legal_name", "name": "legal_name", "vendorname": "legal_name",
    "tradename": "trade_name", "businessname": "trade_name", "dba": "trade_name",
    "entitytype": "entity_type", "companytype": "entity_type", "legalform": "entity_type",
    "address": "address", "registeredaddress": "address", "registeredoffice": "address",
    "contactemail": "contact_email", "email": "contact_email",
    "bankname": "bank_name", "bankholdername": "bank_holder_name", "accountholdername": "bank_holder_name",
    "accountholder": "bank_holder_name", "accountname": "bank_holder_name",
    "bankaccountnumber": "bank_account_number", "accountnumber": "bank_account_number", "accountno": "bank_account_number",
    "pan": "pan", "pannumber": "pan", "gstin": "gstin", "gstnumber": "gstin", "gstno": "gstin",
    "cin": "cin", "ifsc": "ifsc", "ifsccode": "ifsc",
    "ein": "ein", "stateofregistration": "state_of_registration", "state": "state_of_registration",
    "statefilenumber": "state_file_number", "routingnumber": "routing_number", "abaroutingnumber": "routing_number",
    "companynumber": "company_number", "crn": "company_number", "vatnumber": "vat_number", "vat": "vat_number",
    "sortcode": "sort_code",
}


def canonical_field_name(key: str) -> Optional[str]:
    return _ALIASES.get(re.sub(r"[^a-z0-9]", "", (key or "").lower()))


# ------------------------------------------------------------------ labels used inside documents
FIELD_LABELS: Dict[str, List[str]] = {
    "legal_name": ["Legal Name", "Legal Name of the Company", "Legal Name of Business", "Name of Company", "Company Name",
                   "Entity Name", "Name (as shown on your income tax return)", "Name as shown on your income tax return",
                   "Name"],
    "trade_name": ["Trade Name", "Business Name", "Business name/disregarded entity name", "Doing Business As", "DBA"],
    "entity_type": ["Entity Type", "Constitution of Business", "Type of Entity", "Federal tax classification", "Legal Form"],
    "address": ["Registered Office", "Registered Office Address", "Registered Address", "Address", "Principal Address",
                "Principal Place of Business", "Address of Principal Place of Business"],
    "contact_email": ["Contact Email", "Email", "E-mail"],
    "bank_name": ["Bank Name", "Name of Bank"],
    "bank_holder_name": ["Account Holder Name", "Account Holder", "Account Name", "Beneficiary Name", "Name of Account Holder"],
    "bank_account_number": ["Account Number", "Bank Account Number", "A/C No", "Account No"],
    "pan": ["PAN", "Permanent Account Number", "PAN Number"],
    "gstin": ["GSTIN", "GST Registration Number", "GST Identification Number", "Registration Number"],
    "cin": ["CIN", "Corporate Identity Number", "Corporate Identification Number"],
    "ifsc": ["IFSC", "IFSC Code"],
    "ein": ["EIN", "Employer Identification Number", "Employer identification number (EIN)"],
    "state_of_registration": ["State of Formation", "State of Registration", "State of Organization", "Jurisdiction"],
    "state_file_number": ["State File Number", "File Number", "Entity File Number", "Filing Number"],
    "routing_number": ["Routing Number", "ABA Routing Number", "ABA Number", "ABA/Routing Number"],
    "company_number": ["Company Number", "Registered Number", "Company Registration Number"],
    "vat_number": ["VAT Number", "VAT Registration Number", "VAT Registration No"],
    "sort_code": ["Sort Code"],
}

# What each document type is expected to contain: (field, required_on_document)
_F = Tuple[str, bool]
DOC_FIELDS: Dict[Tuple[str, str], List[_F]] = {
    ("IN", "registration_form"): [("legal_name", True), ("trade_name", False), ("entity_type", True), ("address", True),
                                  ("contact_email", True), ("pan", True), ("gstin", False), ("cin", True),
                                  ("bank_name", True), ("bank_holder_name", True), ("bank_account_number", True), ("ifsc", True)],
    ("IN", "registration_certificate"): [("legal_name", True), ("cin", True), ("address", False)],
    ("IN", "tax_certificate"): [("legal_name", True), ("trade_name", False), ("gstin", True), ("address", False)],
    ("IN", "tax_id_proof"): [("legal_name", True), ("pan", True)],
    ("IN", "bank_proof"): [("bank_holder_name", True), ("bank_name", True), ("bank_account_number", True), ("ifsc", True)],
    ("US", "registration_form"): [("legal_name", True), ("trade_name", False), ("entity_type", True), ("address", True),
                                  ("contact_email", True), ("ein", True), ("state_of_registration", True),
                                  ("state_file_number", True), ("bank_name", True), ("bank_holder_name", True),
                                  ("bank_account_number", True), ("routing_number", True)],
    ("US", "registration_certificate"): [("legal_name", True), ("state_of_registration", True), ("state_file_number", True),
                                         ("address", False)],
    ("US", "tax_certificate"): [("legal_name", True), ("trade_name", False), ("entity_type", False), ("address", False),
                                ("ein", True)],
    ("US", "bank_proof"): [("bank_holder_name", True), ("bank_name", True), ("bank_account_number", True), ("routing_number", True)],
    ("GB", "registration_form"): [("legal_name", True), ("trade_name", False), ("entity_type", True), ("address", True),
                                  ("contact_email", True), ("company_number", True), ("vat_number", False),
                                  ("bank_name", True), ("bank_holder_name", True), ("bank_account_number", True), ("sort_code", True)],
    ("GB", "registration_certificate"): [("legal_name", True), ("company_number", True), ("address", False)],
    ("GB", "tax_certificate"): [("legal_name", True), ("vat_number", True), ("address", False)],
    ("GB", "bank_proof"): [("bank_holder_name", True), ("bank_name", True), ("bank_account_number", True), ("sort_code", True)],
}


def expected_fields(country: str, doc_type: str) -> List[_F]:
    return DOC_FIELDS.get((country, doc_type), [])


# ------------------------------------------------------------------ value normalization
_ID_FIELDS_UPPER = {"pan", "gstin", "cin", "ifsc", "company_number"}


def normalize_value(country: str, field: str, value: Optional[str]) -> str:
    """Canonical comparison form of a raw value. Display values are kept separately."""
    if value is None:
        return ""
    v = collapse_ws(nfkc(str(value)))
    if not v or v.upper() in {"N/A", "NA", "NIL", "NONE", "-", "--", "NULL"}:
        return ""
    if field in _ID_FIELDS_UPPER:
        return re.sub(r"[\s\-_.]", "", v).upper()
    if field == "vat_number":
        s = re.sub(r"[\s\-_.]", "", v).upper()
        return s[2:] if s.startswith("GB") else s
    if field in {"ein", "routing_number", "sort_code", "bank_account_number"}:
        return re.sub(r"\D", "", v)
    if field == "contact_email":
        return v.lower()
    if field in {"legal_name", "trade_name", "bank_holder_name"}:
        return normalize_company_name(v)
    if field == "state_file_number":
        return re.sub(r"[\s\-_.]", "", v).upper()
    if field == "state_of_registration":
        return alnum_lower(v)
    return v.lower()


def display_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    v = collapse_ws(nfkc(str(value)))
    return "" if v.upper() in {"N/A", "NA", "NIL", "NONE", "-", "--", "NULL"} else v


def is_blank(value: Optional[str]) -> bool:
    return display_value(value) == ""
