"""Reason codes: machine-readable, stable identifiers with a plain-language title.

Defined once for the whole project. Phase 3 emits the document/completeness/extraction codes;
Phase 4 emits the rest. Codes group into the three decision outcomes via `OUTCOME_CLASS`.
"""
from __future__ import annotations

from typing import Dict, NamedTuple


class ReasonCode(NamedTuple):
    code: str
    title: str
    outcome: str          # which status this code pushes toward: PENDING | REJECTED


REASON_CODES: Dict[str, ReasonCode] = {r.code: r for r in [
    # --- Phase 3: intake and extraction
    ReasonCode("MISSING_FIELD", "Required information is missing", "PENDING"),
    ReasonCode("MISSING_DOCUMENT", "A required document is missing", "PENDING"),
    ReasonCode("WRONG_DOCUMENT_TYPE", "An uploaded document is not the type it was submitted as", "PENDING"),
    ReasonCode("UNREADABLE_DOCUMENT", "A document could not be read", "PENDING"),
    ReasonCode("DECLARATION_NEEDS_REVIEW", "A non-registration declaration needs reviewer acceptance", "PENDING"),
    ReasonCode("LOW_CONFIDENCE_EXTRACTION", "Information read from a document is not reliable enough", "PENDING"),
    # --- Phase 4: validation, verification, consistency, duplicates
    ReasonCode("INVALID_FORMAT", "An identifier has an invalid format for the selected country", "PENDING"),
    ReasonCode("COUNTRY_EVIDENCE_CONFLICT", "Details suggest a different country than selected", "PENDING"),
    ReasonCode("VERIFICATION_UNAVAILABLE", "An external verification service was unavailable", "PENDING"),
    ReasonCode("VERIFICATION_NOT_PERFORMED", "A required external verification was not performed", "PENDING"),
    ReasonCode("VERIFICATION_NOT_LIVE", "Only simulated or sandbox verification is available (PRODUCTION policy)", "PENDING"),
    ReasonCode("VERIFICATION_SIMULATED_NEGATIVE", "A simulated provider reported a negative result; it cannot be treated as evidence", "PENDING"),
    ReasonCode("PROVIDER_NOT_FOUND", "A provider could not find the identifier", "PENDING"),
    ReasonCode("NAME_MISMATCH_BANK", "Bank account holder does not match the registered company", "PENDING"),
    ReasonCode("NAME_MISMATCH_DOCUMENT", "Company name differs between documents", "PENDING"),
    ReasonCode("BANK_OWNERSHIP_UNVERIFIED", "Bank account ownership has not been verified", "PENDING"),
    ReasonCode("DUPLICATE_SUSPECTED", "This vendor may already exist in another application", "PENDING"),
    ReasonCode("BLOCKLIST_MATCH", "Matches a synthetic internal blocklist entry; reviewer confirmation required", "PENDING"),
    ReasonCode("ADDRESS_MISMATCH", "Registered address differs between submission and documents", "PENDING"),
    ReasonCode("IDENTIFIER_MISMATCH", "An identifier differs between the submission and a document", "PENDING"),
    ReasonCode("DOCUMENT_EXPIRED", "A document is past its stated expiry date", "PENDING"),
    ReasonCode("DATE_INCONSISTENT", "A document date is inconsistent (for example in the future)", "PENDING"),
    # --- Rejection: only through documented, evidenced rules
    ReasonCode("REVIEWER_CONFIRMED_DISQUALIFICATION", "A reviewer confirmed a documented disqualifying condition (R1)", "REJECTED"),
    ReasonCode("LIVE_TERMINAL_STATUS", "A LIVE authoritative provider reported a terminal status (R2)", "REJECTED"),
]}


def title(code: str) -> str:
    rc = REASON_CODES.get(code)
    return rc.title if rc else code
