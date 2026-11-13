"""Risk assessment: explainable indicators with severity and evidence. Pure and deterministic.

An indicator is a reason to look more closely. It is NOT an accusation: no indicator says or implies fraud, and risk
indicators never decide the outcome by themselves (the policy in decision/policy.py does, from explicit issues).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from vendorflow.domain.models import Issue
from vendorflow.validation.identifiers import ein_prefix_known

SEVERITY_POINTS = {"HIGH": 40, "MEDIUM": 15, "LOW": 5, "INFO": 0}
_FROM_ISSUE = {
    "NAME_MISMATCH_BANK": ("BANK_HOLDER_DIFFERS_FROM_COMPANY", "The bank account is held under a different name than the company. Third-party accounts are a common payment-diversion pattern but have legitimate explanations."),
    "NAME_MISMATCH_DOCUMENT": ("COMPANY_NAME_INCONSISTENT", "The company name is not consistent across the submission and documents."),
    "IDENTIFIER_MISMATCH": ("IDENTIFIER_INCONSISTENT", "An identifier on a document differs from the submitted one."),
    "ADDRESS_MISMATCH": ("ADDRESS_INCONSISTENT", "The registered address differs across the submission and documents."),
    "DUPLICATE_SUSPECTED": ("POSSIBLE_EXISTING_VENDOR", "Details overlap with an earlier application. This may be a resubmission, a related entity or a coincidence."),
    "BLOCKLIST_MATCH": ("INTERNAL_RECORD_MATCH", "An identifier matches an internal record. Reviewer confirmation is required before any consequence."),
    "LOW_CONFIDENCE_EXTRACTION": ("UNRELIABLE_DOCUMENT_READ", "A value could not be read reliably from a document."),
    "INVALID_FORMAT": ("IDENTIFIER_FORMAT_INVALID", "An identifier does not follow the format for the selected country (usually a typing error)."),
    "VERIFICATION_UNAVAILABLE": ("VERIFICATION_INCOMPLETE", "An external verification could not be completed. This says nothing about the vendor."),
    "VERIFICATION_NOT_PERFORMED": ("VERIFICATION_INCOMPLETE", "A required verification was not performed."),
    "VERIFICATION_NOT_LIVE": ("VERIFICATION_NOT_LIVE", "Verification relied on simulated or sandbox data, not a live authoritative source."),
    "VERIFICATION_SIMULATED_NEGATIVE": ("SIMULATED_NEGATIVE", "A SIMULATED provider returned a negative result. It is not evidence about the vendor."),
    "PROVIDER_NOT_FOUND": ("IDENTIFIER_NOT_FOUND", "A provider could not find the identifier."),
    "LIVE_TERMINAL_STATUS": ("TERMINAL_STATUS_REPORTED", "A live authoritative source reports a terminal status. A reviewer must confirm before rejection."),
    "DOCUMENT_EXPIRED": ("DOCUMENT_EXPIRED", "A document is past its stated expiry date."),
    "DATE_INCONSISTENT": ("DATE_INCONSISTENT", "A document date is inconsistent."),
    "COUNTRY_EVIDENCE_CONFLICT": ("COUNTRY_EVIDENCE_CONFLICT", "Identifiers suggest a different country than the one selected."),
    "DECLARATION_NEEDS_REVIEW": ("TAX_REGISTRATION_DECLARED_ABSENT", "The vendor declares it is not tax-registered; a reviewer must accept this."),
}


def assess(issues: Sequence[Issue], verifications: Sequence[Mapping[str, Any]], duplicate_matches: Sequence[Mapping[str, Any]],
           country: str, norm: Mapping[str, str]) -> Tuple[List[Dict[str, Any]], str, int]:
    out: List[Dict[str, Any]] = []
    for i in issues:
        if i.code in _FROM_ISSUE and i.severity != "INFO":
            code, why = _FROM_ISSUE.get(i.code, (i.code, i.message))
            out.append({"code": code, "severity": i.severity, "explanation": f"{why} {i.message}", "evidence": {"issue": i.key, **i.evidence}})
    for m in duplicate_matches:
        if m["strength"] == "WEAK":
            out.append({"code": "WEAK_NAME_OVERLAP", "severity": "INFO", "evidence": m["basis"],
                        "explanation": f"Some details resemble {m['matched_application_id']} but not enough to suggest the same vendor."})
    sims = [v for v in verifications if v["mode"] != "LIVE" and v["outcome"] == "VERIFIED"]
    if sims:
        out.append({"code": "VERIFIED_WITHOUT_LIVE_SOURCE", "severity": "LOW", "evidence": {"checks": [v["check_type"] for v in sims]},
                    "explanation": "These checks passed using simulated or sandbox data and are not authoritative: " + ", ".join(v["check_type"] + f" [{v['mode']}]" for v in sims)})
    if country == "US" and norm.get("ein") and len(norm["ein"]) == 9 and not ein_prefix_known(norm["ein"]):
        out.append({"code": "EIN_PREFIX_UNASSIGNED", "severity": "LOW", "evidence": {"prefix": norm["ein"][:2]},
                    "explanation": "The EIN prefix is not in the commonly published list of assigned prefixes (list may be out of date)."})
    score = min(100, sum(SEVERITY_POINTS[o["severity"]] for o in out))
    level = ("HIGH" if any(o["severity"] == "HIGH" for o in out) else "MEDIUM" if any(o["severity"] == "MEDIUM" for o in out)
             else "LOW" if any(o["severity"] == "LOW" for o in out) else "NONE")
    return out, level, score
