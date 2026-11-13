"""Completeness checks driven by YAML rules: required fields and required documents.

Findings are persisted to validation_results. This stage NEVER rejects; it only reports what is missing.
Document completeness is evaluated twice: on declared types at intake, and again on detected types after
extraction (so a bank letter uploaded in the W-9 slot is caught).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from vendorflow.domain.fields import field_label, is_blank
from vendorflow.domain.models import Finding

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
_FILES = {"IN": "completeness_in.yaml", "US": "completeness_us.yaml", "GB": "completeness_uk.yaml"}
DOC_LABELS = {
    "registration_form": "Vendor registration form", "registration_certificate": "Company registration certificate",
    "tax_certificate": "Tax registration certificate", "tax_id_proof": "Tax ID proof (PAN)", "bank_proof": "Bank account proof",
}
COUNTRY_DOC_LABELS = {
    ("IN", "tax_certificate"): "GST registration certificate", ("US", "tax_certificate"): "Form W-9",
    ("GB", "tax_certificate"): "VAT registration certificate", ("IN", "registration_certificate"): "Certificate of incorporation",
    ("US", "registration_certificate"): "Articles of organization / certificate of formation",
    ("GB", "registration_certificate"): "Certificate of incorporation", ("IN", "tax_id_proof"): "PAN card / PAN proof",
}
WRONG_TYPE_MIN_CONFIDENCE = 0.75


@lru_cache(maxsize=None)
def load_rules(country: str) -> Dict[str, Any]:
    with open(RULES_DIR / _FILES[country], encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def doc_label(country: str, doc_type: str) -> str:
    return COUNTRY_DOC_LABELS.get((country, doc_type)) or DOC_LABELS.get(doc_type, doc_type)


def check_fields(country: str, fields: Mapping[str, str], declarations: Mapping[str, bool]) -> List[Finding]:
    rules = load_rules(country)
    findings: List[Finding] = []
    for name in rules["required_fields"]:
        missing = is_blank(fields.get(name))
        findings.append(Finding(
            rule_id=f"COMPLETENESS.FIELD.{name}", category="COMPLETENESS", field=name,
            result="FAIL" if missing else "PASS", reason_code="MISSING_FIELD" if missing else None,
            expected="a non-empty value", message=(f"{field_label(name)} is required but was not provided." if missing
                                                    else f"{field_label(name)} provided.")))
    for cond in rules.get("conditional_fields", []):
        name, waiver = cond["field"], cond["unless_declared"]
        if not is_blank(fields.get(name)):
            findings.append(Finding(f"COMPLETENESS.FIELD.{name}", "COMPLETENESS", "PASS", field=name,
                                    message=f"{field_label(name)} provided."))
        elif declarations.get(waiver):
            findings.append(Finding(f"COMPLETENESS.FIELD.{name}", "COMPLETENESS", "SKIPPED", field=name,
                                    reason_code="DECLARATION_NEEDS_REVIEW",
                                    message=f"{field_label(name)} not provided; the vendor declared non-registration "
                                            f"({waiver}). A reviewer must accept the declaration."))
        else:
            findings.append(Finding(f"COMPLETENESS.FIELD.{name}", "COMPLETENESS", "FAIL", field=name, reason_code="MISSING_FIELD",
                                    expected=f"a value, or the declaration '{waiver}'",
                                    message=f"{field_label(name)} is required. Provide it, or declare that the business is not registered."))
    return findings


def _effective_type(doc: Mapping[str, Any], use_detected: bool) -> str:
    det, conf = doc.get("detected_type"), doc.get("detected_confidence") or 0.0
    if use_detected and det and conf >= WRONG_TYPE_MIN_CONFIDENCE:
        return det
    return doc["declared_type"]


def check_documents(country: str, declarations: Mapping[str, bool], docs: Sequence[Mapping[str, Any]],
                    use_detected: bool = False) -> List[Finding]:
    rules = load_rules(country)
    findings: List[Finding] = []
    present = {_effective_type(d, use_detected) for d in docs}
    for req in rules["required_documents"]:
        dtype, waiver = req["type"], req.get("unless_declared")
        label = doc_label(country, dtype)
        if dtype in present:
            findings.append(Finding(f"COMPLETENESS.DOC.{dtype}", "COMPLETENESS", "PASS", field=dtype, message=f"{label} provided."))
        elif waiver and declarations.get(waiver):
            findings.append(Finding(f"COMPLETENESS.DOC.{dtype}", "COMPLETENESS", "SKIPPED", field=dtype,
                                    reason_code="DECLARATION_NEEDS_REVIEW",
                                    message=f"{label} not provided; waived by the vendor's non-registration declaration (needs reviewer acceptance)."))
        else:
            findings.append(Finding(f"COMPLETENESS.DOC.{dtype}", "COMPLETENESS", "FAIL", field=dtype, reason_code="MISSING_DOCUMENT",
                                    expected=label, message=f"{label} is required but was not provided."
                                    + (" A document of a different type was uploaded in its place." if use_detected and any(
                                        d["declared_type"] == dtype for d in docs) else "")))
    if use_detected:
        for d in docs:
            det, conf = d.get("detected_type"), d.get("detected_confidence") or 0.0
            if det and conf >= WRONG_TYPE_MIN_CONFIDENCE and det != d["declared_type"]:
                findings.append(Finding(f"DOCUMENT.WRONG_TYPE.{d['filename']}", "DOCUMENT", "FAIL", field=d["declared_type"],
                                        reason_code="WRONG_DOCUMENT_TYPE", expected=doc_label(country, d["declared_type"]),
                                        observed=doc_label(country, det),
                                        message=f"'{d['filename']}' was uploaded as {doc_label(country, d['declared_type'])} "
                                                f"but its content looks like {doc_label(country, det)}."))
    return findings


def summarize(findings: Sequence[Finding]) -> Dict[str, List[str]]:
    return {
        "missing_fields": [f.field for f in findings if f.category == "COMPLETENESS" and f.rule_id.startswith("COMPLETENESS.FIELD.") and f.result == "FAIL"],
        "missing_documents": [f.field for f in findings if f.rule_id.startswith("COMPLETENESS.DOC.") and f.result == "FAIL"],
        "wrong_type_documents": [f.field for f in findings if f.reason_code == "WRONG_DOCUMENT_TYPE"],
    }
