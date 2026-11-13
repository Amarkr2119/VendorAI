"""Cross-document consistency: compares what the vendor SUBMITTED with what each DOCUMENT says (and documents with each other
through the submission as the common anchor). Pure functions: rows for consistency_results + Issues for the decision engine.

Principles: a mismatch is a reason for review, not proof of fraud; fields read with LOW confidence are not compared (they are
already flagged by extraction, and comparing a probable OCR error would produce a false accusation).
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from vendorflow.analysis import dates as dt
from vendorflow.analysis.names import address_score, classify_names
from vendorflow.domain.fields import field_label, normalize_value
from vendorflow.domain.models import Issue
from vendorflow.ingestion.completeness import WRONG_TYPE_MIN_CONFIDENCE, doc_label
from vendorflow.utils.masking import mask_account
from vendorflow.validation.rules import other_country_matches

_ID_FIELDS = {"IN": ["pan", "gstin", "cin", "ifsc"], "US": ["ein", "state_of_registration", "state_file_number", "routing_number"],
              "GB": ["company_number", "vat_number", "sort_code"]}
ADDRESS_MATCH, ADDRESS_SIMILAR = 0.85, 0.60


def effective_type(doc: Mapping[str, Any]) -> str:
    det, conf = doc.get("detected_type"), doc.get("detected_confidence") or 0.0
    return det if det and conf >= WRONG_TYPE_MIN_CONFIDENCE else doc["declared_type"]


def _shown(field: str, v: str) -> str:
    return mask_account(v) if field == "bank_account_number" else v


class _Values:
    """(document, field) -> value lookup over CURRENT fields; LOW-confidence rows are kept apart."""

    def __init__(self, docs: Sequence[Mapping[str, Any]], fields: Sequence[Mapping[str, Any]]):
        self.docs = {d["document_id"]: d for d in docs}
        self.rows: List[Dict[str, Any]] = []
        for f in fields:
            d = self.docs.get(f["document_id"])
            if d is not None and f["value"]:
                self.rows.append({"doc_type": effective_type(d), "filename": d["filename"], "field": f["field_name"],
                                  "value": f["value"], "low": f["confidence_band"] == "LOW" and f["origin"] != "HUMAN"})

    def of(self, field: str) -> List[Dict[str, Any]]:
        return [r for r in self.rows if r["field"] == field]


def _row(check_id, a, b, outcome, score=None, **evidence) -> Dict[str, Any]:
    return {"check_id": check_id, "source_a": a, "source_b": b, "outcome": outcome, "score": score, "evidence": evidence}


def check_consistency(country: str, payload: Mapping[str, Any], docs: Sequence[Mapping[str, Any]], fields: Sequence[Mapping[str, Any]],
                      doc_text: Optional[Callable[[Mapping[str, Any]], str]] = None, today: Optional[date] = None
                      ) -> Tuple[List[Dict[str, Any]], List[Issue]]:
    disp, norm = payload["fields"], payload["normalized"]
    vals = _Values(docs, fields)
    rows: List[Dict[str, Any]] = []
    issues: List[Issue] = []
    legal = disp.get("legal_name", "")

    # ---- company name: submission vs every document that states a legal name
    for r in vals.of("legal_name"):
        src = f"{doc_label(country, r['doc_type'])} ({r['filename']})"
        if r["low"]:
            rows.append(_row("NAME.legal_name", "submission", src, "SKIPPED_LOW_CONFIDENCE", submitted=legal, document=r["value"]))
            continue
        level, score = classify_names(legal, r["value"])
        rows.append(_row("NAME.legal_name", "submission", src, level, score, submitted=legal, document=r["value"]))
        if level != "MATCH":
            issues.append(Issue("NAME_MISMATCH_DOCUMENT", r["doc_type"], f"Company name on {doc_label(country, r['doc_type'])} "
                                f"('{r['value']}') does not match the submitted name ('{legal}'): {level.lower()}, score {score}.",
                                "MEDIUM", "REVIEW", f"Confirm the legal name; supply a document that shows '{legal}', or correct the submitted name.",
                                {"submitted": legal, "document": r["value"], "score": score, "level": level}))

    # ---- bank account holder vs the registered company (the classic payment-fraud signal; stated as a mismatch, nothing more)
    holders = [("submitted form", disp.get("bank_holder_name", ""), False)]
    holders += [(f"{doc_label(country, r['doc_type'])} ({r['filename']})", r["value"], r["low"]) for r in vals.of("bank_holder_name")
                if r["doc_type"] != "registration_form"]
    for src, holder, low in holders:
        if not holder:
            continue
        if low:
            rows.append(_row("NAME.bank_holder", src, "company legal name", "SKIPPED_LOW_CONFIDENCE", holder=holder, company=legal))
            continue
        level, score = classify_names(holder, legal)
        rows.append(_row("NAME.bank_holder", src, "company legal name", level, score, holder=holder, company=legal))
        if level != "MATCH":
            issues.append(Issue("NAME_MISMATCH_BANK", "bank_holder_name" if src == "submitted form" else "bank_proof",
                                f"The bank account holder on the {src} ('{holder}') does not match the company ('{legal}'): {level.lower()}, score {score}.",
                                "HIGH" if level == "MISMATCH" else "MEDIUM", "REVIEW",
                                "Explain the relationship between the account holder and the company, or supply an account held in the company's name.",
                                {"holder": holder, "company": legal, "score": score, "level": level}))

    # ---- identifiers and bank details: submission vs documents
    id_fields = _ID_FIELDS[country] + ["bank_account_number"]
    for f in id_fields:
        anchor = norm.get(f, "")
        if not anchor:
            continue
        for r in vals.of(f):
            src = f"{doc_label(country, r['doc_type'])} ({r['filename']})"
            if r["low"]:
                rows.append(_row(f"ID.{f}", "submission", src, "SKIPPED_LOW_CONFIDENCE", field=field_label(f)))
                continue
            same = normalize_value(country, f, r["value"]) == anchor
            rows.append(_row(f"ID.{f}", "submission", src, "MATCH" if same else "MISMATCH",
                             1.0 if same else 0.0, submitted=_shown(f, disp.get(f, anchor)), document=_shown(f, r["value"])))
            if not same:
                issues.append(Issue("IDENTIFIER_MISMATCH", f"{f}:{r['doc_type']}",
                                    f"{field_label(f)} on {doc_label(country, r['doc_type'])} ('{_shown(f, r['value'])}') differs from "
                                    f"the submitted value ('{_shown(f, disp.get(f, anchor))}').", "HIGH", "REVIEW",
                                    f"Confirm the correct {field_label(f)} and provide a consistent document.",
                                    {"field": f, "document_type": r["doc_type"]}))

    # ---- address
    for r in vals.of("address"):
        src = f"{doc_label(country, r['doc_type'])} ({r['filename']})"
        if r["low"] or not disp.get("address"):
            continue
        s = address_score(disp["address"], r["value"])
        level = "MATCH" if s >= ADDRESS_MATCH else "SIMILAR" if s >= ADDRESS_SIMILAR else "MISMATCH"
        rows.append(_row("ADDRESS.registered", "submission", src, level, s, submitted=disp["address"], document=r["value"]))
        if level != "MATCH":
            issues.append(Issue("ADDRESS_MISMATCH", r["doc_type"], f"Address on {doc_label(country, r['doc_type'])} ('{r['value']}') "
                                f"{'resembles' if level == 'SIMILAR' else 'differs from'} the submitted address ('{disp['address']}').",
                                "INFO" if level == "SIMILAR" else "MEDIUM", "REVIEW", "Confirm the registered address.",
                                {"score": s, "level": level}))

    # ---- country evidence
    other = other_country_matches(country, norm)
    if other:
        rows.append(_row("COUNTRY.evidence", "selected country", "identifier format", "MISMATCH", selected=country, suggests=other))
        issues.append(Issue("COUNTRY_EVIDENCE_CONFLICT", "country", f"The tax identifier is invalid for {country} but valid for {other}. "
                            "The country selected may be wrong.", "MEDIUM", "VENDOR_ACTION", "Confirm the country of registration and the tax identifier."))
    else:
        rows.append(_row("COUNTRY.evidence", "selected country", "identifier format", "MATCH", selected=country))

    # ---- dates (text-layer documents only)
    if doc_text:
        _date_checks(country, docs, doc_text, rows, issues, today or date.today())
    return rows, issues


def _date_checks(country, docs, doc_text, rows, issues, today: date) -> None:
    sub: Optional[List[date]] = None
    inc: Optional[List[date]] = None
    per_doc = []
    for d in docs:
        try:
            items = dt.labelled_dates(doc_text(d))
        except Exception:
            continue
        per_doc.append((d, items))
        sub = sub or dt.reference_date(items, "submission")
        inc = inc or dt.reference_date(items, "incorporation")
    for d, items in per_doc:
        for kind, label, cands in items:
            iso = cands[0].isoformat()
            if kind == "EXPIRY":
                bad = dt.expired(cands, today)
                rows.append(_row("DATE.expiry", d["filename"], label, "EXPIRED" if bad else "VALID", date=iso))
                if bad:
                    issues.append(Issue("DOCUMENT_EXPIRED", d["filename"], f"'{d['filename']}' states '{label}: {iso}', which is in the past.",
                                        "MEDIUM", "VENDOR_ACTION", "Provide a current, unexpired document.", {"label": label, "date": iso}))
            elif dt.in_future(cands, today):
                rows.append(_row("DATE.future", d["filename"], label, "FUTURE", date=iso))
                issues.append(Issue("DATE_INCONSISTENT", d["filename"], f"'{d['filename']}' has '{label}: {iso}', a date in the future.",
                                    "MEDIUM", "REVIEW", "Confirm the date shown on the document.", {"label": label, "date": iso}))
    if sub and inc:
        bad = dt.after(inc, sub)
        rows.append(_row("DATE.incorporation_vs_submission", "incorporation date", "submission date", "INCONSISTENT" if bad else "CONSISTENT",
                         incorporation=inc[0].isoformat(), submission=sub[0].isoformat()))
        if bad:
            issues.append(Issue("DATE_INCONSISTENT", "incorporation_vs_submission",
                                "The incorporation date is later than the submission date.", "MEDIUM", "REVIEW", "Confirm the dates."))
