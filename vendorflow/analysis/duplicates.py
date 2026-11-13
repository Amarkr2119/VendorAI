"""Duplicate detection against EARLIER applications, using several fields and an explainable additive score.

Evidence weights (a match needs enough independent evidence, never name similarity alone):
    exact tax / registration identifier   0.60  (+0.10 per additional distinct identifier, capped at 0.80)
    exact bank account (bank code + no.)  0.50
    company name MATCH 0.40 | SIMILAR 0.25
    same contact email                    0.25
    address overlap >= 0.85               0.15
Score >= 0.50  -> suspected duplicate: the application goes to MANUAL REVIEW (STRONG when >= 0.90, else POSSIBLE)
Score 0.30-0.49 -> WEAK: recorded as an informational risk indicator only
Below 0.30     -> not recorded. 'Similar names' alone (max 0.40) can never reach the review threshold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from vendorflow.analysis.names import address_score, classify_names
from vendorflow.config import ROOT
from vendorflow.db.repositories import Repository
from vendorflow.domain.models import Issue

REVIEW_MIN, STRONG_MIN, WEAK_MIN = 0.50, 0.90, 0.30
_STRONG_ID_TYPES = {"PAN", "GSTIN", "CIN", "EIN", "STATE_FILE_NO", "CRN", "VAT"}
BLOCKLIST_PATH = ROOT / "data" / "synthetic" / "internal_blocklist.json"


def score_pair(mine: Mapping[str, Any], other: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    basis: Dict[str, Any] = {}
    score = 0.0
    shared = sorted({i["id_type"] + "=" + i["value_norm"] for i in mine["ids"]} & {i["id_type"] + "=" + i["value_norm"] for i in other["ids"]})
    tax = [s.split("=")[0] for s in shared if s.split("=")[0] in _STRONG_ID_TYPES]
    bank = [s for s in shared if s.startswith("BANK_ACCOUNT=")]
    if tax:
        score += min(0.80, 0.60 + 0.10 * (len(tax) - 1))
        basis["exact_identifiers"] = tax
    if bank:
        score += 0.50
        basis["same_bank_account"] = True
    level, ns = classify_names(mine["fields"].get("legal_name", ""), other["fields"].get("legal_name", ""))
    if level in ("MATCH", "SIMILAR"):
        score += 0.40 if level == "MATCH" else 0.25
        basis["name"] = {"level": level, "score": ns}
    em, eo = mine["fields"].get("contact_email", "").lower(), other["fields"].get("contact_email", "").lower()
    if em and em == eo:
        score += 0.25
        basis["same_contact_email"] = True
    a = address_score(mine["fields"].get("address", ""), other["fields"].get("address", ""))
    if a >= 0.85:
        score += 0.15
        basis["address_overlap"] = a
    return round(min(score, 1.0), 3), basis


def detect_duplicates(repo: Repository, application_id: str) -> Tuple[List[Dict[str, Any]], List[Issue]]:
    app = repo.get_application(application_id)
    mine = {"ids": repo.current_identifiers(application_id), "fields": repo.get_revision(application_id)["payload"]["fields"]}
    matches: List[Dict[str, Any]] = []
    issues: List[Issue] = []
    for other_app in repo.applications_before(application_id):
        oid = other_app["application_id"]
        other = {"ids": repo.current_identifiers(oid), "fields": repo.get_revision(oid)["payload"]["fields"]}
        score, basis = score_pair(mine, other)
        if score < WEAK_MIN:
            continue
        strength = "STRONG" if score >= STRONG_MIN else "POSSIBLE" if score >= REVIEW_MIN else "WEAK"
        basis = {**basis, "score": score, "other_name": other["fields"].get("legal_name", ""), "other_source_ref": other_app.get("source_ref")}
        matches.append({"matched_application_id": oid, "basis": basis, "strength": strength})
        if strength != "WEAK":
            what = ", ".join(basis.get("exact_identifiers", []) + (["bank account"] if basis.get("same_bank_account") else [])
                             + (["company name"] if "name" in basis else []) + (["contact email"] if basis.get("same_contact_email") else []))
            issues.append(Issue("DUPLICATE_SUSPECTED", oid, f"This application may duplicate {oid} ('{basis['other_name']}'): shared {what or 'details'} "
                                f"(match score {score}). A reviewer must decide whether it is the same vendor.", "MEDIUM", "REVIEW",
                                f"Reviewer confirms whether this is the same vendor as {oid}.", {"matched": oid, **basis}))
    return matches, issues


def check_blocklist(identifiers: List[Mapping[str, str]], path: Path = BLOCKLIST_PATH) -> List[Issue]:
    """Return internal-record matches, including the evidence needed by the decision audit trail."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    mine = {(i["id_type"], i["value_norm"]) for i in identifiers}
    out = []
    for e in data.get("entries", []):
        if (e["id_type"], e["value_norm"]) in mine:
            cond = data.get("conditions", {}).get(e["condition_code"], e["condition_code"])
            confirmed = bool(e.get("confirmed", False))
            out.append(Issue("BLOCKLIST_MATCH", e["condition_code"], f"An identifier matches internal record {e['record_ref']} "
                             f"(condition {e['condition_code']}: {cond}).", "HIGH", "REVIEW",
                             "Contact procurement if you believe this match is incorrect.",
                             {"condition_code": e["condition_code"], "record_ref": e["record_ref"], "id_type": e["id_type"],
                              "matched_identifier": e["value_norm"], "evidence_source": e.get("source", "internal record"),
                              "verification_timestamp": e.get("verified_at") or e.get("added", ""), "confirmed": confirmed}))
    return out
