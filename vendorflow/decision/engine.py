"""Deterministic decision engine: Issues + verification results + policy  ->  decision with reason codes, actions and a rule trace.

Pure functions (no I/O). The same inputs always yield the same decision; the trace says exactly which rule fired and why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from vendorflow.domain.fields import field_label
from vendorflow.domain.models import Finding, Issue
from vendorflow.domain.reason_codes import title
from vendorflow.ingestion.completeness import doc_label
from vendorflow.utils.masking import mask_account
from vendorflow.decision.policy import ACCEPTED_MODES, POLICY_VERSION

BLOCKING = {"LOW", "MEDIUM", "HIGH"}          # INFO never blocks


# ------------------------------------------------------------------ findings / fields -> issues
def issues_from_findings(country: str, findings: Sequence[Mapping[str, Any]]) -> List[Issue]:
    """Completeness, document and format findings (validation_results rows or Finding objects) -> Issues. EXTRACTION findings are
    skipped on purpose: low confidence is derived from the CURRENT fields so human corrections take effect."""
    out: List[Issue] = []
    for f in findings:
        g = f if isinstance(f, dict) else f.__dict__
        code, res, cat = g.get("reason_code"), g.get("result"), g.get("category")
        if cat == "EXTRACTION" or not code or res == "PASS":
            continue
        rid, fld, msg = g.get("rule_id", ""), g.get("field") or "", g.get("message") or ""
        if res == "SKIPPED" and code == "DECLARATION_NEEDS_REVIEW":
            out.append(Issue(code, fld, msg, "MEDIUM", "REVIEW", "Reviewer accepts or rejects the vendor's declaration of non-registration."))
        elif res == "FAIL" and code == "MISSING_FIELD":
            out.append(Issue(code, fld, msg, "MEDIUM", "VENDOR_ACTION", f"Provide: {field_label(fld)}."))
        elif res == "FAIL" and code == "MISSING_DOCUMENT":
            out.append(Issue(code, fld, msg, "MEDIUM", "VENDOR_ACTION", f"Upload: {doc_label(country, fld)}."))
        elif res == "FAIL" and code == "WRONG_DOCUMENT_TYPE":
            out.append(Issue(code, rid.split(".", 2)[-1], msg, "MEDIUM", "VENDOR_ACTION", f"Upload the correct document: {g.get('expected') or fld}."))
        elif res == "FAIL" and code == "UNREADABLE_DOCUMENT":
            out.append(Issue(code, rid.split(".", 2)[-1], msg, "MEDIUM", "VENDOR_ACTION", "Upload a clear, readable copy of the document."))
        elif res == "FAIL" and code == "INVALID_FORMAT":
            out.append(Issue(code, rid.replace("FORMAT.", ""), msg, "MEDIUM", "VENDOR_ACTION",
                             f"Correct {field_label(fld)}: expected {g.get('expected')}.", {"field": fld, "observed": g.get("observed")}))
    return out


def extraction_issues(docs: Sequence[Mapping[str, Any]], fields: Sequence[Mapping[str, Any]], effective_type) -> List[Issue]:
    by_doc = {d["document_id"]: d for d in docs}
    out: List[Issue] = []
    for f in fields:
        d = by_doc.get(f["document_id"])
        if d is None or f["confidence_band"] != "LOW" or f["origin"] == "HUMAN":
            continue
        dt = effective_type(d)
        flags = json.loads(f.get("flags_json") or "[]")
        shown = mask_account(f["value"]) if f["field_name"] == "bank_account_number" else f["value"]
        if "NOT_FOUND" in flags or not f["value"]:
            msg = f"{field_label(f['field_name'])} was not found in '{d['filename']}'."
        else:
            msg = (f"{field_label(f['field_name'])} read from '{d['filename']}' as '{shown}' is not reliable "
                   f"(confidence {f['confidence']:.2f}: {', '.join(x.replace('_', ' ').lower() for x in flags) or 'low'}).")
        out.append(Issue("LOW_CONFIDENCE_EXTRACTION", f"{dt}:{f['field_name']}", msg, "MEDIUM", "REVIEW",
                         "Reviewer compares the value with the document image, then confirms or corrects it.",
                         {"document": d["filename"], "field": f["field_name"], "confidence": f["confidence"], "flags": flags,
                          "value": shown, "document_id": f["document_id"]}))
    return out


# ------------------------------------------------------------------ verification results -> issues
def verification_issues(results: Sequence[Mapping[str, Any]], mode: str, data_issue_open: bool = False) -> List[Issue]:
    """`data_issue_open`: a MISSING_FIELD/INVALID_FORMAT/MISSING_DOCUMENT issue already explains inputs that could not be verified."""
    out: List[Issue] = []
    accepted = ACCEPTED_MODES[mode]
    for v in results:
        if not v.get("mandatory", True) or v["reason"] == "WAIVED_BY_DECLARATION":
            continue
        ct, oc, md, why, summ = v["check_type"], v["outcome"], v["mode"], v.get("reason", ""), v.get("summary", "")
        ev = {"check": ct, "provider": v["provider"], "mode": md, "outcome": oc, "reason": why}
        if oc == "VERIFIED":
            if md not in accepted:
                out.append(Issue("VERIFICATION_NOT_LIVE", ct, f"{ct} passed only in {md} mode, which PRODUCTION policy does not accept.", "MEDIUM", "SYSTEM",
                                 "Connect an authorised live provider for this check, then reprocess.", ev))
        elif oc == "SKIPPED":
            if why in ("INVALID_FORMAT", "MISSING_INPUT") and data_issue_open:
                continue                                       # already raised as INVALID_FORMAT / MISSING_FIELD
            out.append(Issue("VERIFICATION_NOT_PERFORMED", ct, f"{ct} was not performed ({why or 'skipped'}): {summ}", "MEDIUM", "SYSTEM",
                             "Configure the provider or complete this verification manually, then reprocess.", ev))
        elif oc == "UNAVAILABLE":
            out.append(Issue("VERIFICATION_UNAVAILABLE", ct, f"{ct} could not be completed ({why}). This does not mean the vendor is invalid.", "MEDIUM", "SYSTEM",
                             "Retry the verification later (reprocess), or complete it manually.", ev))
        elif oc == "NEGATIVE":
            if md != "LIVE":
                out.append(Issue("VERIFICATION_SIMULATED_NEGATIVE", ct, f"A {md} provider reported a negative result for {ct} ({why}). "
                                 f"{md.capitalize()} data is not evidence about the vendor.", "MEDIUM", "REVIEW",
                                 "Obtain evidence from the vendor or a LIVE check.", ev))
            elif why == "TERMINAL_STATUS":
                out.append(Issue("LIVE_TERMINAL_STATUS", ct, f"A LIVE provider reports a terminal status for {ct}: {summ}", "HIGH", "REVIEW",
                                 "Reviewer confirms with evidence (this can lead to rejection) or requests clarification.", ev))
            elif why == "NAME_MISMATCH":
                code = "NAME_MISMATCH_BANK" if ct == "BANK_VERIFY" else "NAME_MISMATCH_DOCUMENT"
                out.append(Issue(code, ct, summ, "HIGH" if ct == "BANK_VERIFY" else "MEDIUM", "REVIEW",
                                 "Confirm the account holder / registered name matches the company.", ev))
            else:
                out.append(Issue("PROVIDER_NOT_FOUND", ct, f"{ct}: {summ or 'not found'}", "MEDIUM", "REVIEW",
                                 "Check the identifier with the vendor; a provider miss is not proof the vendor is invalid.", ev))
    return out


# ------------------------------------------------------------------ the decision
@dataclass
class DecisionResult:
    status: str
    reason_codes: List[str]
    explanation: str
    required_actions: List[str]
    trace: Dict[str, Any]
    issues: List[Issue] = field(default_factory=list)
    basis: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def blocking_keys(self) -> List[str]:
        return sorted(i.key for i in self.issues if i.severity in BLOCKING)


def decide(mode: str, issues: Sequence[Issue], verifications: Sequence[Mapping[str, Any]], *, risk_level: str = "NONE",
           risk_score: int = 0, revision_no: int = 1) -> DecisionResult:
    """APPROVED iff no blocking issue and at least one mandatory check exists and all mandatory checks are accepted in this mode."""
    data_open = any(i.code in ("MISSING_FIELD", "INVALID_FORMAT", "MISSING_DOCUMENT") and i.severity in BLOCKING for i in issues)
    all_issues = list(issues) + verification_issues(verifications, mode, data_open)
    blocking = [i for i in all_issues if i.severity in BLOCKING]
    mandatory = [v for v in verifications if v.get("mandatory", True) and v["reason"] != "WAIVED_BY_DECLARATION"]
    accepted = ACCEPTED_MODES[mode]
    trace: Dict[str, Any] = {"policy_version": POLICY_VERSION, "mode": mode, "revision_no": revision_no, "risk_level": risk_level,
                             "risk_score": risk_score, "rules": []}
    rules = trace["rules"]
    rules.append({"rule": "P0.mode", "result": mode, "detail": f"accepted verification modes: {sorted(accepted)}"})
    rules.append({"rule": "P1.mandatory_checks", "result": "OK" if mandatory else "NONE_FOUND",
                  "detail": [f"{v['check_type']}={v['outcome']}/{v['mode']}" for v in mandatory]})
    if not mandatory and not any(i.code in ("MISSING_FIELD", "MISSING_DOCUMENT") for i in blocking):
        blocking.append(Issue("VERIFICATION_NOT_PERFORMED", "all", "No mandatory verification result exists for this application.", "MEDIUM", "SYSTEM",
                              "Run verification (reprocess)."))
    rules.append({"rule": "P2.blocking_issues", "result": "NONE" if not blocking else f"{len(blocking)} open", "detail": [i.key for i in blocking]})
    rejection_issue = next((i for i in blocking if i.code == "BLOCKLIST_MATCH" and i.evidence.get("confirmed")), None)
    if rejection_issue is None:
        rejection_issue = next((i for i in blocking if i.code == "LIVE_TERMINAL_STATUS" and i.evidence.get("mode") == "LIVE"), None)
    cand = rejection_issue is not None
    rules.append({"rule": "P3.rejection", "result": "AUTOMATIC_REJECTION" if cand else "NOT_APPLICABLE",
                  "detail": "Only a confirmed internal disqualification or LIVE terminal status can be rejected automatically."})
    basis = [{"check": v["check_type"], "provider": v["provider"], "outcome": v["outcome"], "mode": v["mode"], "reason": v.get("reason", ""),
              "mandatory": v.get("mandatory", True)} for v in verifications]
    trace["issues"] = [{"key": i.key, "code": i.code, "subject": i.subject, "severity": i.severity, "kind": i.kind, "message": i.message,
                        "required_action": i.required_action, "evidence": i.evidence} for i in all_issues]
    trace["rejection_candidate"] = cand
    if rejection_issue:
        evidence = rejection_issue.evidence
        rule_id = "R1_CONFIRMED_INTERNAL_RECORD" if rejection_issue.code == "BLOCKLIST_MATCH" else "R2_LIVE_TERMINAL_STATUS"
        vendor_explanation = ("We cannot approve this application because an exact identifier match was found in a confirmed internal "
                              "disqualification record. Contact procurement if you believe this is incorrect."
                              if rejection_issue.code == "BLOCKLIST_MATCH" else
                              "We cannot approve this application because an authoritative live registry reports a terminal company status. "
                              "Contact procurement if you believe this is incorrect.")
        trace["automatic_rejection"] = {"policy_rule_id": rule_id, "evidence_source": evidence.get("evidence_source") or evidence.get("provider"),
                                        "matched_identifier": evidence.get("matched_identifier") or evidence.get("check"),
                                        "verification_timestamp": evidence.get("verification_timestamp", "recorded with this decision"),
                                        "vendor_explanation": vendor_explanation}
        rules.append({"rule": rule_id, "result": "REJECTED", "detail": trace["automatic_rejection"]})
        return DecisionResult("REJECTED", [rejection_issue.code], vendor_explanation, [], trace, all_issues, basis)
    if blocking:
        codes = list(dict.fromkeys(i.code for i in blocking))
        actions = list(dict.fromkeys(i.required_action for i in blocking if i.required_action))
        expl = (f"PENDING under {mode} policy {POLICY_VERSION}: {len(blocking)} unresolved issue(s): "
                + "; ".join(f"{title(c)}" for c in codes) + ".")
        return DecisionResult("PENDING", codes, expl, actions, trace, all_issues, basis)
    unproven = [v["check_type"] for v in mandatory if v["outcome"] != "VERIFIED" or v["mode"] not in accepted]
    if unproven:                                              # defensive: APPROVED is impossible unless every mandatory check passed
        i = Issue("VERIFICATION_NOT_PERFORMED", ",".join(unproven), f"Mandatory check(s) not satisfied: {', '.join(unproven)}.", "MEDIUM", "SYSTEM",
                  "Complete verification, then reprocess.")
        trace["issues"].append({"key": i.key, "code": i.code, "subject": i.subject, "severity": i.severity, "kind": i.kind,
                                "message": i.message, "required_action": i.required_action, "evidence": {}})
        return DecisionResult("PENDING", [i.code], f"PENDING under {mode} policy {POLICY_VERSION}: {i.message}", [i.required_action], trace,
                              all_issues + [i], basis)
    modes = sorted({v["mode"] for v in mandatory})
    expl = (f"APPROVED under {mode} policy {POLICY_VERSION}: all {len(mandatory)} mandatory check(s) passed "
            f"(verification mode: {', '.join(modes)}); no open issues.")
    return DecisionResult("APPROVED", [], expl, [], trace, all_issues, basis)
