"""Human review: queue, notes, field corrections, requests for information, validated overrides, reprocessing, audit history.

Reviewer identity is a self-declared name (no authentication) - a documented limitation. Every action is recorded with a
timestamp in review_actions; decisions are append-only, so nothing a reviewer does can erase earlier history.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from vendorflow.config import Settings
from vendorflow.db.repositories import Repository
from vendorflow.decision import communication
from vendorflow.decision.engine import BLOCKING
from vendorflow.decision.policy import MIN_REASON_CHARS, POLICY_VERSION, REJECTION_RULES, overridable
from vendorflow.domain.models import DocumentInput
from vendorflow.ingestion.revisions import create_revision
from vendorflow.utils.masking import mask_account


class ReviewError(ValueError):
    """A review action was refused. `problems` lists every reason so the UI can show them all."""

    def __init__(self, problems: List[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class ReviewService:
    def __init__(self, repo: Repository, settings: Settings, pipeline: Any):
        self.repo, self.settings, self.pipeline = repo, settings, pipeline

    # ------------------------------------------------------------------ queue
    def queue(self, states: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        rows = self.repo.list_tasks(states or ["OPEN", "IN_REVIEW", "AWAITING_VENDOR"])
        for r in rows:
            d = self.repo.current_decision(r["application_id"])
            r["decision_status"] = d["status"] if d else None
            r["risk_level"] = (d or {}).get("rule_trace", {}).get("risk_level")
            r["rejection_candidate"] = bool((d or {}).get("rule_trace", {}).get("rejection_candidate"))
        return rows

    def claim(self, task_id: int, reviewer: str) -> None:
        self._need_name(reviewer)
        self.repo.set_task_state(task_id, "IN_REVIEW", assigned_to=reviewer)

    def current_issues(self, application_id: str) -> List[Dict[str, Any]]:
        d = self.repo.current_decision(application_id)
        return [i for i in (d or {}).get("rule_trace", {}).get("issues", []) if i["severity"] in BLOCKING]

    # ------------------------------------------------------------------ simple actions
    def add_note(self, application_id: str, reviewer: str, comment: str) -> int:
        self._need_name(reviewer)
        if not comment or not comment.strip():
            raise ReviewError(["A comment is required."])
        task = self.repo.active_task(application_id)
        return self.repo.add_review_action(application_id, task["task_id"] if task else None, reviewer, "NOTE", {"comment": comment.strip()})

    def request_info(self, application_id: str, reviewer: str, message: str) -> int:
        """Ask the vendor for something. The decision stays PENDING; the task moves to AWAITING_VENDOR; a draft message is stored."""
        self._need_name(reviewer)
        if not message or len(message.strip()) < 5:
            raise ReviewError(["Say what information is needed."])
        cur = self._pending(application_id)
        task = self.repo.active_task(application_id)
        aid = self.repo.add_review_action(application_id, task["task_id"] if task else None, reviewer, "REQUEST_INFO", {"message": message.strip()})
        if task:
            self.repo.set_task_state(task["task_id"], "AWAITING_VENDOR")
        app = self.repo.get_application(application_id)
        self.repo.add_notification(application_id, cur["decision_id"], f"Information needed: vendor application {application_id}",
                                   f"Dear {app['legal_name']},\n\nOur reviewer needs the following to continue your application:\n\n{message.strip()}\n\n"
                                   f"Your application is on hold, not rejected.\n")
        return aid

    # ------------------------------------------------------------------ corrections
    def correct_field(self, application_id: str, reviewer: str, document_id: int, field_name: str, new_value: str, reason: str,
                      reevaluate: bool = True) -> List[Any]:
        """Confirm or correct a value read from a document (a HUMAN row is appended; automated rows are never overwritten)."""
        self._need_name(reviewer)
        problems = []
        if not reason or len(reason.strip()) < MIN_REASON_CHARS:
            problems.append(f"Give a reason of at least {MIN_REASON_CHARS} characters (what you checked on the document).")
        if not new_value or not new_value.strip():
            problems.append("The value cannot be empty.")
        app = self.repo.get_application(application_id)
        doc = next((d for d in self.repo.list_documents(application_id, app["current_revision"]) if d["document_id"] == document_id), None)
        if not doc:
            problems.append("That document does not belong to the current revision of this application.")
        if problems:
            raise ReviewError(problems)
        old = next((f for f in self.repo.current_fields_rev(application_id, app["current_revision"])
                    if f["document_id"] == document_id and f["field_name"] == field_name), None)
        run = self.repo.latest_run(application_id)
        self.repo.insert_human_field(run["run_id"], document_id, field_name, new_value.strip(), f"Reviewer {reviewer}: {reason.strip()}")
        task = self.repo.active_task(application_id)
        self.repo.add_review_action(application_id, task["task_id"] if task else None, reviewer, "CORRECT_FIELD",
                                    {"document_id": document_id, "filename": doc["filename"], "field": field_name,
                                     "old_value": (old or {}).get("value"), "new_value": new_value.strip(), "reason": reason.strip()})
        return self.pipeline.reevaluate_all(application_id) if reevaluate else []

    def correct_submission(self, application_id: str, reviewer: str, field_updates: Dict[str, str], reason: str,
                           new_documents: Optional[List[DocumentInput]] = None) -> List[Any]:
        """Reviewer edits the submitted data on the vendor's behalf: a NEW revision, fully reprocessed."""
        self._need_name(reviewer)
        if not reason or len(reason.strip()) < MIN_REASON_CHARS:
            raise ReviewError([f"Give a reason of at least {MIN_REASON_CHARS} characters."])
        rev = create_revision(self.repo, self.settings, application_id, field_updates=field_updates, new_documents=new_documents,
                              source="REVIEWER", created_by=reviewer)
        task = self.repo.active_task(application_id)
        self.repo.add_review_action(application_id, task["task_id"] if task else None, reviewer, "CORRECT_FIELD",
                                    {"revision": rev, "fields": list(field_updates), "reason": reason.strip(), "scope": "submission"})
        return self.pipeline.run_all(application_id, trigger="revision")

    def vendor_correction(self, application_id: str, field_updates: Optional[Dict[str, str]] = None,
                          new_documents: Optional[List[DocumentInput]] = None, declaration_updates: Optional[Dict[str, Any]] = None) -> List[Any]:
        """The vendor answers a request: corrected fields and/or replacement documents -> new revision -> reprocess."""
        create_revision(self.repo, self.settings, application_id, field_updates=field_updates, declaration_updates=declaration_updates,
                        new_documents=new_documents, source="FORM", created_by="vendor")
        return self.pipeline.run_all(application_id, trigger="revision")

    def reprocess(self, application_id: str, reviewer: str, reason: str = "", full: bool = False) -> List[Any]:
        self._need_name(reviewer)
        task = self.repo.active_task(application_id)
        self.repo.add_review_action(application_id, task["task_id"] if task else None, reviewer, "REPROCESS",
                                    {"mode": "full (re-extract)" if full else "re-evaluate", "reason": reason})
        return self.pipeline.run_all(application_id, trigger="reprocess") if full else self.pipeline.reevaluate_all(application_id)

    # ------------------------------------------------------------------ override (the only way to APPROVE past an issue or to REJECT)
    def override(self, application_id: str, reviewer: str, decision: str, reason: str, condition_code: Optional[str] = None,
                 evidence: str = "") -> int:
        self._need_name(reviewer)
        cur = self._pending(application_id)
        trace = cur["rule_trace"]
        mode = self.settings.app_mode
        problems: List[str] = []
        if not reason or len(reason.strip()) < MIN_REASON_CHARS:
            problems.append(f"An override reason of at least {MIN_REASON_CHARS} characters is required.")
        blocking = [i for i in trace.get("issues", []) if i["severity"] in BLOCKING]
        decision = decision.upper()
        if decision == "APPROVED":
            for i in blocking:
                if not overridable(i["code"], mode):
                    problems.append(f"'{i['code']}' ({i['subject']}) cannot be overridden in {mode} mode; it must be fixed "
                                    f"(correction, new document, or a successful live verification).")
            codes = sorted({i["code"] for i in blocking})
            reason_codes, resolved = codes, [i["key"] for i in blocking]
        elif decision == "REJECTED":
            rule = REJECTION_RULES.get(condition_code or "")
            if not rule:
                problems.append("Rejection requires a documented condition: " + ", ".join(f"{k} ({v[1]})" for k, v in REJECTION_RULES.items()) + ".")
            else:
                issue = next((i for i in blocking if i["code"] == rule[0]), None)
                if not issue:
                    problems.append(f"Condition {condition_code} requires an open {rule[0]} on this application; none exists.")
                elif condition_code == "R2" and issue["evidence"].get("mode") != "LIVE":
                    problems.append("R2 requires a LIVE provider result; simulated or sandbox results can never support rejection.")
            if not evidence or len(evidence.strip()) < MIN_REASON_CHARS:
                problems.append("Rejection requires an evidence reference or description (record reference, document, provider response).")
            reason_codes = ["REVIEWER_CONFIRMED_DISQUALIFICATION" if condition_code == "R1" else "LIVE_TERMINAL_STATUS"]
            resolved = []
        else:
            problems.append("Decision must be APPROVED or REJECTED.")
            reason_codes, resolved = [], []
        if problems:
            raise ReviewError(problems)
        task = self.repo.active_task(application_id)
        new_trace = {"policy_version": POLICY_VERSION, "mode": mode, "revision_no": trace.get("revision_no"), "resolved_issue_keys": resolved,
                     "condition_code": condition_code, "evidence": evidence.strip() if evidence else "", "overrides_decision_id": cur["decision_id"],
                     "issues": trace.get("issues", []), "risk_level": trace.get("risk_level")}
        did = self.repo.insert_decision(application_id, run_id=(self.repo.latest_run(application_id) or {}).get("run_id"), status=decision,
                                        decided_by=reviewer, decision_type="OVERRIDE", policy_mode=mode,
                                        verification_basis=cur["verification_basis_data"], reason_codes=reason_codes,
                                        explanation=f"{decision} by reviewer {reviewer} ({mode} policy {POLICY_VERSION}): {reason.strip()}",
                                        required_actions=[], rule_trace=new_trace)
        self.repo.add_review_action(application_id, task["task_id"] if task else None, reviewer, "OVERRIDE",
                                    {"decision": decision, "reason": reason.strip(), "condition_code": condition_code, "evidence": (evidence or "").strip(),
                                     "resolved_issue_keys": resolved, "decision_id": did, "previous_decision_id": cur["decision_id"]})
        self.repo.resolve_open_tasks(application_id)
        d = self.repo.current_decision(application_id)
        subj, body = communication.draft(self.repo.get_application(application_id), d)
        self.repo.add_notification(application_id, did, subj, body)
        return did

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _need_name(reviewer: str) -> None:
        if not reviewer or not reviewer.strip():
            raise ReviewError(["Reviewer name is required (it is recorded in the audit trail)."])

    def _pending(self, application_id: str) -> Dict[str, Any]:
        cur = self.repo.current_decision(application_id)
        if not cur:
            raise ReviewError(["This application has no decision yet; process it first."])
        if cur["status"] != "PENDING":
            raise ReviewError([f"This application is already {cur['status']}; only PENDING applications can be reviewed."])
        return cur


def audit_history(repo: Repository, application_id: str) -> List[Dict[str, Any]]:
    """One chronological list of everything that happened to an application (for the details page and for export)."""
    ev: List[Dict[str, Any]] = []
    for r in repo.list_revisions(application_id):
        ev.append({"ts": r["created_at"], "kind": "REVISION", "actor": r["created_by"], "summary": f"Revision {r['revision_no']} created ({r['source']})"})
    for r in repo.list_runs(application_id):
        ev.append({"ts": r["started_at"], "kind": "RUN", "actor": "system", "summary": f"Run {r['run_id']} ({r['trigger']}, revision {r['revision_no']}): {r['status']}"})
    for d in repo.list_decisions(application_id):
        ev.append({"ts": d["created_at"], "kind": "DECISION", "actor": d["decided_by"],
                   "summary": f"{d['status']} ({d['decision_type']}): {d['explanation']}", "reason_codes": d["reason_codes"]})
    for a in repo.list_review_actions(application_id):
        ev.append({"ts": a["created_at"], "kind": a["action_type"], "actor": a["reviewer"], "summary": _action_summary(a), "payload": a["payload"]})
    for n in repo.list_notifications(application_id):
        ev.append({"ts": n["created_at"], "kind": "NOTIFICATION", "actor": "system", "summary": f"Message drafted (not sent): {n['subject']}"})
    return sorted(ev, key=lambda e: e["ts"])


def _action_summary(a: Dict[str, Any]) -> str:
    p, t = a["payload"], a["action_type"]
    if t == "NOTE":
        return f"Note: {p.get('comment')}"
    if t == "REQUEST_INFO":
        return f"Requested information from the vendor: {p.get('message')}"
    if t == "CORRECT_FIELD":
        if p.get("scope") == "submission":
            return f"Corrected submission fields {', '.join(p.get('fields', []))} (revision {p.get('revision')}): {p.get('reason')}"
        return f"Corrected {p.get('field')} on {p.get('filename')}: {p.get('reason')}"
    if t == "OVERRIDE":
        return f"Override to {p.get('decision')}" + (f" under {p['condition_code']}" if p.get("condition_code") else "") + f": {p.get('reason')}"
    return f"Reprocess requested ({p.get('mode')}): {p.get('reason') or ''}".strip()
