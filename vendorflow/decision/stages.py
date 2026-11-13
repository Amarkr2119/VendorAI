"""Phase 4 stages, run after extraction by the existing Pipeline generator (no second pipeline):
VALIDATION -> VERIFICATION -> CONSISTENCY -> DUPLICATES -> RISK -> DECISION -> COMMUNICATION.

Each stage persists its evidence, writes processing_logs, and yields a StageEvent so the UI shows REAL progress.
A failing stage stops the run WITHOUT writing a decision (the application is never approved or rejected by a crash).
"""
from __future__ import annotations

import time
from datetime import date
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from vendorflow.analysis.consistency import check_consistency, effective_type
from vendorflow.analysis.duplicates import BLOCKLIST_PATH, check_blocklist, detect_duplicates
from vendorflow.analysis.risk import assess
from vendorflow.config import Settings
from vendorflow.db.repositories import Repository
from vendorflow.decision import communication
from vendorflow.decision.engine import BLOCKING, DecisionResult, decide, extraction_issues, issues_from_findings, verification_issues
from vendorflow.domain.enums import Stage
from vendorflow.domain.models import Finding, Issue, StageEvent
from vendorflow.ingestion.completeness import check_documents, check_fields
from vendorflow.logging_setup import get_logger
from vendorflow.pipeline.events import make_event
from vendorflow.validation.rules import validate_formats
from vendorflow.verification.service import VerificationService

log = get_logger("decision")
TextLoader = Callable[[Mapping[str, Any]], str]


class DecisionStages:
    def __init__(self, repo: Repository, settings: Settings, verifier: Optional[VerificationService] = None,
                 blocklist_path=BLOCKLIST_PATH, today: Optional[date] = None):
        self.repo, self.settings = repo, settings
        self.verifier = verifier or VerificationService(repo, settings)
        self.blocklist_path, self.today = blocklist_path, today

    # ------------------------------------------------------------------ public
    def run(self, run_id: int, application_id: str, text_loader: Optional[TextLoader] = None) -> Iterator[StageEvent]:
        app = self.repo.get_application(application_id)
        rev_no = app["current_revision"]
        payload = self.repo.get_revision(application_id, rev_no)["payload"]
        country, mode = app["country"], self.settings.app_mode
        norm, disp, decl = payload["normalized"], payload["fields"], payload["declarations"]
        docs = self.repo.list_documents(application_id, rev_no)
        t = {"t0": time.monotonic()}

        def done(stage: str, status: str, msg: str, **detail: Any) -> StageEvent:
            ev = make_event(stage, status, msg, run_id=run_id, **detail)
            self.repo.log_stage(run_id, application_id, stage, status, msg, detail or None,
                                error=msg if status == "error" else None, duration_ms=int((time.monotonic() - t["t0"]) * 1000))
            t["t0"] = time.monotonic()
            return ev

        try:
            # ---------------------------------------------------------- VALIDATION
            yield make_event(Stage.VALIDATION.value, "started", "Validating completeness, identifier formats and reading reliability", run_id=run_id)
            findings: List[Finding] = (check_fields(country, disp, decl) + check_documents(country, decl, docs, use_detected=True)
                                       + validate_formats(country, norm))
            fields = self.repo.current_fields_rev(application_id, rev_no)
            for f in fields:                                         # current reading state, so human corrections are visible in the audit trail
                if f["confidence_band"] == "LOW" and f["origin"] != "HUMAN":
                    findings.append(Finding(f"EXTRACTION.{f['document_id']}.{f['field_name']}", "EXTRACTION", "FAIL", field=f["field_name"],
                                            reason_code="LOW_CONFIDENCE_EXTRACTION", message="Low-confidence reading awaiting human confirmation."))
                elif f["origin"] == "HUMAN":
                    findings.append(Finding(f"EXTRACTION.{f['document_id']}.{f['field_name']}", "EXTRACTION", "PASS", field=f["field_name"],
                                            message="Value confirmed or corrected by a human reviewer."))
            self.repo.insert_findings(run_id, application_id, findings)
            unreadable = self.repo.unreadable_findings(application_id, rev_no)      # recorded by extraction in the same or an earlier run
            issues: List[Issue] = (issues_from_findings(country, findings + [Finding(**{k: f[k] for k in ("rule_id", "category", "result", "field", "reason_code", "message")})
                                                                             for f in unreadable]) + extraction_issues(docs, fields, effective_type))
            n_block = sum(1 for i in issues if i.severity in BLOCKING)
            fmt_bad = [f.field for f in findings if f.category == "FORMAT" and f.result == "FAIL"]
            yield done(Stage.VALIDATION.value, "warning" if n_block else "ok",
                       f"{n_block} issue(s) found" + (f"; invalid format: {', '.join(fmt_bad)}" if fmt_bad else "; all required data present and well-formed"),
                       issues=[i.key for i in issues], invalid_format=fmt_bad)

            # ---------------------------------------------------------- VERIFICATION
            yield make_event(Stage.VERIFICATION.value, "started",
                             f"Verifying tax, registration and bank details ({mode} mode; providers: {self.settings.verification_providers})", run_id=run_id)
            results: List[Dict[str, Any]] = []
            for res in self.verifier.iter_run(run_id, application_id, country, norm, decl, disp):
                results.append({"check_type": res.check_type, "provider": res.provider, "outcome": res.outcome, "mode": res.mode, "reason": res.reason,
                                "summary": res.summary, "mandatory": res.mandatory, "attempts": res.attempts})
                st = {"VERIFIED": "ok", "NEGATIVE": "warning", "UNAVAILABLE": "warning", "SKIPPED": "skipped"}[res.outcome]
                yield done(Stage.VERIFICATION.value, st, f"{res.check_type}: {res.outcome} [{res.mode}] {res.summary}",
                           check=res.check_type, outcome=res.outcome, mode=res.mode, provider=res.provider, attempts=res.attempts, reason=res.reason)
            ok = sum(1 for r in results if r["outcome"] == "VERIFIED")
            yield done(Stage.VERIFICATION.value, "ok" if ok == len([r for r in results if r["mandatory"]]) else "warning",
                       f"{ok} of {len(results)} check(s) verified", summary=True, modes=sorted({r["mode"] for r in results}))

            # ---------------------------------------------------------- CONSISTENCY
            yield make_event(Stage.CONSISTENCY.value, "started", "Comparing the submission with every document", run_id=run_id)
            rows, c_issues = check_consistency(country, payload, docs, fields, text_loader, self.today)
            self.repo.insert_consistency(run_id, application_id, rows)
            issues += c_issues
            bad = [r for r in rows if r["outcome"] not in ("MATCH", "VALID", "CONSISTENT", "SKIPPED_LOW_CONFIDENCE")]
            yield done(Stage.CONSISTENCY.value, "warning" if any(i.severity in BLOCKING for i in c_issues) else "ok",
                       f"{len(rows)} comparison(s); {len(bad)} difference(s)" + (": " + ", ".join(sorted({r['check_id'] for r in bad})) if bad else ""),
                       comparisons=len(rows), differences=len(bad))

            # ---------------------------------------------------------- DUPLICATES
            yield make_event(Stage.DUPLICATES.value, "started", "Checking earlier applications and the internal record list", run_id=run_id)
            matches, d_issues = detect_duplicates(self.repo, application_id)
            b_issues = check_blocklist(self.repo.current_identifiers(application_id), self.blocklist_path)
            self.repo.insert_duplicates(run_id, application_id, matches)
            issues += d_issues + b_issues
            yield done(Stage.DUPLICATES.value, "warning" if (d_issues or b_issues) else "ok",
                       (f"{len(d_issues)} suspected duplicate(s)" if d_issues else "no duplicates") + (f"; {len(matches) - len(d_issues)} weak overlap(s) noted" if len(matches) > len(d_issues) else "")
                       + (f"; internal record match: {', '.join(i.subject for i in b_issues)}" if b_issues else ""),
                       suspected=[m["matched_application_id"] for m in matches if m["strength"] != "WEAK"], blocklist=[i.subject for i in b_issues])

            # ---------------------------------------------------------- RISK
            yield make_event(Stage.RISK.value, "started", "Assessing risk indicators", run_id=run_id)
            data_open = any(i.code in ("MISSING_FIELD", "INVALID_FORMAT", "MISSING_DOCUMENT") for i in issues)
            indicators, level, score = assess(issues + verification_issues(results, self.settings.app_mode, data_open), results, matches, country, norm)
            self.repo.insert_risk(run_id, application_id, indicators)
            yield done(Stage.RISK.value, "ok" if level in ("NONE", "LOW") else "warning",
                       f"Risk level {level} (score {score}); {len(indicators)} indicator(s). Indicators prompt review; they are not findings of wrongdoing.",
                       level=level, score=score, indicators=[i["code"] for i in indicators])

            # ---------------------------------------------------------- DECISION
            yield make_event(Stage.DECISION.value, "started", f"Applying decision policy ({mode})", run_id=run_id)
            result = decide(mode, issues, results, risk_level=level, risk_score=score, revision_no=rev_no)
            decision_id, action = self._persist(run_id, app, result, rev_no)
            yield done(Stage.DECISION.value, {"APPROVED": "ok", "PENDING": "warning", "REJECTED": "error"}[result.status],
                       f"{result.status}: {result.explanation}" + ("" if action == "recorded" else f" ({action})"),
                       decision=result.status, reason_codes=result.reason_codes, required_actions=result.required_actions,
                       recorded=action, decision_id=decision_id, risk_level=level)

            # ---------------------------------------------------------- COMMUNICATION
            cur = self.repo.current_decision(application_id)
            if action == "recorded" and cur:
                subj, body = communication.draft(self.repo.get_application(application_id), cur)
                nid = self.repo.add_notification(application_id, cur["decision_id"], subj, body)
                yield done(Stage.COMMUNICATION.value, "ok", f"Message to the vendor drafted (not sent): {subj}", notification_id=nid)
            else:
                yield done(Stage.COMMUNICATION.value, "skipped", "No new decision, so no new message was drafted")
        except Exception as e:
            log.exception("Phase 4 stage failed for %s", application_id)
            yield done("DECISION", "error", f"Evaluation failed, no decision was recorded: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ persistence rules
    def _persist(self, run_id: int, app: Mapping[str, Any], r: DecisionResult, rev_no: int):
        """Append-only decisions. Returns (decision_id, 'recorded' | explanation of why nothing was appended)."""
        aid = app["application_id"]
        cur = self.repo.current_decision(aid)
        fingerprint = {"status": r.status, "blocking": r.blocking_keys, "policy": r.trace["policy_version"], "mode": r.trace["mode"], "revision": rev_no}
        r.trace["fingerprint"] = fingerprint
        explanation = r.explanation
        if cur:
            ct = cur["rule_trace"]
            if cur["status"] == "REJECTED":
                return cur["decision_id"], "reprocessed; the reviewer's REJECTION stands and is never reversed automatically"
            if cur["decision_type"] == "OVERRIDE" and ct.get("revision_no") == rev_no:
                resolved = set(ct.get("resolved_issue_keys", []))
                new = [k for k in r.blocking_keys if k not in resolved]
                if not new:
                    return cur["decision_id"], "reprocessed; the reviewer override still stands (no new issues)"
                explanation = f"New issue(s) since the reviewer override: {', '.join(new)}. " + r.explanation
            elif cur["decision_type"] == "AUTO" and ct.get("fingerprint") == fingerprint:
                return cur["decision_id"], "reprocessed; the evidence and result are unchanged"
        did = self.repo.insert_decision(aid, run_id=run_id, status=r.status, decided_by="system", decision_type="AUTO", policy_mode=r.trace["mode"],
                                        verification_basis=r.basis, reason_codes=r.reason_codes, explanation=explanation,
                                        required_actions=r.required_actions, rule_trace=r.trace)
        self._sync_tasks(aid, run_id, r)
        return did, "recorded"

    def _sync_tasks(self, aid: str, run_id: int, r: DecisionResult) -> None:
        self.repo.resolve_open_tasks(aid)
        if r.status != "PENDING":
            return
        blocking = [i for i in r.issues if i.severity in BLOCKING]
        needs_human = any(i.kind in ("REVIEW", "SYSTEM") for i in blocking)
        self.repo.open_review_task(aid, run_id, r.reason_codes, "OPEN" if needs_human else "AWAITING_VENDOR")
