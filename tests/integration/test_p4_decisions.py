"""Phase 4 integration tests: the real pipeline (real PDFs, real extraction, real DB) with SIMULATED providers from fixtures.
Where a test needs a LIVE-mode provider it uses a STUB adapter, clearly named. No test here calls a real external API."""
from __future__ import annotations

import json
import sqlite3
import unittest

from tests.helpers import ResponderClient, load_truth, make_services, perfect_responder, process_ref, stub_router
from vendorflow.decision.review import ReviewError, audit_history
from vendorflow.domain.models import DocumentInput
from vendorflow.ingestion.intake import register_application, submission_from_json
from vendorflow.llm.base import LlmError, LlmErrorKind
from vendorflow.services import register_synthetic, synthetic_index
from vendorflow.validation import identifiers as ids
from vendorflow.verification.base import Adapter, ProviderError

APPS = {a["ref"]: a for a in synthetic_index()}
TRUTH = load_truth()
DATA = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "synthetic"


def svc_for(**kw):
    kw.setdefault("app_mode", "DEMO")
    kw.setdefault("verify_backoff_s", 0.0)
    return make_services(llm_enabled=False, **kw)


def decision(svc, app_id):
    return svc.repo.current_decision(app_id)


def verifs(svc, app_id):
    return {v["check_type"]: v for v in svc.repo.run_verifications(svc.repo.latest_run(app_id)["run_id"])}


class LiveStub(Adapter):
    """STUB standing in for a LIVE provider (mode LIVE) so production-mode logic can be exercised offline."""
    def __init__(self, check_type, name_for, status="ACTIVE", found=True, error=None):
        self.name, self.check_type, self.mode = f"live-stub-{check_type.lower()}", check_type, "LIVE"
        self.name_for, self.status, self.found, self.error, self.calls = name_for, status, found, error, 0

    def lookup(self, req, timeout):
        self.calls += 1
        if self.error:
            raise self.error
        return {"found": self.found, "name": self.name_for or req.claimed_name, "holder_name": self.name_for or req.claimed_name,
                "status": self.status, "match": True}


class TestHappyPathAndModes(unittest.TestCase):
    def test_all_three_countries_approve_in_demo_with_simulated_label(self):
        svc = svc_for()
        for ref in ("IN-001", "US-001", "UK-001"):
            app, events = process_ref(svc, ref)
            d = decision(svc, app)
            self.assertEqual((d["status"], d["decision_type"], d["policy_mode"]), ("APPROVED", "AUTO", "DEMO"), ref)
            self.assertIn("SIMULATED", d["explanation"])
            self.assertTrue(all(v["mode"] == "SIMULATED" and v["outcome"] == "VERIFIED" for v in verifs(svc, app).values()), ref)
            self.assertEqual(events[-1].stage, "DONE")
            self.assertEqual(events[-1].detail["decision"], "APPROVED")

    def test_production_with_simulated_or_missing_providers_never_approves(self):
        for kw, code in (({}, "VERIFICATION_NOT_PERFORMED"), ({"verification_providers": "simulated"}, "VERIFICATION_NOT_LIVE")):
            svc = svc_for(app_mode="PRODUCTION", **kw)
            app, _ = process_ref(svc, "IN-001")
            d = decision(svc, app)
            self.assertEqual(d["status"], "PENDING", kw)
            self.assertIn(code, d["reason_codes"])

    def test_production_approves_only_with_LIVE_mode_results_stub(self):
        svc = svc_for(app_mode="PRODUCTION")
        for ct in ("PAN_VERIFY", "GSTIN_VERIFY", "BANK_VERIFY"):
            svc.registry.register("IN", ct, LiveStub(ct, None))
        app, _ = process_ref(svc, "IN-001")
        d = decision(svc, app)
        self.assertEqual((d["status"], d["policy_mode"]), ("APPROVED", "PRODUCTION"))
        self.assertTrue(all(v["mode"] == "LIVE" for v in verifs(svc, app).values()))

    def test_confirmed_synthetic_disqualification_is_rejected_and_other_cases_are_not(self):
        svc = svc_for()
        statuses = {process_ref(svc, ref)[0]: None for ref in APPS}
        rejected = [app for app in statuses if decision(svc, app)["status"] == "REJECTED"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(svc.repo.get_application(rejected[0])["source_ref"], "US-010")

    def test_decision_carries_policy_version_and_rule_trace(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-001")
        t = decision(svc, app)["rule_trace"]
        self.assertEqual(t["policy_version"], "2026.2")
        self.assertIn("P2.blocking_issues", [r["rule"] for r in t["rules"]])


class TestSixMandatoryEdgeCases(unittest.TestCase):
    def test_1_missing_information_or_documents_is_pending_and_requests_them(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-003")
        d = decision(svc, app)
        self.assertEqual(d["status"], "PENDING")
        self.assertTrue({"MISSING_DOCUMENT", "MISSING_FIELD"} <= set(d["reason_codes"]))
        self.assertTrue(any(a.startswith("Upload:") for a in d["required_actions"]))
        self.assertEqual(svc.repo.active_task(app)["state"], "AWAITING_VENDOR")            # waiting for the vendor, not a reviewer
        note = svc.repo.list_notifications(app)[-1]
        self.assertIn("on hold (not rejected)", note["body"])
        self.assertEqual(note["generation_method"], "TEMPLATE")

    def test_2_invalid_tax_identifier_format_is_pending_with_correction_request(self):
        svc = svc_for()
        for ref, field in (("IN-005", "GSTIN"), ("US-004", "EIN"), ("UK-004", "VAT")):
            app, _ = process_ref(svc, ref)
            d = decision(svc, app)
            self.assertEqual(d["status"], "PENDING", ref)
            self.assertIn("INVALID_FORMAT", d["reason_codes"])
            self.assertTrue(any(a.startswith("Correct") and field in a.upper() for a in d["required_actions"]), (ref, d["required_actions"]))
            tax = {"IN-005": "GSTIN_VERIFY", "US-004": "TIN_MATCH", "UK-004": "VAT_VERIFY"}[ref]
            v = verifs(svc, app)[tax]
            self.assertEqual((v["outcome"], v["response"]["reason"]), ("SKIPPED", "INVALID_FORMAT"))   # never sent to a provider
            calls = svc.repo._all("SELECT * FROM api_audit_log WHERE category='VERIFICATION' AND purpose=? AND run_id=?", (tax, v["run_id"]))
            self.assertEqual(calls, [])

    def test_3_bank_holder_company_mismatch_goes_to_manual_review(self):
        svc = svc_for()
        for ref in ("IN-007", "US-006", "UK-005"):
            app, _ = process_ref(svc, ref)
            d = decision(svc, app)
            self.assertEqual(d["status"], "PENDING", ref)
            self.assertIn("NAME_MISMATCH_BANK", d["reason_codes"])
            self.assertEqual(svc.repo.active_task(app)["state"], "OPEN")
            rows = svc.repo.run_consistency(svc.repo.latest_run(app)["run_id"])
            self.assertTrue(any(r["check_id"] == "NAME.bank_holder" and r["outcome"] != "MATCH" for r in rows))
            self.assertNotIn("fraud", d["explanation"].lower())                                # no unsupported accusations
            self.assertTrue(all("fraud" not in r["explanation"].lower() for r in svc.repo.run_risk(svc.repo.latest_run(app)["run_id"])))

    def test_4_duplicate_application_goes_to_manual_review_original_untouched(self):
        svc = svc_for()
        orig, _ = process_ref(svc, "IN-001")
        dup, _ = process_ref(svc, "IN-008")
        d = decision(svc, dup)
        self.assertEqual(d["status"], "PENDING")
        self.assertIn("DUPLICATE_SUSPECTED", d["reason_codes"])
        m = svc.repo.run_duplicates(svc.repo.latest_run(dup)["run_id"])[0]
        self.assertEqual(m["matched_application_id"], orig)
        self.assertIn("PAN", m["basis"]["exact_identifiers"])
        self.assertEqual(decision(svc, orig)["status"], "APPROVED")                          # the original is not disturbed
        self.assertEqual(len(svc.repo.list_decisions(orig)), 1)

    def test_4b_an_application_is_not_a_duplicate_of_a_LATER_one(self):
        svc = svc_for()
        for ref in ("IN-001", "IN-008"):                                                    # both registered before IN-001 is processed
            register_synthetic(svc, ref)
        app = register_synthetic(svc, "IN-001")
        svc.pipeline.run_all(app)
        self.assertEqual(decision(svc, app)["status"], "APPROVED")

    def test_5_provider_timeout_or_failure_is_pending_never_invalid(self):
        svc = svc_for()
        for ref, check, code in (("US-008", "BUSINESS_REGISTRY", "TIMEOUT"), ("UK-007", "VAT_VERIFY", "HTTP_503")):
            app, _ = process_ref(svc, ref)
            d = decision(svc, app)
            self.assertEqual((d["status"], d["reason_codes"]), ("PENDING", ["VERIFICATION_UNAVAILABLE"]), ref)
            v = verifs(svc, app)[check]
            self.assertEqual((v["outcome"], v["attempts"], v["response"]["reason"]), ("UNAVAILABLE", 3, code))
            self.assertFalse(any(x["outcome"] == "NEGATIVE" for x in verifs(svc, app).values()))
            self.assertNotIn("INVALID_FORMAT", d["reason_codes"])
            self.assertTrue(any("Retry" in a for a in d["required_actions"]))

    def test_5b_after_the_provider_recovers_reprocess_approves(self):
        svc = svc_for()
        app, _ = process_ref(svc, "UK-007")
        self.assertEqual(decision(svc, app)["status"], "PENDING")
        svc.registry.register("GB", "VAT_VERIFY", LiveStub("VAT_VERIFY", None))
        svc.review.reprocess(app, "ana", "provider is back", full=False)
        d = decision(svc, app)
        self.assertEqual(d["status"], "APPROVED")                                           # previous PENDING decision is preserved below
        self.assertEqual([x["status"] for x in svc.repo.list_decisions(app)], ["PENDING", "APPROVED"])
        self.assertEqual(svc.repo.list_decisions(app)[1]["supersedes_decision_id"], svc.repo.list_decisions(app)[0]["decision_id"])

    def test_6_low_confidence_extraction_goes_to_human_review_then_correction_resolves(self):
        svc = svc_for()
        for ref in ("IN-009", "US-009", "UK-008"):
            app, _ = process_ref(svc, ref)
            d = decision(svc, app)
            self.assertEqual((d["status"], d["reason_codes"]), ("PENDING", ["LOW_CONFIDENCE_EXTRACTION"]), ref)
            self.assertEqual(svc.review.queue()[-1]["state"], "OPEN")
            for issue in svc.review.current_issues(app):
                ev = issue["evidence"]
                truth = TRUTH[ref][ev["document"]]["fields"][ev["field"]]
                svc.review.correct_field(app, "ana", ev["document_id"], ev["field"], truth, "Compared with the scanned document image.")
            d = decision(svc, app)
            self.assertEqual((d["status"], d["decision_type"]), ("APPROVED", "AUTO"), ref)
            kinds = [e["kind"] for e in audit_history(svc.repo, app)]
            self.assertIn("CORRECT_FIELD", kinds)

    def test_6b_wrong_confident_extraction_cannot_be_silently_accepted(self):
        """An LLM that 'corrects' an identifier to something not in the document must not produce an approval."""
        from tests.helpers import hallucinating_responder
        svc0 = svc_for()
        router = stub_router(svc0.repo, svc0.settings, ResponderClient("gemini", hallucinating_responder("gstin", "29ABCDE1234F1Z5")))
        svc = make_services(router=router, llm_enabled=True, app_mode="DEMO", verify_backoff_s=0.0)
        app, _ = process_ref(svc, "IN-001")
        self.assertNotEqual(decision(svc, app)["status"], "REJECTED")
        self.assertEqual(decision(svc, app)["status"], "PENDING")


class TestAllThreeDecisions(unittest.TestCase):
    def test_approved_pending_rejected(self):
        svc = svc_for()
        a, _ = process_ref(svc, "IN-001")
        p, _ = process_ref(svc, "IN-003")
        r, _ = process_ref(svc, "US-010")
        self.assertEqual(decision(svc, a)["status"], "APPROVED")
        self.assertEqual(decision(svc, p)["status"], "PENDING")
        d = decision(svc, r)
        self.assertEqual((d["status"], d["decision_type"], d["decided_by"]), ("REJECTED", "AUTO", "system"))
        self.assertEqual(d["rule_trace"]["automatic_rejection"]["policy_rule_id"], "R1_CONFIRMED_INTERNAL_RECORD")
        self.assertEqual([x["status"] for x in svc.repo.list_decisions(r)], ["REJECTED"])
        self.assertIsNone(svc.repo.active_task(r))


class TestReviewerOverrides(unittest.TestCase):
    def setUp(self):
        self.svc = svc_for()

    def test_override_requires_reviewer_and_reason(self):
        app, _ = process_ref(self.svc, "UK-009")
        with self.assertRaises(ReviewError):
            self.svc.review.override(app, "", "APPROVED", "Accepting the declaration of non-registration.")
        with self.assertRaises(ReviewError) as cm:
            self.svc.review.override(app, "ana", "APPROVED", "ok")
        self.assertIn("reason", cm.exception.problems[0])
        self.assertEqual(len(self.svc.repo.list_decisions(app)), 1)

    def test_override_approves_human_judgement_issue_and_records_everything(self):
        app, _ = process_ref(self.svc, "UK-009")
        self.assertEqual(decision(self.svc, app)["reason_codes"], ["DECLARATION_NEEDS_REVIEW"])
        self.svc.review.add_note(app, "ana", "Checked gov.uk: company is below the VAT threshold.")
        self.svc.review.override(app, "ana", "APPROVED", "Declaration accepted; turnover below the VAT threshold.")
        d = decision(self.svc, app)
        self.assertEqual((d["status"], d["decision_type"], d["decided_by"]), ("APPROVED", "OVERRIDE", "ana"))
        self.assertEqual(d["rule_trace"]["resolved_issue_keys"][0].split(":")[0], "DECLARATION_NEEDS_REVIEW")
        hist = audit_history(self.svc.repo, app)
        self.assertEqual([e["kind"] for e in hist if e["kind"] in ("NOTE", "OVERRIDE")], ["NOTE", "OVERRIDE"])
        self.assertTrue(all(e["ts"] for e in hist))

    def test_data_defects_cannot_be_overridden(self):
        for ref in ("IN-005", "IN-003", "IN-009"):
            app, _ = process_ref(self.svc, ref)
            with self.assertRaises(ReviewError, msg=ref):
                self.svc.review.override(app, "ana", "APPROVED", "I would like to approve this one anyway please.")
            self.assertEqual(decision(self.svc, app)["status"], "PENDING")

    def test_override_stands_across_reprocessing_when_no_new_issue(self):
        app, _ = process_ref(self.svc, "UK-009")
        self.svc.review.override(app, "ana", "APPROVED", "Declaration accepted after checking records.")
        did = decision(self.svc, app)["decision_id"]
        self.svc.review.reprocess(app, "ana", "routine re-check")
        self.assertEqual(decision(self.svc, app)["decision_id"], did)
        self.assertEqual(len(self.svc.repo.list_decisions(app)), 2)

    def test_new_issue_after_override_supersedes_it_with_history_kept(self):
        app, _ = process_ref(self.svc, "UK-009")
        self.svc.review.override(app, "ana", "APPROVED", "Declaration accepted after checking records.")
        self.svc.registry.register("GB", "COMPANIES_HOUSE", LiveStub("COMPANIES_HOUSE", None, status="ACTIVE", found=False))
        self.svc.review.reprocess(app, "ana", "re-check against live source")
        statuses = [(x["status"], x["decision_type"]) for x in self.svc.repo.list_decisions(app)]
        self.assertEqual(statuses, [("PENDING", "AUTO"), ("APPROVED", "OVERRIDE"), ("PENDING", "AUTO")])
        self.assertIn("New issue", decision(self.svc, app)["explanation"])

    def test_production_blocks_overriding_verification_gaps(self):
        svc = svc_for(app_mode="PRODUCTION", verification_providers="simulated")
        app, _ = process_ref(svc, "UK-009")
        with self.assertRaises(ReviewError) as cm:
            svc.review.override(app, "ana", "APPROVED", "Declaration accepted after checking records.")
        self.assertIn("VERIFICATION_NOT_LIVE", " ".join(cm.exception.problems))

    def test_rejection_needs_documented_condition_and_evidence(self):
        svc = self.svc
        app, _ = process_ref(svc, "IN-003")
        bad = [dict(condition_code=None, evidence="x" * 30), dict(condition_code="R1", evidence=""), dict(condition_code="R2", evidence="x" * 30)]
        for kw in bad:
            with self.assertRaises(ReviewError, msg=str(kw)):
                svc.review.override(app, "ana", "REJECTED", "This vendor looks suspicious to me overall.", **kw)
        self.assertEqual(decision(svc, app)["status"], "PENDING")
        other, _ = process_ref(svc, "IN-003")                                           # no blocklist match: R1 impossible
        with self.assertRaises(ReviewError):
            svc.review.override(other, "ana", "REJECTED", "Incomplete forms are annoying to us.", condition_code="R1", evidence="x" * 30)

    def test_simulated_terminal_status_can_never_support_rejection(self):
        app, _ = process_ref(self.svc, "IN-010")                                        # SIMULATED cancelled GSTIN
        self.assertEqual(decision(self.svc, app)["reason_codes"], ["VERIFICATION_SIMULATED_NEGATIVE"])
        with self.assertRaises(ReviewError):
            self.svc.review.override(app, "ana", "REJECTED", "GST registration is cancelled so reject it.", condition_code="R2", evidence="provider response")

    def test_live_terminal_status_allows_reviewer_rejection_R2_with_stub(self):
        svc = svc_for(app_mode="PRODUCTION")
        svc.registry.register("GB", "COMPANIES_HOUSE", LiveStub("COMPANIES_HOUSE", None, status="DISSOLVED"))
        app, _ = process_ref(svc, "UK-001")
        d = decision(svc, app)
        self.assertIn("LIVE_TERMINAL_STATUS", d["reason_codes"])
        self.assertEqual(d["status"], "REJECTED")
        self.assertTrue(d["rule_trace"]["rejection_candidate"])
        self.assertEqual(d["rule_trace"]["automatic_rejection"]["policy_rule_id"], "R2_LIVE_TERMINAL_STATUS")

    def test_rejection_is_never_reversed_automatically(self):
        app, _ = process_ref(self.svc, "US-010")
        self.svc.pipeline.run_all(app, trigger="reprocess")
        self.assertEqual(decision(self.svc, app)["status"], "REJECTED")

    def test_only_pending_applications_can_be_reviewed(self):
        app, _ = process_ref(self.svc, "IN-001")
        with self.assertRaises(ReviewError):
            self.svc.review.override(app, "ana", "REJECTED", "Trying to reject an approved vendor.", condition_code="R1", evidence="x" * 30)


class TestRequestInfoAndQueue(unittest.TestCase):
    def test_request_information_keeps_pending_and_drafts_message(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-007")
        svc.review.request_info(app, "ana", "Please send a letter from the bank explaining the account holder name.")
        self.assertEqual(decision(svc, app)["status"], "PENDING")
        self.assertEqual(svc.repo.active_task(app)["state"], "AWAITING_VENDOR")
        self.assertIn("letter from the bank", svc.repo.list_notifications(app)[-1]["body"])
        self.assertIn("REQUEST_INFO", [e["kind"] for e in audit_history(svc.repo, app)])

    def test_queue_lists_open_and_awaiting_with_context(self):
        svc = svc_for()
        for ref in ("IN-001", "IN-003", "IN-007", "US-010"):
            process_ref(svc, ref)
        q = svc.review.queue()
        self.assertEqual({r["legal_name"] and r["state"] for r in q}, {"OPEN", "AWAITING_VENDOR"})
        self.assertEqual(len(q), 2)                                                     # approved IN-001 and rejected US-010 are not queued
        self.assertEqual(len(svc.review.queue(["OPEN"])), 1)


class TestReprocessingAndHistory(unittest.TestCase):
    def test_reprocessing_same_evidence_appends_nothing(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-007")
        svc.pipeline.run_all(app, trigger="reprocess")
        svc.pipeline.run_all(app, trigger="reprocess")
        self.assertEqual(len(svc.repo.list_decisions(app)), 1)
        self.assertEqual(len(svc.repo.list_runs(app)), 3)                               # the runs themselves are all preserved

    def test_vendor_correction_creates_revision_and_reprocesses(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-005")
        first = decision(svc, app)
        self.assertIn("INVALID_FORMAT", first["reason_codes"])
        bad = APPS["IN-005"]["fields"]["gstin"]
        fixed = next(bad[:-1] + c for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if ids.is_valid_gstin(bad[:-1] + c))
        svc.review.vendor_correction(app, {"gstin": fixed})
        d = decision(svc, app)
        revs = svc.repo.list_revisions(app)
        self.assertEqual([r["revision_no"] for r in revs], [1, 2])
        self.assertEqual(revs[1]["payload"]["fields"]["gstin"], fixed)
        self.assertEqual(revs[0]["payload"]["fields"]["gstin"], bad)                    # revision 1 untouched
        self.assertNotIn("INVALID_FORMAT", d["reason_codes"])                           # the typo is fixed...
        self.assertEqual(d["status"], "PENDING")                                        # ...but the documents still show the old GSTIN: not ignored
        self.assertTrue({"IDENTIFIER_MISMATCH", "LOW_CONFIDENCE_EXTRACTION"} & set(d["reason_codes"]), d["reason_codes"])
        self.assertEqual(svc.repo.list_decisions(app)[0]["decision_id"], first["decision_id"])     # the first decision is preserved
        self.assertEqual(d["rule_trace"]["revision_no"], 2)

    def test_missing_document_then_vendor_uploads_it_is_approved(self):
        svc = svc_for()
        entry = dict(APPS["IN-001"])
        entry["documents"] = [d for d in entry["documents"] if d["type"] != "tax_certificate"]
        sub = submission_from_json(entry, base_dir=DATA, source="FORM")
        sub.source_ref = None
        app = register_application(svc.repo, svc.settings, sub)
        svc.pipeline.run_all(app)
        self.assertEqual((decision(svc, app)["status"], decision(svc, app)["reason_codes"]), ("PENDING", ["MISSING_DOCUMENT"]))
        pdf = (DATA / next(d["path"] for d in APPS["IN-001"]["documents"] if d["type"] == "tax_certificate")).read_bytes()
        svc.review.vendor_correction(app, new_documents=[DocumentInput("tax_certificate", "gst_certificate.pdf", pdf)])
        self.assertEqual(decision(svc, app)["status"], "APPROVED")
        self.assertEqual([d["status"] for d in svc.repo.list_decisions(app)], ["PENDING", "APPROVED"])
        self.assertEqual(svc.repo.active_task(app), None)                               # the waiting task was closed

    def test_document_replacement_revision_and_human_corrections_carry_forward(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-009")
        # Poor scans may produce one or more reviewer-visible low-confidence
        # readings across OCR versions. Confirm every field the policy asks the
        # reviewer to inspect, rather than assuming a fixed first issue.
        for issue in svc.review.current_issues(app):
            evidence = issue["evidence"]
            truth = TRUTH["IN-009"][evidence["document"]]["fields"][evidence["field"]]
            svc.review.correct_field(app, "ana", evidence["document_id"], evidence["field"], truth,
                                     "Verified against the document image.", reevaluate=False)
        old_doc = svc.repo.list_documents(app)[0]
        new_bytes = (DATA / APPS["IN-009"]["documents"][0]["path"]).read_bytes()
        svc.review.vendor_correction(app, new_documents=[DocumentInput("registration_form", "registration_form_v2.pdf", new_bytes)])
        self.assertEqual(svc.repo.get_application(app)["current_revision"], 2)
        docs2 = svc.repo.list_documents(app, 2)
        self.assertEqual(len(docs2), len(svc.repo.list_documents(app, 1)))
        self.assertIn("registration_form_v2.pdf", [d["filename"] for d in docs2])
        carried = [f for f in svc.repo.current_fields_rev(app, 2) if f["origin"] == "HUMAN"]
        self.assertTrue(carried, "the reviewer's confirmation of an unchanged file must carry forward")
        self.assertEqual(decision(svc, app)["status"], "APPROVED")

    def test_repeated_application_by_same_vendor_is_flagged_and_history_separate(self):
        svc = svc_for()
        first, _ = process_ref(svc, "IN-001")
        sub = submission_from_json(APPS["IN-001"], base_dir=DATA, source="FORM")
        sub.source_ref = None
        second = register_application(svc.repo, svc.settings, sub)
        svc.pipeline.run_all(second)
        d = decision(svc, second)
        self.assertEqual((d["status"], d["reason_codes"]), ("PENDING", ["DUPLICATE_SUSPECTED"]))
        self.assertEqual(svc.repo.run_duplicates(svc.repo.latest_run(second)["run_id"])[0]["strength"], "STRONG")
        self.assertEqual(decision(svc, first)["status"], "APPROVED")
        self.assertEqual(len(svc.repo.list_decisions(first)), 1)

    def test_decisions_are_append_only_at_the_database_level(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-001")
        with self.assertRaises(sqlite3.DatabaseError):
            with svc.repo.tx() as c:
                c.execute("UPDATE decisions SET status='REJECTED' WHERE application_id=?", (app,))
        with self.assertRaises(sqlite3.DatabaseError):
            with svc.repo.tx() as c:
                c.execute("DELETE FROM decisions WHERE application_id=?", (app,))

    def test_sensitive_values_are_masked_in_everything_phase4_stores(self):
        svc = svc_for()
        app, _ = process_ref(svc, "IN-007")
        acct = APPS["IN-007"]["fields"]["bank_account_number"]
        blob = ""
        for table in ("verification_results", "consistency_results", "risk_indicators", "decisions", "duplicate_matches", "notifications", "validation_results",
                      "api_audit_log", "processing_logs"):
            blob += json.dumps(svc.repo._all(f"SELECT * FROM {table}"), default=str)
        self.assertNotIn(acct, blob)


class TestFailureHandling(unittest.TestCase):
    def test_llm_failure_falls_back_and_decision_is_identical(self):
        base, _ = process_ref(svc_for(), "IN-001")        # (different DB: only the outcome is compared)
        down = lambda: ResponderClient("gemini", error=LlmError(LlmErrorKind.TRANSIENT, "503", 503))
        for label, make in (("all LLMs down", lambda: [down(), ResponderClient("groq", error=LlmError(LlmErrorKind.TRANSIENT, "503", 503))]),
                            ("gemini down, groq answers", lambda: [down(), ResponderClient("groq", responder=perfect_responder)])):
            svc = make_services(llm_enabled=True, app_mode="DEMO", verify_backoff_s=0.0)
            clients = make()
            svc.extraction.router = stub_router(svc.repo, svc.settings, *clients)
            app, _ = process_ref(svc, "IN-001")
            self.assertEqual(decision(svc, app)["status"], "APPROVED", label)
            self.assertTrue(sum(c.calls for c in clients) > 0, label)                 # the LLM path really was exercised

    def test_corrupted_pdf_leads_to_pending_with_clear_action(self):
        svc = svc_for()
        sub = submission_from_json({"country": "IN", "fields": APPS["IN-001"]["fields"]})
        sub.documents = [DocumentInput("bank_proof", "broken.pdf", b"%PDF-1.4 this is not really a pdf")]
        app = register_application(svc.repo, svc.settings, sub)
        events = svc.pipeline.run_all(app)
        d = decision(svc, app)
        self.assertEqual(d["status"], "PENDING")
        self.assertIn("UNREADABLE_DOCUMENT", d["reason_codes"])
        self.assertTrue(any("readable" in a for a in d["required_actions"]))
        self.assertEqual(events[-1].stage, "DONE")

    def test_evaluation_crash_records_no_decision(self):
        svc = svc_for()
        app = register_synthetic(svc, "IN-001")
        svc.pipeline.stages.verifier.iter_run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        events = svc.pipeline.run_all(app)
        self.assertIsNone(decision(svc, app))                                          # a crash is never an approval or rejection
        self.assertTrue(any(e.status == "error" for e in events))
        self.assertEqual(svc.repo.list_runs(app)[-1]["status"], "COMPLETED_WITH_ERRORS")


class TestStatusSemantics(unittest.TestCase):
    def test_live_http_ok_with_name_mismatch_is_not_approval_stub(self):
        svc = svc_for(app_mode="PRODUCTION")
        for ct in ("PAN_VERIFY", "GSTIN_VERIFY"):
            svc.registry.register("IN", ct, LiveStub(ct, None))
        svc.registry.register("IN", "BANK_VERIFY", LiveStub("BANK_VERIFY", "Some Other Company Pvt Ltd"))      # answers 'found' but a different holder
        app, _ = process_ref(svc, "IN-001")
        d = decision(svc, app)
        self.assertEqual((d["status"], d["reason_codes"]), ("PENDING", ["NAME_MISMATCH_BANK"]))

    def test_live_not_found_is_pending_not_rejected(self):
        svc = svc_for(app_mode="PRODUCTION")
        svc.registry.register("IN", "PAN_VERIFY", LiveStub("PAN_VERIFY", None, found=False))
        for ct in ("GSTIN_VERIFY", "BANK_VERIFY"):
            svc.registry.register("IN", ct, LiveStub(ct, None))
        app, _ = process_ref(svc, "IN-001")
        self.assertEqual((decision(svc, app)["status"], decision(svc, app)["reason_codes"]), ("PENDING", ["PROVIDER_NOT_FOUND"]))


if __name__ == "__main__":
    unittest.main()
