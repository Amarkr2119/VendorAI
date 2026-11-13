"""Integration tests: real synthetic documents through the real pipeline (PDF -> OCR -> extraction -> SQLite)."""
import json
import re
import tempfile
import unittest
from pathlib import Path

from tests.helpers import (DATA, FaultyOcr, ResponderClient, hallucinating_responder, load_truth, make_services, perfect_responder,
                           process_ref, stub_router, temp_settings)
from vendorflow.db.repositories import Repository
from vendorflow.domain.fields import normalize_value
from vendorflow.domain.models import DocumentInput
from vendorflow.evaluation import score_against_truth
from vendorflow.extraction import pdf_text
from vendorflow.extraction.ocr import get_ocr_engine
from vendorflow.ingestion.intake import register_application, submission_from_json
from vendorflow.ingestion.normalize import normalize_fields
from vendorflow.ingestion.pdf_form import submission_from_pdf_form
from vendorflow.llm.base import LlmError, LlmErrorKind
from vendorflow.logging_setup import setup_logging

TRUTH = load_truth()
APPS = {a["ref"]: a for a in json.loads((DATA / "applications.json").read_text())}


def fields_of(svc, app_id, doc_name=None):
    docs = {d["document_id"]: d["filename"] for d in svc.repo.list_documents(app_id)}
    rows = svc.repo.current_fields(app_id)
    return [dict(r, filename=docs[r["document_id"]]) for r in rows if doc_name in (None, docs[r["document_id"]])]


def one(rows, name):
    return [r for r in rows if r["field_name"] == name][0]


class TestIngestionPaths(unittest.TestCase):
    def test_AT05_pdf_form_gives_same_canonical_submission_as_json_for_all_29(self):
        be, ocr = pdf_text.get_backend("auto"), None            # registration forms are machine-readable: no OCR needed
        for ref, a in APPS.items():
            with self.subTest(ref=ref):
                form = DATA / "documents" / ref / "registration_form.pdf"
                sub = submission_from_pdf_form(form.read_bytes(), "registration_form.pdf", backend=be, ocr=ocr)
                from_json = submission_from_json(a, base_dir=DATA)
                self.assertEqual(sub.country, from_json.country)
                d_pdf, n_pdf = normalize_fields(sub.country, sub.fields)
                d_json, n_json = normalize_fields(from_json.country, from_json.fields)
                self.assertEqual(n_pdf, n_json)                  # every field identical after normalization
                self.assertEqual(sub.declarations, from_json.declarations)

    def test_pdf_form_rejects_unreadable_or_countryless_input(self):
        be = pdf_text.get_backend("auto")
        with self.assertRaises(ValueError) as cm:
            submission_from_pdf_form(b"garbage", "x.pdf", backend=be, ocr=None)
        self.assertIn("could not be read", str(cm.exception))
        # a genuine, readable PDF that carries no country evidence at all
        from reportlab.pdfgen import canvas
        import io
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(50, 800, "Minutes of the quarterly facilities meeting held in the main conference room.")
        c.drawString(50, 780, "Action items were assigned to the facilities team and reviewed by the committee.")
        c.save()
        with self.assertRaises(ValueError) as cm:
            submission_from_pdf_form(buf.getvalue(), "minutes.pdf", backend=be, ocr=None)
        self.assertIn("country", str(cm.exception))

    def test_registered_pdf_form_application_runs_through_the_pipeline(self):
        svc = make_services(llm_enabled=False)
        be = svc.extraction.backend
        folder = DATA / "documents" / "UK-001"
        others = [DocumentInput(t, f"{t}.pdf", (folder / f"{t}.pdf").read_bytes()) for t in ("registration_certificate", "tax_certificate", "bank_proof")]
        sub = submission_from_pdf_form((folder / "registration_form.pdf").read_bytes(), "registration_form.pdf", others, backend=be, ocr=None, source_ref="UK-001")
        app_id = register_application(svc.repo, svc.settings, sub)
        self.assertEqual(svc.repo.get_revision(app_id)["source"], "PDF_FORM")
        done = svc.pipeline.run_all(app_id)[-1]
        self.assertEqual(done.detail["state"], "EXTRACTED")


class TestTextLayerAndOcr(unittest.TestCase):
    def test_AT07_text_layer_vs_scanned_detection(self):
        svc = make_services(llm_enabled=False)
        app_id, _ = process_ref(svc, "IN-002")
        docs = {d["filename"]: d for d in svc.repo.list_documents(app_id)}
        self.assertEqual(docs["tax_certificate.pdf"]["is_scanned"], 0)
        self.assertGreater(docs["tax_certificate.pdf"]["text_chars"], 100)
        self.assertEqual(docs["bank_proof.pdf"]["is_scanned"], 1)                 # clean scan: image-only PDF
        self.assertEqual(docs["bank_proof.pdf"]["page_count"], 1)

    @unittest.skipIf(get_ocr_engine("tesseract") is None, "Tesseract not installed")
    def test_AT08_clean_scan_is_read_by_ocr_with_high_confidence(self):
        svc = make_services(llm_enabled=False, ocr_engine="tesseract")
        app_id, _ = process_ref(svc, "IN-002")
        rows = fields_of(svc, app_id, "bank_proof.pdf")
        truth = TRUTH["IN-002"]["bank_proof.pdf"]["fields"]
        for name, tv in truth.items():
            r = one(rows, name)
            self.assertEqual(normalize_value("IN", name, r["value"]), normalize_value("IN", name, tv), name)
            self.assertNotEqual(r["confidence_band"], "LOW", name)
            self.assertEqual(r["source_kind"], "OCR")

    def test_AT08_without_ocr_scanned_documents_are_flagged_unreadable_never_guessed(self):
        svc = make_services(llm_enabled=False, ocr=None)
        app_id, events = process_ref(svc, "IN-002")
        self.assertEqual(fields_of(svc, app_id, "bank_proof.pdf"), [])
        findings = svc.repo.latest_findings(svc.repo.list_runs(app_id)[-1]["run_id"])
        self.assertTrue(any(f["reason_code"] == "UNREADABLE_DOCUMENT" and "bank_proof" in f["rule_id"] for f in findings))
        self.assertEqual(events[-1].detail["state"], "NEEDS_ATTENTION")            # surfaced, not crashed, not silently skipped
        self.assertTrue(any("OCR" in " ".join(e.detail.get("notes", [])) for e in events))        # the reason is stated

    def test_unreadable_garbage_scan_is_never_a_confident_wrong_value(self):
        import sys
        sys.path.insert(0, str(DATA.parents[1] / "scripts"))
        import generate_dataset as gd
        svc = make_services(llm_enabled=False)
        tmp = Path(tempfile.mkdtemp())
        src = DATA / "documents" / "IN-001" / "bank_proof.pdf"
        img = gd.degrade(gd.pdf_to_image(src, 200), "bad", 5)
        png = tmp / "bad.png"
        img.save(png)
        sub = submission_from_json({"country": "IN", "fields": APPS["IN-001"]["fields"], "documents": []})
        sub.documents = [DocumentInput("bank_proof", "bad.png", png.read_bytes())]
        app_id = register_application(svc.repo, svc.settings, sub)
        svc.pipeline.run_all(app_id)
        truth = TRUTH["IN-001"]["bank_proof.pdf"]["fields"]
        for r in fields_of(svc, app_id, "bad.png"):
            right = normalize_value("IN", r["field_name"], r["value"] or "") == normalize_value("IN", r["field_name"], truth.get(r["field_name"], ""))
            self.assertTrue(right or r["confidence_band"] == "LOW", f"confident wrong value: {r['field_name']}={r['value']!r}")


class TestClassificationAndWrongDocument(unittest.TestCase):
    def test_AT09_all_122_documents_classified_correctly(self):
        be, ocr = pdf_text.get_backend("auto"), get_ocr_engine("auto")
        from vendorflow.extraction.doc_classifier import classify
        wrong, n = [], 0
        for ref, docs in TRUTH.items():
            for fname, info in docs.items():
                data = (DATA / "documents" / ref / fname).read_bytes()
                c = pdf_text.read_document(data, fname, be)
                text = c.pages[0].text if not c.pages[0].needs_ocr else ocr.recognize(pdf_text.render_page(data, fname, 0, be)).text
                n += 1
                if classify(text)[0] != info["actual_type"]:
                    wrong.append((ref, fname))
        self.assertEqual(n, 122)
        self.assertEqual(wrong, [])

    def test_AT09_bank_letter_in_the_w9_slot_raises_wrong_document_type(self):
        svc = make_services(llm_enabled=False)
        app_id, events = process_ref(svc, "US-003")
        findings = {f["rule_id"]: f for f in svc.repo.latest_findings(svc.repo.list_runs(app_id)[-1]["run_id"])}
        self.assertEqual(findings["DOCUMENT.WRONG_TYPE.tax_certificate.pdf"]["reason_code"], "WRONG_DOCUMENT_TYPE")
        self.assertEqual(findings["COMPLETENESS.DOC.tax_certificate"]["result"], "FAIL")      # effectively missing after extraction
        self.assertEqual(events[-1].detail["state"], "NEEDS_ATTENTION")


class TestLlmPath(unittest.TestCase):
    def _svc(self, *clients, ocr="__default__", **kw):
        s = temp_settings(llm_enabled=True, llm_min_interval_s=0.0, **kw)
        repo = Repository(s.db_path)
        repo.init_schema()
        from vendorflow.services import build_services
        extra = {} if ocr == "__default__" else {"ocr": ocr}
        return build_services(s, router=stub_router(repo, s, *clients), **extra)

    def test_agreeing_llm_and_regex_give_high_confidence_with_llm_origin(self):
        c = ResponderClient()
        svc = self._svc(c)
        app_id, _ = process_ref(svc, "IN-001")
        r = one(fields_of(svc, app_id, "tax_certificate.pdf"), "gstin")
        self.assertEqual((r["origin"], r["regex_agrees"], r["grounded"], r["confidence_band"]), ("TEXT_LAYER_LLM", 1, 1, "HIGH"))
        self.assertEqual(c.calls, 5)                                                # one call per document
        run = svc.repo.list_runs(app_id)[-1]["run_id"]
        self.assertTrue(all(a["status"] == "OK" for a in svc.repo.list_audit(run)))

    def test_AT13_second_run_makes_zero_llm_calls_thanks_to_the_cache(self):
        c = ResponderClient()
        svc = self._svc(c)
        app_id, _ = process_ref(svc, "IN-001")
        first = c.calls
        svc.pipeline.run_all(app_id)
        self.assertEqual(c.calls, first)
        self.assertEqual(len(svc.repo._all("SELECT * FROM extraction_cache")), first)

    def test_llm_mode_when_needed_skips_clean_text_documents(self):
        c = ResponderClient()
        svc = self._svc(c, llm_mode="when_needed")
        process_ref(svc, "IN-001")
        self.assertEqual(c.calls, 0)

    def test_hallucinated_correction_is_caught_by_grounding_and_flagged(self):
        from vendorflow.validation.identifiers import gstin_check_char, is_valid_gstin
        fake = "27AABCK1234F1Z" + gstin_check_char("27AABCK1234F1Z")
        self.assertTrue(is_valid_gstin(fake))                                      # format cannot save us: only grounding can
        svc = self._svc(ResponderClient(responder=hallucinating_responder("gstin", fake)))
        app_id, _ = process_ref(svc, "IN-001")
        r = one(fields_of(svc, app_id, "tax_certificate.pdf"), "gstin")
        printed = TRUTH["IN-001"]["tax_certificate.pdf"]["fields"]["gstin"]
        self.assertEqual(r["value"], printed)                                      # the value actually in the document wins
        self.assertEqual(r["confidence_band"], "LOW")                              # but the disagreement is surfaced to a human
        self.assertIn("EXTRACTORS_DISAGREE", json.loads(r["flags_json"]))
        alts = json.loads(r["alternatives_json"])
        self.assertTrue(any(a["source"] == "llm" and a["value"] == fake for a in alts))

    def test_llm_only_value_with_fabricated_snippet_is_ungrounded_and_low(self):
        """The LLM is the ONLY source (regex finds nothing). It returns the right-looking address but quotes text that is not
        in the document. Nothing else can reject it; grounding must."""
        truth = TRUTH["IN-009"]["tax_certificate.pdf"]["fields"]
        text = (f"GST REGISTRATION CERTIFICATE\nForm GST REG-06 (synthetic issuer)\nRegistration Number: {truth['gstin']}\n"
                f"Legal Name: {truth['legal_name']}\nTrade Name: {truth['trade_name']}\n")      # no address line in the OCR text

        def respond(user, system=""):
            data = json.loads(perfect_responder(user))
            data["fields"]["address"] = {"value": truth["address"], "snippet": "Address: " + truth["address"], "page": 1}
            return json.dumps(data)

        svc = self._svc(ResponderClient(responder=respond), ocr=FaultyOcr(text))
        app_id, _ = process_ref(svc, "IN-009")
        r = one(fields_of(svc, app_id, "tax_certificate.pdf"), "address")
        self.assertEqual(r["value"], truth["address"])
        self.assertEqual((r["grounded"], r["regex_agrees"], r["form_agrees"]), (0, None, 1))   # form agrees, yet it is ungrounded
        self.assertEqual(r["confidence_band"], "LOW")
        self.assertIn("UNGROUNDED", json.loads(r["flags_json"]))

    def test_all_llm_providers_failing_falls_back_to_regex_and_audits_every_attempt(self):
        g = ResponderClient("gemini", error=LlmError(LlmErrorKind.QUOTA, "429 quota", 429))
        q = ResponderClient("groq", error=LlmError(LlmErrorKind.AUTH, "bad key", 401))
        svc = self._svc(g, q)
        app_id, events = process_ref(svc, "IN-001")
        self.assertEqual(events[-1].detail["state"], "EXTRACTED")                  # still fully functional
        r = one(fields_of(svc, app_id, "tax_certificate.pdf"), "gstin")
        self.assertEqual((r["origin"], r["confidence_band"]), ("REGEX", "HIGH"))
        audit = svc.repo.list_audit()
        self.assertTrue(any(a["provider"] == "gemini" and a["status"] == "QUOTA" for a in audit))
        self.assertTrue(any(a["provider"] == "groq" and a["status"] == "AUTH" for a in audit))
        self.assertTrue(any(a["status"] == "SKIPPED" for a in audit))              # later docs skip the cooling/disabled providers
        self.assertTrue(any("LLM unavailable" in str(e.detail.get("notes")) for e in events))

    def test_quota_on_primary_is_served_by_fallback_and_recorded(self):
        g = ResponderClient("gemini", error=LlmError(LlmErrorKind.QUOTA, "429", 429))
        q = ResponderClient("groq")
        svc = self._svc(g, q)
        app_id, _ = process_ref(svc, "IN-001")
        self.assertEqual(g.calls, 1)
        self.assertEqual(q.calls, 5)
        self.assertTrue(any(a["provider"] == "groq" and a["fallback_from"] == "gemini" and a["status"] == "OK" for a in svc.repo.list_audit()))
        self.assertEqual(one(fields_of(svc, app_id, "tax_certificate.pdf"), "gstin")["origin"], "TEXT_LAYER_LLM")


class TestEdgeCase6Deterministic(unittest.TestCase):
    """Edge case 6 on any machine: OCR returns a specific misread; the system must not trust it."""

    def test_misread_gstin_on_a_scan_is_low_confidence_and_other_fields_stay_high(self):
        truth = TRUTH["IN-009"]["tax_certificate.pdf"]["fields"]
        misread = truth["gstin"].replace("Z1Z7", "21Z7")                          # classic OCR confusion Z -> 2
        text = (f"GST REGISTRATION CERTIFICATE\nForm GST REG-06 (synthetic issuer)\nRegistration Number: {misread}\n"
                f"Legal Name: {truth['legal_name']}\nTrade Name: {truth['trade_name']}\n"
                f"Address of Principal Place of Business: {truth['address']}\n")
        svc = make_services(llm_enabled=False, ocr=FaultyOcr(text))
        app_id, events = process_ref(svc, "IN-009")
        rows = fields_of(svc, app_id, "tax_certificate.pdf")
        g = one(rows, "gstin")
        self.assertEqual(g["value"], misread)
        self.assertEqual(g["confidence_band"], "LOW")
        flags = set(json.loads(g["flags_json"]))
        self.assertTrue({"FORMAT_INVALID", "DIFFERS_FROM_FORM"} <= flags, flags)
        self.assertEqual(g["format_valid"], 0)
        self.assertEqual(g["form_agrees"], 0)
        self.assertEqual(one(rows, "legal_name")["confidence_band"], "HIGH")
        self.assertIn("EXTRACTION", [f["category"] for f in svc.repo.latest_findings(svc.repo.list_runs(app_id)[-1]["run_id"]) if f["result"] == "FAIL"])
        self.assertEqual(events[-1].detail["state"], "NEEDS_ATTENTION")


class TestFullDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = make_services(llm_enabled=False)
        cls.events = {}
        for ref in APPS:
            cls.events[ref] = process_ref(cls.svc, ref)

    def test_AT14_all_29_process_without_error_and_no_confident_wrong_values(self):
        for ref, (app_id, events) in self.events.items():
            self.assertNotEqual(events[-1].detail["state"], "FAILED", ref)
            self.assertNotIn("FAILED", [r["status"] for r in self.svc.repo.list_runs(app_id)], ref)
        rows = score_against_truth(self.svc.repo, TRUTH)
        self.assertEqual(len(rows), 672)
        self.assertEqual([r for r in rows if r["verdict"] == "CONFIDENT_WRONG"], [])

    def test_AT14_clean_documents_extract_with_perfect_accuracy(self):
        rows = [r for r in score_against_truth(self.svc.repo, TRUTH) if r["scan"] == "none"]
        bad = [(r["ref"], r["document"], r["field"], r["verdict"]) for r in rows if r["verdict"] != "CORRECT" or r["band"] == "LOW"]
        self.assertEqual(bad, [])
        self.assertGreater(len(rows), 550)

    def test_honestly_invalid_identifiers_are_extracted_faithfully_and_not_flagged_low(self):
        for ref, doc, field in [("IN-005", "tax_certificate.pdf", "gstin"), ("US-004", "tax_certificate.pdf", "ein"),
                                ("US-005", "bank_proof.pdf", "routing_number"), ("UK-004", "tax_certificate.pdf", "vat_number")]:
            with self.subTest(ref=ref):
                app_id = self.events[ref][0]
                r = one(fields_of(self.svc, app_id, doc), field)
                self.assertEqual(r["format_valid"], 0)
                self.assertEqual(r["form_agrees"], 1)
                self.assertNotEqual(r["confidence_band"], "LOW")

    def test_AT15_poor_scans_are_correct_or_flagged_never_confidently_wrong(self):
        poor = [r for r in score_against_truth(self.svc.repo, TRUTH) if r["scan"] == "poor"]
        self.assertGreater(len(poor), 8)
        for r in poor:
            self.assertIn(r["verdict"], {"CORRECT", "FLAGGED_WRONG", "MISSING"}, r)
            if r["verdict"] == "MISSING":
                self.assertTrue(r["band"] == "" and r["extracted"] == "")

    def test_manifest_scenarios_needing_attention_match_expected_edge_cases(self):
        attention = {ref for ref, (_, ev) in self.events.items() if ev[-1].detail["state"] == "NEEDS_ATTENTION"}
        must = {"IN-003", "IN-004", "US-003", "UK-003"}                         # missing info / wrong document (deterministic)
        self.assertTrue(must <= attention, must - attention)
        clean = {"IN-001", "IN-002", "US-001", "US-002", "UK-001", "UK-002"}
        self.assertEqual(clean & attention, set())

    def test_AT16_reprocessing_is_idempotent_and_preserves_history(self):
        svc = make_services(llm_enabled=False)
        app_id, _ = process_ref(svc, "IN-001")
        docs_before = len(svc.repo.list_documents(app_id))
        fields_run1 = len(svc.repo.run_fields(svc.repo.list_runs(app_id)[-1]["run_id"]))
        logs_before = len(svc.repo.list_logs(app_id))
        svc.pipeline.run_all(app_id, trigger="reprocess")
        runs = svc.repo.list_runs(app_id)
        self.assertEqual([r["trigger"] for r in runs], ["initial", "reprocess"])
        self.assertEqual(len(svc.repo.list_documents(app_id)), docs_before)           # no duplicate documents
        self.assertEqual(len(svc.repo.run_fields(runs[0]["run_id"])), fields_run1)    # earlier rows untouched
        self.assertGreater(len(svc.repo.list_logs(app_id)), logs_before)              # history grew
        self.assertEqual(len(svc.repo.current_fields(app_id)), fields_run1)           # current view is still one value per field
        self.assertEqual(json.loads(runs[1]["config_snapshot_json"])["pipeline_version"], "phase4-verification-decision")  # Phase 4 intentionally bumped the version


class TestMaskingAndLogging(unittest.TestCase):
    def test_AT17_full_account_numbers_never_reach_logs_or_audit_tables(self):
        import logging
        from vendorflow.domain.models import Finding
        from vendorflow.logging_setup import get_logger
        tmp = Path(tempfile.mkdtemp())
        log_file = tmp / "app.log"
        svc = make_services(llm_enabled=False, log_file=log_file)
        setup_logging("DEBUG", log_file)
        repo = svc.repo
        app_id, _ = process_ref(svc, "IN-009")                                     # real pipeline run
        account = APPS["IN-009"]["fields"]["bank_account_number"]
        self.assertEqual(len(account), 12)
        run = repo.list_runs(app_id)[-1]["run_id"]
        # Deliberately push the account number through EVERY sink; each must mask it.
        repo.log_stage(run, app_id, "TEST", "ok", f"paid to account {account}", {"bank_account_number": account, "n": f"x {account}"},
                       error=f"failed for {account}")
        repo.audit(run_id=run, category="VERIFICATION", provider="p", status="ERR", request={"bank_account_number": account, "memo": account},
                   error=f"timeout for account {account}")
        repo.insert_findings(run, app_id, [Finding("TEST.MASK", "TEST", "FAIL", observed=account, message=f"account {account}")])
        get_logger("test").info("processing payment to %s", account)
        for h in logging.getLogger("vendorflow").handlers:
            h.flush()
        texts = {"log file": log_file.read_text(encoding="utf-8", errors="ignore")}
        for table, cols in {"processing_logs": ["summary", "detail_json", "error_message"], "api_audit_log": ["request_masked", "error"],
                            "validation_results": ["observed_masked", "message"]}.items():
            texts[table] = " ".join(str(v) for row in repo._all(f"SELECT {', '.join(cols)} FROM {table}") for v in row.values() if v)
        for name, text in texts.items():
            self.assertNotIn(account, text, f"full account number leaked into {name}")
            self.assertTrue(len(text) > 0, name)
        masked = "*" * 8 + account[-4:]
        for name in ("log file", "processing_logs", "api_audit_log", "validation_results"):
            self.assertIn(masked, texts[name], f"{name} should contain the masked form")


class TestResilience(unittest.TestCase):
    def test_missing_file_on_disk_is_flagged_not_crashed(self):
        svc = make_services(llm_enabled=False)
        app_id = register_application(svc.repo, svc.settings, submission_from_json(APPS["IN-001"], base_dir=DATA))
        Path(svc.repo.list_documents(app_id)[2]["stored_path"]).unlink()
        events = svc.pipeline.run_all(app_id)
        self.assertEqual(events[-1].detail["state"], "NEEDS_ATTENTION")
        self.assertTrue(any(f["reason_code"] == "UNREADABLE_DOCUMENT" for f in svc.repo.latest_findings(svc.repo.list_runs(app_id)[-1]["run_id"])))

    def test_corrupt_pdf_is_flagged_not_crashed(self):
        svc = make_services(llm_enabled=False)
        sub = submission_from_json({"country": "IN", "fields": APPS["IN-001"]["fields"]})
        sub.documents = [DocumentInput("bank_proof", "broken.pdf", b"%PDF-1.4 this is not really a pdf")]
        app_id = register_application(svc.repo, svc.settings, sub)
        events = svc.pipeline.run_all(app_id)
        self.assertEqual(events[-1].detail["state"], "NEEDS_ATTENTION")
        self.assertEqual(svc.repo.list_runs(app_id)[-1]["status"], "COMPLETED_WITH_ERRORS")


if __name__ == "__main__":
    unittest.main()
