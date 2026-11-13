import csv
import json
import unittest
from pathlib import Path

from tests.helpers import temp_repo, temp_settings
from vendorflow.domain.fields import canonical_field_name, normalize_country, normalize_value
from vendorflow.ingestion.completeness import check_documents, check_fields, summarize
from vendorflow.ingestion.intake import register_application, submission_from_form, submission_from_json
from vendorflow.ingestion.normalize import extract_identifiers, normalize_fields

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic"


class TestNormalization(unittest.TestCase):
    def test_AT04_table_driven(self):
        cases = [
            ("IN", "gstin", " 27aabcr1234f1zs ", "27AABCR1234F1ZS"), ("IN", "pan", "aabcr-1234-f", "AABCR1234F"),
            ("IN", "ifsc", "synb0 123456", "SYNB0123456"), ("US", "ein", "12-3456789", "123456789"),
            ("US", "routing_number", "021 000 021", "021000021"), ("GB", "sort_code", "20-45-67", "204567"),
            ("GB", "vat_number", "GB 123 4567 89", "123456789"), ("GB", "vat_number", "123456789", "123456789"),
            ("GB", "company_number", "sc 123456", "SC123456"), ("IN", "bank_account_number", "1234 5678 9012", "123456789012"),
            ("IN", "contact_email", "Accounts@X.EXAMPLE", "accounts@x.example"), ("IN", "gstin", "N/A", ""), ("IN", "gstin", None, ""),
        ]
        for country, field, raw, expected in cases:
            with self.subTest(field=field, raw=raw):
                self.assertEqual(normalize_value(country, field, raw), expected)

    def test_country_aliases(self):
        for raw, exp in [("India", "IN"), ("in", "IN"), ("USA", "US"), ("United States", "US"), ("UK", "GB"),
                         ("United Kingdom", "GB"), ("GB", "GB"), ("Germany", None), ("", None)]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_country(raw), exp)

    def test_field_aliases(self):
        self.assertEqual(canonical_field_name("GST No."), "gstin")
        self.assertEqual(canonical_field_name("Account Holder"), "bank_holder_name")
        self.assertEqual(canonical_field_name("Companies House Number"), None)  # unknown key is dropped, not guessed

    def test_company_name_variants_normalize_equal(self):
        pairs = [("Lumen Software Pvt. Ltd.", "LUMEN SOFTWARE PRIVATE LIMITED"), ("Redwood Harbor Supply LLC", "Redwood Harbor Supply Limited Liability Company"),
                 ("Marsh & Fenn Ltd", "Marsh and Fenn Limited")]
        for a, b in pairs:
            with self.subTest(a=a):
                self.assertEqual(normalize_value("IN", "legal_name", a), normalize_value("IN", "legal_name", b))
        self.assertNotEqual(normalize_value("IN", "legal_name", "ABC Technologies Pvt Ltd"), normalize_value("IN", "legal_name", "XYZ Enterprises Pvt Ltd"))

    def test_identifiers_for_duplicate_detection(self):
        disp, norm = normalize_fields("IN", {"pan": "aabcr1234f", "gstin": "27AABCR1234F1ZS", "ifsc": "synb0000001", "bank_account_number": "123456789012"})
        ids = {(i["id_type"], i["value_norm"]) for i in extract_identifiers("IN", norm, disp)}
        self.assertIn(("PAN", "AABCR1234F"), ids)
        self.assertIn(("BANK_ACCOUNT", "SYNB0000001:123456789012"), ids)


class TestIntakeCompleteness(unittest.TestCase):
    def test_blank_values_are_missing(self):
        f = check_fields("IN", {"legal_name": "  ", "pan": "N/A"}, {})
        failed = {x.field for x in f if x.result == "FAIL"}
        self.assertIn("legal_name", failed)
        self.assertIn("pan", failed)
        self.assertIn("gstin", failed)

    def test_declaration_waives_vat_but_needs_review(self):
        fields = check_fields("GB", {}, {"vat_not_registered": True})
        vat = [x for x in fields if x.field == "vat_number"][0]
        self.assertEqual((vat.result, vat.reason_code), ("SKIPPED", "DECLARATION_NEEDS_REVIEW"))
        docs = check_documents("GB", {"vat_not_registered": True}, [])
        tax = [x for x in docs if x.field == "tax_certificate"][0]
        self.assertEqual(tax.result, "SKIPPED")

    def test_wrong_type_detected_only_after_extraction(self):
        docs = [{"declared_type": t, "detected_type": t, "detected_confidence": 1.0, "filename": f"{t}.pdf"}
                for t in ["registration_certificate", "bank_proof"]]
        docs.append({"declared_type": "tax_certificate", "detected_type": "bank_proof", "detected_confidence": 1.0, "filename": "w9.pdf"})
        self.assertEqual(summarize(check_documents("US", {}, docs, use_detected=False))["missing_documents"], [])
        after = summarize(check_documents("US", {}, docs, use_detected=True))
        self.assertEqual(after["missing_documents"], ["tax_certificate"])
        self.assertEqual(after["wrong_type_documents"], ["tax_certificate"])

    def test_unknown_detection_falls_back_to_declared(self):
        docs = [{"declared_type": "bank_proof", "detected_type": None, "detected_confidence": None, "filename": "b.pdf"}]
        self.assertEqual(summarize(check_documents("US", {}, docs, use_detected=True))["wrong_type_documents"], [])


class TestAllManifestScenarios(unittest.TestCase):
    """AT-06: every synthetic application gets exactly the completeness result the manifest expects (intake level)."""

    def test_AT06_manifest_vs_actual_completeness(self):
        apps = {a["ref"]: a for a in json.loads((DATA / "applications.json").read_text())}
        with open(DATA / "manifest.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 29)
        for row in rows:
            ref = row["ref"]
            with self.subTest(ref=ref):
                a = apps[ref]
                sub = submission_from_json(a, base_dir=DATA)
                disp, _ = normalize_fields(sub.country, sub.fields)
                f_missing = set(summarize(check_fields(sub.country, disp, sub.declarations))["missing_fields"])
                docs = [{"declared_type": d.declared_type, "filename": d.filename} for d in sub.documents]
                d_missing = set(summarize(check_documents(sub.country, sub.declarations, docs))["missing_documents"])
                exp_f = set(filter(None, row["expected_missing_fields"].split(";")))
                wrong = set(filter(None, row["expected_wrong_type_documents"].split(";")))
                exp_d = set(filter(None, row["expected_missing_documents"].split(";"))) - wrong   # wrong-slot docs pass at intake
                self.assertEqual(f_missing, exp_f)
                self.assertEqual(d_missing, exp_d)

    def test_register_persists_everything(self):
        repo, settings = temp_repo(), temp_settings()
        a = [x for x in json.loads((DATA / "applications.json").read_text()) if x["ref"] == "IN-001"][0]
        sub = submission_from_json(a, base_dir=DATA)
        app_id = register_application(repo, settings, sub)
        self.assertEqual(app_id, "APP-" + app_id[4:8] + "-0001")
        rev = repo.get_revision(app_id)
        self.assertEqual(rev["payload"]["country"], "IN")
        self.assertEqual(rev["payload"]["normalized"]["gstin"], a["fields"]["gstin"])
        docs = repo.list_documents(app_id)
        self.assertEqual(len(docs), 5)
        self.assertTrue(all(Path(d["stored_path"]).exists() for d in docs))
        self.assertTrue(any(i["id_type"] == "GSTIN" for i in repo.list_identifiers(app_id)))

    def test_form_json_and_dict_give_same_canonical_submission(self):
        a = [x for x in json.loads((DATA / "applications.json").read_text()) if x["ref"] == "US-001"][0]
        s_json = submission_from_json(json.dumps({k: v for k, v in a.items() if k != "documents"}))
        s_form = submission_from_form("USA", {k.upper(): v for k, v in a["fields"].items()}, a["declarations"])
        self.assertEqual(s_json.fields, s_form.fields)
        self.assertEqual(s_json.country, s_form.country)

    def test_bad_input_is_rejected_cleanly(self):
        from vendorflow.domain.models import DocumentInput
        repo, settings = temp_repo(), temp_settings()
        with self.assertRaises(ValueError):
            submission_from_form("Germany", {})
        with self.assertRaises(ValueError):
            register_application(repo, settings, submission_from_form("IN", {}, documents=[DocumentInput("passport", "x.pdf", b"x")]))
        with self.assertRaises(ValueError):
            register_application(repo, settings, submission_from_form("IN", {}, documents=[DocumentInput("bank_proof", "x.exe", b"x")]))


if __name__ == "__main__":
    unittest.main()
