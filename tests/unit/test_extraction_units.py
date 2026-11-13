import json
import unittest

from vendorflow.config import Settings
from vendorflow.domain.fields import expected_fields
from vendorflow.extraction import pdf_text
from vendorflow.extraction.confidence import WEIGHTS, band, score_field
from vendorflow.extraction.doc_classifier import classify
from vendorflow.extraction.grounding import is_grounded
from vendorflow.extraction.llm_extractor import make_parser
from vendorflow.extraction.ocr import OcrResult
from vendorflow.extraction.regex_extractor import regex_candidates
from vendorflow.logging_setup import MaskingFilter
from vendorflow.utils.masking import mask_account, mask_payload, mask_text
from vendorflow.validation import identifiers as ids


class TestConfig(unittest.TestCase):
    def test_AT03_missing_keys_do_not_crash_and_providers_show_disabled(self):
        s = Settings.from_env({})
        self.assertEqual(s.app_mode, "DEMO")
        self.assertFalse(s.provider_configured("gemini"))
        self.assertFalse(s.provider_configured("groq"))

    def test_keys_need_a_model_too_and_are_not_in_repr(self):
        s = Settings.from_env({"GEMINI_API_KEY": "SECRET123", "GEMINI_MODEL": ""})
        self.assertFalse(s.provider_configured("gemini"))
        s = Settings.from_env({"GEMINI_API_KEY": "SECRET123", "GEMINI_MODEL": "m", "APP_MODE": "production"})
        self.assertTrue(s.provider_configured("gemini"))
        self.assertEqual(s.app_mode, "PRODUCTION")
        self.assertNotIn("SECRET123", repr(s))

    def test_inline_comments_and_bad_values_are_tolerated(self):
        s = Settings.from_env({"LLM_MIN_INTERVAL_S": "4   # spacing", "LLM_ENABLED": "false", "APP_MODE": "weird", "LLM_TIMEOUT_S": "abc"})
        self.assertEqual((s.llm_min_interval_s, s.llm_enabled, s.app_mode, s.llm_timeout_s), (4.0, False, "DEMO", 30.0))


class TestGrounding(unittest.TestCase):
    TEXT = "Registration Number: 27AABCK1234F1ZS\nLegal Name: Kestrel Logistics Private Limited"

    def test_AT10_grounded_and_ungrounded(self):
        ok, _ = is_grounded("27AABCK1234F1ZS", "Registration Number: 27AABCK1234F1ZS", self.TEXT)
        self.assertTrue(ok)
        ok, why = is_grounded("27AABCK1234F1ZX", "Registration Number: 27AABCK1234F1ZX", self.TEXT)   # snippet not in document
        self.assertFalse(ok)
        self.assertIn("does not appear", why)
        ok, why = is_grounded("27AABCK1234F1ZX", "Registration Number: 27AABCK1234F1ZS", self.TEXT)   # value not in its snippet
        self.assertFalse(ok)
        self.assertIn("does not contain", why)
        self.assertFalse(is_grounded("x", None, self.TEXT)[0])
        self.assertFalse(is_grounded(None, "y", self.TEXT)[0])

    def test_whitespace_and_case_tolerant(self):
        self.assertTrue(is_grounded("kestrel logistics private limited", "LEGAL NAME:  Kestrel   Logistics Private Limited", self.TEXT)[0])


class TestConfidence(unittest.TestCase):
    PERFECT = dict(grounded=True, regex_agrees=True, format_valid=True, source_quality=1.0, form_agrees=True, source_kind="TEXT_LAYER")

    def test_AT11_deterministic_and_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)
        a, b = score_field(**self.PERFECT), score_field(**self.PERFECT)
        self.assertEqual(a, b)
        self.assertEqual(a[:2], (1.0, "HIGH"))

    def test_weights_renormalize_when_signals_are_missing(self):
        score, bnd, _ = score_field(grounded=True, regex_agrees=None, format_valid=None, source_quality=1.0, form_agrees=None, source_kind="TEXT_LAYER")
        self.assertEqual((score, bnd), (1.0, "HIGH"))              # only the available signals count, all perfect

    def test_ungrounded_is_low(self):
        score, bnd, flags = score_field(**{**self.PERFECT, "grounded": False})
        self.assertEqual(bnd, "LOW")
        self.assertLessEqual(score, 0.5)
        self.assertIn("UNGROUNDED", flags)

    def test_extractors_disagree_is_low(self):
        score, bnd, flags = score_field(**{**self.PERFECT, "regex_agrees": False})
        self.assertEqual(bnd, "LOW")
        self.assertIn("EXTRACTORS_DISAGREE", flags)

    def test_invalid_format_low_unless_it_faithfully_matches_the_submitted_value(self):
        _, bnd, flags = score_field(**{**self.PERFECT, "format_valid": False, "form_agrees": None})
        self.assertEqual(bnd, "LOW")
        _, bnd, flags = score_field(**{**self.PERFECT, "format_valid": False, "form_agrees": False})
        self.assertEqual(bnd, "LOW")
        score, bnd, flags = score_field(**{**self.PERFECT, "format_valid": False, "form_agrees": True})
        self.assertNotEqual(bnd, "LOW")                            # faithful extraction of the vendor's own invalid value
        self.assertIn("FORMAT_INVALID", flags)                     # ...but still flagged for the validator

    def test_form_disagreement_caps_ocr_harder_than_text_layer(self):
        _, bnd_ocr, _ = score_field(**{**self.PERFECT, "form_agrees": False, "source_kind": "OCR"})
        _, bnd_txt, _ = score_field(**{**self.PERFECT, "form_agrees": False, "source_kind": "TEXT_LAYER"})
        self.assertEqual(bnd_ocr, "LOW")
        self.assertEqual(bnd_txt, "MEDIUM")                        # text layer cannot have an OCR error: a real vendor inconsistency

    def test_bands(self):
        self.assertEqual([band(0.9), band(0.85), band(0.7), band(0.6), band(0.59)], ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW"])


class TestClassifier(unittest.TestCase):
    def test_titles(self):
        cases = {"GST REGISTRATION CERTIFICATE\nForm GST REG-06": "tax_certificate", "FORM W-9\nRequest for Taxpayer": "tax_certificate",
                 "VAT REGISTRATION CERTIFICATE\nHMRC (synthetic issuer)": "tax_certificate",
                 "CERTIFICATE OF INCORPORATION\nMinistry of Corporate Affairs": "registration_certificate",
                 "ARTICLES OF ORGANIZATION\nState of Oregon": "registration_certificate",
                 "BANK ACCOUNT CONFIRMATION LETTER\nIssued by the bank": "bank_proof",
                 "INCOME TAX DEPARTMENT - PERMANENT ACCOUNT NUMBER CARD\nGovernment of India": "tax_id_proof",
                 "VENDOR REGISTRATION FORM\nSupplier onboarding": "registration_form"}
        for text, expected in cases.items():
            with self.subTest(text=text[:30]):
                self.assertEqual(classify(text)[0], expected)

    def test_ocr_noise_merged_words_still_classify(self):
        self.assertEqual(classify("VATREGISTRATIONCERTIFICATE = 3 :\nO° -YRT Regentaton ember GB 577")[0], "tax_certificate")

    def test_unknown_text_is_none_not_a_guess(self):
        self.assertIsNone(classify("Invoice 4471 for consulting services rendered in October, payable in 30 days")[0])
        self.assertIsNone(classify("")[0])

    def test_footer_text_cannot_change_the_type(self):
        body = "BANK ACCOUNT CONFIRMATION LETTER\nBank Name: X\n" + ("filler line\n" * 40) + "Doc Ref: registration_certificate VAT REGISTRATION CERTIFICATE"
        self.assertEqual(classify(body)[0], "bank_proof")


class TestRegexExtractor(unittest.TestCase):
    def test_labels_patterns_and_dedup(self):
        text = "GST REGISTRATION CERTIFICATE\nRegistration Number: 27JUFCK9755F1ZR\nLegal Name: Kestrel Logistics Private Limited\nnote 27JUFCK9755F1ZR again"
        c = regex_candidates(text, expected_fields("IN", "tax_certificate"))
        self.assertEqual([x.value for x in c["gstin"]], ["27JUFCK9755F1ZR"])          # label + pattern hit deduplicated
        self.assertEqual(c["legal_name"][0].value, "Kestrel Logistics Private Limited")
        self.assertEqual(c["address"], [])

    def test_label_scoped_to_expected_fields(self):
        # 'Name' means legal_name on a PAN card, but 'Bank Name' must never be read as legal_name
        c = regex_candidates("Name: Acme Pvt Ltd\nBank Name: Synthetic National Bank", expected_fields("IN", "tax_id_proof"))
        self.assertEqual(c["legal_name"][0].value, "Acme Pvt Ltd")

    def test_pattern_scan_finds_unlabelled_identifier(self):
        c = regex_candidates("blah blah AABCR1234F more text", expected_fields("IN", "tax_id_proof"))
        self.assertEqual(c["pan"][0].kind, "pattern")


class TestLlmParser(unittest.TestCase):
    P = staticmethod(make_parser(["pan", "gstin"]))

    def test_valid_fenced_and_surrounded_json(self):
        good = '{"document_type_guess": "tax_certificate", "fields": {"pan": {"value": "AABCR1234F", "snippet": "PAN: AABCR1234F", "page": 1}, "gstin": null}}'
        for text in (good, "```json\n" + good + "\n```", "Here you go: " + good + " Thanks"):
            with self.subTest(text=text[:12]):
                r = self.P(text)
                self.assertEqual(r.items["pan"].value, "AABCR1234F")
                self.assertIsNone(r.items["gstin"].value)

    def test_unknown_fields_are_dropped_not_trusted(self):
        r = self.P('{"fields": {"pan": {"value": "X", "snippet": "X"}, "approved": {"value": "yes"}}}')
        self.assertNotIn("approved", r.items)

    def test_unusable_output_raises_value_error(self):
        for bad in ("not json", "[1,2]", '{"nofields": 1}', '{"fields": {"pan": "AABCR1234F"}}', ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.P(bad)


class TestPdfBackends(unittest.TestCase):
    def test_fallback_when_pymupdf_missing(self):
        orig = pdf_text.PyMuPdfBackend

        class Missing:
            def __init__(self):
                raise pdf_text.BackendUnavailable("PyMuPDF (fitz) is not installed")

        pdf_text.PyMuPdfBackend = Missing
        try:
            self.assertEqual(pdf_text.get_backend("auto").name, "pypdfium2")
            with self.assertRaises(pdf_text.BackendUnavailable):
                pdf_text.get_backend("pymupdf")
        finally:
            pdf_text.PyMuPdfBackend = orig

    def test_corrupt_pdf_returns_error_not_exception(self):
        c = pdf_text.read_document(b"this is not a pdf", "x.pdf", pdf_text.get_backend("auto"))
        self.assertTrue(c.error)
        self.assertEqual(c.page_count, 0)
        c = pdf_text.read_document(b"not an image", "x.png", pdf_text.get_backend("auto"))
        self.assertTrue(c.error)

    @unittest.skipUnless("pymupdf" in pdf_text.available_backends(), "PyMuPDF not installed here; this test runs on your machine")
    def test_pymupdf_matches_pdfium_on_a_synthetic_pdf(self):
        from tests.helpers import DATA
        data = (DATA / "documents" / "IN-001" / "tax_certificate.pdf").read_bytes()
        a = pdf_text.PyMuPdfBackend().read(data)
        b = pdf_text.PdfiumBackend().read(data)
        self.assertEqual(a.page_count, b.page_count)
        self.assertIn("27", a.pages[0].text)
        self.assertFalse(a.is_scanned)


class TestOcrConfidenceLookup(unittest.TestCase):
    def test_word_level_confidence_for_a_value(self):
        r = OcrResult("x", [("Kestrel", 0.95), ("Logistics", 0.90), ("GSTIN", 0.99), ("27AABCK1234F1ZS", 0.40)], 0.8, "t")
        self.assertAlmostEqual(r.confidence_for("27AABCK1234F1ZS"), 0.40)
        self.assertAlmostEqual(r.confidence_for("Kestrel Logistics"), 0.925)
        self.assertAlmostEqual(r.confidence_for("Unknown Words"), 0.6)       # not found: penalised page mean


class TestMasking(unittest.TestCase):
    def test_account_numbers_and_keys(self):
        self.assertEqual(mask_account("123456789012"), "********9012")
        self.assertEqual(mask_account(""), "")
        self.assertEqual(mask_text("acct 123456789012, sort 20-45-67, 2026-11-10"), "acct ********9012, sort 20-45-67, 2026-11-10")
        self.assertIn("[REDACTED_KEY]", mask_text("key AIzaSyA1234567890abcdefghijklmnop"))
        self.assertIn("[REDACTED_KEY]", mask_text("Authorization: Bearer abcdefghijklmnopqrstuvwxyz"))

    def test_payload_masking(self):
        out = mask_payload({"bank_account_number": "123456789012", "nested": {"api_key": "s3cr3t", "note": "pay 987654321098"}})
        self.assertEqual(out["bank_account_number"], "********9012")
        self.assertEqual(out["nested"]["api_key"], "[REDACTED]")
        self.assertNotIn("987654321098", json.dumps(out))

    def test_logging_filter_masks_record(self):
        import logging
        rec = logging.LogRecord("x", logging.INFO, "f", 1, "account %s", ("123456789012",), None)
        MaskingFilter().filter(rec)
        self.assertNotIn("123456789012", rec.getMessage())


class TestIdentifiers(unittest.TestCase):
    def test_check_digit_algorithms(self):
        self.assertTrue(ids.is_valid_aba("021000021"))
        self.assertFalse(ids.is_valid_aba("021000022"))
        self.assertFalse(ids.is_valid_aba("12345"))
        g = "27AABCR1234F1Z" + ids.gstin_check_char("27AABCR1234F1Z")
        self.assertTrue(ids.is_valid_gstin(g))
        self.assertFalse(ids.is_valid_gstin(g[:-1] + ("A" if g[-1] != "A" else "B")))
        self.assertEqual(ids.gstin_embedded_pan(g), "AABCR1234F")
        v = "123456" + "7" + ids.uk_vat_check_digits("1234567")
        self.assertTrue(ids.is_valid_uk_vat(v[:0] + "1234567" + ids.uk_vat_check_digits("1234567")))
        self.assertTrue(ids.is_valid_uk_vat("1234567" + ids.uk_vat_check_digits("1234567", new_style=True)))
        self.assertFalse(ids.is_valid_uk_vat("123456789"))
        self.assertTrue(ids.is_valid_ein("12-3456789") and not ids.is_valid_ein("12-345678"))
        self.assertTrue(ids.is_valid_ifsc("SYNB0123456") and not ids.is_valid_ifsc("SYNB1123456"))
        self.assertTrue(ids.is_valid_pan("AABCR1234F") and not ids.is_valid_pan("AABCR12345"))
        self.assertTrue(ids.is_valid_sort_code("204567") and ids.is_valid_uk_account("12345678") and not ids.is_valid_uk_account("1234567"))

    def test_format_valid_dispatch(self):
        self.assertIsNone(ids.format_valid("IN", "legal_name", "Anything"))
        self.assertIs(ids.format_valid("US", "bank_account_number", "1234567890"), True)
        self.assertIs(ids.format_valid("GB", "bank_account_number", "1234567890"), False)


if __name__ == "__main__":
    unittest.main()
