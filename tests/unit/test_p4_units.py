"""Phase 4 unit tests: name matching, format rules, dates, the decision policy, provider adapters, retries, circuit breaker.
Everything here uses fixtures or STUB HTTP sessions. No real external API is called by any test in this repository."""
from __future__ import annotations

import itertools
import unittest
from datetime import date

from tests.helpers import make_services
from vendorflow.analysis import dates as dt
from vendorflow.analysis.duplicates import score_pair
from vendorflow.analysis.names import address_score, classify_names
from vendorflow.decision.engine import decide
from vendorflow.decision.policy import overridable
from vendorflow.domain.models import Issue
from vendorflow.services import register_synthetic
from vendorflow.validation.rules import other_country_matches, validate_formats
from vendorflow.verification.base import Adapter, Interpretation, ProviderError, VerificationRequest
from vendorflow.verification.service import VerificationService, plan_checks
from vendorflow.verification.uk import CompaniesHouseAdapter, HmrcVatAdapter
from vendorflow.config import Settings


class TestNames(unittest.TestCase):
    def test_equivalent_forms_match(self):
        for a, b in [("Marsh and Fenn Limited", "MARSH & FENN LTD"), ("Kestrel Pvt. Ltd.", "Kestrel Private Limited"),
                     ("Cobalt Ridge Systems Inc.", "Cobalt Ridge Systems Incorporated")]:
            self.assertEqual(classify_names(a, b)[0], "MATCH", (a, b))

    def test_different_companies_do_not_match(self):
        for a, b in [("Harrow Bridge Services Limited", "Westgate Ventures Limited"),
                     ("Prairie Wind Analytics LLC", "Prairie Wind Data Services LLC"),
                     ("Kestrel Logistics Private Limited", "Kestrel Freight Services Private Limited")]:
            self.assertEqual(classify_names(a, b)[0], "MISMATCH", (a, b))

    def test_same_core_different_legal_form_is_only_similar(self):
        self.assertEqual(classify_names("Acme Limited", "Acme Private Limited")[0], "SIMILAR")

    def test_address_overlap(self):
        self.assertGreaterEqual(address_score("Plot 14, Phase II, Pune 411019", "Plot 14 Phase II Industrial Area Pune 411019 Maharashtra"), 0.85)
        self.assertLess(address_score("12 Oak Street London", "9 Elm Avenue Leeds"), 0.5)


class TestFormatRules(unittest.TestCase):
    def test_in_gstin_pan_link(self):
        f = validate_formats("IN", {"pan": "JUFCK9755F", "gstin": "27JUFCK9755F1ZR"})
        self.assertTrue(all(x.result == "PASS" for x in f))
        f = validate_formats("IN", {"pan": "AAACK1111A", "gstin": "27JUFCK9755F1ZR"})
        self.assertIn("FORMAT.gstin_pan_link", [x.rule_id for x in f if x.result == "FAIL"])

    def test_invalid_formats_give_invalid_format_reason(self):
        self.assertEqual({x.reason_code for x in validate_formats("US", {"ein": "45015135"}) if x.result == "FAIL"}, {"INVALID_FORMAT"})
        self.assertEqual([x.result for x in validate_formats("GB", {"vat_number": "123456789"})], ["FAIL"])

    def test_absent_fields_are_not_format_errors(self):
        self.assertEqual(validate_formats("IN", {}), [])

    def test_bank_account_is_masked_in_finding(self):
        f = validate_formats("IN", {"bank_account_number": "1234567890123456789"})[0]      # 19 digits: invalid
        self.assertEqual(f.result, "FAIL")
        self.assertNotIn("1234567890123456789", f.observed + f.message)
        self.assertTrue(f.observed.endswith("6789") and f.observed.startswith("*"))

    def test_country_evidence(self):
        # a valid GSTIN typed where a UK VAT number belongs: the evidence points to India
        self.assertEqual(other_country_matches("GB", {"vat_number": "27JUFCK9755F1ZR"}), "IN")
        self.assertIsNone(other_country_matches("GB", {"vat_number": ""}))


class TestDates(unittest.TestCase):
    def test_ambiguous_date_flagged_only_if_wrong_both_ways(self):
        today = date(2026, 11, 13)
        self.assertFalse(dt.in_future(dt.candidates("03/09/2026"), today))          # 3 Sep or 9 Mar: past either way
        self.assertTrue(dt.in_future(dt.candidates("20/12/2027"), today))
        self.assertFalse(dt.expired(dt.candidates("01/12/2026"), today))            # Dec 1 (future) or Jan 12 (past): not wrong both ways, so not flagged
        self.assertTrue(dt.expired(dt.candidates("01/02/2020"), today))

    def test_labelled_dates_only(self):
        items = dt.labelled_dates("Valid Until: 31/12/2020\nRandom 12/12/2012 text\nDate of Incorporation: 12/03/2018")
        self.assertEqual([k for k, _, _ in items], ["EXPIRY", "OTHER"])


def V(ct, outcome, mode="SIMULATED", reason="", mandatory=True):
    return {"check_type": ct, "provider": "p", "outcome": outcome, "mode": mode, "reason": reason, "summary": "", "mandatory": mandatory}


ALL_OK = [V("A", "VERIFIED"), V("B", "VERIFIED")]


class TestPolicy(unittest.TestCase):
    def test_approved_demo_when_all_mandatory_verified(self):
        r = decide("DEMO", [], ALL_OK)
        self.assertEqual((r.status, r.reason_codes), ("APPROVED", []))
        self.assertIn("SIMULATED", r.explanation)

    def test_production_never_approves_simulated_or_sandbox(self):
        for mode in ("SIMULATED", "SANDBOX"):
            r = decide("PRODUCTION", [], [V("A", "VERIFIED", mode), V("B", "VERIFIED", "LIVE")])
            self.assertEqual(r.status, "PENDING")
            self.assertIn("VERIFICATION_NOT_LIVE", r.reason_codes)

    def test_production_approves_when_all_live(self):
        self.assertEqual(decide("PRODUCTION", [], [V("A", "VERIFIED", "LIVE"), V("B", "VERIFIED", "LIVE")]).status, "APPROVED")

    def test_unavailable_is_pending_never_invalid(self):
        r = decide("DEMO", [], [V("A", "UNAVAILABLE", reason="TIMEOUT"), V("B", "VERIFIED")])
        self.assertEqual((r.status, r.reason_codes), ("PENDING", ["VERIFICATION_UNAVAILABLE"]))
        self.assertNotIn("INVALID_FORMAT", r.reason_codes)

    def test_skipped_not_performed_in_production(self):
        r = decide("PRODUCTION", [], [V("A", "SKIPPED", "LIVE", "NOT_CONFIGURED"), V("B", "VERIFIED", "LIVE")])
        self.assertEqual((r.status, r.reason_codes), ("PENDING", ["VERIFICATION_NOT_PERFORMED"]))

    def test_skipped_for_missing_input_cannot_slip_through(self):
        r = decide("DEMO", [], [V("A", "SKIPPED", reason="MISSING_INPUT"), V("B", "VERIFIED")])    # no MISSING_FIELD issue supplied
        self.assertEqual(r.status, "PENDING")

    def test_simulated_negative_is_not_evidence(self):
        r = decide("DEMO", [], [V("A", "NEGATIVE", reason="TERMINAL_STATUS"), V("B", "VERIFIED")])
        self.assertEqual(r.reason_codes, ["VERIFICATION_SIMULATED_NEGATIVE"])
        self.assertFalse(r.trace["rejection_candidate"])

    def test_live_terminal_status_is_rejected_with_an_audit_trace(self):
        r = decide("PRODUCTION", [], [V("A", "NEGATIVE", "LIVE", "TERMINAL_STATUS"), V("B", "VERIFIED", "LIVE")])
        self.assertEqual((r.status, r.reason_codes), ("REJECTED", ["LIVE_TERMINAL_STATUS"]))
        self.assertTrue(r.trace["rejection_candidate"])
        self.assertEqual(r.trace["automatic_rejection"]["policy_rule_id"], "R2_LIVE_TERMINAL_STATUS")

    def test_waived_check_does_not_block_and_no_results_blocks(self):
        self.assertEqual(decide("DEMO", [], [V("A", "VERIFIED"), V("B", "SKIPPED", reason="WAIVED_BY_DECLARATION")]).status, "APPROVED")
        self.assertEqual(decide("DEMO", [], []).status, "PENDING")

    def test_info_issue_does_not_block(self):
        self.assertEqual(decide("DEMO", [Issue("ADDRESS_MISMATCH", "x", "m", "INFO")], ALL_OK).status, "APPROVED")

    def test_unconfirmed_blocklist_match_never_rejects_over_all_combinations(self):
        issues_pool = [Issue("BLOCKLIST_MATCH", "DQ-01", "m", "HIGH"), Issue("DUPLICATE_SUSPECTED", "APP-1", "m"), Issue("MISSING_FIELD", "pan", "m")]
        outcomes = ["VERIFIED", "NEGATIVE", "UNAVAILABLE", "SKIPPED"]
        for mode, md, oc in itertools.product(("DEMO", "PRODUCTION"), ("LIVE", "SANDBOX", "SIMULATED"), outcomes):
            for n in range(len(issues_pool) + 1):
                r = decide(mode, issues_pool[:n], [V("A", oc, md, "NOT_FOUND"), V("B", "VERIFIED", md)])
                self.assertNotEqual(r.status, "REJECTED")
                if r.status == "APPROVED":                          # APPROVED implies a clean, fully verified, accepted-mode state
                    self.assertEqual(n, 0)
                    self.assertEqual(oc, "VERIFIED")
                    self.assertTrue(mode == "DEMO" or md == "LIVE")

    def test_same_input_same_decision(self):
        a, b = decide("DEMO", [], ALL_OK), decide("DEMO", [], ALL_OK)
        self.assertEqual((a.status, a.explanation, a.trace["rules"]), (b.status, b.explanation, b.trace["rules"]))

    def test_overridable_sets(self):
        self.assertTrue(overridable("DUPLICATE_SUSPECTED", "PRODUCTION"))
        self.assertTrue(overridable("VERIFICATION_UNAVAILABLE", "DEMO"))
        self.assertFalse(overridable("VERIFICATION_UNAVAILABLE", "PRODUCTION"))
        for c in ("INVALID_FORMAT", "MISSING_DOCUMENT", "VERIFICATION_NOT_LIVE", "LIVE_TERMINAL_STATUS", "LOW_CONFIDENCE_EXTRACTION"):
            self.assertFalse(overridable(c, "DEMO"), c)


class TestDuplicateScoring(unittest.TestCase):
    def _app(self, name, ids=(), email="", addr=""):
        return {"ids": [{"id_type": t, "value_norm": v} for t, v in ids], "fields": {"legal_name": name, "contact_email": email, "address": addr}}

    def test_similar_name_alone_never_reaches_review_threshold(self):
        s, _ = score_pair(self._app("Prairie Wind Analytics LLC"), self._app("Prairie Wind Analytics Limited Liability Company"))
        self.assertLess(s, 0.5)

    def test_exact_identifier_reaches_review_threshold_even_with_different_name(self):
        s, b = score_pair(self._app("A Ltd", [("PAN", "X")]), self._app("Totally Different Ltd", [("PAN", "X")]))
        self.assertGreaterEqual(s, 0.5)
        self.assertEqual(b["exact_identifiers"], ["PAN"])

    def test_identifier_plus_name_is_strong(self):
        s, _ = score_pair(self._app("Same Ltd", [("EIN", "1")]), self._app("Same Limited", [("EIN", "1")]))
        self.assertGreaterEqual(s, 0.9)


# ----------------------------------------------------------------------------- provider adapters (STUB http)
class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._b = status, body or {}

    def json(self):
        return self._b


class StubSession:
    """Stands in for `requests`. Records calls. NOT a real API."""

    def __init__(self, get=None, post=None, exc=None):
        self._get, self._post, self.exc, self.calls = get, post, exc, []

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        if self.exc:
            raise self.exc
        return self._get

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self._post


def S(**kw):
    return Settings(db_path=None, upload_dir=None, **kw)


def REQ(ct="COMPANIES_HOUSE", ident="12345678", name="Acme Limited"):
    return VerificationRequest(ct, "GB", ident, name)


class TestUkLiveAdaptersWithStubSession(unittest.TestCase):
    def test_companies_house_active_name_match_is_verified_live(self):
        sess = StubSession(get=Resp(200, {"company_name": "ACME LIMITED", "company_status": "active"}))
        a = CompaniesHouseAdapter(S(companies_house_api_key="k"), sess)
        i = a.interpret(a.lookup(REQ(), 5), REQ())
        self.assertEqual((i.outcome, a.mode), ("VERIFIED", "LIVE"))
        self.assertEqual(sess.calls[0][2]["auth"], ("k", ""))                       # key as Basic-auth username
        self.assertTrue(sess.calls[0][1].endswith("/company/12345678"))

    def test_http_200_is_not_automatically_verified(self):
        a = CompaniesHouseAdapter(S(companies_house_api_key="k"), StubSession(get=Resp(200, {"company_name": "OTHER CO LIMITED", "company_status": "active"})))
        self.assertEqual(a.interpret(a.lookup(REQ(), 5), REQ()).reason, "NAME_MISMATCH")

    def test_dissolved_is_terminal_not_found_is_negative_not_terminal(self):
        a = CompaniesHouseAdapter(S(companies_house_api_key="k"), StubSession(get=Resp(200, {"company_name": "ACME LIMITED", "company_status": "dissolved"})))
        self.assertEqual(a.interpret(a.lookup(REQ(), 5), REQ()).reason, "TERMINAL_STATUS")
        a = CompaniesHouseAdapter(S(companies_house_api_key="k"), StubSession(get=Resp(404)))
        self.assertEqual(a.interpret(a.lookup(REQ(), 5), REQ()).reason, "NOT_FOUND")

    def test_error_statuses_map_to_provider_errors(self):
        for status, kind in [(500, "TRANSIENT"), (429, "TRANSIENT"), (401, "AUTH"), (418, "CONFIG")]:
            a = CompaniesHouseAdapter(S(companies_house_api_key="k"), StubSession(get=Resp(status)))
            with self.assertRaises(ProviderError) as cm:
                a.lookup(REQ(), 5)
            self.assertEqual(cm.exception.kind, kind, status)

    def test_sandbox_environment_labels_mode_sandbox(self):
        self.assertEqual(CompaniesHouseAdapter(S(companies_house_api_key="k", companies_house_env="sandbox"), StubSession()).mode, "SANDBOX")
        self.assertEqual(HmrcVatAdapter(S(hmrc_client_id="i", hmrc_client_secret="s", hmrc_env="sandbox"), StubSession()).mode, "SANDBOX")
        self.assertEqual(HmrcVatAdapter(S(hmrc_client_id="i", hmrc_client_secret="s", hmrc_env="live"), StubSession()).mode, "LIVE")

    def test_hmrc_fetches_token_then_looks_up(self):
        sess = StubSession(post=Resp(200, {"access_token": "T"}), get=Resp(200, {"target": {"name": "Acme Limited"}}))
        a = HmrcVatAdapter(S(hmrc_client_id="i", hmrc_client_secret="s"), sess)
        i = a.interpret(a.lookup(REQ("VAT_VERIFY", "123456789"), 5), REQ("VAT_VERIFY", "123456789"))
        self.assertEqual(i.outcome, "VERIFIED")
        self.assertEqual(sess.calls[0][0], "POST")
        self.assertEqual(sess.calls[1][2]["headers"]["Authorization"], "Bearer T")

    def test_network_exceptions_are_transient(self):
        import requests
        a = CompaniesHouseAdapter(S(companies_house_api_key="k"), StubSession(exc=requests.Timeout()))
        with self.assertRaises(ProviderError) as cm:
            a.lookup(REQ(), 5)
        self.assertEqual((cm.exception.kind, cm.exception.code), ("TRANSIENT", "TIMEOUT"))


class FlakyAdapter(Adapter):
    """Stub adapter: raises the given errors in order, then answers. Counts calls."""
    name, check_type, mode = "flaky", "GSTIN_VERIFY", "LIVE"

    def __init__(self, errors=(), answer=None, mode="LIVE"):
        self.errors, self.calls, self.mode = list(errors), 0, mode
        self.answer = answer or {"found": True, "name": "X", "status": "ACTIVE"}

    def lookup(self, req, timeout):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.answer


class TestVerificationService(unittest.TestCase):
    def setUp(self):
        self.svc = make_services(app_mode="DEMO", verify_backoff_s=0.0)
        self.app = register_synthetic(self.svc, "IN-001")
        self.run = self.svc.repo.start_run(self.app, 1, "initial", {})
        self.payload = self.svc.repo.get_revision(self.app)["payload"]

    def _go(self):
        vs = VerificationService(self.svc.repo, self.svc.settings, self.svc.registry, sleep=lambda s: None)
        p = self.payload
        return {r.check_type: r for r in vs.run(self.run, self.app, "IN", p["normalized"], p["declarations"], p["fields"])}

    def test_transient_errors_retried_then_succeed(self):
        a = FlakyAdapter([ProviderError("TRANSIENT", "HTTP_503"), ProviderError("TRANSIENT", "TIMEOUT")], {"found": True, "name": "Kestrel Logistics Private Limited", "status": "ACTIVE"})
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)
        r = self._go()["GSTIN_VERIFY"]
        self.assertEqual((r.outcome, r.attempts, a.calls, r.mode), ("VERIFIED", 3, 3, "LIVE"))

    def test_retries_exhausted_is_unavailable_and_not_simulated(self):
        a = FlakyAdapter([ProviderError("TRANSIENT", "TIMEOUT")] * 9)
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)       # a LIVE adapter exists, DEMO mode has simulation available
        r = self._go()["GSTIN_VERIFY"]
        self.assertEqual((r.outcome, r.mode, r.provider, a.calls), ("UNAVAILABLE", "LIVE", "flaky", 3))   # NO silent fallback to simulation

    def test_auth_error_not_retried(self):
        a = FlakyAdapter([ProviderError("AUTH", "HTTP_401")] * 3)
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)
        r = self._go()["GSTIN_VERIFY"]
        self.assertEqual((r.outcome, a.calls), ("UNAVAILABLE", 1))

    def test_restricted_provider_is_skipped_not_failed(self):
        a = FlakyAdapter([ProviderError("RESTRICTED", "NO_ACCESS")])
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)
        self.assertEqual(self._go()["GSTIN_VERIFY"].outcome, "SKIPPED")

    def test_unexpected_exception_is_unavailable_not_negative(self):
        a = FlakyAdapter([KeyError("boom")])
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)
        self.assertEqual(self._go()["GSTIN_VERIFY"].outcome, "UNAVAILABLE")

    def test_circuit_opens_after_repeated_failures_and_stops_calling(self):
        a = FlakyAdapter([ProviderError("TRANSIENT", "TIMEOUT")] * 100)
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)
        for _ in range(3):
            self._go()
        before = a.calls
        r = self._go()["GSTIN_VERIFY"]
        self.assertEqual((r.reason, a.calls), ("CIRCUIT_OPEN", before))

    def test_invalid_format_identifier_is_never_sent_to_provider(self):
        a = FlakyAdapter()
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)
        self.payload["normalized"]["gstin"] = "27JUFCK9755F1ZX"      # bad check character
        r = self._go()["GSTIN_VERIFY"]
        self.assertEqual((r.outcome, r.reason, a.calls), ("SKIPPED", "INVALID_FORMAT", 0))

    def test_each_attempt_is_audited_with_masked_request(self):
        a = FlakyAdapter([ProviderError("TRANSIENT", "TIMEOUT")], {"found": True, "name": "Kestrel Logistics Private Limited", "status": "ACTIVE"})
        self.svc.registry.register("IN", "GSTIN_VERIFY", a)
        self._go()
        rows = [r for r in self.svc.repo._all("SELECT * FROM api_audit_log WHERE category='VERIFICATION' AND purpose='GSTIN_VERIFY'")]
        self.assertEqual([r["status"] for r in rows], ["TIMEOUT", "OK"])


class TestRegistryResolution(unittest.TestCase):
    def test_demo_auto_uses_simulated(self):
        a = make_services(app_mode="DEMO").registry.resolve("IN", "GSTIN_VERIFY")
        self.assertEqual(a.mode, "SIMULATED")

    def test_production_auto_without_live_provider_is_not_configured(self):
        self.assertIsNone(make_services(app_mode="PRODUCTION").registry.resolve("IN", "GSTIN_VERIFY"))

    def test_production_can_still_run_simulated_but_labelled(self):
        self.assertEqual(make_services(app_mode="PRODUCTION", verification_providers="simulated").registry.resolve("IN", "GSTIN_VERIFY").mode, "SIMULATED")

    def test_none_disables_everything(self):
        self.assertIsNone(make_services(app_mode="DEMO", verification_providers="none").registry.resolve("UK", "x"))

    def test_credentials_enable_live_uk_adapters(self):
        reg = make_services(app_mode="PRODUCTION", companies_house_api_key="k", hmrc_client_id="i", hmrc_client_secret="s").registry
        self.assertEqual(reg.resolve("GB", "COMPANIES_HOUSE").name, "companies-house")
        self.assertEqual(reg.resolve("GB", "VAT_VERIFY").name, "hmrc-vat")
        self.assertIsNone(reg.resolve("GB", "BANK_VERIFY"))        # no real bank-ownership provider exists: not silently simulated

    def test_simulated_unknown_identifier_behaviour_configurable(self):
        svc = make_services(app_mode="DEMO", sim_unknown="not_found")
        a = svc.registry.resolve("IN", "PAN_VERIFY")
        self.assertFalse(a.lookup(VerificationRequest("PAN_VERIFY", "IN", "ZZZZZ9999Z", "X"), 1)["found"])

    def test_plan_marks_declared_nonregistration_as_waived(self):
        p = {c.check_type: c for c in plan_checks("GB", {"company_number": "00442597", "vat_number": "", "bank_account_number": "12345678", "sort_code": "112233"},
                                                  {"vat_not_registered": True})}
        self.assertEqual(p["VAT_VERIFY"].skip_reason, "WAIVED_BY_DECLARATION")


if __name__ == "__main__":
    unittest.main()
