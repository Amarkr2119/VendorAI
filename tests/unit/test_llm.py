"""AT-12: LLM router and REST clients, tested with stubs only (no network)."""
import json
import unittest

import requests

from tests.helpers import temp_settings
from vendorflow.llm.base import LlmError, LlmErrorKind, LlmResponse, LlmUnavailable
from vendorflow.llm.gemini_client import GeminiClient
from vendorflow.llm.groq_client import GroqClient
from vendorflow.llm.router import LlmRouter


class Clock:
    def __init__(self):
        self.t, self.slept = 1000.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class FakeClient:
    """Scripted client: each call pops the next item (an LlmError to raise, or a string to return)."""

    def __init__(self, name, script):
        self.name, self.model, self.script, self.calls = name, f"{name}-model", list(script), 0

    def generate_json(self, system, user, timeout):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, LlmError):
            raise item
        return LlmResponse(item, 5)

    def list_models(self, timeout=20.0):
        return []


def parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}")


def make_router(clients, **settings):
    clock, audit = Clock(), []
    s = temp_settings(llm_min_interval_s=settings.pop("min_interval", 0.0), llm_max_retries=settings.pop("retries", 2), **settings)
    r = LlmRouter(clients, s, audit=lambda **kw: audit.append(kw), clock=clock.now, sleep=clock.sleep)
    return r, clock, audit


OK = '{"fields": {}}'
quota = lambda: LlmError(LlmErrorKind.QUOTA, "429 quota", 429)
transient = lambda: LlmError(LlmErrorKind.TRANSIENT, "timeout")
auth = lambda: LlmError(LlmErrorKind.AUTH, "bad key", 401)


class TestRouter(unittest.TestCase):
    def test_primary_success(self):
        g, q = FakeClient("gemini", [OK]), FakeClient("groq", [OK])
        r, _, audit = make_router([g, q])
        res = r.complete_json("s", "u", parse)
        self.assertEqual((res.provider, res.fallback_from, g.calls, q.calls), ("gemini", None, 1, 0))
        self.assertEqual([a["status"] for a in audit], ["OK"])

    def test_quota_moves_to_fallback_and_primary_cools_down(self):
        g, q = FakeClient("gemini", [quota()]), FakeClient("groq", [OK, OK])
        r, clock, audit = make_router([g, q])
        res = r.complete_json("s", "u", parse)
        self.assertEqual((res.provider, res.fallback_from), ("groq", "gemini"))
        self.assertEqual(g.calls, 1)                                    # no retry on quota
        r.complete_json("s", "u", parse)                                # primary is cooling down: skipped
        self.assertEqual(g.calls, 1)
        self.assertIn("SKIPPED", [a["status"] for a in audit])
        clock.t += 61                                                   # cooldown (60 s) elapsed
        self.assertEqual(r.status()["gemini"], "ready")

    def test_transient_retried_with_backoff_then_succeeds(self):
        g = FakeClient("gemini", [transient(), transient(), OK])
        r, clock, audit = make_router([g])
        res = r.complete_json("s", "u", parse)
        self.assertEqual((res.provider, res.attempts, g.calls), ("gemini", 3, 3))
        self.assertEqual(clock.slept, [1, 2])
        self.assertEqual([a["status"] for a in audit], ["TRANSIENT", "TRANSIENT", "OK"])

    def test_transient_exhausted_falls_back(self):
        g, q = FakeClient("gemini", [transient()] * 3), FakeClient("groq", [OK])
        r, _, _ = make_router([g, q])
        res = r.complete_json("s", "u", parse)
        self.assertEqual((res.provider, res.fallback_from, g.calls), ("groq", "gemini", 3))

    def test_auth_disables_provider_for_session(self):
        g, q = FakeClient("gemini", [auth()]), FakeClient("groq", [OK, OK])
        r, _, _ = make_router([g, q])
        r.complete_json("s", "u", parse)
        r.complete_json("s", "u", parse)
        self.assertEqual(g.calls, 1)
        self.assertTrue(r.status()["gemini"].startswith("disabled"))

    def test_bad_json_gets_one_repair_retry_on_same_provider(self):
        g = FakeClient("gemini", ["not json", OK])
        r, _, audit = make_router([g])
        res = r.complete_json("s", "u", parse)
        self.assertEqual((res.provider, g.calls), ("gemini", 2))
        self.assertEqual([a["status"] for a in audit], ["BAD_OUTPUT", "OK"])

    def test_persistent_bad_json_falls_back(self):
        g, q = FakeClient("gemini", ["x", "y"]), FakeClient("groq", [OK])
        r, _, _ = make_router([g, q])
        res = r.complete_json("s", "u", parse)
        self.assertEqual((res.provider, g.calls), ("groq", 2))

    def test_all_providers_fail_raises_unavailable(self):
        g, q = FakeClient("gemini", [quota()]), FakeClient("groq", [auth()])
        r, _, _ = make_router([g, q])
        with self.assertRaises(LlmUnavailable) as cm:
            r.complete_json("s", "u", parse)
        self.assertEqual(len(cm.exception.attempts), 2)

    def test_no_providers_configured_raises_unavailable(self):
        r, _, _ = make_router([])
        self.assertFalse(r.has_providers())
        with self.assertRaises(LlmUnavailable):
            r.complete_json("s", "u", parse)

    def test_circuit_breaker_after_three_consecutive_transient_failures(self):
        g = FakeClient("gemini", [transient()] * 3 + [OK])
        r, clock, _ = make_router([g])
        with self.assertRaises(LlmUnavailable):
            r.complete_json("s", "u", parse)
        self.assertEqual(r.status()["gemini"], "cooling down")
        with self.assertRaises(LlmUnavailable):                         # skipped, no new call
            r.complete_json("s", "u", parse)
        self.assertEqual(g.calls, 3)

    def test_client_side_rate_limit_spaces_calls(self):
        g = FakeClient("gemini", [OK, OK])
        r, clock, _ = make_router([g], min_interval=4.0)
        r.complete_json("s", "u", parse)
        r.complete_json("s", "u", parse)
        self.assertTrue(any(abs(s - 4.0) < 1e-6 for s in clock.slept))

    def test_audit_never_contains_document_text(self):
        g = FakeClient("gemini", [OK])
        r, _, audit = make_router([g])
        r.complete_json("sys", "SECRET DOCUMENT TEXT 123456789012", parse, run_id=7, purpose="extraction")
        blob = json.dumps(audit)
        self.assertNotIn("SECRET", blob)
        self.assertEqual(audit[0]["run_id"], 7)


class FakeResp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code, self._body, self.text, self.headers = status, body, text or json.dumps(body or {}), headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.last = resp, exc, None

    def post(self, url, **kw):
        self.last = (url, kw)
        if self.exc:
            raise self.exc
        return self.resp

    get = post


class TestRestClients(unittest.TestCase):
    def test_gemini_success_and_key_only_in_header(self):
        body = {"candidates": [{"content": {"parts": [{"text": '{"fields": {}}'}]}}], "usageMetadata": {"totalTokenCount": 9}}
        s = FakeSession(FakeResp(200, body))
        resp = GeminiClient("SECRETKEY", "gm", s).generate_json("sys", "usr", 5)
        self.assertEqual(resp.text, '{"fields": {}}')
        url, kw = s.last
        self.assertNotIn("SECRETKEY", url)
        self.assertEqual(kw["headers"]["x-goog-api-key"], "SECRETKEY")
        self.assertEqual(kw["json"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(kw["json"]["generationConfig"]["temperature"], 0)

    def test_gemini_image_ocr_sends_image_inline_and_key_only_in_header(self):
        body = {"candidates": [{"content": {"parts": [{"text": "GSTIN: 27ABCDE1234F1Z5"}]}}]}
        session = FakeSession(FakeResp(200, body))
        response = GeminiClient("SECRETKEY", "gm", session).generate_text_from_image("transcribe", b"png-bytes", "image/png", 5)
        self.assertEqual(response.text, "GSTIN: 27ABCDE1234F1Z5")
        url, kwargs = session.last
        self.assertNotIn("SECRETKEY", url)
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "SECRETKEY")
        part = kwargs["json"]["contents"][0]["parts"][1]["inline_data"]
        self.assertEqual(part["mime_type"], "image/png")
        self.assertEqual(part["data"], "cG5nLWJ5dGVz")

    def test_gemini_error_mapping(self):
        cases = [(429, "RESOURCE_EXHAUSTED", LlmErrorKind.QUOTA), (400, "API key not valid. Please pass a valid API key", LlmErrorKind.AUTH),
                 (400, "bad schema", LlmErrorKind.BAD_REQUEST), (403, "denied", LlmErrorKind.AUTH), (404, "model not found", LlmErrorKind.CONFIG),
                 (500, "oops", LlmErrorKind.TRANSIENT), (503, "unavailable", LlmErrorKind.TRANSIENT)]
        for status, text, kind in cases:
            with self.subTest(status=status, text=text):
                with self.assertRaises(LlmError) as cm:
                    GeminiClient("k", "m", FakeSession(FakeResp(status, None, text))).generate_json("s", "u", 5)
                self.assertEqual(cm.exception.kind, kind)

    def test_gemini_retry_after_and_network_errors_and_blocked(self):
        with self.assertRaises(LlmError) as cm:
            GeminiClient("k", "m", FakeSession(FakeResp(429, None, "x", {"Retry-After": "12"}))).generate_json("s", "u", 5)
        self.assertEqual(cm.exception.retry_after, 12.0)
        for exc in (requests.Timeout("t"), requests.ConnectionError("c")):
            with self.assertRaises(LlmError) as cm:
                GeminiClient("k", "m", FakeSession(exc=exc)).generate_json("s", "u", 5)
            self.assertEqual(cm.exception.kind, LlmErrorKind.TRANSIENT)
        with self.assertRaises(LlmError) as cm:
            GeminiClient("k", "m", FakeSession(FakeResp(200, {"promptFeedback": {"blockReason": "SAFETY"}}))).generate_json("s", "u", 5)
        self.assertEqual(cm.exception.kind, LlmErrorKind.BAD_OUTPUT)

    def test_groq_success_and_error_mapping(self):
        s = FakeSession(FakeResp(200, {"choices": [{"message": {"content": '{"a":1}'}}], "usage": {"total_tokens": 5}}))
        self.assertEqual(GroqClient("gk", "gm", s).generate_json("s", "u", 5).text, '{"a":1}')
        self.assertEqual(s.last[1]["headers"]["Authorization"], "Bearer gk")
        self.assertEqual(s.last[1]["json"]["response_format"], {"type": "json_object"})
        for status, kind in [(429, LlmErrorKind.QUOTA), (401, LlmErrorKind.AUTH), (404, LlmErrorKind.CONFIG), (502, LlmErrorKind.TRANSIENT)]:
            with self.subTest(status=status):
                with self.assertRaises(LlmError) as cm:
                    GroqClient("k", "m", FakeSession(FakeResp(status, None, "e"))).generate_json("s", "u", 5)
                self.assertEqual(cm.exception.kind, kind)
        with self.assertRaises(LlmError) as cm:
            GroqClient("k", "m", FakeSession(FakeResp(200, {"weird": 1}))).generate_json("s", "u", 5)
        self.assertEqual(cm.exception.kind, LlmErrorKind.BAD_OUTPUT)

    def test_list_models_parsing(self):
        g = GeminiClient("k", "m", FakeSession(FakeResp(200, {"models": [
            {"name": "models/flash-x", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embed-y", "supportedGenerationMethods": ["embedContent"]}]})))
        self.assertEqual(g.list_models(), ["flash-x"])
        q = GroqClient("k", "m", FakeSession(FakeResp(200, {"data": [{"id": "b"}, {"id": "a"}]})))
        self.assertEqual(q.list_models(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
