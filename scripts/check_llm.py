#!/usr/bin/env python3
"""LIVE connectivity checks for Gemini and Groq. Run these on YOUR machine with YOUR keys:

  python scripts/check_llm.py --list-models                 # list models your keys can use (then set GEMINI_MODEL / GROQ_MODEL)
  python scripts/check_llm.py --ping gemini                 # one minimal real request
  python scripts/check_llm.py --ping groq
  python scripts/check_llm.py --extract-sample gemini       # real structured extraction on a synthetic GST certificate text
  python scripts/check_llm.py --fallback-test               # invalid Gemini key on purpose -> must fall back to Groq

The build sandbox had no network, so these paths were only tested with stubs. Keys are never printed.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendorflow.config import load_settings                          # noqa: E402
from vendorflow.domain.fields import expected_fields                 # noqa: E402
from vendorflow.extraction.llm_extractor import make_parser          # noqa: E402
from vendorflow.llm.base import LlmError, LlmUnavailable             # noqa: E402
from vendorflow.llm.gemini_client import GeminiClient                # noqa: E402
from vendorflow.llm.groq_client import GroqClient                    # noqa: E402
from vendorflow.llm.prompts import build_extraction_prompt           # noqa: E402
from vendorflow.llm.router import LlmRouter                          # noqa: E402

SAMPLE = """GST REGISTRATION CERTIFICATE
Form GST REG-06 (synthetic issuer)
Registration Number: 27AABCK1234F1ZS
Legal Name: Kestrel Logistics Private Limited
Trade Name: Kestrel Logistics Private Limited
Address of Principal Place of Business: Plot 14, Phase II, Industrial Area, Pune 411019, Maharashtra
SYNTHETIC SAMPLE - NOT A REAL DOCUMENT"""

ap = argparse.ArgumentParser()
ap.add_argument("--list-models", action="store_true")
ap.add_argument("--ping", choices=["gemini", "groq"])
ap.add_argument("--extract-sample", choices=["gemini", "groq"])
ap.add_argument("--fallback-test", action="store_true")
args = ap.parse_args()
s = load_settings()


def client(name, key=None, model=None):
    if name == "gemini":
        return GeminiClient(key or s.gemini_api_key, model or s.gemini_model)
    return GroqClient(key or s.groq_api_key, model or s.groq_model)


def need(name):
    key, model = (s.gemini_api_key, s.gemini_model) if name == "gemini" else (s.groq_api_key, s.groq_model)
    if not key:
        sys.exit(f"{name.upper()}_API_KEY is not set in .env")
    if not model:
        sys.exit(f"{name.upper()}_MODEL is not set in .env (run --list-models first)")


if args.list_models:
    for name, key in (("gemini", s.gemini_api_key), ("groq", s.groq_api_key)):
        if not key:
            print(f"{name}: no key set, skipped")
            continue
        try:
            print(f"{name}: " + ", ".join(client(name, model='x').list_models()))
        except LlmError as e:
            print(f"{name}: FAILED [{e.kind}] {e.message}")

if args.ping:
    need(args.ping)
    try:
        r = client(args.ping).generate_json("Reply with JSON only.", 'Return exactly {"ok": true}', s.llm_timeout_s)
        print(f"{args.ping}: OK in {r.latency_ms} ms -> {r.text.strip()[:200]}")
    except LlmError as e:
        print(f"{args.ping}: FAILED [{e.kind}] {e.message}")

if args.extract_sample:
    need(args.extract_sample)
    expected = expected_fields("IN", "tax_certificate")
    system, user = build_extraction_prompt("IN", "tax_certificate", expected, SAMPLE)
    router = LlmRouter([client(args.extract_sample)], s)
    try:
        res = router.complete_json(system, user, make_parser([n for n, _ in expected]))
        print(f"{res.provider}/{res.model}: extracted")
        for k, v in res.parsed.items.items():
            print(f"  {k:12} value={v.value!r} snippet={(v.snippet or '')[:50]!r}")
    except LlmUnavailable as e:
        print("FAILED:", e.attempts)

if args.fallback_test:
    need("groq")
    bad = client("gemini", key="INVALID-KEY-FOR-FALLBACK-TEST", model=s.gemini_model or "x")
    router = LlmRouter([bad, client("groq")], dataclasses.replace(s, llm_min_interval_s=0))
    expected = expected_fields("IN", "tax_certificate")
    system, user = build_extraction_prompt("IN", "tax_certificate", expected, SAMPLE)
    try:
        res = router.complete_json(system, user, make_parser([n for n, _ in expected]))
        print(f"Served by {res.provider} (fell back from {res.fallback_from}). Router status: {router.status()}")
    except LlmUnavailable as e:
        print("FAILED:", e.attempts)
