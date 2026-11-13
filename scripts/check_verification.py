"""Show how each verification check is resolved under your current .env, and optionally make ONE real lookup.

    python scripts/check_verification.py                          # configuration only, no network
    python scripts/check_verification.py --live --company-number 00445790     # real Companies House call (needs COMPANIES_HOUSE_API_KEY)
    python scripts/check_verification.py --live --vat 123456789                # real HMRC call (needs HMRC_CLIENT_ID/SECRET)

This is the only way to exercise the LIVE/SANDBOX adapters against the real services. The automated test suite uses stub HTTP sessions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendorflow.config import Settings  # noqa: E402
from vendorflow.verification.base import ProviderError, VerificationRequest  # noqa: E402
from vendorflow.verification.registry import ProviderRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--company-number")
    ap.add_argument("--vat")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    s = Settings.from_env()
    reg = ProviderRegistry(s)
    print(f"APP_MODE={s.app_mode}  VERIFICATION_PROVIDERS={s.verification_providers}")
    for k, v in reg.describe().items():
        print(f"  {k:28s} -> {v}")
    if not a.live:
        return 0
    jobs = []
    if a.company_number:
        jobs.append(("COMPANIES_HOUSE", a.company_number))
    if a.vat:
        jobs.append(("VAT_VERIFY", a.vat))
    for ct, ident in jobs:
        ad = reg.resolve("GB", ct)
        if ad is None or ad.mode == "SIMULATED":
            print(f"{ct}: no LIVE/SANDBOX adapter configured (set credentials in .env). Not calling anything.")
            continue
        req = VerificationRequest(ct, "GB", ident, a.name)
        try:
            raw = ad.lookup(req, s.verify_timeout_s)
            print(f"{ct} [{ad.mode}] raw: {raw}\n   interpretation: {ad.interpret(raw, req)}")
        except ProviderError as e:
            print(f"{ct} [{ad.mode}] provider error {e.kind}/{e.code}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
