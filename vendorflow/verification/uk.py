"""UK adapters. Companies House and HMRC have real public APIs; they are used only when credentials are configured.

NOTE (honest status): these LIVE/SANDBOX adapters were written against the providers' published documentation and are
exercised in this repository only with a stub HTTP session. Run `python scripts/check_verification.py --live` with your own
keys to test them for real. UK bank-account ownership (Confirmation of Payee) is only available through banks/PSPs: SIMULATED only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from vendorflow.config import Settings
from vendorflow.verification.base import Adapter, Interpretation, ProviderError, VerificationRequest, judge_entity
from vendorflow.verification.simulated import SimulatedAdapter

CH_URLS = {"live": "https://api.company-information.service.gov.uk", "sandbox": "https://api-sandbox.company-information.service.gov.uk"}
HMRC_URLS = {"live": "https://api.service.hmrc.gov.uk", "sandbox": "https://test-api.service.hmrc.gov.uk"}


def _http_error(status: int) -> ProviderError:
    if status in (401, 403):
        return ProviderError("AUTH", f"HTTP_{status}", "provider rejected the credentials")
    if status == 429 or status >= 500:
        return ProviderError("TRANSIENT", f"HTTP_{status}", f"provider returned HTTP {status}")
    return ProviderError("CONFIG", f"HTTP_{status}", f"unexpected HTTP {status}")


def _call(fn, *args, **kw):
    import requests
    try:
        return fn(*args, **kw)
    except requests.Timeout as e:
        raise ProviderError("TRANSIENT", "TIMEOUT", "request timed out") from e
    except requests.ConnectionError as e:
        raise ProviderError("TRANSIENT", "CONNECTION", "could not connect") from e


class CompaniesHouseAdapter(Adapter):
    """GET /company/{number} with HTTP Basic auth (API key as the username, empty password)."""
    check_type, name = "COMPANIES_HOUSE", "companies-house"

    def __init__(self, settings: Settings, session: Any = None):
        import requests
        self.session = session or requests
        self.key = settings.companies_house_api_key
        env = settings.companies_house_env if settings.companies_house_env in CH_URLS else "live"
        self.mode = "SANDBOX" if env == "sandbox" else "LIVE"
        self.base = CH_URLS[env]
        self.endpoint = f"{self.base}/company/{{number}}"

    def lookup(self, req: VerificationRequest, timeout: float) -> Dict[str, Any]:
        r = _call(self.session.get, f"{self.base}/company/{req.identifier}", auth=(self.key, ""), timeout=timeout)
        if r.status_code == 404:
            return {"found": False}
        if r.status_code != 200:
            raise _http_error(r.status_code)
        d = r.json()
        return {"found": True, "name": d.get("company_name", ""), "status": (d.get("company_status") or "").upper()}


class HmrcVatAdapter(Adapter):
    """HMRC 'Check a UK VAT number' API: OAuth2 client-credentials token, then GET lookup/{vrn}."""
    check_type, name = "VAT_VERIFY", "hmrc-vat"

    def __init__(self, settings: Settings, session: Any = None):
        import requests
        self.session = session or requests
        self.cid, self.secret = settings.hmrc_client_id, settings.hmrc_client_secret
        env = settings.hmrc_env if settings.hmrc_env in HMRC_URLS else "sandbox"
        self.mode = "SANDBOX" if env == "sandbox" else "LIVE"
        self.base = HMRC_URLS[env]
        self.endpoint = f"{self.base}/organisations/vat/check-vat-number/lookup/{{vrn}}"

    def _token(self, timeout: float) -> str:
        r = _call(self.session.post, f"{self.base}/oauth/token", timeout=timeout,
                  data={"grant_type": "client_credentials", "client_id": self.cid, "client_secret": self.secret})
        if r.status_code != 200:
            raise _http_error(r.status_code)
        return r.json()["access_token"]

    def lookup(self, req: VerificationRequest, timeout: float) -> Dict[str, Any]:
        token = self._token(timeout)
        r = _call(self.session.get, f"{self.base}/organisations/vat/check-vat-number/lookup/{req.identifier}", timeout=timeout,
                  headers={"Accept": "application/vnd.hmrc.2.0+json", "Authorization": f"Bearer {token}"})
        if r.status_code in (400, 404):
            return {"found": False}
        if r.status_code != 200:
            raise _http_error(r.status_code)
        return {"found": True, "name": (r.json().get("target") or {}).get("name", ""), "status": "ACTIVE"}


def simulated_adapters(unknown: str = "accept", fixtures: Optional[Dict[str, Any]] = None):
    mk = lambda ct, sec, nm, nk="name": SimulatedAdapter("GB", ct, sec, nm, name_key=nk, unknown=unknown, fixtures=fixtures)
    return [mk("COMPANIES_HOUSE", "companies_house", "sim-gb-companies-house"),
            mk("VAT_VERIFY", "hmrc_vat", "sim-gb-hmrc-vat"),
            mk("BANK_VERIFY", "bank", "sim-gb-bank", "holder_name")]
