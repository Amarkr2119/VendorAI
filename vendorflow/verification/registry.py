"""Provider registry: which adapter serves (country, check_type).

Selection is decided ONCE, from configuration, before any call is made, and is never revisited after a failure:
    * a LIVE/SANDBOX adapter exists (credentials configured)  -> it is used. If it fails, the check is UNAVAILABLE.
      There is NO fallback to a simulated adapter after a live failure.
    * none exists and VERIFICATION_PROVIDERS=simulated, or auto in DEMO mode  -> the SIMULATED adapter (labelled SIMULATED).
    * none exists in PRODUCTION (auto), or VERIFICATION_PROVIDERS=none        -> no adapter: the check is SKIPPED (NOT_CONFIGURED).
Any adapter can be replaced with `register()` (for a licensed provider, or a test stub).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from vendorflow.config import Settings
from vendorflow.verification import india, uk, usa
from vendorflow.verification.base import Adapter


class ProviderRegistry:
    def __init__(self, settings: Settings, session: Any = None, fixtures: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self._live: Dict[Tuple[str, str], Adapter] = {}
        self._sim: Dict[Tuple[str, str], Adapter] = {}
        for mod, country in ((india, "IN"), (usa, "US"), (uk, "GB")):
            for a in mod.simulated_adapters(settings.sim_unknown, fixtures):
                self._sim[(country, a.check_type)] = a
        if settings.companies_house_api_key:
            self._live[("GB", "COMPANIES_HOUSE")] = uk.CompaniesHouseAdapter(settings, session)
        if settings.hmrc_client_id and settings.hmrc_client_secret:
            self._live[("GB", "VAT_VERIFY")] = uk.HmrcVatAdapter(settings, session)

    def register(self, country: str, check_type: str, adapter: Adapter) -> None:
        """Plug in (or replace with) a real or stub provider. Adapters in mode SIMULATED go to the simulated table."""
        (self._sim if adapter.mode == "SIMULATED" else self._live)[(country, check_type)] = adapter

    def resolve(self, country: str, check_type: str) -> Optional[Adapter]:
        mode = self.settings.verification_providers
        if mode == "none":
            return None
        live = self._live.get((country, check_type))
        if live is not None and mode != "simulated":
            return live
        if mode == "simulated" or (mode == "auto" and self.settings.app_mode == "DEMO"):
            return self._sim.get((country, check_type))
        return None

    def describe(self) -> Dict[str, str]:
        """For the UI/status: what each check would use right now."""
        out = {}
        for key in sorted(set(self._sim) | set(self._live)):
            a = self.resolve(*key)
            out[f"{key[0]}:{key[1]}"] = f"{a.name} [{a.mode}]" if a else "not configured (SKIPPED)"
        return out
