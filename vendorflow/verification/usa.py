"""USA adapters. EIN format is validated locally (validation/rules.py). IRS TIN Matching is restricted to enrolled
e-Services users and has no open API, so only a SIMULATED adapter exists; in PRODUCTION without an authorised provider
the check stays NOT performed and the vendor remains Pending. State business registries have no single public API."""
from __future__ import annotations

from typing import Any, Dict, Optional

from vendorflow.verification.simulated import SimulatedAdapter


def _registry_key(req) -> str:
    return f"{req.extra.get('state', '')}:{req.identifier}".lower()


def simulated_adapters(unknown: str = "accept", fixtures: Optional[Dict[str, Any]] = None):
    mk = lambda ct, sec, nm, **kw: SimulatedAdapter("US", ct, sec, nm, unknown=unknown, fixtures=fixtures, **kw)
    return [mk("TIN_MATCH", "tin_match", "sim-us-tin"),
            mk("BUSINESS_REGISTRY", "registry", "sim-us-registry", key_fn=_registry_key),
            mk("BANK_VERIFY", "bank", "sim-us-bank", name_key="holder_name")]
