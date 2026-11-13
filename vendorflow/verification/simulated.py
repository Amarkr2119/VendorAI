"""SIMULATED provider: answers from data/synthetic/simulation_fixtures.json. It is NEVER real verification.

Every result it produces carries mode=SIMULATED, and the policy (decision/policy.py) treats it accordingly:
allowed to satisfy mandatory checks only in DEMO mode, never in PRODUCTION, and a simulated NEGATIVE can never
be evidence of a disqualifying condition.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from vendorflow.analysis.names import classify_names
from vendorflow.config import ROOT
from vendorflow.domain.fields import normalize_value
from vendorflow.utils.text_normalize import alnum_lower
from vendorflow.verification.base import Adapter, Interpretation, ProviderError, VerificationRequest, judge_entity

FIXTURE_PATH = ROOT / "data" / "synthetic" / "simulation_fixtures.json"
_FAULTS = {"TIMEOUT": ("TRANSIENT", "TIMEOUT", "simulated provider timed out"),
           "HTTP_503": ("TRANSIENT", "HTTP_503", "simulated provider returned HTTP 503")}


def load_fixtures(path: Path = FIXTURE_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class SimulatedAdapter(Adapter):
    """Generic fixture-backed adapter. `section` selects the fixture table; `key_fn` builds the lookup key from the request."""
    mode = "SIMULATED"

    def __init__(self, country: str, check_type: str, section: str, name: str, key_fn=None, name_key: str = "name",
                 unknown: str = "accept", fixtures: Optional[Dict[str, Any]] = None):
        self.country, self.check_type, self.section, self.name = country, check_type, section, name
        self.key_fn = key_fn or (lambda r: r.identifier)
        self.name_key, self.unknown = name_key, unknown
        self._fx = fixtures if fixtures is not None else load_fixtures()
        self.endpoint = f"simulated://{country.lower()}/{section}"

    def _table(self) -> Dict[str, Any]:
        return {self._norm(k): v for k, v in (self._fx.get(self.country, {}).get(self.section) or {}).items()}

    @staticmethod
    def _norm(k: str) -> str:
        return k.lower() if ":" in k else k.upper() if not k.isdigit() else k

    def _faults(self) -> Dict[str, str]:
        return {self._norm(k): v for k, v in (self._fx.get(self.country, {}).get("fault") or {}).items()}

    def lookup(self, req: VerificationRequest, timeout: float) -> Dict[str, Any]:
        key = self._norm(self.key_fn(req))
        fault = self._faults().get(key)
        if fault in _FAULTS:
            raise ProviderError(*_FAULTS[fault])
        hit = self._table().get(key)
        if hit is not None:
            return {"found": True, "fixture_hit": True, **hit}
        if self.unknown == "not_found":
            return {"found": False, "fixture_hit": False}
        # Not in the fixtures: a synthetic acceptance so new demo vendors can flow end to end. Labelled as such in the evidence.
        echo = {self.name_key: req.claimed_name} if req.claimed_name else {}
        extra = {"status": "ACTIVE"} if self.section in {"gstin", "registry", "companies_house"} else {}
        if self.section == "tin_match":
            extra = {"match": True}
        return {"found": True, "fixture_hit": False, "synthetic_echo": True, **echo, **extra}

    def interpret(self, raw: Dict[str, Any], req: VerificationRequest) -> Interpretation:
        note = " (SIMULATED: identifier not in fixtures, synthetic acceptance)" if raw.get("synthetic_echo") else " (SIMULATED)"
        if self.section == "tin_match" and raw.get("found"):
            if not raw.get("match", False):
                return Interpretation("NEGATIVE", "NO_MATCH", "TIN and name do not match" + note, {})
            reg = raw.get("name") or ""
            lvl = classify_names(reg, req.claimed_name)[0] if reg and req.claimed_name else "MATCH"
            if lvl != "MATCH":
                return Interpretation("NEGATIVE", "NAME_MISMATCH", f"TIN match record is for '{reg}', not the submitted name" + note,
                                      {"registered_name": reg, "name_level": lvl})
            return Interpretation("VERIFIED", "", "TIN and name match" + note, {"registered_name": reg})
        i = judge_entity(raw, req, self.name_key)
        return Interpretation(i.outcome, i.reason, i.summary.rstrip(".") + note, i.data)


def bank_key(req: VerificationRequest) -> str:
    return req.identifier
