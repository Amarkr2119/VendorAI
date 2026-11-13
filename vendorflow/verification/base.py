"""Provider-neutral types for verification adapters.

OUTCOME (what the check found) and MODE (how it was obtained) are separate on purpose:
    outcome: VERIFIED | NEGATIVE | UNAVAILABLE | SKIPPED
    mode:    LIVE | SANDBOX | SIMULATED
An HTTP 200 is not a VERIFIED outcome, a timeout is not a NEGATIVE one, and a valid format is not a verification at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional

from vendorflow.analysis.names import classify_names

TERMINAL_STATUSES = {"CANCELLED", "DISSOLVED", "CLOSED", "CONVERTED-CLOSED", "REMOVED"}
ACTIVE_STATUSES = {"ACTIVE"}


@dataclass
class VerificationRequest:
    check_type: str
    country: str
    identifier: str                                  # normalized identifier (EIN digits, GSTIN, account number ...)
    claimed_name: str = ""                           # the company name the vendor submitted
    extra: Dict[str, str] = field(default_factory=dict)


class Interpretation(NamedTuple):
    outcome: str                                     # VERIFIED | NEGATIVE
    reason: str                                      # '' | NOT_FOUND | TERMINAL_STATUS | STATUS_NOT_ACTIVE | NAME_MISMATCH | NO_MATCH
    summary: str
    data: Dict[str, Any]


@dataclass
class VerificationResult:
    check_type: str
    provider: str
    outcome: str
    mode: str
    attempts: int = 0
    reason: str = ""                                 # why SKIPPED/UNAVAILABLE/NEGATIVE (machine readable)
    summary: str = ""
    response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    mandatory: bool = True

    @property
    def label(self) -> str:
        return f"{self.mode} {self.outcome}"


class ProviderError(Exception):
    """kind: TRANSIENT (retry) | AUTH | CONFIG | RESTRICTED (not permitted to call: SKIPPED, not a failure of the vendor)."""

    def __init__(self, kind: str, code: str, message: str = ""):
        super().__init__(message or code)
        self.kind, self.code = kind, code


class Adapter:
    """A replaceable provider. Subclasses implement `lookup` (I/O only) and, if needed, `interpret` (pure)."""
    name = "adapter"
    check_type = ""
    mode = "SIMULATED"
    endpoint = ""

    def lookup(self, req: VerificationRequest, timeout: float) -> Dict[str, Any]:
        raise NotImplementedError

    def interpret(self, raw: Dict[str, Any], req: VerificationRequest) -> Interpretation:
        return judge_entity(raw, req)


def judge_entity(raw: Dict[str, Any], req: VerificationRequest, name_key: str = "name") -> Interpretation:
    """Shared reading of an entity-style response: found? active? does the registered name match the submitted one?"""
    if not raw.get("found", True):
        return Interpretation("NEGATIVE", "NOT_FOUND", "The provider has no record of this identifier.", {})
    status = str(raw.get("status") or "").upper()
    if status in TERMINAL_STATUSES:
        return Interpretation("NEGATIVE", "TERMINAL_STATUS", f"The provider reports status {status}.", {"status": status})
    if status and status not in ACTIVE_STATUSES:
        return Interpretation("NEGATIVE", "STATUS_NOT_ACTIVE", f"The provider reports status {status} (not active).", {"status": status})
    registered = raw.get(name_key) or ""
    if registered and req.claimed_name:
        level, score = classify_names(registered, req.claimed_name)
        data = {"registered_name": registered, "name_level": level, "name_score": score, "status": status or None}
        if level != "MATCH":
            return Interpretation("NEGATIVE", "NAME_MISMATCH",
                                  f"Provider name '{registered}' does not match the submitted name '{req.claimed_name}' ({level.lower()}).", data)
        return Interpretation("VERIFIED", "", f"Found; registered name matches the submitted name{f'; status {status}' if status else ''}.", data)
    return Interpretation("VERIFIED", "", "Found; no name available to compare.", {"status": status or None})
