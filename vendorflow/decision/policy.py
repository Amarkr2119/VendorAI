"""The decision POLICY: explicit, versioned, deterministic. No LLM is involved anywhere in this module.

    APPROVED  every mandatory check passed under the operating mode, and no blocking issue is open
    PENDING   anything else that is unresolved (missing/incorrect data, mismatch, duplicate, API failure, low confidence ...)
    REJECTED  may be automatic only when conclusive evidence is retained in the audit trail:
              R1  an exact identifier match against a confirmed internal disqualification record
              R2  a LIVE authoritative provider reports a terminal status (e.g. company dissolved)
              Simulated or sandbox provider results can never support automatic rejection.

Operating modes
    DEMO        SIMULATED / SANDBOX / LIVE verification may satisfy mandatory checks (simulated ones are always labelled)
    PRODUCTION  only LIVE verification can satisfy a mandatory check
"""
from __future__ import annotations

POLICY_VERSION = "2026.2"
MODES = ("DEMO", "PRODUCTION")
ACCEPTED_MODES = {"DEMO": {"LIVE", "SANDBOX", "SIMULATED"}, "PRODUCTION": {"LIVE"}}

# Issue codes a reviewer may resolve through an override, in every mode (human judgement, not data defects)
OVERRIDABLE_ALWAYS = {"DUPLICATE_SUSPECTED", "NAME_MISMATCH_BANK", "NAME_MISMATCH_DOCUMENT", "ADDRESS_MISMATCH", "DATE_INCONSISTENT",
                      "DECLARATION_NEEDS_REVIEW", "BLOCKLIST_MATCH"}
# ...and these only in DEMO (in PRODUCTION they must be re-verified live, not waved through)
OVERRIDABLE_DEMO_ONLY = {"VERIFICATION_UNAVAILABLE", "VERIFICATION_SIMULATED_NEGATIVE", "PROVIDER_NOT_FOUND"}
# Everything else (missing/invalid data, unreadable/wrong documents, unreliable reads, identifier conflicts, not-live/not-performed
# verification, live terminal status) must be FIXED (correction, new revision, retry), not overridden.


def overridable(code: str, mode: str) -> bool:
    return code in OVERRIDABLE_ALWAYS or (mode == "DEMO" and code in OVERRIDABLE_DEMO_ONLY)


# documented rejection conditions: condition id -> (issue code that must be present, description)
REJECTION_RULES = {
    "R1": ("BLOCKLIST_MATCH", "Reviewer confirms a documented internal disqualification (evidence: internal record reference)"),
    "R2": ("LIVE_TERMINAL_STATUS", "Reviewer confirms a terminal status reported by a LIVE authoritative provider"),
}
MIN_REASON_CHARS = 15
