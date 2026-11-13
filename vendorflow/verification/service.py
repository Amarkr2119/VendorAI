"""Verification orchestration: plan the checks for a country, execute each with timeout/retry/circuit-breaker,
audit every call, and persist structured results with their evidence.

Hard rules implemented here:
  * format-invalid identifiers are NEVER sent to a provider (SKIPPED / INVALID_FORMAT)
  * a transient failure is retried; after the last attempt the outcome is UNAVAILABLE (never NEGATIVE)
  * a RESTRICTED provider (not permitted to call) is SKIPPED, not failed
  * the adapter chosen by the registry is the only one tried: no silent switch to simulation
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from vendorflow.config import Settings
from vendorflow.db.repositories import Repository
from vendorflow.logging_setup import get_logger
from vendorflow.utils.masking import mask_account
from vendorflow.validation import identifiers as ids
from vendorflow.verification.base import Adapter, ProviderError, VerificationRequest, VerificationResult
from vendorflow.verification.registry import ProviderRegistry

log = get_logger("verification")


@dataclass
class CheckPlan:
    check_type: str
    request: VerificationRequest
    mandatory: bool = True
    skip_reason: str = ""            # INVALID_FORMAT | MISSING_INPUT | WAIVED_BY_DECLARATION
    skip_summary: str = ""


def _plan(check_type, country, ident, name, extra=None, *, valid=True, waived=False, mandatory=True, what="") -> CheckPlan:
    req = VerificationRequest(check_type, country, ident or "", name, extra or {})
    if waived:
        return CheckPlan(check_type, req, False, "WAIVED_BY_DECLARATION", "The vendor declared it is not registered; a reviewer must accept the declaration.")
    if not ident:
        return CheckPlan(check_type, req, mandatory, "MISSING_INPUT", f"{what or 'The identifier'} was not provided, so nothing could be verified.")
    if not valid:
        return CheckPlan(check_type, req, mandatory, "INVALID_FORMAT", f"{what or 'The identifier'} has an invalid format and was not sent to the provider.")
    return CheckPlan(check_type, req, mandatory)


def plan_checks(country: str, norm: Mapping[str, str], declarations: Mapping[str, bool],
                display: Optional[Mapping[str, str]] = None) -> List[CheckPlan]:
    n = lambda k: norm.get(k, "")
    name = (display or {}).get("legal_name") or n("legal_name")      # providers and evidence see the name as submitted
    if country == "IN":
        gst_waived = bool(declarations.get("gst_not_registered")) and not n("gstin")
        gst_ok = ids.is_valid_gstin(n("gstin")) and (not n("pan") or ids.gstin_embedded_pan(n("gstin")) == n("pan"))
        return [_plan("PAN_VERIFY", "IN", n("pan"), name, valid=ids.is_valid_pan(n("pan")), what="PAN"),
                _plan("GSTIN_VERIFY", "IN", n("gstin"), name, valid=gst_ok, waived=gst_waived, what="GSTIN"),
                _plan("BANK_VERIFY", "IN", n("bank_account_number"), name, {"ifsc": n("ifsc"), "submitted_holder": n("bank_holder_name")},
                      valid=ids.is_valid_in_account(n("bank_account_number")) and ids.is_valid_ifsc(n("ifsc")), what="Bank account / IFSC")]
    if country == "US":
        return [_plan("TIN_MATCH", "US", n("ein"), name, valid=ids.is_valid_ein(n("ein")), what="EIN"),
                _plan("BUSINESS_REGISTRY", "US", n("state_file_number"), name, {"state": n("state_of_registration")},
                      valid=bool(n("state_of_registration")), what="State file number and state"),
                _plan("BANK_VERIFY", "US", n("bank_account_number"), name, {"routing": n("routing_number"), "submitted_holder": n("bank_holder_name")},
                      valid=ids.is_valid_us_account(n("bank_account_number")) and ids.is_valid_aba(n("routing_number")), what="Bank account / routing number")]
    vat_waived = bool(declarations.get("vat_not_registered")) and not n("vat_number")
    return [_plan("COMPANIES_HOUSE", "GB", n("company_number"), name, valid=ids.is_valid_uk_company_number(n("company_number")), what="Company number"),
            _plan("VAT_VERIFY", "GB", n("vat_number"), name, valid=ids.is_valid_uk_vat(n("vat_number")), waived=vat_waived, what="VAT number"),
            _plan("BANK_VERIFY", "GB", n("bank_account_number"), name, {"sort_code": n("sort_code"), "submitted_holder": n("bank_holder_name")},
                  valid=ids.is_valid_uk_account(n("bank_account_number")) and ids.is_valid_sort_code(n("sort_code")), what="Bank account / sort code")]


class VerificationService:
    def __init__(self, repo: Repository, settings: Settings, registry: Optional[ProviderRegistry] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.repo, self.settings = repo, settings
        self.registry = registry or ProviderRegistry(settings)
        self.sleep = sleep

    # ------------------------------------------------------------------ public
    def run(self, *args, **kw) -> List[VerificationResult]:
        return list(self.iter_run(*args, **kw))

    def iter_run(self, run_id: int, application_id: str, country: str, norm: Mapping[str, str],
                 declarations: Mapping[str, bool], display: Optional[Mapping[str, str]] = None):
        """Yields each VerificationResult as soon as that check finished (so a UI can show real progress)."""
        for plan in plan_checks(country, norm, declarations, display):
            res = self._run_one(run_id, country, plan)
            self.repo.insert_verification(run_id, application_id, provider=res.provider, check_type=res.check_type, outcome=res.outcome,
                                          mode=res.mode, attempts=res.attempts, response=self._evidence(res, plan), error=res.error)
            res.mandatory = plan.mandatory
            yield res

    # ------------------------------------------------------------------ internals
    def _run_one(self, run_id: int, country: str, plan: CheckPlan) -> VerificationResult:
        adapter = self.registry.resolve(country, plan.check_type)
        provider = adapter.name if adapter else "none"
        # the mode an actual call WOULD use; with no adapter, the mode PRODUCTION requires (a LIVE provider)
        mode = adapter.mode if adapter else "LIVE"
        if plan.skip_reason:
            return VerificationResult(plan.check_type, provider, "SKIPPED", mode, 0, plan.skip_reason, plan.skip_summary)
        if adapter is None:
            return VerificationResult(plan.check_type, provider, "SKIPPED", mode, 0, "NOT_CONFIGURED",
                                      "No verification provider is configured for this check under the current policy; it was not performed.")
        health = self.repo.get_provider_health(adapter.name)
        if health and health.get("circuit_open_until") and health["circuit_open_until"] > datetime.now(timezone.utc).isoformat(timespec="seconds"):
            self._audit(run_id, adapter, plan, "CIRCUIT_OPEN", 0, 0, "circuit open after repeated failures")
            return VerificationResult(plan.check_type, adapter.name, "UNAVAILABLE", adapter.mode, 0, "CIRCUIT_OPEN",
                                      "The provider has failed repeatedly and is paused briefly; retry later.", error="circuit open")
        total = 1 + max(0, self.settings.verify_max_retries)
        for attempt in range(1, total + 1):
            t0 = time.monotonic()
            try:
                raw = adapter.lookup(plan.request, self.settings.verify_timeout_s)
            except ProviderError as e:
                self._audit(run_id, adapter, plan, e.code, attempt, int((time.monotonic() - t0) * 1000), str(e))
                if e.kind == "RESTRICTED":
                    return VerificationResult(plan.check_type, adapter.name, "SKIPPED", adapter.mode, attempt, "RESTRICTED",
                                              "This provider may not be called with the current access rights.", error=str(e))
                if e.kind == "TRANSIENT" and attempt < total:
                    self.sleep(self.settings.verify_backoff_s * (2 ** (attempt - 1)))
                    continue
                self._health(adapter, False)
                return VerificationResult(plan.check_type, adapter.name, "UNAVAILABLE", adapter.mode, attempt, e.code,
                                          f"The provider could not be reached or refused the request ({e.code}). This says nothing about the vendor.",
                                          error=f"{e.kind}: {e}")
            except Exception as e:                                      # a bug or odd response must not look like a vendor problem
                log.exception("Unexpected provider failure %s", adapter.name)
                self._audit(run_id, adapter, plan, "UNEXPECTED", attempt, int((time.monotonic() - t0) * 1000), f"{type(e).__name__}: {e}")
                self._health(adapter, False)
                return VerificationResult(plan.check_type, adapter.name, "UNAVAILABLE", adapter.mode, attempt, "UNEXPECTED",
                                          "The provider call failed unexpectedly.", error=f"{type(e).__name__}: {e}")
            self._audit(run_id, adapter, plan, "OK", attempt, int((time.monotonic() - t0) * 1000), None)
            self._health(adapter, True)
            interp = adapter.interpret(raw, plan.request)
            return VerificationResult(plan.check_type, adapter.name, interp.outcome, adapter.mode, attempt, interp.reason, interp.summary,
                                      response={"provider_data": raw, "interpretation": interp.data})
        raise AssertionError("unreachable")

    def _health(self, adapter: Adapter, ok: bool) -> None:
        self.repo.record_provider_result(adapter.name, ok, self.settings.verify_circuit_threshold, self.settings.verify_circuit_cooldown_s)

    def _audit(self, run_id, adapter, plan, status, attempt, latency, error) -> None:
        req = {"check_type": plan.check_type, "identifier": plan.request.identifier, "claimed_name": plan.request.claimed_name}
        self.repo.audit(run_id=run_id, category="VERIFICATION", provider=adapter.name, status=status, endpoint=adapter.endpoint,
                        purpose=plan.check_type, request=req, latency_ms=latency, attempt=attempt, error=error)

    @staticmethod
    def _evidence(res: VerificationResult, plan: CheckPlan) -> Dict[str, Any]:
        ident = plan.request.identifier
        shown = mask_account(ident) if plan.check_type == "BANK_VERIFY" else ident
        return {"label": res.label, "reason": res.reason, "summary": res.summary, "mandatory": plan.mandatory,
                "request": {"identifier": shown, "claimed_name": plan.request.claimed_name}, **res.response}
