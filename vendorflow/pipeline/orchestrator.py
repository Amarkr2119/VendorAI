"""Pipeline orchestrator (Phase 3 covers stages 1-2: INTAKE and EXTRACTION).

`Pipeline.run()` is a generator that yields a StageEvent as each stage/document completes, so a UI can show
real progress as it happens. Every terminal event is also written to processing_logs (the audit trail).
Phase 4 adds validation, verification, consistency, duplicates, risk and decision stages to this same generator.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

from vendorflow.config import Settings
from vendorflow.db.repositories import Repository
from vendorflow.domain.enums import ProcessingState, Stage, StageStatus
from vendorflow.domain.models import Finding, StageEvent
from vendorflow.extraction.service import DocResult, ExtractionService
from vendorflow.ingestion.completeness import check_documents, check_fields, summarize
from vendorflow.llm.prompts import PROMPT_VERSION
from vendorflow.logging_setup import get_logger
from vendorflow.decision.policy import POLICY_VERSION
from vendorflow.pipeline.events import make_event

log = get_logger("pipeline")
PIPELINE_VERSION = "phase4-verification-decision"


class Pipeline:
    def __init__(self, repo: Repository, settings: Settings, extraction: ExtractionService, stages: Any = None):
        self.repo, self.settings, self.extraction = repo, settings, extraction
        self.stages = stages            # Phase 4 DecisionStages (validation .. communication); None = Phase 3 behaviour

    def config_snapshot(self) -> Dict[str, Any]:
        router = self.extraction.router
        return {"pipeline_version": PIPELINE_VERSION, "app_mode": self.settings.app_mode, "llm_enabled": self.settings.llm_enabled,
                "llm_mode": self.settings.llm_mode, "llm_providers": router.providers() if router else [],
                "prompt_version": PROMPT_VERSION, "pdf_backend": self.extraction.backend.name,
                "ocr_engine": getattr(self.extraction.ocr, "name", None),
                "policy_version": POLICY_VERSION, "verification_providers": self.settings.verification_providers,
                "providers": self.stages.verifier.registry.describe() if self.stages else {}}

    # ------------------------------------------------------------------ public
    def run(self, application_id: str, trigger: str = "initial") -> Iterator[StageEvent]:
        app = self.repo.get_application(application_id)
        if not app:
            raise ValueError(f"Unknown application {application_id}")
        revision_no = app["current_revision"]
        payload = self.repo.get_revision(application_id, revision_no)["payload"]
        country = app["country"]
        run_id = self.repo.start_run(application_id, revision_no, trigger, self.config_snapshot())
        had_error = False
        failed = False

        def finish(ev: StageEvent, t0: float) -> StageEvent:
            self.repo.log_stage(run_id, application_id, ev.stage, ev.status, ev.message, ev.detail or None,
                                error=ev.message if ev.status == StageStatus.ERROR.value else None,
                                duration_ms=int((time.monotonic() - t0) * 1000))
            return ev

        # ---------------------------------------------------------------- stage 1: intake + completeness
        t0 = time.monotonic()
        yield make_event(Stage.INTAKE.value, "started", f"Submission {application_id} received ({country}); checking completeness", run_id=run_id)
        try:
            findings = check_fields(country, payload["fields"], payload["declarations"])
            docs = self.repo.list_documents(application_id, revision_no)
            findings += check_documents(country, payload["declarations"], docs, use_detected=False)
            self.repo.insert_findings(run_id, application_id, findings)
            s = summarize(findings)
            n_fail = sum(1 for f in findings if f.result == "FAIL")
            msg = (f"{len(docs)} document(s) received. " +
                   ("All required fields and documents are present." if not n_fail else
                    f"Missing fields: {', '.join(s['missing_fields']) or 'none'}; missing documents: {', '.join(s['missing_documents']) or 'none'}."))
            yield finish(make_event(Stage.INTAKE.value, "warning" if n_fail else "ok", msg, run_id=run_id, **s), t0)
        except Exception as e:
            log.exception("Intake stage failed for %s", application_id)
            had_error = failed = True
            yield finish(make_event(Stage.INTAKE.value, "error", f"Intake failed: {type(e).__name__}: {e}", run_id=run_id), t0)

        # ---------------------------------------------------------------- stage 2: extraction
        if not failed:
            t0 = time.monotonic()
            docs = self.repo.list_documents(application_id, revision_no)
            yield make_event(Stage.EXTRACTION.value, "started", f"Extracting information from {len(docs)} document(s)", run_id=run_id)
            results: List[DocResult] = []
            try:
                for d in docs:
                    t1 = time.monotonic()
                    yield make_event(Stage.EXTRACTION.value, "started", f"Reading {d['filename']}", run_id=run_id, document=d["filename"])
                    r = self.extraction.process_document(run_id, application_id, country, d, payload["normalized"])
                    results.append(r)
                    if revision_no > 1:
                        for h in self.repo.human_fields_for_sha(application_id, d["sha256"], revision_no):
                            self.repo.insert_human_field(run_id, d["document_id"], h["field_name"], h["value"], h.get("source_snippet") or "")
                    if r.findings:
                        self.repo.insert_findings(run_id, application_id, r.findings)
                    low = [f.field_name for f in r.fields if f.band == "LOW"]
                    had_error |= r.status == "error"
                    yield finish(make_event(
                        Stage.EXTRACTION.value, {"ok": "ok", "warning": "warning", "error": "error"}[r.status],
                        self._doc_message(r, low), run_id=run_id, document=r.filename, detected_type=r.detected_type,
                        scanned=r.is_scanned, fields=len(r.fields), low_confidence=low, llm=r.llm, notes=r.notes), t1)
                # re-evaluate document completeness using DETECTED types (catches wrong documents)
                fresh = self.repo.list_documents(application_id, revision_no)
                self.repo.insert_findings(run_id, application_id, check_documents(country, payload["declarations"], fresh, use_detected=True))
                n_fields = sum(len(r.fields) for r in results)
                n_low = sum(1 for r in results for f in r.fields if f.band == "LOW")
                latest = self.repo.latest_findings(run_id)
                wrong = [f["field"] for f in latest if f["reason_code"] == "WRONG_DOCUMENT_TYPE"]
                unreadable = [r.filename for r in results if any(f.reason_code == "UNREADABLE_DOCUMENT" for f in r.findings)]
                yield finish(make_event(
                    Stage.EXTRACTION.value, "warning" if (n_low or wrong or unreadable) else "ok",
                    f"Extracted {n_fields} field(s) from {len(results)} document(s); {n_low} low-confidence"
                    + (f"; wrong document type: {', '.join(wrong)}" if wrong else "")
                    + (f"; unreadable: {', '.join(unreadable)}" if unreadable else ""),
                    run_id=run_id, summary=True, documents=len(results), fields=n_fields, low_confidence=n_low,
                    wrong_type=wrong, unreadable=unreadable), t0)
            except Exception as e:
                log.exception("Extraction stage failed for %s", application_id)
                had_error = failed = True
                yield finish(make_event(Stage.EXTRACTION.value, "error", f"Extraction failed: {type(e).__name__}: {e}", run_id=run_id), t0)

        # ---------------------------------------------------------------- wrap up
        # processing_state is the technical extraction state (Phase 3 contract); the business outcome lives in `decisions`.
        latest = self.repo.latest_findings(run_id)
        needs_attention = any(f["result"] == "FAIL" for f in latest)
        state = (ProcessingState.FAILED if failed else ProcessingState.NEEDS_ATTENTION if needs_attention else ProcessingState.EXTRACTED)
        open_findings = [f["rule_id"] for f in latest if f["result"] == "FAIL"]
        self.repo.set_processing_state(application_id, state.value)
        if self.stages is not None and not failed:
            decision_error = False
            for ev in self.stages.run(run_id, application_id, self.text_loader()):
                decision_error |= ev.status == "error"
                yield ev
            had_error |= decision_error
        yield from self._finish(run_id, application_id, state, had_error, failed, needs_attention, open_findings)

    def _finish(self, run_id, application_id, state, had_error, failed, needs_attention, open_findings) -> Iterator[StageEvent]:
        self.repo.finish_run(run_id, "FAILED" if failed else "COMPLETED_WITH_ERRORS" if had_error else "COMPLETED")
        cur = self.repo.current_decision(application_id)
        yield make_event("DONE", "error" if failed else "warning" if needs_attention else "ok",
                         f"Run {run_id} finished: {state.value}", run_id=run_id, state=state.value, open_findings=open_findings,
                         decision=cur["status"] if cur else None)

    # ------------------------------------------------------------------ re-evaluation (no re-extraction)
    def reevaluate(self, application_id: str, trigger: str = "reprocess") -> Iterator[StageEvent]:
        """Run only the Phase 4 stages on the CURRENT fields (including human corrections). Used after a reviewer correction."""
        if self.stages is None:
            raise RuntimeError("Phase 4 stages are not configured")
        app = self.repo.get_application(application_id)
        if not app:
            raise ValueError(f"Unknown application {application_id}")
        run_id = self.repo.start_run(application_id, app["current_revision"], trigger, self.config_snapshot())
        yield make_event(Stage.INTAKE.value, "started", f"Re-evaluating {application_id} on current data (no re-extraction)", run_id=run_id)
        had_error = False
        for ev in self.stages.run(run_id, application_id, self.text_loader()):
            had_error |= ev.status == "error"
            yield ev
        latest = self.repo.latest_findings(run_id)
        state = ProcessingState(app["processing_state"])
        yield from self._finish(run_id, application_id, state, had_error, False, any(f["result"] == "FAIL" for f in latest),
                                [f["rule_id"] for f in latest if f["result"] == "FAIL"])

    def reevaluate_all(self, application_id: str, trigger: str = "reprocess") -> List[StageEvent]:
        return list(self.reevaluate(application_id, trigger))

    def text_loader(self):
        """Returns document text for date checks: text-layer pages only (scanned pages are not trusted for blocking date rules)."""
        from pathlib import Path
        from vendorflow.extraction import pdf_text
        backend, cache = self.extraction.backend, {}

        def load(doc) -> str:
            if doc["document_id"] not in cache:
                try:
                    c = pdf_text.read_document(Path(doc["stored_path"]).read_bytes(), doc["filename"], backend)
                    cache[doc["document_id"]] = "\n".join(p.text for p in c.pages if not p.needs_ocr) if not c.error else ""
                except Exception:
                    cache[doc["document_id"]] = ""
            return cache[doc["document_id"]]
        return load

    def run_all(self, application_id: str, trigger: str = "initial") -> List[StageEvent]:
        return list(self.run(application_id, trigger))

    @staticmethod
    def _doc_message(r: DocResult, low: List[str]) -> str:
        how = "OCR" if r.is_scanned else "text layer"
        base = f"{r.filename}: read via {how}, detected as {r.detected_type or 'unknown'}, {len(r.fields)} field(s)"
        if r.status == "error":
            return f"{r.filename}: could not be read ({'; '.join(r.notes) or 'error'})"
        return base + (f"; LOW confidence: {', '.join(low)}" if low else "")
