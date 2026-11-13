"""Extraction service: one document in -> classified, read (text layer / OCR), structured fields with
confidence and source evidence out, persisted to SQLite.

Order of evidence:  text layer -> OCR (scans)  ->  regex candidates  ->  LLM (optional, cached)  ->  merge + ground
+ score.  The app works with no LLM and no OCR: it then extracts what the regex path can and flags the rest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from vendorflow.config import Settings
from vendorflow.db.repositories import Repository
from vendorflow.domain.fields import expected_fields, field_label, normalize_value
from vendorflow.domain.models import FieldResult, Finding
from vendorflow.extraction import pdf_text
from vendorflow.extraction.confidence import score_field
from vendorflow.extraction.doc_classifier import classify
from vendorflow.extraction.grounding import is_grounded
from vendorflow.extraction.llm_extractor import (LlmExtraction, LlmItem, extract_with_llm, extraction_from_cache,
                                                 extraction_to_cache)
from vendorflow.extraction.ocr import OcrResult, get_ocr_engine
from vendorflow.extraction.regex_extractor import Candidate, regex_candidates
from vendorflow.ingestion.completeness import WRONG_TYPE_MIN_CONFIDENCE
from vendorflow.llm.base import LlmUnavailable
from vendorflow.llm.prompts import PROMPT_VERSION
from vendorflow.llm.router import LlmRouter
from vendorflow.logging_setup import get_logger
from vendorflow.utils.hashing import sha256_bytes
from vendorflow.utils.masking import mask_account
from vendorflow.utils.text_normalize import alnum_lower
from vendorflow.validation.identifiers import format_valid

log = get_logger("extraction")
_UNSET: Any = object()
MIN_READABLE_ALNUM = 30


@dataclass
class DocResult:
    document_id: int
    filename: str
    declared_type: str
    detected_type: Optional[str] = None
    detected_confidence: Optional[float] = None
    page_count: int = 0
    is_scanned: bool = False
    text_chars: int = 0
    backend: str = ""
    ocr_engine: Optional[str] = None
    fields: List[FieldResult] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    llm: Dict[str, Any] = field(default_factory=lambda: {"used": False})
    status: str = "ok"                      # ok | warning | error
    duration_ms: int = 0


def values_equal(country: str, name: str, a: str, b: str) -> bool:
    if name == "address":
        return alnum_lower(a) == alnum_lower(b)
    return normalize_value(country, name, a) == normalize_value(country, name, b)


class ExtractionService:
    def __init__(self, repo: Repository, settings: Settings, router: Optional[LlmRouter] = None,
                 ocr: Any = _UNSET, backend: Any = None):
        self.repo, self.settings, self.router = repo, settings, router
        self.backend = backend or pdf_text.get_backend(settings.pdf_backend)
        self.ocr = get_ocr_engine(settings.ocr_engine) if ocr is _UNSET else ocr

    # ------------------------------------------------------------------ public
    def process_document(self, run_id: int, application_id: str, country: str, doc: Mapping[str, Any],
                         form_norm: Optional[Mapping[str, str]] = None) -> DocResult:
        t0 = time.monotonic()
        res = DocResult(doc["document_id"], doc["filename"], doc["declared_type"], backend=self.backend.name,
                        ocr_engine=getattr(self.ocr, "name", None))
        try:
            self._process(run_id, country, doc, form_norm or {}, res)
        except Exception as e:                                   # one bad document must never stop the run
            log.exception("Extraction failed for document %s", doc["filename"])
            res.status = "error"
            res.notes.append(f"Unexpected extraction error: {type(e).__name__}: {e}")
            res.findings.append(Finding(f"DOCUMENT.UNREADABLE.{doc['filename']}", "DOCUMENT", "FAIL", reason_code="UNREADABLE_DOCUMENT",
                                        message=f"'{doc['filename']}' could not be processed ({type(e).__name__})."))
        res.duration_ms = int((time.monotonic() - t0) * 1000)
        self.repo.update_document_meta(doc["document_id"], detected_type=res.detected_type, detected_confidence=res.detected_confidence,
                                       page_count=res.page_count, is_scanned=res.is_scanned, text_chars=res.text_chars)
        self.repo.insert_fields(run_id, doc["document_id"], res.fields)
        return res

    # ------------------------------------------------------------------ internals
    def _unreadable(self, res: DocResult, why: str) -> None:
        res.status = "error"
        res.notes.append(why)
        res.findings.append(Finding(f"DOCUMENT.UNREADABLE.{res.filename}", "DOCUMENT", "FAIL", reason_code="UNREADABLE_DOCUMENT",
                                    message=f"'{res.filename}' could not be read: {why}"))

    def _read_pages(self, data: bytes, filename: str, content: pdf_text.PdfContent, res: DocResult) -> List[Tuple[int, str, str, Optional[OcrResult]]]:
        pages: List[Tuple[int, str, str, Optional[OcrResult]]] = []
        for p in content.pages:
            if not p.needs_ocr:
                pages.append((p.page_no, p.text, "TEXT_LAYER", None))
                continue
            if self.ocr is None:
                res.notes.append(f"Page {p.page_no} is scanned and no OCR engine is available.")
                pages.append((p.page_no, "", "OCR", None))
                continue
            try:
                img = pdf_text.render_page(data, filename, p.page_no - 1, self.backend, 200)
                r = self.ocr.recognize(img)
                pages.append((p.page_no, r.text, "OCR", r))
            except Exception as e:
                res.notes.append(f"OCR failed on page {p.page_no}: {type(e).__name__}: {e}")
                pages.append((p.page_no, "", "OCR", None))
        return pages

    def _process(self, run_id: int, country: str, doc: Mapping[str, Any], form_norm: Mapping[str, str], res: DocResult) -> None:
        path = Path(doc["stored_path"])
        if not path.exists():
            return self._unreadable(res, "file is missing from storage")
        data = path.read_bytes()
        content = pdf_text.read_document(data, doc["filename"], self.backend)
        if content.error:
            return self._unreadable(res, content.error)
        res.page_count, res.is_scanned, res.text_chars = content.page_count, content.is_scanned, content.text_chars

        pages = self._read_pages(data, doc["filename"], content, res)
        full_text = "\n".join(t for _, t, _, _ in pages)
        if len(alnum_lower(full_text)) < MIN_READABLE_ALNUM:
            why = "no readable text (scanned document and OCR " + ("returned almost nothing)" if self.ocr else "is not available)")
            return self._unreadable(res, why)
        res.text_chars = max(res.text_chars, len(full_text))

        dtype, dconf, _ = classify(pages[0][1] or full_text)
        res.detected_type, res.detected_confidence = dtype, dconf if dtype else None
        trusted = dtype is not None and dconf >= WRONG_TYPE_MIN_CONFIDENCE
        effective = dtype if trusted else doc["declared_type"]
        expected = expected_fields(country, effective)
        if not expected:
            res.notes.append(f"No extraction template for document type '{effective}' in this country.")
            return

        rx = regex_candidates(full_text, expected)
        llm = self._run_llm(run_id, doc, country, effective, expected, full_text, rx, res)
        res.fields = [f for f in (self._merge(country, name, required, rx.get(name, []), llm, full_text, pages, form_norm)
                                  for name, required in expected) if f]
        for f in res.fields:
            if f.band == "LOW":
                res.findings.append(self._low_finding(res, f))
        if any(f.band == "LOW" for f in res.fields):
            res.status = "warning"

    def _run_llm(self, run_id, doc, country, effective, expected, text, rx, res) -> Optional[LlmExtraction]:
        if not self.router or not self.router.has_providers():
            res.llm = {"used": False, "reason": "no LLM configured (regex-only extraction)"}
            return None
        if self.settings.llm_mode == "when_needed" and not res.is_scanned:
            if all(rx.get(n) for n, req in expected if req):
                res.llm = {"used": False, "reason": "LLM_MODE=when_needed and the regex extractor found every required field"}
                return None
        key = sha256_bytes(f"{doc['sha256']}|{effective}|{PROMPT_VERSION}".encode())
        cached = self.repo.cache_get(key)
        if cached:
            res.llm = {"used": True, "cache_hit": True, "provider": cached["provider"], "model": cached["model"]}
            return extraction_from_cache(cached["result"])
        try:
            parsed, provider, model = extract_with_llm(self.router, country, effective, expected, text, run_id)
        except LlmUnavailable as e:
            res.llm = {"used": False, "error": "all LLM providers failed", "attempts": e.attempts}
            res.notes.append("LLM unavailable; used deterministic regex extraction instead.")
            return None
        self.repo.cache_put(key, doc["sha256"], effective, provider, model, PROMPT_VERSION, extraction_to_cache(parsed))
        res.llm = {"used": True, "cache_hit": False, "provider": provider, "model": model}
        return parsed

    def _merge(self, country: str, name: str, required: bool, cands: List[Candidate], llm: Optional[LlmExtraction], text: str,
               pages, form_norm: Mapping[str, str]) -> Optional[FieldResult]:
        norm = lambda v: normalize_value(country, name, v)
        item: Optional[LlmItem] = llm.items.get(name) if llm else None
        lv = item.value if item and item.value else None
        rbest = cands[0] if cands else None
        regex_norms = {norm(c.value) for c in cands}
        alternatives: List[Dict[str, Any]] = []

        value = snippet = None
        page = None
        origin_llm = False
        regex_agrees: Optional[bool] = None
        if lv and rbest:
            l_ok, _ = is_grounded(lv, item.snippet, text)
            if norm(lv) in regex_norms:
                value, snippet, page, origin_llm, regex_agrees = lv, item.snippet, item.page, True, True
            else:
                regex_agrees = False
                fl, fr = format_valid(country, name, norm(lv)), format_valid(country, name, norm(rbest.value))
                use_regex = (not l_ok) or (fr is True and fl is False)
                alternatives = [{"source": "llm", "value": lv, "snippet": item.snippet},
                                *[{"source": "regex", "value": c.value, "snippet": c.snippet} for c in cands[:3]]]
                if use_regex:
                    value, snippet = rbest.value, rbest.snippet
                else:
                    value, snippet, page, origin_llm = lv, item.snippet, item.page, True
        elif lv:
            value, snippet, page, origin_llm = lv, item.snippet, item.page, True
        elif rbest:
            value, snippet = rbest.value, rbest.snippet

        if value is None:
            if not required:
                return None
            return FieldResult(name, None, 0.0, "LOW", "REGEX", "TEXT_LAYER", None, None, None, None, None, None,
                               ["NOT_FOUND"], [])

        grounded, why = is_grounded(value, snippet, text)
        pno, kind, ocr_res = self._locate(snippet or value, pages)
        fvalid = format_valid(country, name, norm(value))
        fagree: Optional[bool] = None
        if form_norm.get(name):             # the form PDF is compared with the stored submission too
            fagree = values_equal(country, name, value, form_norm[name])
        quality = 1.0 if kind == "TEXT_LAYER" else (ocr_res.confidence_for(value) if ocr_res else 0.0)

        score, band, flags = score_field(grounded=grounded, regex_agrees=regex_agrees, format_valid=fvalid,
                                         source_quality=quality, form_agrees=fagree, source_kind=kind)
        if kind == "OCR" and quality < self.settings.ocr_min_confidence:
            flags.append("LOW_OCR_QUALITY")
        origin = ("OCR_LLM" if kind == "OCR" else "TEXT_LAYER_LLM") if origin_llm else "REGEX"
        return FieldResult(name, value, score, band, origin, kind, page or pno, (snippet or "")[:300], grounded, regex_agrees,
                           fvalid, fagree, flags, alternatives)

    @staticmethod
    def _locate(snippet: str, pages) -> Tuple[int, str, Optional[OcrResult]]:
        target = alnum_lower(snippet)
        for no, t, kind, ocr in pages:
            if target and target in alnum_lower(t):
                return no, kind, ocr
        no, _, kind, ocr = next(((n, t, k, o) for n, t, k, o in pages if t), pages[0])
        return no, kind, ocr

    @staticmethod
    def _low_finding(res: DocResult, f: FieldResult) -> Finding:
        shown = mask_account(f.value) if f.field_name == "bank_account_number" else f.value
        if "NOT_FOUND" in f.flags:
            msg = f"{field_label(f.field_name)} was not found in '{res.filename}'."
        else:
            reasons = ", ".join(x.replace("_", " ").lower() for x in f.flags) or "low confidence"
            msg = (f"{field_label(f.field_name)} read from '{res.filename}' as '{shown}' is not reliable "
                   f"(confidence {f.confidence:.2f}: {reasons}). Compare with the document image.")
        return Finding(f"EXTRACTION.{res.document_id}.{f.field_name}", "EXTRACTION", "FAIL", field=f.field_name,
                       reason_code="LOW_CONFIDENCE_EXTRACTION", expected="a value confirmed by the document",
                       observed=shown, message=msg)
