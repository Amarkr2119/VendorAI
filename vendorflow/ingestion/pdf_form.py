"""Intake from an uploaded PDF registration form (deterministic labelled-line parsing; OCR for scanned forms).

Deliberately LLM-free: a vendor registration form has a known layout. Anything the parser cannot read is left
blank and is then reported by the completeness check, never guessed. The form PDF is stored as a document
and is re-extracted (with confidence and evidence) by the normal extraction stage.
"""
from __future__ import annotations

from typing import Any, List, Optional

from vendorflow.domain.enums import SubmissionSource
from vendorflow.domain.fields import DECLARATIONS, expected_fields, normalize_country
from vendorflow.domain.models import DocumentInput, Submission
from vendorflow.extraction import pdf_text
from vendorflow.extraction.ocr import get_ocr_engine
from vendorflow.extraction.regex_extractor import labelled_lines, regex_candidates
from vendorflow.ingestion.normalize import parse_bool
from vendorflow.utils.text_normalize import alnum_lower

_DECL_LABELS = {"gstregistered": ("gst_not_registered", "IN"), "vatregistered": ("vat_not_registered", "GB")}


def _form_text(data: bytes, filename: str, backend: Any, ocr: Any) -> str:
    content = pdf_text.read_document(data, filename, backend)
    if content.error:
        raise ValueError(f"The registration form could not be read: {content.error}")
    parts: List[str] = []
    for p in content.pages:
        if not p.needs_ocr:
            parts.append(p.text)
        elif ocr is not None:
            parts.append(ocr.recognize(pdf_text.render_page(data, filename, p.page_no - 1, backend)).text)
    text = "\n".join(parts)
    if len(alnum_lower(text)) < 30:
        raise ValueError("The registration form contains no readable text (scanned form and no OCR available?)")
    return text


def submission_from_pdf_form(data: bytes, filename: str, other_documents: Optional[List[DocumentInput]] = None,
                             backend: Any = None, ocr: Any = "auto", source_ref: Optional[str] = None) -> Submission:
    backend = backend or pdf_text.get_backend("auto")
    ocr = get_ocr_engine("auto") if ocr == "auto" else ocr
    text = _form_text(data, filename, backend, ocr)
    lines = labelled_lines(text)

    country = next((normalize_country(v) for lab, v, _ in lines if lab == "country"), None)
    if not country:                                           # infer from country-specific identifier labels
        labels = {lab for lab, _, _ in lines}
        country = ("IN" if labels & {"gstin", "pan", "cin", "ifsccode"} else "US" if labels & {"ein", "abaroutingnumber"}
                   else "GB" if labels & {"companynumber", "vatnumber", "sortcode"} else None)
    if not country:
        raise ValueError("Could not determine the country from the registration form (expected a 'Country:' line)")

    cands = regex_candidates(text, expected_fields(country, "registration_form"))
    fields = {name: cs[0].value for name, cs in cands.items() if cs}
    decl = {d: False for d in DECLARATIONS[country]}
    for lab, val, _ in lines:
        if lab in _DECL_LABELS and _DECL_LABELS[lab][1] == country:
            decl[_DECL_LABELS[lab][0]] = not parse_bool(val)        # "GST Registered: No" -> not registered
    docs = [DocumentInput("registration_form", filename, data)] + list(other_documents or [])
    return Submission(country=country, fields=fields, declarations=decl, documents=docs,
                      source=SubmissionSource.PDF_FORM.value, source_ref=source_ref)
