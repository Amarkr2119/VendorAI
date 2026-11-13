"""Intake: accept a vendor submission from a form, JSON, or (see pdf_form.py) a PDF, and register it.

All paths converge on one `Submission`, stored as revision 1 together with its documents.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from vendorflow.config import Settings
from vendorflow.db.repositories import Repository
from vendorflow.domain.enums import DocType, SubmissionSource
from vendorflow.domain.models import DocumentInput, Submission
from vendorflow.ingestion.normalize import (build_payload, canonicalize_declarations, canonicalize_fields, country_or_raise,
                                            extract_identifiers, normalize_fields)
from vendorflow.logging_setup import get_logger
from vendorflow.utils.hashing import sha256_bytes

log = get_logger("intake")
_VALID_DOC_TYPES = {d.value for d in DocType}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg"}


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return name or "document"


def _check_doc(d: DocumentInput) -> None:
    if d.declared_type not in _VALID_DOC_TYPES:
        raise ValueError(f"Unknown document type {d.declared_type!r}; expected one of {sorted(_VALID_DOC_TYPES)}")
    if Path(d.filename).suffix.lower() not in ALLOWED_EXT:
        raise ValueError(f"Unsupported file type for {d.filename!r}; allowed: {sorted(ALLOWED_EXT)}")
    if not d.content:
        raise ValueError(f"Document {d.filename!r} is empty")
    if len(d.content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Document {d.filename!r} exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")


def submission_from_form(country: str, form: Mapping[str, Any], declarations: Optional[Mapping[str, Any]] = None,
                         documents: Optional[List[DocumentInput]] = None, source: str = SubmissionSource.FORM.value,
                         source_ref: Optional[str] = None) -> Submission:
    c = country_or_raise(country)
    return Submission(country=c, fields=canonicalize_fields(form), declarations=canonicalize_declarations(c, declarations or {}),
                      documents=list(documents or []), source=source, source_ref=source_ref)


def submission_from_json(obj: Union[str, Path, Mapping[str, Any]], base_dir: Optional[Path] = None,
                         source: str = SubmissionSource.JSON.value) -> Submission:
    """Accept a dict, a JSON string, or a path to a JSON file.

    Document entries may be {"type", "path"} (relative to base_dir) or {"type", "filename", "content_base64"}.
    """
    if isinstance(obj, (str, Path)):
        text = str(obj)
        if text.lstrip().startswith("{"):
            data = json.loads(text)
        else:
            p = Path(text)
            data = json.loads(p.read_text(encoding="utf-8"))
            base_dir = base_dir or p.parent
    else:
        data = dict(obj)
    if "country" not in data:
        raise ValueError("JSON submission must contain 'country'")
    docs: List[DocumentInput] = []
    for entry in data.get("documents", []):
        if "path" in entry:
            p = Path(entry["path"])
            if not p.is_absolute():
                p = (base_dir or Path(".")) / p
            docs.append(DocumentInput(entry["type"], p.name, p.read_bytes()))
        elif "content_base64" in entry:
            docs.append(DocumentInput(entry["type"], entry.get("filename", "document.pdf"), base64.b64decode(entry["content_base64"])))
        else:
            raise ValueError("Each document needs 'path' or 'content_base64'")
    return submission_from_form(data["country"], data.get("fields", {}), data.get("declarations", {}), docs, source,
                                data.get("ref") or data.get("source_ref"))


def register_application(repo: Repository, settings: Settings, sub: Submission, created_by: str = "vendor") -> str:
    """Persist a submission as a new application (revision 1) and store its documents on disk."""
    sub.country = country_or_raise(sub.country)
    for d in sub.documents:
        _check_doc(d)
    application_id = repo.allocate_application_id()
    folder = settings.upload_dir / application_id
    folder.mkdir(parents=True, exist_ok=True)

    doc_rows, doc_meta, used = [], [], set()
    for d in sub.documents:
        name = _safe_name(d.filename)
        stem, suf, n = Path(name).stem, Path(name).suffix, 1
        while name in used:
            n += 1
            name = f"{stem}_{n}{suf}"
        used.add(name)
        path = folder / name
        path.write_bytes(d.content)
        digest = sha256_bytes(d.content)
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        doc_rows.append({"declared_type": d.declared_type, "filename": name, "stored_path": str(path), "sha256": digest, "mime": mime})
        doc_meta.append({"declared_type": d.declared_type, "filename": name, "sha256": digest})

    payload = build_payload(sub, doc_meta)
    display, norm = normalize_fields(sub.country, sub.fields)
    repo.create_application(application_id=application_id, country=sub.country, legal_name=display.get("legal_name", ""),
                            source_ref=sub.source_ref, source=sub.source, payload=payload,
                            identifiers=extract_identifiers(sub.country, norm, display), documents=doc_rows, created_by=created_by)
    log.info("Registered %s (%s, source=%s, %d documents)", application_id, sub.country, sub.source, len(doc_rows))
    return application_id
