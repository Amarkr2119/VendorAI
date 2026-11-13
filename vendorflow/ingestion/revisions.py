"""Corrections after submission: a vendor (or a reviewer, on the vendor's behalf) supplies changed fields and/or replacement
documents. This creates revision N+1; revision N and its documents, fields and decisions are kept untouched."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from vendorflow.config import Settings
from vendorflow.db.repositories import Repository
from vendorflow.domain.fields import display_value
from vendorflow.domain.models import DocumentInput
from vendorflow.ingestion.intake import _check_doc, _safe_name
from vendorflow.ingestion.normalize import canonicalize_declarations, canonicalize_fields, extract_identifiers, normalize_fields
from vendorflow.utils.hashing import sha256_bytes


def create_revision(repo: Repository, settings: Settings, application_id: str, *, field_updates: Optional[Mapping[str, Any]] = None,
                    declaration_updates: Optional[Mapping[str, Any]] = None, new_documents: Optional[List[DocumentInput]] = None,
                    source: str = "FORM", created_by: str = "vendor") -> int:
    app = repo.get_application(application_id)
    if not app:
        raise ValueError(f"Unknown application {application_id}")
    if not (field_updates or declaration_updates or new_documents):
        raise ValueError("Nothing to change: provide at least one corrected field, declaration or document.")
    country = app["country"]
    cur_rev = app["current_revision"]
    payload = dict(repo.get_revision(application_id, cur_rev)["payload"])
    fields = dict(payload["fields"])
    for k, v in canonicalize_fields(field_updates or {}).items():
        fields[k] = display_value(v)
    decl = dict(payload["declarations"])
    decl.update(canonicalize_declarations(country, declaration_updates or {}) if declaration_updates else {})
    display, norm = normalize_fields(country, fields)

    for d in new_documents or []:
        _check_doc(d)
    replaced = {d.declared_type for d in (new_documents or [])}
    new_rev = cur_rev + 1
    folder = settings.upload_dir / application_id / f"rev{new_rev}"
    folder.mkdir(parents=True, exist_ok=True)
    doc_rows, meta = [], []
    for old in repo.list_documents(application_id, cur_rev):         # carry forward what was not replaced (same stored file, same hash)
        if old["declared_type"] in replaced:
            continue
        doc_rows.append({k: old[k] for k in ("declared_type", "filename", "stored_path", "sha256", "mime")})
        meta.append({"declared_type": old["declared_type"], "filename": old["filename"], "sha256": old["sha256"]})
    for d in new_documents or []:
        name = _safe_name(d.filename)
        path = folder / name
        path.write_bytes(d.content)
        digest = sha256_bytes(d.content)
        doc_rows.append({"declared_type": d.declared_type, "filename": name, "stored_path": str(path), "sha256": digest,
                         "mime": mimetypes.guess_type(name)[0] or "application/octet-stream"})
        meta.append({"declared_type": d.declared_type, "filename": name, "sha256": digest})
    payload.update({"fields": display, "normalized": norm, "declarations": decl, "documents": meta, "source": source})
    return repo.add_revision(application_id, payload=payload, source=source, created_by=created_by,
                             identifiers=extract_identifiers(country, norm, display), documents=doc_rows,
                             legal_name=display.get("legal_name", ""))
