"""Score extraction against synthetic ground truth. Shared by scripts/report_extraction.py and the tests.

Verdicts per field: CORRECT | FLAGGED_WRONG (wrong but confidence LOW, so a human will see it) |
CONFIDENT_WRONG (wrong AND not LOW: the dangerous case, must be zero) | MISSING.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from vendorflow.db.repositories import Repository
from vendorflow.domain.fields import normalize_value


def score_against_truth(repo: Repository, truth: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ref, docs in truth.items():
        app = repo.find_by_source_ref(ref)
        if not app or not repo.list_runs(app["application_id"]):
            continue                                       # not registered / not processed yet
        cur = repo.current_fields(app["application_id"])
        by_doc = {d["filename"]: d["document_id"] for d in repo.list_documents(app["application_id"])}
        for fname, info in docs.items():
            did = by_doc.get(fname)
            for field, tval in info["fields"].items():
                row = next((f for f in cur if f["document_id"] == did and f["field_name"] == field), None)
                if row is None or row["value"] is None:
                    verdict, got, conf, band = "MISSING", "", None, ""
                else:
                    got, conf, band = row["value"], row["confidence"], row["confidence_band"]
                    ok = normalize_value(app["country"], field, got) == normalize_value(app["country"], field, tval)
                    verdict = "CORRECT" if ok else ("FLAGGED_WRONG" if band == "LOW" else "CONFIDENT_WRONG")
                rows.append({"ref": ref, "document": fname, "scan": info["scan"], "field": field, "truth": tval,
                             "extracted": got, "confidence": conf, "band": band, "verdict": verdict})
    return rows
