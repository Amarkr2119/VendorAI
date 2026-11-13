"""SQLite access layer. All SQL lives here; business modules never build SQL strings.

Single-process, single-user usage: one connection per Repository, guarded by a lock so Streamlit's
script threads cannot interleave statements.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from vendorflow.domain.models import FieldResult, Finding
from vendorflow.utils.masking import mask_text

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _b(v: Optional[bool]) -> Optional[int]:
    return None if v is None else int(bool(v))


class Repository:
    def __init__(self, db_path: Path | str):
        self.path = str(db_path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            with self.conn:                       # commit on success, rollback on exception
                yield self.conn

    def _all(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def _one(self, sql: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------ schema
    def init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            if not self._one("SELECT 1 AS x FROM schema_version"):
                self.conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (?,?)", (SCHEMA_VERSION, utcnow()))
            self.conn.commit()

    def table_names(self) -> List[str]:
        return [r["name"] for r in self._all("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

    def object_names(self, kind: str) -> List[str]:
        return [r["name"] for r in self._all("SELECT name FROM sqlite_master WHERE type=?", (kind,))]

    def foreign_keys_on(self) -> bool:
        return bool(self._one("PRAGMA foreign_keys")["foreign_keys"])

    # ------------------------------------------------------------------ applications
    def allocate_application_id(self, year: Optional[int] = None) -> str:
        year = year or datetime.now(timezone.utc).year
        row = self._one("SELECT COUNT(*) AS n FROM applications")
        return f"APP-{year}-{row['n'] + 1:04d}"

    def create_application(self, *, application_id: str, country: str, legal_name: str, source_ref: Optional[str],
                           source: str, payload: Dict[str, Any], identifiers: List[Dict[str, str]],
                           documents: List[Dict[str, Any]], created_by: str = "vendor") -> str:
        """Create vendor + application + revision 1 + identifiers + document rows in ONE transaction."""
        now = utcnow()
        with self.tx() as c:
            vendor_id = f"VND-{c.execute('SELECT COUNT(*) FROM vendors').fetchone()[0] + 1:04d}"
            c.execute("INSERT INTO vendors(vendor_id, legal_name, country, created_at) VALUES (?,?,?,?)",
                      (vendor_id, legal_name or "(unnamed)", country, now))
            c.execute("INSERT INTO applications(application_id, vendor_id, source_ref, country, created_at) VALUES (?,?,?,?,?)",
                      (application_id, vendor_id, source_ref, country, now))
            c.execute("INSERT INTO application_revisions(application_id, revision_no, payload_json, source, created_by, created_at)"
                      " VALUES (?,?,?,?,?,?)", (application_id, 1, json.dumps(payload, ensure_ascii=False), source, created_by, now))
            for ident in identifiers:
                c.execute("INSERT INTO application_identifiers(application_id, revision_no, id_type, value_norm, value_display)"
                          " VALUES (?,?,?,?,?)", (application_id, 1, ident["id_type"], ident["value_norm"], ident.get("value_display")))
            for d in documents:
                c.execute("INSERT INTO documents(application_id, revision_no, declared_type, filename, stored_path, sha256, mime, uploaded_at)"
                          " VALUES (?,?,?,?,?,?,?,?)",
                          (application_id, 1, d["declared_type"], d["filename"], d["stored_path"], d["sha256"], d.get("mime"), now))
        return application_id

    def get_application(self, application_id: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT a.*, v.legal_name FROM applications a JOIN vendors v USING(vendor_id) WHERE application_id=?",
                         (application_id,))

    def find_by_source_ref(self, source_ref: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM applications WHERE source_ref=? ORDER BY created_at LIMIT 1", (source_ref,))

    def list_applications(self) -> List[Dict[str, Any]]:
        return self._all("SELECT a.*, v.legal_name FROM applications a JOIN vendors v USING(vendor_id) ORDER BY a.application_id")

    def set_processing_state(self, application_id: str, state: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE applications SET processing_state=? WHERE application_id=?", (state, application_id))

    def get_revision(self, application_id: str, revision_no: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if revision_no is None:
            row = self._one("SELECT current_revision FROM applications WHERE application_id=?", (application_id,))
            if not row:
                return None
            revision_no = row["current_revision"]
        r = self._one("SELECT * FROM application_revisions WHERE application_id=? AND revision_no=?", (application_id, revision_no))
        if r:
            r["payload"] = json.loads(r["payload_json"])
        return r

    def list_identifiers(self, application_id: str, revision_no: Optional[int] = None) -> List[Dict[str, Any]]:
        if revision_no is None:
            return self._all("SELECT * FROM application_identifiers WHERE application_id=?", (application_id,))
        return self._all("SELECT * FROM application_identifiers WHERE application_id=? AND revision_no=?", (application_id, revision_no))

    # ------------------------------------------------------------------ documents
    def list_documents(self, application_id: str, revision_no: Optional[int] = None) -> List[Dict[str, Any]]:
        if revision_no is None:
            return self._all("SELECT * FROM documents WHERE application_id=? ORDER BY document_id", (application_id,))
        return self._all("SELECT * FROM documents WHERE application_id=? AND revision_no=? ORDER BY document_id",
                         (application_id, revision_no))

    def update_document_meta(self, document_id: int, *, detected_type: Optional[str], detected_confidence: Optional[float],
                             page_count: Optional[int], is_scanned: Optional[bool], text_chars: Optional[int]) -> None:
        with self.tx() as c:
            c.execute("UPDATE documents SET detected_type=?, detected_confidence=?, page_count=?, is_scanned=?, text_chars=?"
                      " WHERE document_id=?", (detected_type, detected_confidence, page_count, _b(is_scanned), text_chars, document_id))

    # ------------------------------------------------------------------ runs and logs
    def start_run(self, application_id: str, revision_no: int, trigger: str, config_snapshot: Dict[str, Any]) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO processing_runs(application_id, revision_no, trigger, config_snapshot_json, started_at)"
                            " VALUES (?,?,?,?,?)", (application_id, revision_no, trigger, json.dumps(config_snapshot), utcnow()))
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE processing_runs SET finished_at=?, status=? WHERE run_id=?", (utcnow(), status, run_id))

    def list_runs(self, application_id: str) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM processing_runs WHERE application_id=? ORDER BY run_id", (application_id,))

    def log_stage(self, run_id: int, application_id: str, stage: str, status: str, summary: str = "",
                  detail: Optional[Dict[str, Any]] = None, error: Optional[str] = None, duration_ms: Optional[int] = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO processing_logs(run_id, application_id, stage, status, summary, detail_json, error_message,"
                      " duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                      (run_id, application_id, stage, status, mask_text(summary),
                       mask_text(json.dumps(detail, ensure_ascii=False)) if detail else None,
                       mask_text(error) if error else None, duration_ms, utcnow()))

    def list_logs(self, application_id: str) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM processing_logs WHERE application_id=? ORDER BY log_id", (application_id,))

    # ------------------------------------------------------------------ extracted fields
    def insert_fields(self, run_id: int, document_id: int, fields: List[FieldResult]) -> None:
        now = utcnow()
        with self.tx() as c:
            for f in fields:
                c.execute(
                    "INSERT INTO extracted_fields(run_id, document_id, field_name, value, confidence, confidence_band, origin,"
                    " source_kind, source_page, source_snippet, grounded, regex_agrees, format_valid, form_agrees, flags_json,"
                    " alternatives_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, document_id, f.field_name, f.value, f.confidence, f.band, f.origin, f.source_kind, f.source_page,
                     f.source_snippet, _b(f.grounded), _b(f.regex_agrees), _b(f.format_valid), _b(f.form_agrees),
                     json.dumps(f.flags), json.dumps(f.alternatives) if f.alternatives else None, now))

    def current_fields(self, application_id: str) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM v_current_fields WHERE application_id=? ORDER BY document_id, field_id", (application_id,))

    def run_fields(self, run_id: int) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM extracted_fields WHERE run_id=? ORDER BY field_id", (run_id,))

    # ------------------------------------------------------------------ extraction cache
    def cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        r = self._one("SELECT * FROM extraction_cache WHERE cache_key=?", (key,))
        if r:
            r["result"] = json.loads(r["result_json"])
        return r

    def cache_put(self, key: str, doc_sha256: str, doc_type: str, provider: str, model: str, prompt_version: str,
                  result: Dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO extraction_cache(cache_key, doc_sha256, doc_type, provider, model, prompt_version,"
                      " result_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (key, doc_sha256, doc_type, provider, model, prompt_version, json.dumps(result), utcnow()))

    # ------------------------------------------------------------------ audit of external calls
    def audit(self, *, run_id: Optional[int], category: str, provider: str, status: str, endpoint: str = "",
              purpose: str = "", request: Any = None, latency_ms: Optional[int] = None, attempt: Optional[int] = None,
              fallback_from: Optional[str] = None, error: Optional[str] = None) -> None:
        from vendorflow.utils.masking import mask_payload
        req = json.dumps(mask_payload(request), ensure_ascii=False)[:2000] if request is not None else None
        with self.tx() as c:
            c.execute("INSERT INTO api_audit_log(run_id, category, provider, endpoint, purpose, request_masked, status, latency_ms,"
                      " attempt, fallback_from, error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (run_id, category, provider, endpoint, purpose, req, status, latency_ms, attempt, fallback_from,
                       mask_text(error) if error else None, utcnow()))

    def list_audit(self, run_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if run_id is None:
            return self._all("SELECT * FROM api_audit_log ORDER BY id")
        return self._all("SELECT * FROM api_audit_log WHERE run_id=? ORDER BY id", (run_id,))

    # ------------------------------------------------------------------ validation / completeness findings
    def insert_findings(self, run_id: int, application_id: str, findings: List[Finding]) -> None:
        now = utcnow()
        with self.tx() as c:
            for f in findings:
                c.execute("INSERT INTO validation_results(run_id, application_id, rule_id, category, field, result, reason_code,"
                          " expected, observed_masked, message, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (run_id, application_id, f.rule_id, f.category, f.field, f.result, f.reason_code,
                           mask_text(f.expected) if f.expected else None,
                           mask_text(f.observed) if f.observed else None, mask_text(f.message), now))

    def latest_findings(self, run_id: int) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM v_latest_validation WHERE run_id=? ORDER BY id", (run_id,))

    # ================================================================== Phase 4 additions
    # ------------------------------------------------------------------ revisions (vendor / reviewer corrections)
    def list_revisions(self, application_id: str) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM application_revisions WHERE application_id=? ORDER BY revision_no", (application_id,))
        for r in rows:
            r["payload"] = json.loads(r["payload_json"])
        return rows

    def add_revision(self, application_id: str, *, payload: Dict[str, Any], source: str, created_by: str,
                     identifiers: List[Dict[str, str]], documents: List[Dict[str, Any]], legal_name: str = "") -> int:
        """Append revision N+1 (never overwrites revision N). `documents` are the FULL document set of the new revision."""
        now = utcnow()
        with self.tx() as c:
            cur = c.execute("SELECT current_revision FROM applications WHERE application_id=?", (application_id,)).fetchone()
            if not cur:
                raise ValueError(f"Unknown application {application_id}")
            rev = cur[0] + 1
            c.execute("INSERT INTO application_revisions(application_id, revision_no, payload_json, source, created_by, created_at)"
                      " VALUES (?,?,?,?,?,?)", (application_id, rev, json.dumps(payload, ensure_ascii=False), source, created_by, now))
            for ident in identifiers:
                c.execute("INSERT INTO application_identifiers(application_id, revision_no, id_type, value_norm, value_display)"
                          " VALUES (?,?,?,?,?)", (application_id, rev, ident["id_type"], ident["value_norm"], ident.get("value_display")))
            for d in documents:
                c.execute("INSERT INTO documents(application_id, revision_no, declared_type, filename, stored_path, sha256, mime, uploaded_at)"
                          " VALUES (?,?,?,?,?,?,?,?)",
                          (application_id, rev, d["declared_type"], d["filename"], d["stored_path"], d["sha256"], d.get("mime"), now))
            c.execute("UPDATE applications SET current_revision=?, processing_state='RECEIVED' WHERE application_id=?", (rev, application_id))
            if legal_name:
                c.execute("UPDATE vendors SET legal_name=? WHERE vendor_id=(SELECT vendor_id FROM applications WHERE application_id=?)",
                          (legal_name, application_id))
            return rev

    # ------------------------------------------------------------------ revision-aware field access
    def current_fields_rev(self, application_id: str, revision_no: int) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM v_current_fields WHERE application_id=? AND revision_no=? ORDER BY document_id, field_id",
                         (application_id, revision_no))

    def human_fields_for_sha(self, application_id: str, sha256: str, before_revision: int) -> List[Dict[str, Any]]:
        """Current HUMAN corrections made on an identical file in an earlier revision (carried forward to the new revision)."""
        return self._all(
            "SELECT f.* FROM v_current_fields f JOIN documents d ON d.document_id=f.document_id "
            "WHERE d.application_id=? AND d.sha256=? AND d.revision_no<? AND f.origin='HUMAN'", (application_id, sha256, before_revision))

    def insert_human_field(self, run_id: int, document_id: int, field_name: str, value: str, snippet: str = "") -> None:
        self.insert_fields(run_id, document_id, [FieldResult(field_name, value, 1.0, "HIGH", "HUMAN", "TEXT_LAYER", None,
                                                             snippet[:300], None, None, None, None, ["HUMAN_CORRECTED"], [])])

    def latest_run(self, application_id: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM processing_runs WHERE application_id=? ORDER BY run_id DESC LIMIT 1", (application_id,))

    # ------------------------------------------------------------------ verification
    def insert_verification(self, run_id: int, application_id: str, *, provider: str, check_type: str, outcome: str, mode: str,
                            attempts: int, response: Optional[Dict[str, Any]], error: Optional[str]) -> None:
        from vendorflow.utils.masking import mask_payload
        with self.tx() as c:
            c.execute("INSERT INTO verification_results(run_id, application_id, provider, check_type, outcome, mode, attempts,"
                      " response_masked_json, error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (run_id, application_id, provider, check_type, outcome, mode, attempts,
                       json.dumps(mask_payload(response), ensure_ascii=False) if response is not None else None,
                       mask_text(error) if error else None, utcnow()))

    def run_verifications(self, run_id: int) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM verification_results WHERE run_id=? ORDER BY id", (run_id,))
        for r in rows:
            r["response"] = json.loads(r["response_masked_json"]) if r["response_masked_json"] else {}
        return rows

    def get_provider_health(self, provider: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM provider_health WHERE provider=?", (provider,))

    def record_provider_result(self, provider: str, ok: bool, threshold: int, cooldown_s: float) -> None:
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        h = self.get_provider_health(provider) or {"consecutive_failures": 0}
        fails = 0 if ok else h["consecutive_failures"] + 1
        opened = (now + timedelta(seconds=cooldown_s)).isoformat(timespec="seconds") if (not ok and fails >= threshold) else None
        with self.tx() as c:
            c.execute("INSERT INTO provider_health(provider, state, consecutive_failures, last_success_at, last_failure_at, circuit_open_until)"
                      " VALUES (?,?,?,?,?,?) ON CONFLICT(provider) DO UPDATE SET state=excluded.state,"
                      " consecutive_failures=excluded.consecutive_failures,"
                      " last_success_at=COALESCE(excluded.last_success_at, provider_health.last_success_at),"
                      " last_failure_at=COALESCE(excluded.last_failure_at, provider_health.last_failure_at),"
                      " circuit_open_until=excluded.circuit_open_until",
                      (provider, "OPEN" if opened else "CLOSED", fails, now.isoformat(timespec="seconds") if ok else None,
                       None if ok else now.isoformat(timespec="seconds"), opened))

    # ------------------------------------------------------------------ consistency / duplicates / risk
    def insert_consistency(self, run_id: int, application_id: str, rows: List[Dict[str, Any]]) -> None:
        now = utcnow()
        with self.tx() as c:
            for r in rows:
                c.execute("INSERT INTO consistency_results(run_id, application_id, check_id, source_a, source_b, score, outcome,"
                          " evidence_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                          (run_id, application_id, r["check_id"], r.get("source_a"), r.get("source_b"), r.get("score"), r["outcome"],
                           mask_text(json.dumps(r.get("evidence") or {}, ensure_ascii=False)), now))

    def run_consistency(self, run_id: int) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM consistency_results WHERE run_id=? ORDER BY id", (run_id,))
        for r in rows:
            r["evidence"] = json.loads(r["evidence_json"] or "{}")
        return rows

    def insert_duplicates(self, run_id: int, application_id: str, matches: List[Dict[str, Any]]) -> None:
        now = utcnow()
        with self.tx() as c:
            for m in matches:
                c.execute("INSERT INTO duplicate_matches(run_id, application_id, matched_application_id, match_basis, strength, created_at)"
                          " VALUES (?,?,?,?,?,?)", (run_id, application_id, m["matched_application_id"],
                                                    mask_text(json.dumps(m["basis"], ensure_ascii=False)), m["strength"], now))

    def run_duplicates(self, run_id: int) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM duplicate_matches WHERE run_id=? ORDER BY id", (run_id,))
        for r in rows:
            r["basis"] = json.loads(r["match_basis"])
        return rows

    def insert_risk(self, run_id: int, application_id: str, indicators: List[Dict[str, Any]]) -> None:
        now = utcnow()
        with self.tx() as c:
            for i in indicators:
                c.execute("INSERT INTO risk_indicators(run_id, application_id, code, severity, evidence_json, explanation, created_at)"
                          " VALUES (?,?,?,?,?,?,?)", (run_id, application_id, i["code"], i["severity"],
                                                      mask_text(json.dumps(i.get("evidence") or {}, ensure_ascii=False)),
                                                      mask_text(i.get("explanation", "")), now))

    def run_risk(self, run_id: int) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM risk_indicators WHERE run_id=? ORDER BY id", (run_id,))
        for r in rows:
            r["evidence"] = json.loads(r["evidence_json"] or "{}")
        return rows

    # ------------------------------------------------------------------ duplicate-detection lookups (other applications)
    def applications_before(self, application_id: str) -> List[Dict[str, Any]]:
        """Applications registered earlier than this one (the earlier one is the 'original'; ids are allocated sequentially)."""
        return self._all("SELECT a.*, v.legal_name FROM applications a JOIN vendors v USING(vendor_id)"
                         " WHERE a.application_id < ? ORDER BY a.application_id", (application_id,))

    def current_identifiers(self, application_id: str) -> List[Dict[str, Any]]:
        return self._all("SELECT i.* FROM application_identifiers i JOIN applications a ON a.application_id=i.application_id"
                         " AND a.current_revision=i.revision_no WHERE i.application_id=?", (application_id,))

    # ------------------------------------------------------------------ decisions
    def insert_decision(self, application_id: str, *, run_id: Optional[int], status: str, decided_by: str, decision_type: str,
                        policy_mode: str, verification_basis: Any, reason_codes: List[str], explanation: str,
                        required_actions: List[str], rule_trace: Dict[str, Any]) -> int:
        """Append a decision (append-only table), link it as the supersessor of the current one, and move the pointer."""
        with self.tx() as c:
            prev = c.execute("SELECT current_decision_id FROM applications WHERE application_id=?", (application_id,)).fetchone()
            prev_id = prev[0] if prev else None
            cur = c.execute(
                "INSERT INTO decisions(application_id, run_id, status, decided_by, decision_type, policy_mode, verification_basis,"
                " reason_codes_json, explanation, required_actions_json, rule_trace_json, supersedes_decision_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (application_id, run_id, status, decided_by, decision_type, policy_mode,
                 json.dumps(verification_basis, ensure_ascii=False), json.dumps(reason_codes), mask_text(explanation),
                 json.dumps([mask_text(a) for a in required_actions], ensure_ascii=False),
                 mask_text(json.dumps(rule_trace, ensure_ascii=False)), prev_id, utcnow()))
            c.execute("UPDATE applications SET current_decision_id=? WHERE application_id=?", (cur.lastrowid, application_id))
            return int(cur.lastrowid)

    @staticmethod
    def _decision_row(r: Dict[str, Any]) -> Dict[str, Any]:
        r["reason_codes"] = json.loads(r["reason_codes_json"] or "[]")
        r["required_actions"] = json.loads(r["required_actions_json"] or "[]")
        r["rule_trace"] = json.loads(r["rule_trace_json"] or "{}")
        r["verification_basis_data"] = json.loads(r["verification_basis"] or "null")
        return r

    def list_decisions(self, application_id: str) -> List[Dict[str, Any]]:
        return [self._decision_row(r) for r in self._all("SELECT * FROM decisions WHERE application_id=? ORDER BY decision_id", (application_id,))]

    def current_decision(self, application_id: str) -> Optional[Dict[str, Any]]:
        r = self._one("SELECT d.* FROM decisions d JOIN applications a ON a.current_decision_id=d.decision_id WHERE a.application_id=?",
                      (application_id,))
        return self._decision_row(r) if r else None

    # ------------------------------------------------------------------ review tasks / actions / notifications
    def open_review_task(self, application_id: str, run_id: Optional[int], reason_codes: List[str], state: str) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO review_tasks(application_id, run_id, reason_codes_json, state, created_at) VALUES (?,?,?,?,?)",
                            (application_id, run_id, json.dumps(reason_codes), state, utcnow()))
            return int(cur.lastrowid)

    def resolve_open_tasks(self, application_id: str) -> int:
        with self.tx() as c:
            cur = c.execute("UPDATE review_tasks SET state='RESOLVED', resolved_at=? WHERE application_id=? AND state!='RESOLVED'",
                            (utcnow(), application_id))
            return cur.rowcount

    def set_task_state(self, task_id: int, state: str, assigned_to: Optional[str] = None) -> None:
        with self.tx() as c:
            c.execute("UPDATE review_tasks SET state=?, assigned_to=COALESCE(?, assigned_to), resolved_at=CASE WHEN ?='RESOLVED' THEN ? ELSE resolved_at END"
                      " WHERE task_id=?", (state, assigned_to, state, utcnow(), task_id))

    def active_task(self, application_id: str) -> Optional[Dict[str, Any]]:
        r = self._one("SELECT * FROM review_tasks WHERE application_id=? AND state!='RESOLVED' ORDER BY task_id DESC LIMIT 1", (application_id,))
        if r:
            r["reason_codes"] = json.loads(r["reason_codes_json"] or "[]")
        return r

    def list_tasks(self, states: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        sql = ("SELECT t.*, a.country, v.legal_name FROM review_tasks t JOIN applications a USING(application_id) "
               "JOIN vendors v ON v.vendor_id=a.vendor_id")
        params: List[Any] = []
        if states:
            sql += f" WHERE t.state IN ({','.join('?' * len(states))})"
            params = list(states)
        rows = self._all(sql + " ORDER BY t.task_id", params)
        for r in rows:
            r["reason_codes"] = json.loads(r["reason_codes_json"] or "[]")
        return rows

    def add_review_action(self, application_id: str, task_id: Optional[int], reviewer: str, action_type: str, payload: Dict[str, Any]) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO review_actions(task_id, application_id, reviewer, action_type, payload_json, created_at) VALUES (?,?,?,?,?,?)",
                            (task_id, application_id, reviewer, action_type, mask_text(json.dumps(payload, ensure_ascii=False)), utcnow()))
            return int(cur.lastrowid)

    def list_review_actions(self, application_id: str) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM review_actions WHERE application_id=? ORDER BY action_id", (application_id,))
        for r in rows:
            r["payload"] = json.loads(r["payload_json"] or "{}")
        return rows

    def add_notification(self, application_id: str, decision_id: Optional[int], subject: str, body: str, method: str = "TEMPLATE") -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO notifications(application_id, decision_id, subject, body, generation_method, created_at) VALUES (?,?,?,?,?,?)",
                            (application_id, decision_id, subject, mask_text(body), method, utcnow()))
            return int(cur.lastrowid)

    def list_notifications(self, application_id: str) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM notifications WHERE application_id=? ORDER BY id", (application_id,))

    # ------------------------------------------------------------------ dashboard / search (everything derives from the database)
    def application_overview(self) -> List[Dict[str, Any]]:
        """One row per application with its CURRENT decision status (None = not yet decided)."""
        return self._all(
            "SELECT a.application_id, a.source_ref, a.country, a.processing_state, a.current_revision, a.created_at, v.legal_name,"
            " d.status AS decision_status, d.decision_type, d.created_at AS decided_at, d.reason_codes_json,"
            " (SELECT COUNT(*) FROM review_tasks t WHERE t.application_id=a.application_id AND t.state IN ('OPEN','IN_REVIEW')) AS open_reviews,"
            " (SELECT COUNT(*) FROM review_tasks t WHERE t.application_id=a.application_id AND t.state='AWAITING_VENDOR') AS awaiting_vendor"
            " FROM applications a JOIN vendors v USING(vendor_id) LEFT JOIN decisions d ON d.decision_id=a.current_decision_id"
            " ORDER BY a.application_id")

    def verification_stats(self) -> List[Dict[str, Any]]:
        """Counts of verification results from each application's LATEST run, grouped by outcome and mode."""
        return self._all(
            "SELECT vr.outcome, vr.mode, vr.check_type, COUNT(*) AS n FROM verification_results vr "
            "WHERE vr.run_id = (SELECT MAX(run_id) FROM verification_results x WHERE x.application_id=vr.application_id) "
            "GROUP BY vr.outcome, vr.mode, vr.check_type")

    def unreadable_findings(self, application_id: str, revision_no: int) -> List[Dict[str, Any]]:
        """UNREADABLE_DOCUMENT findings from the latest run of this revision that actually performed extraction.
        (A re-evaluation run does not re-read documents, so the readability verdict of the last extraction still applies.)"""
        row = self._one("SELECT MAX(l.run_id) AS run_id FROM processing_logs l JOIN processing_runs r ON r.run_id=l.run_id "
                        "WHERE l.application_id=? AND r.revision_no=? AND l.stage='EXTRACTION'", (application_id, revision_no))
        if not row or not row["run_id"]:
            return []
        return [f for f in self.latest_findings(row["run_id"]) if f["reason_code"] == "UNREADABLE_DOCUMENT" and f["result"] == "FAIL"]
