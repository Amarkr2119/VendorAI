-- VendorFlow AI: one SQLite database. All tables from the approved design are created now;
-- tables used from Phase 4 onward simply stay empty in Phase 3.
-- Audit rule: history is append-only. Triggers below make that a database guarantee.

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS vendors (
    vendor_id  TEXT PRIMARY KEY,
    legal_name TEXT NOT NULL,
    country    TEXT NOT NULL CHECK (country IN ('IN','US','GB')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    application_id      TEXT PRIMARY KEY,                 -- APP-2026-0001
    vendor_id           TEXT NOT NULL REFERENCES vendors(vendor_id),
    source_ref          TEXT,                              -- synthetic ID such as IN-008 (reconciles with the manifest)
    country             TEXT NOT NULL CHECK (country IN ('IN','US','GB')),
    processing_state    TEXT NOT NULL DEFAULT 'RECEIVED'
                        CHECK (processing_state IN ('RECEIVED','EXTRACTED','NEEDS_ATTENTION','FAILED')),
    current_revision    INTEGER NOT NULL DEFAULT 1,
    current_decision_id INTEGER,                           -- pointer into decisions (Phase 4); no FK to avoid a cycle
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_app_source_ref ON applications(source_ref);
CREATE INDEX IF NOT EXISTS idx_app_country    ON applications(country);

CREATE TABLE IF NOT EXISTS application_revisions (
    revision_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    revision_no  INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    source       TEXT NOT NULL CHECK (source IN ('FORM','JSON','PDF_FORM','SEED','REVIEWER')),
    created_by   TEXT NOT NULL DEFAULT 'vendor',
    created_at   TEXT NOT NULL,
    UNIQUE (application_id, revision_no)
);

CREATE TABLE IF NOT EXISTS application_identifiers (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    revision_no    INTEGER NOT NULL,
    id_type        TEXT NOT NULL,                         -- PAN GSTIN CIN EIN STATE_FILE_NO CRN VAT BANK_ACCOUNT
    value_norm     TEXT NOT NULL,
    value_display  TEXT
);
CREATE INDEX IF NOT EXISTS idx_ident_lookup ON application_identifiers(id_type, value_norm);

CREATE TABLE IF NOT EXISTS documents (
    document_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id     TEXT NOT NULL REFERENCES applications(application_id),
    revision_no        INTEGER NOT NULL,
    declared_type      TEXT NOT NULL,
    detected_type      TEXT,
    detected_confidence REAL,
    filename           TEXT NOT NULL,
    stored_path        TEXT NOT NULL,
    sha256             TEXT NOT NULL,
    mime               TEXT,
    page_count         INTEGER,
    is_scanned         INTEGER,
    text_chars         INTEGER,
    uploaded_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_app    ON documents(application_id);
CREATE INDEX IF NOT EXISTS idx_doc_sha256 ON documents(sha256);

CREATE TABLE IF NOT EXISTS processing_runs (
    run_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id     TEXT NOT NULL REFERENCES applications(application_id),
    revision_no        INTEGER NOT NULL,
    trigger            TEXT NOT NULL CHECK (trigger IN ('initial','reprocess','revision')),
    config_snapshot_json TEXT,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    status             TEXT NOT NULL DEFAULT 'RUNNING'
                       CHECK (status IN ('RUNNING','COMPLETED','COMPLETED_WITH_ERRORS','FAILED'))
);
CREATE INDEX IF NOT EXISTS idx_run_app ON processing_runs(application_id);

CREATE TABLE IF NOT EXISTS processing_logs (
    log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES processing_runs(run_id),
    application_id TEXT NOT NULL,
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL,
    summary        TEXT,
    detail_json    TEXT,
    error_message  TEXT,
    duration_ms    INTEGER,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_app ON processing_logs(application_id);

CREATE TABLE IF NOT EXISTS extracted_fields (
    field_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES processing_runs(run_id),
    document_id      INTEGER NOT NULL REFERENCES documents(document_id),
    field_name       TEXT NOT NULL,
    value            TEXT,
    confidence       REAL,
    confidence_band  TEXT CHECK (confidence_band IN ('HIGH','MEDIUM','LOW')),
    origin           TEXT NOT NULL CHECK (origin IN ('TEXT_LAYER_LLM','OCR_LLM','REGEX','HUMAN')),
    source_kind      TEXT,                                 -- TEXT_LAYER | OCR
    source_page      INTEGER,
    source_snippet   TEXT,
    grounded         INTEGER,
    regex_agrees     INTEGER,
    format_valid     INTEGER,
    form_agrees      INTEGER,
    flags_json       TEXT,
    alternatives_json TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ef_doc ON extracted_fields(document_id, field_name);

CREATE TABLE IF NOT EXISTS extraction_cache (
    cache_key      TEXT PRIMARY KEY,
    doc_sha256     TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    provider       TEXT,
    model          TEXT,
    prompt_version TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_audit_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER,
    category       TEXT NOT NULL CHECK (category IN ('LLM','VERIFICATION')),
    provider       TEXT NOT NULL,
    endpoint       TEXT,
    purpose        TEXT,
    request_masked TEXT,
    status         TEXT NOT NULL,                          -- OK | QUOTA | TRANSIENT | AUTH | CONFIG | BAD_OUTPUT | SKIPPED ...
    latency_ms     INTEGER,
    attempt        INTEGER,
    fallback_from  TEXT,
    error          TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES processing_runs(run_id),
    application_id TEXT NOT NULL,
    rule_id        TEXT NOT NULL,
    category       TEXT NOT NULL,                          -- COMPLETENESS | DOCUMENT | FORMAT
    field          TEXT,
    result         TEXT NOT NULL CHECK (result IN ('PASS','FAIL','SKIPPED')),
    reason_code    TEXT,
    expected       TEXT,
    observed_masked TEXT,
    message        TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_val_run ON validation_results(run_id, rule_id);

-- ---------------------------------------------------------------- tables used from Phase 4
CREATE TABLE IF NOT EXISTS verification_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES processing_runs(run_id),
    application_id TEXT NOT NULL,
    provider       TEXT NOT NULL,
    check_type     TEXT NOT NULL,
    outcome        TEXT NOT NULL CHECK (outcome IN ('VERIFIED','NEGATIVE','UNAVAILABLE','SKIPPED')),
    mode           TEXT NOT NULL CHECK (mode IN ('LIVE','SANDBOX','SIMULATED')),
    attempts       INTEGER,
    response_masked_json TEXT,
    error          TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_health (
    provider            TEXT PRIMARY KEY,
    state               TEXT NOT NULL DEFAULT 'CLOSED',
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_success_at     TEXT,
    last_failure_at     TEXT,
    circuit_open_until  TEXT
);

CREATE TABLE IF NOT EXISTS consistency_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES processing_runs(run_id),
    application_id TEXT NOT NULL,
    check_id       TEXT NOT NULL,
    source_a       TEXT,
    source_b       TEXT,
    score          REAL,
    outcome        TEXT NOT NULL,
    evidence_json  TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_matches (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 INTEGER NOT NULL REFERENCES processing_runs(run_id),
    application_id         TEXT NOT NULL,
    matched_application_id TEXT NOT NULL,
    match_basis            TEXT NOT NULL,
    strength               TEXT NOT NULL,
    created_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_indicators (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES processing_runs(run_id),
    application_id TEXT NOT NULL,
    code           TEXT NOT NULL,
    severity       TEXT NOT NULL,
    evidence_json  TEXT,
    explanation    TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id        TEXT NOT NULL REFERENCES applications(application_id),
    run_id                INTEGER REFERENCES processing_runs(run_id),
    status                TEXT NOT NULL CHECK (status IN ('APPROVED','PENDING','REJECTED')),
    decided_by            TEXT NOT NULL,
    decision_type         TEXT NOT NULL CHECK (decision_type IN ('AUTO','OVERRIDE')),
    policy_mode           TEXT,
    verification_basis    TEXT,
    reason_codes_json     TEXT,
    explanation           TEXT,
    required_actions_json TEXT,
    rule_trace_json       TEXT,
    supersedes_decision_id INTEGER REFERENCES decisions(decision_id),
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_tasks (
    task_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    run_id         INTEGER REFERENCES processing_runs(run_id),
    reason_codes_json TEXT,
    state          TEXT NOT NULL DEFAULT 'OPEN' CHECK (state IN ('OPEN','IN_REVIEW','AWAITING_VENDOR','RESOLVED')),
    assigned_to    TEXT,
    created_at     TEXT NOT NULL,
    resolved_at    TEXT
);

CREATE TABLE IF NOT EXISTS review_actions (
    action_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        INTEGER REFERENCES review_tasks(task_id),
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    reviewer       TEXT NOT NULL,
    action_type    TEXT NOT NULL CHECK (action_type IN ('CORRECT_FIELD','NOTE','REQUEST_INFO','OVERRIDE','REPROCESS')),
    payload_json   TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL REFERENCES applications(application_id),
    decision_id    INTEGER REFERENCES decisions(decision_id),
    channel        TEXT NOT NULL DEFAULT 'email_draft',
    subject        TEXT,
    body           TEXT,
    generation_method TEXT,                                -- LLM | TEMPLATE
    status         TEXT NOT NULL DEFAULT 'DRAFT',
    created_at     TEXT NOT NULL
);

-- ---------------------------------------------------------------- append-only guarantees
CREATE TRIGGER IF NOT EXISTS trg_ef_no_update  BEFORE UPDATE ON extracted_fields BEGIN SELECT RAISE(ABORT, 'extracted_fields is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_ef_no_delete  BEFORE DELETE ON extracted_fields BEGIN SELECT RAISE(ABORT, 'extracted_fields is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_pl_no_update  BEFORE UPDATE ON processing_logs  BEGIN SELECT RAISE(ABORT, 'processing_logs is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_pl_no_delete  BEFORE DELETE ON processing_logs  BEGIN SELECT RAISE(ABORT, 'processing_logs is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_aa_no_update  BEFORE UPDATE ON api_audit_log    BEGIN SELECT RAISE(ABORT, 'api_audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_aa_no_delete  BEFORE DELETE ON api_audit_log    BEGIN SELECT RAISE(ABORT, 'api_audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_dc_no_update  BEFORE UPDATE ON decisions        BEGIN SELECT RAISE(ABORT, 'decisions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_dc_no_delete  BEFORE DELETE ON decisions        BEGIN SELECT RAISE(ABORT, 'decisions is append-only'); END;

-- ---------------------------------------------------------------- views
-- Latest value per (document, field). A HUMAN correction wins over any automated row, and the
-- most recent human correction wins among human rows. Nothing is overwritten.
CREATE VIEW IF NOT EXISTS v_current_fields AS
SELECT * FROM (
    SELECT ef.*, d.application_id AS application_id, d.revision_no AS revision_no,
           d.declared_type AS declared_type, d.detected_type AS detected_type,
           ROW_NUMBER() OVER (PARTITION BY ef.document_id, ef.field_name
                              ORDER BY (ef.origin = 'HUMAN') DESC, ef.field_id DESC) AS rn
    FROM extracted_fields ef JOIN documents d ON d.document_id = ef.document_id
) WHERE rn = 1;

-- Latest result per (run, rule): a later stage may re-evaluate a rule (e.g. document completeness
-- using detected types) by appending a newer row; consumers read this view.
CREATE VIEW IF NOT EXISTS v_latest_validation AS
SELECT vr.* FROM validation_results vr
WHERE vr.id = (SELECT MAX(v2.id) FROM validation_results v2 WHERE v2.run_id = vr.run_id AND v2.rule_id = vr.rule_id);
