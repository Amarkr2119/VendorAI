import sqlite3
import unittest

from tests.helpers import temp_repo
from vendorflow.domain.models import FieldResult


def _make_app(repo):
    app_id = repo.allocate_application_id()
    repo.create_application(application_id=app_id, country="IN", legal_name="Test Co", source_ref="T-1", source="SEED",
                            payload={"country": "IN", "fields": {}}, identifiers=[{"id_type": "PAN", "value_norm": "AAAAA1111A"}],
                            documents=[{"declared_type": "bank_proof", "filename": "b.pdf", "stored_path": "/x/b.pdf", "sha256": "ab"}])
    return app_id


class TestSchema(unittest.TestCase):
    def test_AT01_tables_triggers_views_and_idempotent_init(self):
        repo = temp_repo()
        repo.init_schema()  # second call must not fail
        names = set(repo.table_names())
        for t in ["vendors", "applications", "application_revisions", "application_identifiers", "documents", "processing_runs",
                  "processing_logs", "extracted_fields", "extraction_cache", "api_audit_log", "validation_results",
                  "verification_results", "provider_health", "consistency_results", "duplicate_matches", "risk_indicators",
                  "decisions", "review_tasks", "review_actions", "notifications", "schema_version"]:
            self.assertIn(t, names)
        self.assertEqual(len(repo.object_names("trigger")), 8)
        self.assertEqual(set(repo.object_names("view")), {"v_current_fields", "v_latest_validation"})
        self.assertTrue(repo.foreign_keys_on())

    def test_foreign_keys_enforced(self):
        repo = temp_repo()
        with self.assertRaises(sqlite3.IntegrityError):
            with repo.tx() as c:
                c.execute("INSERT INTO applications(application_id, vendor_id, country, created_at) VALUES ('A','nope','IN','x')")

    def test_country_check_constraint(self):
        repo = temp_repo()
        with self.assertRaises(sqlite3.IntegrityError):
            repo.create_application(application_id="A1", country="XX", legal_name="n", source_ref=None, source="SEED",
                                    payload={}, identifiers=[], documents=[])


class TestAppendOnly(unittest.TestCase):
    def test_AT02_append_only_tables_reject_update_and_delete(self):
        repo = temp_repo()
        app_id = _make_app(repo)
        run = repo.start_run(app_id, 1, "initial", {})
        doc = repo.list_documents(app_id)[0]["document_id"]
        repo.log_stage(run, app_id, "INTAKE", "ok", "x")
        repo.insert_fields(run, doc, [FieldResult("pan", "AAAAA1111A", 0.9, "HIGH", "REGEX", "TEXT_LAYER")])
        repo.audit(run_id=run, category="LLM", provider="gemini", status="OK")
        with repo.tx() as c:
            c.execute("INSERT INTO decisions(application_id, status, decided_by, decision_type, created_at)"
                      " VALUES (?,?,?,?,?)", (app_id, "PENDING", "SYSTEM", "AUTO", "now"))
        for table in ["extracted_fields", "processing_logs", "api_audit_log", "decisions"]:
            with self.subTest(table=table, op="update"):
                with self.assertRaises(sqlite3.DatabaseError) as cm:
                    with repo.tx() as c:
                        c.execute(f"UPDATE {table} SET created_at='tampered'")
                self.assertIn("append-only", str(cm.exception))
            with self.subTest(table=table, op="delete"):
                with self.assertRaises(sqlite3.DatabaseError) as cm:
                    with repo.tx() as c:
                        c.execute(f"DELETE FROM {table}")
                self.assertIn("append-only", str(cm.exception))


class TestHumanCorrectionView(unittest.TestCase):
    def test_human_row_wins_without_overwriting(self):
        repo = temp_repo()
        app_id = _make_app(repo)
        run = repo.start_run(app_id, 1, "initial", {})
        doc = repo.list_documents(app_id)[0]["document_id"]
        repo.insert_fields(run, doc, [FieldResult("pan", "AAAAB1111A", 0.4, "LOW", "OCR_LLM", "OCR")])
        repo.insert_fields(run, doc, [FieldResult("pan", "AAAAA1111A", 1.0, "HIGH", "HUMAN", "TEXT_LAYER")])
        run2 = repo.start_run(app_id, 1, "reprocess", {})
        repo.insert_fields(run2, doc, [FieldResult("pan", "AAAAB1111A", 0.4, "LOW", "OCR_LLM", "OCR")])  # later automated row
        cur = [f for f in repo.current_fields(app_id) if f["field_name"] == "pan"]
        self.assertEqual(len(cur), 1)
        self.assertEqual(cur[0]["value"], "AAAAA1111A")            # human correction still wins
        self.assertEqual(len([f for f in repo.run_fields(run) if f["field_name"] == "pan"]), 2)  # history preserved


class TestLatestValidationView(unittest.TestCase):
    def test_latest_row_per_rule(self):
        from vendorflow.domain.models import Finding
        repo = temp_repo()
        app_id = _make_app(repo)
        run = repo.start_run(app_id, 1, "initial", {})
        repo.insert_findings(run, app_id, [Finding("COMPLETENESS.DOC.tax_certificate", "COMPLETENESS", "PASS")])
        repo.insert_findings(run, app_id, [Finding("COMPLETENESS.DOC.tax_certificate", "COMPLETENESS", "FAIL", reason_code="MISSING_DOCUMENT")])
        latest = repo.latest_findings(run)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
