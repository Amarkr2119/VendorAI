# Phase 3 test report (backend foundation and document processing)

Environment: Python 3.12.3, pypdfium2 PDF backend, no network. The current application uses Gemini image OCR for scanned documents.

## Test run
`python -m unittest discover -s tests -t . -v` -> **96 tests, OK, 2 skipped** (41.7 s).
Skipped (packages not installable offline): real-Streamlit `AppTest`; PyMuPDF-vs-pdfium comparison. Both run automatically on a machine that has the packages.

| Module | Tests | Covers |
|---|---|---|
| unit/test_db | 6 | AT-01 schema/triggers/views, AT-02 append-only, FK + CHECK enforcement, human-correction view, latest-validation view |
| unit/test_normalize_completeness | 12 | AT-04 normalization, AT-06 completeness for all 29 manifest scenarios, intake persistence, input rejection |
| unit/test_llm | 13 | AT-12 router (quota, backoff, auth, bad JSON repair, circuit breaker, rate limit, audit) and REST clients, with stubs |
| unit/test_extraction_units | 31 (1 skipped) | AT-03 config, AT-10 grounding, AT-11 confidence, classifier, regex extractor, LLM JSON parser, backend fallback, masking, identifiers |
| integration/test_extraction_pipeline | 28 | AT-05 (PDF form == JSON for all 29), AT-07, AT-08, AT-09 (122 documents), AT-13 cache, AT-14, AT-15, AT-16, AT-17, LLM paths, edge case 6, resilience |
| integration/test_app_smoke | 3 (1 skipped) | AT-18 app.py (recording fake of Streamlit; real AppTest on your machine) |

## Extraction accuracy (29 applications, 672 ground-truth fields, deterministic mode, no LLM)
CORRECT 666 (99.1%) | FLAGGED_WRONG 5 (0.7%) | **CONFIDENT_WRONG 0** | MISSING 1 (flagged LOW).
All 6 clean machine-readable applications: 100% correct, none flagged. Every wrong value came from the deliberately poor scans
(IN-009, US-009, UK-008) and was flagged LOW confidence.

## Mutation checks
Six deliberate code breakages (grounding disabled, invalid-format cap removed, classifier disabled, cache disabled, log masking
removed, human-correction precedence removed) were each caught by at least one test. Two tests were strengthened after the first
mutation pass showed they were weaker than intended.

## Defects found and fixed during Phase 3
1. Footer text leaked the document type into classification (synthetic-data artifact) -> neutral reference + classify top of page only.
2. Honestly-invalid identifiers on the registration form were capped LOW -> form PDF is now compared with the stored submission.
3. `insert_findings` masked `observed` but not `message` -> masking enforced inside the repository for all text columns.
4. Test harness unloaded numpy via `patch.dict(sys.modules)` -> patch a single entry.

## Not tested live (no network in the build environment)
Gemini and Groq REST calls, `google` model listing, real Streamlit rendering, PyMuPDF backend. Use `scripts/check_llm.py` and
`pytest` locally.
