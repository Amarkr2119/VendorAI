# VendorFlow AI

AI-assisted vendor onboarding for India, USA and UK (Zamp case study PS-2). Python + Streamlit + SQLite.

**Status: Phase 5 UI complete.** `app.py` is now a five-screen Streamlit workspace over the Phase 3/4 backend: dashboard,
new submission, live pipeline run, human-review queue, and application details/audit history.

Principles: AI only *reads* documents; deterministic Python makes every decision. Text-based PDFs work without an
external API; scanned documents use Gemini OCR. Simulated verification is never presented as real.

## Setup

### macOS (primary)
```bash
git clone <your repo> && cd vendorflow
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                   # add a Gemini API key and model for scanned-document OCR
```

Scanned documents use Gemini image OCR (`OCR_ENGINE=gemini`). No Tesseract, RapidOCR, or other local OCR installation is required.
Text-based PDFs still extract locally without an API call. If Gemini is unavailable, a scanned document is safely marked unreadable
rather than guessed.

## Run

```bash
python scripts/init_db.py                 # create SQLite database and all tables (idempotent)
python scripts/generate_dataset.py        # regenerate the synthetic dataset (deterministic; already included)
python scripts/seed_db.py                 # register the 29 synthetic applications (inputs only)
python scripts/process_all.py --no-llm    # process text-based PDFs only; scans require Gemini OCR
python scripts/report_extraction.py       # score extraction against ground truth -> data/extraction_report.csv
streamlit run app.py                      # full vendor-onboarding workspace
pytest -v                                 # or: python -m unittest discover -s tests -t . -v
```

## Test the LLM providers LIVE (needs your own free-tier keys)

Put `GEMINI_API_KEY` / `GROQ_API_KEY` in `.env`, then:

```bash
python scripts/check_llm.py --list-models          # choose models, then set GEMINI_MODEL / GROQ_MODEL in .env
python scripts/check_llm.py --ping gemini          # one minimal real request
python scripts/check_llm.py --ping groq
python scripts/check_llm.py --extract-sample gemini    # real structured extraction on a synthetic GST certificate
python scripts/check_llm.py --fallback-test        # invalid Gemini key on purpose: must fall back to Groq
python scripts/process_all.py --only IN-001 IN-009 # full pipeline with the LLM
```

These live paths were only tested with stubs in the build environment. Gemini is also used for scanned-document OCR. Model names are never hardcoded: Google's lineup changes
quickly, so you choose from the list your key can use. Free-tier Gemini content may be used by Google to improve its products:
send only synthetic data.

## Layout

```
app.py                      Streamlit entry point (five-page vendor-onboarding workspace)
vendorflow/                 core package, no Streamlit imports
  config.py logging_setup.py services.py evaluation.py
  domain/      enums, field model, reason codes, dataclasses
  db/          schema.sql (21 tables, 8 append-only triggers, 2 views), repositories.py
  ingestion/   intake (form/JSON), pdf_form (PDF registration form), normalize, completeness
  rules/       completeness_{in,us,uk}.yaml
  extraction/  pdf_text (PyMuPDF + pypdfium2 fallback), ocr, doc_classifier, regex_extractor,
               llm_extractor, grounding, confidence, service
  llm/         gemini_client, groq_client (REST), router (retry / cooldown / fallback / audit), prompts
  pipeline/    orchestrator (generator of stage events), events
  validation/  identifiers.py (format + check-digit primitives; Phase 4 builds rule packs on these)
  utils/       masking, hashing, text_normalize
scripts/       init_db, generate_dataset, seed_db, process_all, check_llm, report_extraction
data/synthetic/ applications.json, ground_truth.json, manifest.csv, simulation_fixtures.json,
                internal_blocklist.json, documents/<ref>/*
tests/         unit/ and integration/ (96 tests)
```

## Synthetic dataset (29 applications, 122 documents)

India 10, USA 10, UK 9. Everything is fictional; identifiers are format-valid (checksums included). `manifest.csv` records, per
application, the scenario, expected completeness result, expected decision (DEMO and PRODUCTION), reason codes and required actions.
`ground_truth.json` holds the value printed on every document for scoring. Rejection is demonstrated only through a documented
synthetic condition confirmed by a human reviewer (US-010); simulated evidence never produces Rejected (IN-010 stays Pending).

## Phase 4: verification, consistency, duplicates, risk, decision, human review

Added **on top of** Phase 3 (same `app.py`, same SQLite database, same pipeline generator). The pipeline now continues after extraction:
`VALIDATION -> VERIFICATION -> CONSISTENCY -> DUPLICATES -> RISK -> DECISION -> COMMUNICATION`.

| Area | Where | What it does |
|---|---|---|
| Format rules | `vendorflow/validation/rules.py` | Country rule packs (IN/US/UK): PAN, GSTIN (+embedded-PAN link), CIN, IFSC, EIN, ABA, company no., VAT mod-97, sort code, accounts. A valid format never implies existence. |
| Providers | `vendorflow/verification/` | `base.py` types, `india.py` `usa.py` `uk.py` adapters, `registry.py` (which adapter serves which check), `service.py` (timeout, retry with backoff, circuit breaker, audit, evidence). |
| Consistency | `vendorflow/analysis/consistency.py`, `dates.py`, `names.py` | Submission vs every document: company name, bank holder, identifiers, address, country evidence, dates. |
| Duplicates | `vendorflow/analysis/duplicates.py` | Weighted multi-field score against EARLIER applications; internal blocklist (fictional). |
| Risk | `vendorflow/analysis/risk.py` | Explainable indicators with severity and evidence. They prompt review; they never accuse. |
| Decision | `vendorflow/decision/policy.py`, `engine.py`, `stages.py` | Deterministic, versioned (`POLICY_VERSION`), no LLM. |
| Review | `vendorflow/decision/review.py`, `ingestion/revisions.py` | Queue, notes, field corrections, request-info, validated overrides, rejection rules R1/R2, reprocessing, audit history. |

### Decision policy in one table

| Status | When |
|---|---|
| **APPROVED** | Every mandatory check is VERIFIED in a mode the operating policy accepts, and no blocking issue is open. |
| **PENDING** | Anything unresolved: missing/invalid data, mismatch, duplicate, provider failure, low-confidence read, unverified check. |
| **REJECTED** | **Never automatic.** Only a reviewer, citing a documented condition with evidence: **R1** internal disqualification record (blocklist) or **R2** a *LIVE* provider reporting a terminal status. Simulated/sandbox results can never support rejection. |

* **DEMO** (`APP_MODE=DEMO`): SIMULATED / SANDBOX / LIVE verification can satisfy mandatory checks. Simulated results are labelled `SIMULATED` everywhere.
* **PRODUCTION**: only LIVE verification can. With no live provider configured every check is `SKIPPED (NOT_CONFIGURED)` and vendors stay Pending.
* Outcome (`VERIFIED | NEGATIVE | UNAVAILABLE | SKIPPED`) and mode (`LIVE | SANDBOX | SIMULATED`) are separate fields. A timeout is `UNAVAILABLE`, never `NEGATIVE`; HTTP 200 is not `VERIFIED`; there is **no fallback from a failed live call to simulation**.
* Reviewer overrides: human-judgement issues (duplicate, name/address mismatch, declaration, blocklist clearance) can be overridden with a reason; data defects (missing/invalid/unreadable/low-confidence, identifier conflicts, verification not live/not performed) must be *fixed*, not overridden. In PRODUCTION verification gaps can never be overridden.
* Decisions are append-only (database triggers). Reprocessing never deletes history: unchanged evidence appends nothing; changed evidence appends a new decision linked to the one it supersedes. A reviewer's override stands unless a NEW issue appears or the vendor submits a new revision. A rejection is never reversed automatically.

### Provider status (honest)

| Check | LIVE possible? | Notes |
|---|---|---|
| UK Companies House | Yes, with `COMPANIES_HOUSE_API_KEY` (free) | adapter written to the published API; tested here only with a stub HTTP session |
| UK HMRC VAT | Yes, with `HMRC_CLIENT_ID` / `HMRC_CLIENT_SECRET` (sandbox or live) | same caveat |
| India GSTIN / PAN / bank | No public API | needs a licensed provider: plug in with `ProviderRegistry.register()` |
| India IFSC | Format only | no authoritative source bundled |
| US TIN match | No | IRS TIN Matching needs e-Services enrolment; simulated only |
| US state registry | No single public API | simulated only |
| UK / US bank ownership | No | only banks/PSPs offer it; simulated only |

Run `python scripts/check_verification.py` to see what each check resolves to under your `.env`, and `--live` to make one real lookup with your own keys.

### Phase 4 tests

```bash
python -m unittest discover -s tests -t .          # everything (also works with: pytest)
python -m unittest tests.unit.test_p4_units tests.integration.test_p4_decisions   # Phase 4 only
python scripts/evaluate_decisions.py --mode DEMO --out docs/PHASE4_DECISION_EVALUATION.md
```

See `docs/PHASE4_REPORT.md` for the actual results and what is mocked vs real.

## Design notes and known limitations

- **Format validity is not verification.** `validation/identifiers.py` only checks patterns and check digits.
- **Confidence is computed deterministically** from grounding, extractor agreement, format validity, source quality and agreement
  with the submitted form. The LLM's self-reported confidence is never used.
- **EIN prefix list** (`ein_prefix_known`) is a commonly published list; verify against current IRS guidance before relying on it.
- **Gemini OCR is a cloud service.** It returns transcription text but no calibrated word-level confidence, so the application
  uses a conservative page confidence and still applies its deterministic validation rules.
- **Storage:** raw bank account numbers are stored in the submission payload because verification needs them; they are masked in
  logs, audit tables and the UI. A production system would add field-level encryption.
- **Reviewer identity** (Phase 4) is a self-declared name, not authentication.
