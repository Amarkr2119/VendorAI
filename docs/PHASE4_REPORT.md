# Phase 4 report: verification and decision engine

## 1. What was inspected before coding
Phase 3 ZIP read in full: README, PHASE3_REPORT, `db/schema.sql`, domain models/enums/reason codes, repositories, pipeline orchestrator,
extraction service, intake/normalization/completeness, identifier validators, masking utilities, synthetic dataset and manifest, test helpers.
Baseline reproduced before any change: **96 tests, OK, 2 skipped** under `unittest` (handover quoted 95 passed / 1 skipped under pytest;
this sandbox has neither PyMuPDF nor Streamlit, which accounts for the extra skip; pytest is not installed here either).

Findings that shaped the design:
* The schema already contained every Phase 4 table (verification_results, consistency_results, duplicate_matches, risk_indicators, decisions,
  review_tasks, review_actions, notifications, provider_health). **No schema change was needed**; decisions/audit tables are append-only via triggers.
* `review_actions.action_type` is CHECK-constrained, so review semantics (accept declaration, confirm rejection ...) live in `payload_json` and in
  `decisions.decision_type='OVERRIDE'`, not in new action types.
* `v_current_fields` is not revision-scoped; a revision-aware query was added.
* The Phase 3 contract "last event is DONE with `processing_state`" is preserved; Phase 4 stages run before it.
* Synthetic documents carry incorporation / liability / submission dates but no expiry dates. Date rules are implemented (expired, future, incorporation after
  submission) and unit-tested, but the dataset contains no expiry case.

## 2. Files
New: `validation/rules.py`; `verification/{base,simulated,india,usa,uk,registry,service}.py`; `analysis/{names,dates,consistency,duplicates,risk}.py`;
`decision/{policy,engine,stages,review,communication}.py`; `ingestion/revisions.py`; `scripts/{evaluate_decisions,check_verification}.py`;
`tests/unit/test_p4_units.py`; `tests/integration/test_p4_decisions.py`; docs.
Modified (additive): `domain/{enums,models,reason_codes}.py`, `config.py`, `db/repositories.py` (methods appended), `pipeline/orchestrator.py`
(optional `stages` hook, `reevaluate`, human-correction carry-forward), `services.py` (wiring), `.env.example`, `README.md`.
Phase 3 tests modified: **one assertion** (`test_AT16...`): it asserted the literal pipeline version string `phase3-intake-extraction`, which Phase 4 bumps to
`phase4-verification-decision` on purpose. No other Phase 3 test was touched.

## 3. Test results (actually run in this environment)
| Suite | Result |
|---|---|
| Phase 3 (unchanged except the one assertion above) | 96 tests, passing; 2 skipped (PyMuPDF, real Streamlit not installed here) |
| Phase 4 unit (`test_p4_units.py`) | 49 tests, OK |
| Phase 4 integration (`test_p4_decisions.py`) | 40 tests, OK |
| **Full suite** | **185 tests, OK, 2 skipped** |
| Dataset evaluation, DEMO mode | 29/29 decisions match the manifest, 29/29 also match expected reason codes. Distribution: 6 APPROVED, 23 PENDING, 0 REJECTED (correct: reject is human-only) |
| Dataset evaluation, PRODUCTION mode (no live providers configured) | 0 APPROVED, 29 PENDING |

### Mocked vs real (read this before claiming anything about live APIs)
* **No real external API was called by any test or evaluation.** Network access is disabled in this environment, and the tests are designed not to need it.
* Provider results come from (a) the SIMULATED fixture adapters, (b) `LiveStub` adapters (mode LIVE, test-only) used to exercise production-mode logic
  and R2 rejection, and (c) a `StubSession` that stands in for `requests` to test the Companies House and HMRC adapters' request construction and
  response/error mapping.
* Therefore: **the Companies House and HMRC adapters are untested against the real services.** They follow the providers' published APIs as understood;
  confirm with `scripts/check_verification.py --live` and your own keys. HMRC endpoint/headers/OAuth flow in particular should be verified against the
  current HMRC developer documentation.
* LLM tests use stub clients (as in Phase 3). Live Gemini/Groq still requires your keys (`scripts/check_llm.py`).

### Known difference from the manifest (PRODUCTION column)
The manifest's production column assumed simulated providers still run in production. This implementation does not run simulation there unless you explicitly
set `VERIFICATION_PROVIDERS=simulated`; with no live provider, checks are `SKIPPED/NOT_CONFIGURED` and the reason code is `VERIFICATION_NOT_PERFORMED`.
The **status** (Pending) matches in all 29; only the specific reason code differs for IN-010, US-008, UK-007 (they would otherwise show their simulated
negative/unavailable codes). I consider this the correct policy, and say so rather than bending the data to fit.

## 4. A defect found and fixed during testing
The first integration run showed a corrupted PDF produced only `MISSING_DOCUMENT`, never `UNREADABLE_DOCUMENT`: the decision layer rebuilt findings itself
and ignored extraction-time findings, so **a corrupted document could have been ignored by the decision**. Fixed by reading the unreadable-document findings of the
latest extraction run for the current revision (`Repository.unreadable_findings`). Covered by `test_corrupted_pdf_leads_to_pending_with_clear_action`.

## 5. The six mandatory edge cases, and where they are tested
| # | Case | Result | Test |
|---|---|---|---|
| 1 | Missing information/documents | PENDING, vendor-action list, task AWAITING_VENDOR, message drafted | `test_1_...` |
| 2 | Invalid tax identifier format (GSTIN / EIN / VAT) | PENDING, "Correct ..." action, check SKIPPED and **never sent to a provider** | `test_2_...` |
| 3 | Bank holder vs company mismatch | PENDING, task OPEN, no fraud wording | `test_3_...` |
| 4 | Duplicate application | PENDING, STRONG/POSSIBLE match with evidence; original untouched; earlier app never flagged by a later one | `test_4_`, `test_4b_`, repeated-application test |
| 5 | Provider timeout/failure | PENDING `VERIFICATION_UNAVAILABLE`, 3 attempts, no NEGATIVE, recovers on reprocess | `test_5_`, `test_5b_` |
| 6 | Low-confidence / incorrect extraction | PENDING human review; reviewer correction -> AUTO approval; hallucinated value cannot approve | `test_6_`, `test_6b_` |

Also tested: all three decisions, LLM failure and fallback, corrupted PDF, reprocessing after document replacement (human corrections carry forward),
vendor correction revision, reviewer overrides (allowed, refused, sticky, superseded by new issues), rejection rules R1/R2, rejection never auto-reversed,
decision history and append-only enforcement, masking of account numbers in every table Phase 4 writes to, crash -> no decision.

## 6. Assumptions and limitations
* Reviewer identity is a typed name (no authentication).
* When a simulated provider has no fixture for an identifier, `SIM_UNKNOWN=accept` (default) returns a *synthetic acceptance*, labelled
  "SIMULATED: identifier not in fixtures". This lets new demo vendors flow end to end; set `SIM_UNKNOWN=not_found` to make unknown identifiers fail instead.
* Date checks read text-layer PDFs only; scanned documents are not parsed for dates.
* Name similarity uses a deterministic token/sequence score, not a trained model; thresholds are in `analysis/names.py` and documented.
* Duplicate detection compares against all earlier applications in the database (O(n) per application); fine at this scale, not at millions.
* Stored verification evidence masks 8-18 digit runs (EIN/VAT/accounts), so reviewers see masked identifiers in evidence JSON.
* Notifications are drafts only; no email is sent.
