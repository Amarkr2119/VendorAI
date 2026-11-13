"""VendorFlow AI - Streamlit interface for vendor onboarding.

The UI is deliberately thin: it collects inputs and makes the deterministic Phase 3/4
services visible. Decisions, verification, review validation and audit history remain
in the domain layer; this module never makes a vendor decision itself.
"""
from __future__ import annotations

import json
import shutil
from typing import Any, Dict, Iterable, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from vendorflow.decision.review import ReviewError, audit_history
from vendorflow.domain.enums import DocType
from vendorflow.domain.fields import COUNTRY_NAMES, DECLARATIONS, field_label, fields_for
from vendorflow.domain.models import DocumentInput
from vendorflow.ingestion.intake import register_application, submission_from_form, submission_from_json
from vendorflow.ingestion.pdf_form import submission_from_pdf_form
from vendorflow.services import build_services, register_synthetic, synthetic_index
from vendorflow.utils.masking import mask_account

st.set_page_config(page_title="VendorFlow AI", page_icon=":material/fact_check:", layout="wide")


@st.cache_resource
def get_services():
    return build_services()


svc = get_services()
STATUS_ICON = {"started": ":material/hourglass_top:", "ok": ":material/check_circle:", "warning": ":material/warning:", "error": ":material/error:", "skipped": ":material/skip_next:"}
DECISION_ICON = {"APPROVED": ":material/verified:", "PENDING": ":material/pending:", "REJECTED": ":material/block:"}
REVIEWER_PAGES = ["Dashboard", "Live run", "Review queue", "Application details"]
VENDOR_PAGES = ["New submission"]
NAV_LABELS = {
    "Dashboard": ":material/dashboard:  Dashboard",
    "New submission": ":material/person_add:  Submit vendor",
    "Live run": ":material/play_circle:  Run processing",
    "Review queue": ":material/fact_check:  Review queue",
    "Application details": ":material/folder_open:  Application details",
}


def safe_frame(rows: Iterable[Dict[str, Any]], columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if columns:
        for col in columns:
            if col not in df:
                df[col] = ""
        return df[columns]
    return df


def masked_value(field: str, value: Any) -> Any:
    return mask_account(str(value or "")) if field == "bank_account_number" else value


def select_application(application_id: str) -> None:
    st.session_state["selected_application"] = application_id


def open_application_details() -> None:
    """Navigate via a widget callback, before Streamlit instantiates the sidebar radio."""
    st.session_state["workspace_nav"] = "Application details"


def sync_navigation_for_role() -> None:
    """Keep the selected workspace page valid when the demo role is changed."""
    role = st.session_state.get("workspace_role", "Reviewer")
    allowed = VENDOR_PAGES if role == "Vendor" else REVIEWER_PAGES
    if st.session_state.get("workspace_nav") not in allowed:
        st.session_state["workspace_nav"] = allowed[0]


def clear_demo_data() -> None:
    """Remove local applications and uploads, then rebuild the cached service on rerun."""
    db_path = svc.settings.db_path
    upload_dir = svc.settings.upload_dir
    if db_path.is_file():
        db_path.unlink()
    if upload_dir.is_dir():
        shutil.rmtree(upload_dir)
    st.cache_resource.clear()
    st.session_state.clear()
    st.rerun()


def request_draft(application_id: str, issues: List[Dict[str, Any]]) -> str:
    """Create a deterministic, application-specific vendor follow-up draft."""
    app = svc.repo.get_application(application_id)
    decision = svc.repo.current_decision(application_id) or {}
    # Prefer issue-level wording where it can identify a particular document.
    # This matters when two applications share UNREADABLE_DOCUMENT but the
    # vendor must replace different files.
    actions = []
    unreadable = [issue.get("subject") for issue in issues if issue.get("code") == "UNREADABLE_DOCUMENT" and issue.get("subject")]
    if unreadable:
        actions.append("Upload a clear, readable replacement for: " + ", ".join(dict.fromkeys(unreadable)) + ".")
    actions.extend(issue.get("required_action") for issue in issues
                   if issue.get("code") != "UNREADABLE_DOCUMENT" and issue.get("required_action"))
    if not actions:
        actions = list(decision.get("required_actions") or [])
    if not actions:
        actions = ["Review the application details and provide the requested correction or document."]
    steps = "\n".join(f"- {action}" for action in dict.fromkeys(actions))
    return (f"Hello {app['legal_name']},\n\n"
            "To continue your vendor onboarding application, please complete the following:\n"
            f"{steps}\n\nThank you.")


def show_run(application_id: str, *, full: bool = True, trigger: str = "initial") -> None:
    """Execute the real generator and show every emitted stage as it happens."""
    events = []
    with st.status(f"Processing {application_id}", expanded=True) as status:
        generator = svc.pipeline.run(application_id, trigger=trigger) if full else svc.pipeline.reevaluate(application_id, trigger=trigger)
        for event in generator:
            events.append(event)
            st.write(f"{STATUS_ICON.get(event.status, '')} **{event.stage}** - {event.message}")
        last = events[-1] if events else None
        status.update(label=f"{application_id}: {last.message if last else 'No events returned'}", state="error" if last and last.status == "error" else "complete")
    select_application(application_id)
    st.success("Live run complete. Review the decision and evidence below.")
    render_application_summary(application_id)


def process_all_synthetic() -> None:
    """Register and process every bundled scenario with a compact, live progress view."""
    entries = synthetic_index()
    if not entries:
        st.error("Synthetic dataset is unavailable. Run scripts/generate_dataset.py.")
        return
    results = []
    with st.status(f"Processing {len(entries)} synthetic scenarios", expanded=True) as status:
        progress = st.progress(0, text="Preparing synthetic applications")
        for number, entry in enumerate(entries, start=1):
            app_id = register_synthetic(svc, entry["ref"])
            # The database audit schema accepts only initial/reprocess/revision.
            # A bulk operation is a reprocess of each registered scenario.
            events = list(svc.pipeline.run(app_id, trigger="reprocess"))
            decision = svc.repo.current_decision(app_id) or {}
            results.append({
                "Scenario": entry["ref"], "Application": app_id, "Vendor": svc.repo.get_application(app_id)["legal_name"],
                "Decision": decision.get("status", "Not recorded"), "Reasons": ", ".join(decision.get("reason_codes", [])) or "-",
            })
            progress.progress(number / len(entries), text=f"Processed {number}/{len(entries)}: {entry['ref']}")
            st.write(f"{STATUS_ICON.get(events[-1].status, '')} **{entry['ref']}** - {events[-1].message}")
        status.update(label=f"Processed all {len(entries)} synthetic scenarios", state="complete")
    st.success("Bulk run complete. The dashboard and review queue now include every scenario.")
    st.dataframe(pd.DataFrame(results), hide_index=True)


def render_application_summary(application_id: str) -> None:
    app = svc.repo.get_application(application_id)
    if not app:
        st.error("Application not found.")
        return
    decision = svc.repo.current_decision(application_id)
    top = st.columns([2, 1, 1, 1])
    top[0].markdown(f"### {app['legal_name']}")
    top[1].metric("Country", COUNTRY_NAMES.get(app["country"], app["country"]))
    top[2].metric("Revision", app["current_revision"])
    top[3].metric("Decision", (decision or {}).get("status", "Not processed"))
    if decision:
        status = decision["status"]
        message = f"{DECISION_ICON.get(status, '')} **{status}** - {decision['explanation']}"
        {"APPROVED": st.success, "PENDING": st.warning, "REJECTED": st.error}.get(status, st.info)(message)
        if decision["required_actions"]:
            st.caption("Required next steps")
            for action in decision["required_actions"]:
                st.write(f"- {action}")
    else:
        st.info("This submission is registered but has not yet been processed.")


def render_dashboard() -> None:
    st.title("Vendor onboarding dashboard")
    st.caption("A single view of submission history, decisions, and review workload.")
    # A one-click, deterministic happy path makes the dashboard useful in a live demo
    # before any vendor has submitted data (and keeps the original smoke-test flow).
    if st.button("Register and process", type="primary", key="dashboard_demo"):
        index = synthetic_index()
        if index:
            show_run(register_synthetic(svc, index[0]["ref"]))
        else:
            st.error("Synthetic dataset is unavailable. Use New submission instead.")
    with st.expander("Clear previous demo data"):
        st.warning("This permanently removes all applications, processing history, review tasks, and uploaded files from this local demo database.")
        confirmed = st.checkbox("I understand that this cannot be undone", key="confirm_clear_data")
        if st.button("Clear all demo data", type="secondary", disabled=not confirmed, key="clear_demo_data"):
            clear_demo_data()
    rows = svc.repo.application_overview()
    counts = pd.Series([r.get("decision_status") or "NOT_RUN" for r in rows]).value_counts().to_dict() if rows else {}
    metrics = st.columns(5)
    metrics[0].metric("Applications", len(rows))
    metrics[1].metric("Approved", counts.get("APPROVED", 0))
    metrics[2].metric("Pending review", counts.get("PENDING", 0))
    metrics[3].metric("Rejected", counts.get("REJECTED", 0))
    metrics[4].metric("Open tasks", sum(int(r.get("open_reviews", 0)) for r in rows))
    if rows:
        decision_data = pd.DataFrame([
            {"Decision": "Approved", "Applications": counts.get("APPROVED", 0)},
            {"Decision": "Pending", "Applications": counts.get("PENDING", 0)},
            {"Decision": "Rejected", "Applications": counts.get("REJECTED", 0)},
        ])
        donut = alt.Chart(decision_data).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("Applications:Q"),
            color=alt.Color("Decision:N", scale=alt.Scale(
                domain=["Approved", "Pending", "Rejected"],
                range=["#22c55e", "#eab308", "#ef4444"],
            ), legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("Decision:N"), alt.Tooltip("Applications:Q")],
        ).properties(height=240)
        country_data = (pd.DataFrame({"Country": [COUNTRY_NAMES.get(row["country"], row["country"]) for row in rows]})
                        .value_counts("Country").rename("Applications").reset_index())
        country_chart = alt.Chart(country_data).mark_bar(color="#60a5fa").encode(
            x=alt.X("Applications:Q", title="Applications"),
            y=alt.Y("Country:N", title=None, sort="-x"),
            tooltip=[alt.Tooltip("Country:N"), alt.Tooltip("Applications:Q")],
        ).properties(height=240)
        decision_column, country_column = st.columns(2)
        with decision_column:
            st.subheader("Decision distribution")
            st.altair_chart(donut)
        with country_column:
            st.subheader("Applications by country")
            st.altair_chart(country_chart)
    if rows:
        st.subheader("All applications")
        st.dataframe(pd.DataFrame([{"Application": r["application_id"], "Vendor": r["legal_name"], "Country": COUNTRY_NAMES.get(r["country"], r["country"]), "Decision": r.get("decision_status") or "Not run", "Processing": r["processing_state"], "Reviewer action now": "Yes" if r["open_reviews"] else "No", "Vendor response needed": "Yes" if r["awaiting_vendor"] else "No", "Revision": r["current_revision"], "Created": r["created_at"]} for r in rows]), hide_index=True)
    else:
        st.info("Start with a new submission, or use a synthetic scenario for a live demo.")
    if rows:
        awaiting_vendor = sum(int(r.get("awaiting_vendor", 0)) for r in rows)
        needs_reviewer = sum(int(r.get("open_reviews", 0)) for r in rows)
        st.subheader("What needs attention")
        st.caption("A plain-language summary of the current workload. Detailed provider checks remain available on each application’s Evidence tab.")
        st.dataframe(pd.DataFrame([
            {"Status": "Approved", "Applications": counts.get("APPROVED", 0), "What it means": "All required checks passed.", "Next step": "No action required."},
            {"Status": "Awaiting vendor", "Applications": awaiting_vendor, "What it means": "A document or correction is required from the vendor.", "Next step": "Wait for the response, then reprocess."},
            {"Status": "Needs reviewer", "Applications": needs_reviewer, "What it means": "A human must assess an issue, match, or verification result.", "Next step": "Claim a task in Review queue."},
            {"Status": "Rejected", "Applications": counts.get("REJECTED", 0), "What it means": "A conclusive disqualification was recorded.", "Next step": "Closed; evidence remains in the audit trail."},
        ]), hide_index=True)


def document_inputs() -> List[DocumentInput]:
    docs: List[DocumentInput] = []
    st.caption("Upload supporting documents. Missing or mismatched documents are surfaced by the live checks.")
    labels = {DocType.REGISTRATION_CERTIFICATE.value: "Registration certificate", DocType.TAX_CERTIFICATE.value: "Tax certificate / W-9 / VAT certificate", DocType.TAX_ID_PROOF.value: "Tax ID proof (India only)", DocType.BANK_PROOF.value: "Bank proof"}
    cols = st.columns(2)
    for index, (dtype, label) in enumerate(labels.items()):
        upload = cols[index % 2].file_uploader(label, type=["pdf", "png", "jpg", "jpeg"], key=f"document_{dtype}")
        if upload is not None:
            docs.append(DocumentInput(dtype, upload.name, upload.getvalue()))
    return docs


def render_new_submission() -> None:
    st.title("New vendor submission")
    st.caption("Submit company, tax, banking, and compliance information. The process will explain every resulting status.")
    input_type = st.radio("Input method", ["Form submission", "JSON submission", "PDF registration form"], horizontal=True)
    if input_type == "JSON submission":
        upload = st.file_uploader("Vendor submission JSON", type=["json"])
        st.caption("Upload the company details as JSON, then attach the supporting documents below.")
        documents = document_inputs()
        if st.button("Register and process", type="primary", key="json_register"):
            if upload is None: st.error("Upload a JSON submission first.")
            else:
                try:
                    submission = submission_from_json(upload.getvalue().decode("utf-8"))
                    # JSON supplies the typed submission fields. Browser uploads
                    # provide the evidence files, which cannot be referenced by
                    # a local path from a cloud-hosted browser session.
                    submission.documents.extend(documents)
                    show_run(register_application(svc.repo, svc.settings, submission, created_by="vendor"))
                except (ValueError, UnicodeDecodeError) as exc: st.error(str(exc))
        return
    if input_type == "PDF registration form":
        registration = st.file_uploader("Registration form", type=["pdf"], key="registration_pdf")
        documents = document_inputs()
        if st.button("Register and process", type="primary", key="pdf_register"):
            if registration is None: st.error("Upload the registration-form PDF first.")
            else:
                try:
                    sub = submission_from_pdf_form(registration.getvalue(), registration.name, documents, backend=svc.extraction.backend, ocr=svc.extraction.ocr)
                    show_run(register_application(svc.repo, svc.settings, sub, created_by="vendor"))
                except ValueError as exc: st.error(str(exc))
        return
    country = st.selectbox("Country of registration", list(COUNTRY_NAMES), format_func=lambda c: COUNTRY_NAMES[c])
    form: Dict[str, str] = {}
    with st.form("vendor_form"):
        identity, compliance, banking = st.tabs(["Company", "Registration & tax", "Banking"])
        bank_fields = {"bank_name", "bank_holder_name", "bank_account_number", "ifsc", "routing_number", "sort_code"}
        with identity:
            for spec in fields_for(country):
                if spec.name in bank_fields or (spec.name not in {"legal_name", "trade_name", "entity_type", "address", "contact_email"}): continue
                form[spec.name] = st.text_input(spec.label, key=f"form_{country}_{spec.name}")
        with compliance:
            for spec in fields_for(country):
                if spec.name in bank_fields or spec.name in form: continue
                form[spec.name] = st.text_input(spec.label, key=f"form_{country}_{spec.name}")
            declarations = {name: st.checkbox(name.replace("_", " ").capitalize(), key=f"declaration_{country}_{name}") for name in DECLARATIONS.get(country, [])}
        with banking:
            for spec in fields_for(country):
                if spec.name in bank_fields:
                    form[spec.name] = st.text_input(spec.label, type="password" if spec.name == "bank_account_number" else "default", key=f"form_{country}_{spec.name}")
        documents = document_inputs()
        submitted = st.form_submit_button("Register and process", type="primary")
    if submitted:
        try: show_run(register_application(svc.repo, svc.settings, submission_from_form(country, form, declarations, documents), created_by="vendor"))
        except ValueError as exc: st.error(str(exc))


def render_live_run() -> None:
    st.title("Live process run")
    st.caption("Use a real submission or a seeded scenario. Each pipeline event appears as it completes.")
    demo, existing = st.tabs(["Synthetic demo", "Existing application"])
    with demo:
        index = synthetic_index()
        if not index: st.error("Synthetic dataset is unavailable. Run scripts/generate_dataset.py.")
        else:
            choices = {f"{item['ref']} - {item['scenario']}": item["ref"] for item in index}
            choice = st.selectbox("Scenario", list(choices), key="synthetic_choice")
            if st.button("Register and process", type="primary", key="synthetic_register"): show_run(register_synthetic(svc, choices[choice]))
            st.divider()
            st.subheader("Run the complete synthetic set")
            st.caption(f"Processes all {len(index)} scenarios using the current configuration: "
                       f"LLM {'enabled' if svc.settings.llm_enabled else 'disabled'}; mode {svc.settings.llm_mode}. "
                       "Existing applications are reprocessed and retain their audit history.")
            if st.button(f"Process all {len(index)} scenarios", type="primary", key="process_all_synthetic"):
                process_all_synthetic()
    with existing:
        apps = svc.repo.application_overview()
        if apps:
            ids = [a["application_id"] for a in apps]
            current = st.selectbox("Application", ids, key="run_application")
            full = st.checkbox("Re-extract documents (slower)", value=False, help="Clear this to re-evaluate current evidence without reading documents again.")
            if st.button("Run checks", type="primary"): show_run(current, full=full, trigger="manual_run")
        else: st.info("No application available yet.")


def render_review_queue() -> None:
    st.title("Human review queue")
    st.caption("Review actions are auditable. The policy refuses overrides for data defects that must be corrected.")
    reviewer = st.text_input("Reviewer name", key="queue_reviewer", placeholder="Required for audit history")
    tasks = svc.review.queue()
    if not tasks:
        st.success("No applications currently need review.")
        return
    st.caption(f"{len(tasks)} active task(s). Resolved tasks are retained in the audit trail but are not shown here.")
    st.dataframe(pd.DataFrame([{"Queue #": position, "Application": t["application_id"], "Vendor": t["legal_name"], "State": t["state"], "Assigned": t.get("assigned_to") or "Unassigned", "Decision": t.get("decision_status"), "Risk": t.get("risk_level") or "-", "Reasons": ", ".join(t["reason_codes"])} for position, t in enumerate(tasks, start=1)]), hide_index=True)
    choices = {f"{t['task_id']} - {t['application_id']} - {t['legal_name']}": t for t in tasks}
    task = choices[st.selectbox("Open review task", list(choices), key="review_task")]
    app_id = task["application_id"]
    select_application(app_id)
    issues = svc.review.current_issues(app_id)
    if issues:
        st.subheader("Blocking issues")
        # Historical decisions created before the UI existed do not have a
        # required_action in their stored rule trace. Show a useful fallback
        # rather than breaking the whole queue for those valid records.
        st.dataframe(safe_frame(({"Code": i["code"], "Subject": i["subject"], "Severity": i["severity"], "Type": i["kind"], "What happened": i["message"], "Required action": i.get("required_action") or "See the decision's required next steps."} for i in issues), ["Code", "Subject", "Severity", "Type", "What happened", "Required action"]), hide_index=True)
    left, middle, right = st.columns(3)
    with left:
        if st.button("Claim task"):
            try: svc.review.claim(task["task_id"], reviewer); st.success("Task claimed.")
            except ReviewError as exc: st.error(" ".join(exc.problems))
    with middle:
        note = st.text_input("Add internal note", key="review_note")
        if st.button("Save note"):
            try: svc.review.add_note(app_id, reviewer, note); st.success("Note saved.")
            except ReviewError as exc: st.error(" ".join(exc.problems))
    with right:
        st.button("Open full details", on_click=open_application_details)
    decision = svc.repo.current_decision(app_id) or {}
    request = st.text_area(
        "Message to vendor",
        value=request_draft(app_id, issues),
        key=f"request_message_{task['task_id']}_{decision.get('decision_id', 'current')}",
        help="This application-specific draft is based on its blocking issues. Edit it before saving if needed.",
    )
    if st.button("Create request-for-information draft"):
        try: svc.review.request_info(app_id, reviewer, request); st.success("Request saved and a vendor-message draft created.")
        except ReviewError as exc: st.error(" ".join(exc.problems))


def render_details() -> None:
    st.title("Application details")
    apps = svc.repo.application_overview()
    if not apps:
        st.info("No applications to inspect yet.")
        return
    ids = [a["application_id"] for a in apps]
    selected = st.session_state.get("selected_application")
    app_id = st.selectbox("Application", ids, index=ids.index(selected) if selected in ids else 0, format_func=lambda x: f"{x} - {svc.repo.get_application(x)['legal_name']}")
    select_application(app_id)
    app, decision = svc.repo.get_application(app_id), svc.repo.current_decision(app_id)
    render_application_summary(app_id)
    evidence, review, messages, audit = st.tabs(["Evidence", "Review actions", "Messages", "Audit trail"])
    docs = svc.repo.list_documents(app_id, app["current_revision"])
    doc_names = {d["document_id"]: d["filename"] for d in docs}
    with evidence:
        st.subheader("Documents")
        st.dataframe(safe_frame(({"Document": d["filename"], "Submitted as": d["declared_type"], "Detected as": d.get("detected_type") or "Not extracted", "Pages": d.get("page_count") or "-", "Scan": "Yes" if d.get("is_scanned") else "No"} for d in docs), ["Document", "Submitted as", "Detected as", "Pages", "Scan"]), hide_index=True)
        st.subheader("Extracted fields")
        fields = svc.repo.current_fields_rev(app_id, app["current_revision"])
        st.dataframe(safe_frame(({"Document": doc_names.get(f["document_id"], ""), "Field": field_label(f["field_name"]), "Value": masked_value(f["field_name"], f["value"]), "Confidence": f["confidence"], "Band": f["confidence_band"], "Origin": f["origin"], "Evidence": (f.get("source_snippet") or "")[:120]} for f in fields), ["Document", "Field", "Value", "Confidence", "Band", "Origin", "Evidence"]), hide_index=True)
        latest = svc.repo.latest_run(app_id)
        if latest:
            st.subheader("Validation and completeness")
            st.dataframe(safe_frame(svc.repo.latest_findings(latest["run_id"]), ["rule_id", "category", "field", "result", "reason_code", "message"]), hide_index=True)
            verifications = svc.repo.run_verifications(latest["run_id"])
            if verifications:
                st.subheader("Verification checks")
                st.dataframe(safe_frame(verifications, ["check_type", "provider", "outcome", "mode", "attempts", "error"]), hide_index=True)
    with review:
        reviewer = st.text_input("Reviewer name", key="details_reviewer")
        if decision and decision["status"] == "PENDING":
            field_options = {f"{doc_names.get(f['document_id'], '')} - {field_label(f['field_name'])}": f for f in svc.repo.current_fields_rev(app_id, app["current_revision"])}
            with st.expander("Confirm or correct an extracted field"):
                if field_options:
                    selected_field = field_options[st.selectbox("Field", list(field_options), key="correct_field_choice")]
                    value = st.text_input("Corrected value", value=str(selected_field["value"] or ""),
                                          type="password" if selected_field["field_name"] == "bank_account_number" else "default",
                                          key="correct_value")
                    reason = st.text_area("What did you check? (minimum 12 characters)", key="correct_reason")
                    if st.button("Save correction and re-evaluate"):
                        try: svc.review.correct_field(app_id, reviewer, selected_field["document_id"], selected_field["field_name"], value, reason); st.success("Correction saved and decision re-evaluated.")
                        except ReviewError as exc: st.error(" ".join(exc.problems))
            with st.expander("Reviewer decision override"):
                action = st.selectbox("Decision", ["APPROVED", "REJECTED"], key="override_decision")
                condition = st.selectbox("Rejection condition", ["R1", "R2"]) if action == "REJECTED" else None
                reason = st.text_area("Decision rationale (minimum 12 characters)", key="override_reason")
                evidence_text = st.text_area("Evidence reference / description", key="override_evidence") if action == "REJECTED" else ""
                if st.button("Record decision override"):
                    try: svc.review.override(app_id, reviewer, action, reason, condition, evidence_text); st.success("Decision recorded and vendor draft created.")
                    except ReviewError as exc: st.error(" ".join(exc.problems))
            if st.button("Re-evaluate current evidence"): show_run(app_id, full=False, trigger="review_recheck")
        else: st.info("Corrections and overrides are available only while the current decision is Pending.")
        actions = svc.repo.list_review_actions(app_id)
        if actions: st.dataframe(safe_frame(({"When": a["created_at"], "Reviewer": a["reviewer"], "Action": a["action_type"], "Details": json.dumps(a["payload"])} for a in actions), ["When", "Reviewer", "Action", "Details"]), hide_index=True)
    with messages:
        notifications = svc.repo.list_notifications(app_id)
        if not notifications: st.info("No vendor communication draft has been created yet.")
        for message in notifications[::-1]:
            with st.expander(f"{message['subject']} - {message['created_at']}"):
                st.code(message["body"], language=None)
                st.download_button("Download draft", message["body"], file_name=f"{app_id}_message.txt", mime="text/plain", key=f"download_message_{message['id']}")
    with audit:
        history = audit_history(svc.repo, app_id)
        st.dataframe(safe_frame(({"When": item["ts"], "Type": item["kind"], "Actor": item["actor"], "Detail": item["summary"]} for item in history), ["When", "Type", "Actor", "Detail"]), hide_index=True)
        st.download_button("Download audit CSV", pd.DataFrame(history).to_csv(index=False).encode("utf-8"), file_name=f"{app_id}_audit.csv", mime="text/csv")


st.markdown("""
<style>
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.4rem; }
[data-testid="stSidebar"] h1 { margin-bottom: 0.05rem; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    border-radius: 0.55rem;
    margin: 0.15rem 0;
    padding: 0.42rem 0.5rem;
    transition: background-color 120ms ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255, 255, 255, 0.06); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255, 75, 75, 0.16);
    font-weight: 650;
}
.sidebar-subtitle { color: #a6a6b0; font-size: 0.92rem; margin: -0.3rem 0 1.75rem; }
.sidebar-status-title { color: #a6a6b0; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("VendorFlow AI")
    st.markdown("<div class='sidebar-subtitle'>AI-assisted, policy-led</div>", unsafe_allow_html=True)
    role = st.selectbox("Role", ["Reviewer", "Vendor"], key="workspace_role", on_change=sync_navigation_for_role)
    st.caption("WORKSPACE")
    allowed_pages = VENDOR_PAGES if role == "Vendor" else REVIEWER_PAGES
    page = st.radio("Workspace navigation", allowed_pages, key="workspace_nav", format_func=lambda item: NAV_LABELS[item], label_visibility="collapsed")
    if page not in allowed_pages: page = allowed_pages[0]  # compatible with the project's lightweight Streamlit test double
    st.divider()
    with st.expander("System status", expanded=False):
        st.markdown("<div class='sidebar-status-title'>Decision mode</div>", unsafe_allow_html=True)
        st.write(f"**{svc.settings.app_mode}**")
        st.caption("AI reads documents. Policy rules make decisions.")
        if not svc.router.has_providers():
            st.info("No LLM is configured. Text-only extraction is active.")
        st.caption("Simulated verification is always labelled.")

if page == "Dashboard": render_dashboard()
elif page == "New submission": render_new_submission()
elif page == "Live run": render_live_run()
elif page == "Review queue": render_review_queue()
else: render_details()
