"""Plain-language messages back to the vendor. TEMPLATE-generated (deterministic); an LLM never writes decisions or rules.
Drafts are stored in `notifications`; nothing is sent automatically (no email integration is claimed)."""
from __future__ import annotations

from typing import Mapping, Tuple

from vendorflow.domain.reason_codes import title


def draft(application: Mapping, decision: Mapping) -> Tuple[str, str]:
    name, app_id, status = application["legal_name"], application["application_id"], decision["status"]
    if status == "APPROVED":
        return (f"Vendor application {app_id} approved",
                f"Dear {name},\n\nYour vendor application {app_id} has been approved.\n\nThank you.\n")
    if status == "REJECTED":
        return (f"Vendor application {app_id}: decision",
                f"Dear {name},\n\nAfter review, we are unable to approve vendor application {app_id}.\n"
                f"Reason recorded by the reviewer: {decision['explanation']}\n\nPlease contact procurement if you believe this is an error.\n")
    vendor_actions = [a for a in decision["required_actions"]]
    lines = "\n".join(f"  {n}. {a}" for n, a in enumerate(vendor_actions, 1)) or "  (our team is completing an internal review; no action is needed from you yet)"
    why = "; ".join(title(c) for c in decision["reason_codes"])
    return (f"Action needed: vendor application {app_id}",
            f"Dear {name},\n\nYour vendor application {app_id} is on hold (not rejected). Summary: {why}.\n\n"
            f"What we need:\n{lines}\n\nReply with the corrected information or documents and we will re-check immediately.\n")
