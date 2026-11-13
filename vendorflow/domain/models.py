"""Typed data containers (stdlib dataclasses; no third-party model library is required)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentInput:
    declared_type: str              # a DocType value chosen by the vendor / uploader
    filename: str
    content: bytes


@dataclass
class Submission:
    """The canonical vendor submission, independent of how it arrived (form, JSON, PDF)."""
    country: str                                                # IN | US | GB
    fields: Dict[str, str] = field(default_factory=dict)       # canonical field -> display value
    declarations: Dict[str, bool] = field(default_factory=dict)
    documents: List[DocumentInput] = field(default_factory=list)
    source: str = "FORM"
    source_ref: Optional[str] = None                            # e.g. synthetic ID "IN-001"


@dataclass
class Finding:
    """One completeness / validation / document finding, persisted to validation_results."""
    rule_id: str
    category: str                   # COMPLETENESS | DOCUMENT | FORMAT
    result: str                     # PASS | FAIL | SKIPPED
    field: Optional[str] = None
    reason_code: Optional[str] = None
    expected: Optional[str] = None
    observed: Optional[str] = None
    message: str = ""


@dataclass
class FieldResult:
    """One extracted field with confidence and evidence."""
    field_name: str
    value: Optional[str]
    confidence: float
    band: str
    origin: str
    source_kind: str                # TEXT_LAYER | OCR
    source_page: Optional[int] = None
    source_snippet: Optional[str] = None
    grounded: Optional[bool] = None
    regex_agrees: Optional[bool] = None
    format_valid: Optional[bool] = None
    form_agrees: Optional[bool] = None
    flags: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StageEvent:
    stage: str
    status: str                     # started | ok | warning | error | skipped
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)
    ts: str = ""


@dataclass
class Issue:
    """One unresolved matter found by any Phase 4 component. The decision engine works only on Issues + verification results.

    `key` identifies the issue precisely (code + subject) so a reviewer's override can resolve exactly these issues
    and so a later reprocess can tell 'same issue' from 'new issue'.
    """
    code: str                       # a REASON_CODES key
    subject: str                    # what it is about: a field, a check type, an application id...
    message: str
    severity: str = "MEDIUM"        # INFO | LOW | MEDIUM | HIGH
    kind: str = "REVIEW"            # VENDOR_ACTION (vendor must fix) | REVIEW (human judgement) | SYSTEM (retry / config)
    required_action: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.code}:{self.subject}"
