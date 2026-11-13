from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:  # so f"{Country.IN}" renders 'IN'
        return str(self.value)


class Country(StrEnum):
    IN = "IN"
    US = "US"
    GB = "GB"


class DocType(StrEnum):
    REGISTRATION_FORM = "registration_form"
    REGISTRATION_CERTIFICATE = "registration_certificate"
    TAX_CERTIFICATE = "tax_certificate"      # IN: GST certificate | US: W-9 | GB: VAT certificate
    TAX_ID_PROOF = "tax_id_proof"            # IN: PAN proof (not used in US/GB)
    BANK_PROOF = "bank_proof"


class SubmissionSource(StrEnum):
    FORM = "FORM"
    JSON = "JSON"
    PDF_FORM = "PDF_FORM"
    SEED = "SEED"
    REVIEWER = "REVIEWER"


class ProcessingState(StrEnum):
    RECEIVED = "RECEIVED"
    EXTRACTED = "EXTRACTED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    FAILED = "FAILED"


class Origin(StrEnum):
    TEXT_LAYER_LLM = "TEXT_LAYER_LLM"
    OCR_LLM = "OCR_LLM"
    REGEX = "REGEX"
    HUMAN = "HUMAN"


class Band(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class StageStatus(StrEnum):
    STARTED = "started"
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class Stage(StrEnum):
    INTAKE = "INTAKE"
    EXTRACTION = "EXTRACTION"
    VALIDATION = "VALIDATION"
    VERIFICATION = "VERIFICATION"
    CONSISTENCY = "CONSISTENCY"
    DUPLICATES = "DUPLICATES"
    RISK = "RISK"
    DECISION = "DECISION"
    COMMUNICATION = "COMMUNICATION"


class VerificationOutcome(StrEnum):
    VERIFIED = "VERIFIED"
    NEGATIVE = "NEGATIVE"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"


class VerificationMode(StrEnum):
    LIVE = "LIVE"
    SANDBOX = "SANDBOX"
    SIMULATED = "SIMULATED"


class DecisionStatus(StrEnum):
    APPROVED = "APPROVED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
