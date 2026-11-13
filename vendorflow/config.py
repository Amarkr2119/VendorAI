"""Configuration: environment variables (+ optional .env) -> a typed Settings object.

Business logic never reads os.environ directly; it receives a Settings instance,
which makes every component easy to test with explicit values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _num(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _clean(value: Optional[str]) -> str:
    """Strip whitespace and trailing '# comment' that .env.example style files may carry."""
    if value is None:
        return ""
    return value.split("#", 1)[0].strip()


@dataclass(frozen=True)
class Settings:
    app_mode: str = "DEMO"
    db_path: Path = ROOT / "data" / "vendorflow.db"
    upload_dir: Path = ROOT / "data" / "uploads"
    log_level: str = "INFO"
    log_file: Optional[Path] = ROOT / "logs" / "vendorflow.log"

    llm_enabled: bool = True
    llm_mode: str = "always"            # always | when_needed (scans, or regex missed a required field)
    llm_primary: str = "gemini"
    llm_fallback: str = "groq"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_min_interval_s: float = 4.0
    llm_quota_cooldown_s: float = 60.0
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = ""
    groq_api_key: str = field(default="", repr=False)
    groq_model: str = ""

    pdf_backend: str = "auto"
    ocr_engine: str = "auto"
    ocr_min_confidence: float = 0.60

    # ---- Phase 4: verification
    verification_providers: str = "auto"    # auto | simulated | none   (auto: LIVE where credentials exist, else DEMO->SIMULATED / PRODUCTION->SKIPPED)
    verify_timeout_s: float = 10.0
    verify_max_retries: int = 2             # retries after the first attempt, transient errors only
    verify_backoff_s: float = 0.5
    verify_circuit_threshold: int = 3
    verify_circuit_cooldown_s: float = 60.0
    sim_unknown: str = "accept"             # simulated provider, identifier absent from fixtures: accept | not_found
    companies_house_api_key: str = field(default="", repr=False)
    companies_house_env: str = "live"       # live | sandbox
    hmrc_client_id: str = field(default="", repr=False)
    hmrc_client_secret: str = field(default="", repr=False)
    hmrc_env: str = "sandbox"               # sandbox | live

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        e = env if env is not None else os.environ

        def path(key: str, default: Path) -> Path:
            raw = _clean(e.get(key))
            if not raw:
                return default
            p = Path(raw)
            return p if p.is_absolute() else ROOT / p

        log_raw = _clean(e.get("LOG_FILE", "logs/vendorflow.log"))
        mode = (_clean(e.get("APP_MODE")) or "DEMO").upper()
        return cls(
            app_mode=mode if mode in {"DEMO", "PRODUCTION"} else "DEMO",
            db_path=path("DB_PATH", ROOT / "data" / "vendorflow.db"),
            upload_dir=path("UPLOAD_DIR", ROOT / "data" / "uploads"),
            log_level=(_clean(e.get("LOG_LEVEL")) or "INFO").upper(),
            log_file=(ROOT / log_raw if log_raw and not Path(log_raw).is_absolute() else (Path(log_raw) if log_raw else None)),
            llm_enabled=_bool(_clean(e.get("LLM_ENABLED")), True),
            llm_mode=(_clean(e.get("LLM_MODE")) or "always").lower(),
            llm_primary=(_clean(e.get("LLM_PRIMARY")) or "gemini").lower(),
            llm_fallback=(_clean(e.get("LLM_FALLBACK")) or "groq").lower(),
            llm_timeout_s=_num(_clean(e.get("LLM_TIMEOUT_S")), 30.0),
            llm_max_retries=int(_num(_clean(e.get("LLM_MAX_RETRIES")), 2)),
            llm_min_interval_s=_num(_clean(e.get("LLM_MIN_INTERVAL_S")), 4.0),
            llm_quota_cooldown_s=_num(_clean(e.get("LLM_QUOTA_COOLDOWN_S")), 60.0),
            gemini_api_key=_clean(e.get("GEMINI_API_KEY")),
            gemini_model=_clean(e.get("GEMINI_MODEL")),
            groq_api_key=_clean(e.get("GROQ_API_KEY")),
            groq_model=_clean(e.get("GROQ_MODEL")),
            pdf_backend=(_clean(e.get("PDF_BACKEND")) or "auto").lower(),
            ocr_engine=(_clean(e.get("OCR_ENGINE")) or "auto").lower(),
            ocr_min_confidence=_num(_clean(e.get("OCR_MIN_CONFIDENCE")), 0.60),
            verification_providers=(_clean(e.get("VERIFICATION_PROVIDERS")) or "auto").lower(),
            verify_timeout_s=_num(_clean(e.get("VERIFY_TIMEOUT_S")), 10.0),
            verify_max_retries=int(_num(_clean(e.get("VERIFY_MAX_RETRIES")), 2)),
            verify_backoff_s=_num(_clean(e.get("VERIFY_BACKOFF_S")), 0.5),
            verify_circuit_threshold=int(_num(_clean(e.get("VERIFY_CIRCUIT_THRESHOLD")), 3)),
            verify_circuit_cooldown_s=_num(_clean(e.get("VERIFY_CIRCUIT_COOLDOWN_S")), 60.0),
            sim_unknown=(_clean(e.get("SIM_UNKNOWN")) or "accept").lower(),
            companies_house_api_key=_clean(e.get("COMPANIES_HOUSE_API_KEY")),
            companies_house_env=(_clean(e.get("COMPANIES_HOUSE_ENV")) or "live").lower(),
            hmrc_client_id=_clean(e.get("HMRC_CLIENT_ID")),
            hmrc_client_secret=_clean(e.get("HMRC_CLIENT_SECRET")),
            hmrc_env=(_clean(e.get("HMRC_ENV")) or "sandbox").lower(),
        )

    def provider_configured(self, name: str) -> bool:
        if name == "gemini":
            return bool(self.gemini_api_key and self.gemini_model)
        if name == "groq":
            return bool(self.groq_api_key and self.groq_model)
        return False


def load_settings() -> Settings:
    """Load .env (if python-dotenv is installed) and build Settings from the environment."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:  # python-dotenv is optional at import time
        pass
    return Settings.from_env()
