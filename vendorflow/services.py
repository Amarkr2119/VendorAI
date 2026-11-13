"""Service wiring: builds the repository, LLM router, extraction service and pipeline from Settings.

The Streamlit app and the scripts call this; business modules never import Streamlit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from vendorflow.config import ROOT, Settings, load_settings
from vendorflow.db.repositories import Repository
from vendorflow.extraction.service import ExtractionService, _UNSET
from vendorflow.ingestion.intake import register_application, submission_from_json
from vendorflow.llm.router import LlmRouter, build_router
from vendorflow.logging_setup import setup_logging
from vendorflow.decision.review import ReviewService
from vendorflow.decision.stages import DecisionStages
from vendorflow.pipeline.orchestrator import Pipeline
from vendorflow.verification.registry import ProviderRegistry
from vendorflow.verification.service import VerificationService

SYNTHETIC_DIR = ROOT / "data" / "synthetic"


@dataclass
class Services:
    settings: Settings
    repo: Repository
    router: LlmRouter
    extraction: ExtractionService
    pipeline: Pipeline
    review: ReviewService
    registry: ProviderRegistry


def build_services(settings: Optional[Settings] = None, router: Optional[LlmRouter] = None, ocr: Any = _UNSET,
                   session: Any = None, registry: Optional[ProviderRegistry] = None, sleep: Any = None, today: Any = None) -> Services:
    settings = settings or load_settings()
    setup_logging(settings.log_level, settings.log_file)
    repo = Repository(settings.db_path)
    repo.init_schema()
    router = router or build_router(settings, audit=repo.audit, session=session)
    extraction = ExtractionService(repo, settings, router, ocr=ocr)
    registry = registry or ProviderRegistry(settings, session=session)
    verifier = VerificationService(repo, settings, registry, **({"sleep": sleep} if sleep else {}))
    pipeline = Pipeline(repo, settings, extraction, DecisionStages(repo, settings, verifier, today=today))
    return Services(settings, repo, router, extraction, pipeline, ReviewService(repo, settings, pipeline), registry)


def synthetic_index(directory: Path = SYNTHETIC_DIR) -> List[Dict[str, Any]]:
    path = directory / "applications.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def register_synthetic(svc: Services, ref: str, directory: Path = SYNTHETIC_DIR) -> str:
    """Register a synthetic application once (idempotent on source_ref); returns its application_id."""
    existing = svc.repo.find_by_source_ref(ref)
    if existing:
        return existing["application_id"]
    entry = next((a for a in synthetic_index(directory) if a["ref"] == ref), None)
    if not entry:
        raise ValueError(f"Unknown synthetic application {ref}")
    sub = submission_from_json(entry, base_dir=directory, source="SEED")
    return register_application(svc.repo, svc.settings, sub, created_by="seed")
