"""Shared test helpers (unittest-style so the suite runs with pytest OR `python -m unittest`)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from vendorflow.config import Settings
from vendorflow.db.repositories import Repository


def temp_settings(**overrides) -> Settings:
    tmp = Path(tempfile.mkdtemp(prefix="vf_test_"))
    base = dict(db_path=tmp / "t.db", upload_dir=tmp / "uploads", log_file=None, llm_enabled=False,
                llm_min_interval_s=0.0, ocr_engine="auto")
    base.update(overrides)
    return Settings(**base)


def temp_repo() -> Repository:
    s = temp_settings()
    r = Repository(s.db_path)
    r.init_schema()
    return r


# ----------------------------------------------------------------------------- integration helpers
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path as _P  # noqa: E402

from vendorflow.domain.fields import expected_fields  # noqa: E402
from vendorflow.extraction.ocr import OcrResult  # noqa: E402
from vendorflow.extraction.regex_extractor import regex_candidates  # noqa: E402
from vendorflow.llm.base import LlmError, LlmResponse  # noqa: E402
from vendorflow.llm.router import LlmRouter  # noqa: E402
from vendorflow.services import build_services, register_synthetic  # noqa: E402

DATA = _P(__file__).resolve().parents[1] / "data" / "synthetic"


def load_truth():
    return json.loads((DATA / "ground_truth.json").read_text())


def make_services(router=None, ocr="__default__", **settings):
    s = temp_settings(**settings)
    kw = {} if ocr == "__default__" else {"ocr": ocr}
    return build_services(s, router=router, **kw)


def process_ref(svc, ref):
    app_id = register_synthetic(svc, ref)
    events = svc.pipeline.run_all(app_id)
    return app_id, events


def _doc_from_prompt(user: str):
    text = user.split("<document>\n", 1)[1].rsplit("\n</document>", 1)[0]
    dtype = re.search(r"declared by the uploader\): (\w+)", user).group(1)
    country = re.search(r"Country: (\w+)", user).group(1)
    return country, dtype, text


def perfect_responder(user: str, system: str = "") -> str:
    """A 'perfect LLM': returns exactly what is printed, with real snippets (derived from the prompt text)."""
    country, dtype, text = _doc_from_prompt(user)
    out = {}
    for name, cands in regex_candidates(text, expected_fields(country, dtype)).items():
        out[name] = {"value": cands[0].value, "snippet": cands[0].snippet, "page": 1} if cands else None
    return json.dumps({"document_type_guess": dtype, "fields": out})


def hallucinating_responder(field: str, fake_value: str):
    """An LLM that 'helpfully corrects' one identifier to a value that is NOT in the document."""
    def respond(user: str, system: str = "") -> str:
        data = json.loads(perfect_responder(user))
        if data["fields"].get(field):
            data["fields"][field] = {"value": fake_value, "snippet": f"{field.upper()}: {fake_value}", "page": 1}
        return json.dumps(data)
    return respond


class ResponderClient:
    """LLM client stub driven by a function; counts calls; can be told to fail."""

    def __init__(self, name="gemini", responder=perfect_responder, error=None):
        self.name, self.model, self.responder, self.error, self.calls = name, f"{name}-stub", responder, error, 0

    def generate_json(self, system, user, timeout):
        self.calls += 1
        if self.error:
            raise self.error
        return LlmResponse(self.responder(user), 3)

    def list_models(self, timeout=20.0):
        return []


def stub_router(repo, settings, *clients):
    return LlmRouter(list(clients), settings, audit=repo.audit, sleep=lambda s: None)


class FaultyOcr:
    """OCR stub that returns fixed text (used to inject a specific misread deterministically)."""
    name = "faulty-stub"

    def __init__(self, text: str, conf: float = 0.9):
        self.text, self.conf = text, conf

    def recognize(self, img):
        words = [(w, self.conf) for w in self.text.split()]
        return OcrResult(self.text, words, self.conf, self.name)
