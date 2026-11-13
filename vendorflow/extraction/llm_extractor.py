"""LLM-based structured extraction: prompt -> router -> validated JSON. The LLM only *reads*; it never decides."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vendorflow.llm.prompts import build_extraction_prompt
from vendorflow.llm.router import LlmRouter


@dataclass
class LlmItem:
    value: Optional[str]
    snippet: Optional[str]
    page: Optional[int]


@dataclass
class LlmExtraction:
    doc_type_guess: Optional[str]
    items: Dict[str, LlmItem] = field(default_factory=dict)


def _json_from_text(text: str) -> Any:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b > a:
            return json.loads(t[a:b + 1])
        raise


def make_parser(expected_names: Sequence[str]):
    """Validator passed to the router. Raises ValueError (=> one repair retry, then fallback) on unusable output."""
    names = set(expected_names)

    def parse(text: str) -> LlmExtraction:
        try:
            data = _json_from_text(text)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"reply is not valid JSON ({e})")
        if not isinstance(data, dict) or not isinstance(data.get("fields"), dict):
            raise ValueError("JSON must be an object with a 'fields' object")
        items: Dict[str, LlmItem] = {}
        for name, raw in data["fields"].items():
            if name not in names:
                continue                                    # unknown fields are dropped, never trusted
            if raw is None:
                items[name] = LlmItem(None, None, None)
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"field '{name}' must be an object with value/snippet/page")
            val, snip, page = raw.get("value"), raw.get("snippet"), raw.get("page")
            items[name] = LlmItem(None if val in (None, "") else str(val).strip(),
                                  None if snip in (None, "") else str(snip),
                                  page if isinstance(page, int) else None)
        guess = data.get("document_type_guess")
        return LlmExtraction(guess if isinstance(guess, str) else None, items)

    return parse


def extraction_to_cache(ex: LlmExtraction) -> Dict[str, Any]:
    return {"doc_type_guess": ex.doc_type_guess,
            "items": {k: {"value": v.value, "snippet": v.snippet, "page": v.page} for k, v in ex.items.items()}}


def extraction_from_cache(d: Dict[str, Any]) -> LlmExtraction:
    return LlmExtraction(d.get("doc_type_guess"), {k: LlmItem(v["value"], v["snippet"], v["page"]) for k, v in d["items"].items()})


def extract_with_llm(router: LlmRouter, country: str, doc_type: str, expected: List[Tuple[str, bool]], text: str,
                     run_id: Optional[int] = None) -> Tuple[LlmExtraction, str, str]:
    system, user = build_extraction_prompt(country, doc_type, expected, text)
    res = router.complete_json(system, user, make_parser([n for n, _ in expected]), run_id=run_id, purpose="extraction")
    return res.parsed, res.provider, res.model
