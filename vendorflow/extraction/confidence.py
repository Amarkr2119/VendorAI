"""Deterministic, reproducible field confidence.

confidence = weighted mean of the signals that APPLY to the field (weights renormalize over available signals):
    grounded        0.30  quoted snippet exists in the document and contains the value
    regex_agrees    0.25  LLM value equals a value found independently by the regex extractor
    format_valid    0.20  identifier passes its format/check-digit rule
    source_quality  0.15  1.0 for a machine-readable text layer, OCR word confidence for scans
    form_agrees     0.10  equals the value the vendor submitted on the form
The LLM's own claimed confidence is never used. Hard caps then prevent a high-scoring weak value:
    ungrounded -> <= 0.50;  extractors disagree -> <= 0.55;
    invalid format -> <= 0.50 UNLESS the document faithfully matches the submitted (invalid) value;
    differs from the form -> <= 0.50 for OCR sources (could be an OCR error), <= 0.80 for text-layer sources.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

WEIGHTS: Dict[str, float] = {"grounded": 0.30, "regex_agrees": 0.25, "format_valid": 0.20, "source_quality": 0.15, "form_agrees": 0.10}
HIGH, MEDIUM = 0.85, 0.60


def _signal(v) -> Optional[float]:
    if v is None:
        return None
    return float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)


def band(score: float) -> str:
    return "HIGH" if score >= HIGH else "MEDIUM" if score >= MEDIUM else "LOW"


def score_field(*, grounded: Optional[bool], regex_agrees: Optional[bool], format_valid: Optional[bool],
                source_quality: Optional[float], form_agrees: Optional[bool], source_kind: str) -> Tuple[float, str, List[str]]:
    signals = {"grounded": _signal(grounded), "regex_agrees": _signal(regex_agrees), "format_valid": _signal(format_valid),
               "source_quality": None if source_quality is None else max(0.0, min(1.0, float(source_quality))),
               "form_agrees": _signal(form_agrees)}
    avail = {k: v for k, v in signals.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in avail)
    score = sum(WEIGHTS[k] * v for k, v in avail.items()) / total_w if total_w else 0.0

    flags: List[str] = []
    if grounded is False:
        score, flags = min(score, 0.50), flags + ["UNGROUNDED"]
    if regex_agrees is False:
        score, flags = min(score, 0.55), flags + ["EXTRACTORS_DISAGREE"]
    if format_valid is False:
        flags.append("FORMAT_INVALID")
        if form_agrees is not True:
            score = min(score, 0.50)
    if form_agrees is False:
        flags.append("DIFFERS_FROM_FORM")
        score = min(score, 0.50 if source_kind == "OCR" else 0.80)
    return round(score, 3), band(score), flags
