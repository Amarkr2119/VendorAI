"""OCR engines. Tesseract preferred; RapidOCR as an optional pip-only alternative; or none.

An engine returns text plus per-word confidences (0..1) so downstream confidence scoring can use
the quality of the specific words that make up an extracted value.
"""
from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import List, Optional, Protocol, Tuple

from PIL import Image, ImageOps

from vendorflow.utils.text_normalize import alnum_lower


@dataclass
class OcrResult:
    text: str
    words: List[Tuple[str, float]] = field(default_factory=list)       # (word, confidence 0..1)
    mean_conf: float = 0.0
    engine: str = ""

    def confidence_for(self, value: str) -> float:
        """Mean OCR confidence of the words that make up `value`; falls back to a penalized page mean."""
        toks = [alnum_lower(t) for t in value.split() if alnum_lower(t)]
        if not toks:
            return self.mean_conf
        confs = []
        for t in toks:
            hits = [c for w, c in self.words if alnum_lower(w) == t]
            if hits:
                confs.append(sum(hits) / len(hits))
        if len(confs) < len(toks):                       # some tokens were not found as standalone OCR words
            confs.append(self.mean_conf * 0.75)
        return sum(confs) / len(confs)


class OcrEngine(Protocol):
    name: str

    def recognize(self, img: Image.Image) -> OcrResult: ...


class OcrUnavailable(RuntimeError):
    pass


class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        # Resolve the binary first.  The direct command-line fallback below
        # keeps OCR usable when Streamlit's Python environment has Tesseract
        # installed but does not include the optional pytesseract package.
        executable = which("tesseract")
        if not executable:
            executable = next((str(path) for path in (
                Path("/opt/homebrew/bin/tesseract"),
                Path("/usr/local/bin/tesseract"),
            ) if path.is_file()), None)
        if not executable:
            raise OcrUnavailable("Tesseract binary was not found")
        self._command = executable
        self._t = None
        try:
            import pytesseract
            # Desktop-launched Python processes do not always inherit Homebrew's
            # bin directory.  Prefer PATH, then support the two conventional
            # Homebrew locations so OCR behaves the same from a shell and
            # Streamlit/IDE launchers.
            pytesseract.pytesseract.tesseract_cmd = executable
            pytesseract.get_tesseract_version()          # raises if the binary is missing
            self._t = pytesseract
        except Exception as e:
            # Verify that the discovered executable works before accepting the
            # CLI fallback.  Do not mark valid scanned documents unreadable
            # merely because a Python wrapper is absent.
            try:
                subprocess.run([self._command, "--version"], check=True, capture_output=True, text=True, timeout=5)
            except Exception as command_error:
                raise OcrUnavailable(f"Tesseract not available: {command_error}") from e

    @staticmethod
    def _prep(img: Image.Image) -> Image.Image:
        img = img.convert("L")
        if img.width < 1600:                              # small scans OCR much better when upscaled
            f = 1600 / img.width
            img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
        return ImageOps.autocontrast(img, cutoff=1)

    def recognize(self, img: Image.Image) -> OcrResult:
        if self._t is None:
            return self._recognize_cli(img)
        d = self._t.image_to_data(self._prep(img), output_type=self._t.Output.DICT, config="--psm 6")
        lines: dict = {}
        words: List[Tuple[str, float]] = []
        for i, txt in enumerate(d["text"]):
            conf = float(d["conf"][i])
            txt = (txt or "").strip()
            if not txt or conf < 0:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append(txt)
            words.append((txt, conf / 100.0))
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        mean = sum(c for _, c in words) / len(words) if words else 0.0
        return OcrResult(text, words, mean, self.name)

    def _recognize_cli(self, img: Image.Image) -> OcrResult:
        """OCR through the installed Tesseract CLI when pytesseract is absent."""
        payload = io.BytesIO()
        self._prep(img).save(payload, format="PNG")
        try:
            completed = subprocess.run(
                [self._command, "stdin", "stdout", "--psm", "6", "tsv"], input=payload.getvalue(),
                check=True, capture_output=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise OcrUnavailable(f"Tesseract OCR failed: {e}") from e
        lines: dict = {}
        words: List[Tuple[str, float]] = []
        for row in csv.DictReader(io.StringIO(completed.stdout.decode("utf-8", errors="replace")), delimiter="\t"):
            txt = (row.get("text") or "").strip()
            try:
                conf = float(row.get("conf", "-1"))
            except ValueError:
                conf = -1.0
            if not txt or conf < 0:
                continue
            key = (row.get("block_num", ""), row.get("par_num", ""), row.get("line_num", ""))
            lines.setdefault(key, []).append(txt)
            words.append((txt, conf / 100.0))
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        mean = sum(c for _, c in words) / len(words) if words else 0.0
        return OcrResult(text, words, mean, self.name)


class RapidOcrEngine:
    """Optional pure-pip alternative (`pip install rapidocr-onnxruntime`). Not exercised by the sandbox tests."""
    name = "rapidocr"

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise OcrUnavailable("rapidocr-onnxruntime is not installed") from e
        self._engine = RapidOCR()

    def recognize(self, img: Image.Image) -> OcrResult:
        import numpy as np
        result, _ = self._engine(np.asarray(img.convert("RGB")))
        items = sorted(result or [], key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))   # (line bucket, x)
        words = [(r[1], float(r[2])) for r in items]
        text = "\n".join(r[1] for r in items)
        mean = sum(c for _, c in words) / len(words) if words else 0.0
        return OcrResult(text, words, mean, self.name)


def get_ocr_engine(setting: str = "auto") -> Optional[OcrEngine]:
    """Return an engine, or None when OCR is disabled/unavailable (scanned docs are then flagged, never guessed)."""
    if setting == "none":
        return None
    order = {"tesseract": [TesseractEngine], "rapidocr": [RapidOcrEngine]}.get(setting, [TesseractEngine, RapidOcrEngine])
    for cls in order:
        try:
            return cls()
        except OcrUnavailable:
            continue
    return None
