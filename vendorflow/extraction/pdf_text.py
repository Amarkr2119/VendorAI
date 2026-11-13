"""PDF text layer + page rendering behind a small backend interface.

PyMuPDF is the primary backend. pypdfium2 is an automatic fallback so the app still runs where PyMuPDF
is not installed. Pages whose text layer is (almost) empty are marked `needs_ocr` (scanned pages).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from PIL import Image

MIN_TEXT_CHARS = 40            # a page with fewer extractable characters is treated as scanned
IMAGE_EXT = {".png", ".jpg", ".jpeg"}


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class PageContent:
    page_no: int                 # 1-based
    text: str
    char_count: int
    needs_ocr: bool


@dataclass
class PdfContent:
    pages: List[PageContent] = field(default_factory=list)
    backend: str = ""
    kind: str = "pdf"            # pdf | image
    error: Optional[str] = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def is_scanned(self) -> bool:
        return bool(self.pages) and all(p.needs_ocr for p in self.pages)


def _page(no: int, text: str) -> PageContent:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    n = len("".join(text.split()))
    return PageContent(no, text, n, n < MIN_TEXT_CHARS)


class PdfBackend(Protocol):
    name: str

    def read(self, data: bytes) -> PdfContent: ...
    def render(self, data: bytes, page_index: int, dpi: int = 200) -> Image.Image: ...


class PyMuPdfBackend:
    name = "pymupdf"

    def __init__(self):
        try:
            import fitz  # type: ignore
        except ImportError as e:
            raise BackendUnavailable("PyMuPDF (fitz) is not installed") from e
        self._fitz = fitz

    def read(self, data: bytes) -> PdfContent:
        doc = self._fitz.open(stream=data, filetype="pdf")
        try:
            if doc.needs_pass:
                return PdfContent(backend=self.name, error="PDF is password protected")
            return PdfContent([_page(i + 1, p.get_text("text")) for i, p in enumerate(doc)], backend=self.name)
        finally:
            doc.close()

    def render(self, data: bytes, page_index: int, dpi: int = 200) -> Image.Image:
        doc = self._fitz.open(stream=data, filetype="pdf")
        try:
            pix = doc[page_index].get_pixmap(dpi=dpi, colorspace=self._fitz.csGRAY)
            return Image.frombytes("L", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()


class PdfiumBackend:
    name = "pypdfium2"

    def __init__(self):
        try:
            import pypdfium2 as pdfium  # type: ignore
        except ImportError as e:
            raise BackendUnavailable("pypdfium2 is not installed") from e
        self._pdfium = pdfium

    def read(self, data: bytes) -> PdfContent:
        pdf = self._pdfium.PdfDocument(data)
        try:
            pages = []
            for i in range(len(pdf)):
                tp = pdf[i].get_textpage()
                pages.append(_page(i + 1, tp.get_text_range()))
                tp.close()
            return PdfContent(pages, backend=self.name)
        finally:
            pdf.close()

    def render(self, data: bytes, page_index: int, dpi: int = 200) -> Image.Image:
        pdf = self._pdfium.PdfDocument(data)
        try:
            return pdf[page_index].render(scale=dpi / 72).to_pil().convert("L")
        finally:
            pdf.close()


def get_backend(preference: str = "auto") -> PdfBackend:
    order = {"pymupdf": [PyMuPdfBackend], "pdfium": [PdfiumBackend]}.get(preference, [PyMuPdfBackend, PdfiumBackend])
    errors = []
    for cls in order:
        try:
            return cls()
        except BackendUnavailable as e:
            errors.append(str(e))
    raise BackendUnavailable("No PDF backend available: " + "; ".join(errors))


def available_backends() -> List[str]:
    out = []
    for cls in (PyMuPdfBackend, PdfiumBackend):
        try:
            out.append(cls().name)
        except BackendUnavailable:
            pass
    return out


def is_image(filename: str) -> bool:
    return any(filename.lower().endswith(e) for e in IMAGE_EXT)


def read_document(data: bytes, filename: str, backend: PdfBackend) -> PdfContent:
    """Read any supported upload. Never raises: failures are returned in `error`."""
    try:
        if is_image(filename):
            Image.open(io.BytesIO(data)).verify()
            return PdfContent([PageContent(1, "", 0, True)], backend="image", kind="image")
        return backend.read(data)
    except Exception as e:                      # corrupt / encrypted / not a real PDF
        return PdfContent(backend=getattr(backend, "name", "?"), error=f"{type(e).__name__}: {e}")


def render_page(data: bytes, filename: str, page_index: int, backend: PdfBackend, dpi: int = 200) -> Image.Image:
    if is_image(filename):
        return Image.open(io.BytesIO(data)).convert("L")
    return backend.render(data, page_index, dpi)
