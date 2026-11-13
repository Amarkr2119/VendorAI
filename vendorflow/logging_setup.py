"""Logging with a masking filter, so bank account numbers and API keys never reach a log file."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from vendorflow.utils.masking import mask_text

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


class MaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = mask_text(record.getMessage())
            record.args = ()
        except Exception:  # never let logging break the app
            pass
        return True


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    root = logging.getLogger("vendorflow")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    for h in list(root.handlers):          # idempotent: Streamlit reruns call this repeatedly
        root.removeHandler(h)
        h.close()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(logging.Formatter(_FORMAT))
        h.addFilter(MaskingFilter())
        root.addHandler(h)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vendorflow.{name}")
