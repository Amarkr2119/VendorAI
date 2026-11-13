#!/usr/bin/env python3
"""Process every registered synthetic application through the Phase 3 pipeline (intake + extraction).

  python scripts/process_all.py              # uses configured LLM providers if keys are set
  python scripts/process_all.py --no-llm     # regex-only baseline (works offline)
  python scripts/process_all.py --only IN-009 US-003
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendorflow.config import load_settings                                            # noqa: E402
from vendorflow.services import build_services, register_synthetic, synthetic_index    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-llm", action="store_true", help="disable the LLM (deterministic extraction only)")
ap.add_argument("--only", nargs="*", help="process only these synthetic refs")
args = ap.parse_args()

settings = load_settings()
if args.no_llm:
    settings = dataclasses.replace(settings, llm_enabled=False)
svc = build_services(settings)
print("PDF backend:", svc.extraction.backend.name, "| OCR:", getattr(svc.extraction.ocr, "name", None),
      "| LLM providers:", svc.router.providers() or "none (regex-only)")
for entry in synthetic_index():
    if args.only and entry["ref"] not in args.only:
        continue
    app_id = register_synthetic(svc, entry["ref"])
    events = svc.pipeline.run_all(app_id)
    done = events[-1]
    print(f"{entry['ref']:7} {app_id}  {done.message}  open: {', '.join(done.detail.get('open_findings', [])) or '-'}")
