#!/usr/bin/env python3
"""Register the synthetic applications (inputs only; nothing is pre-processed). Idempotent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendorflow.services import build_services, register_synthetic, synthetic_index   # noqa: E402

svc = build_services()
refs = [a["ref"] for a in synthetic_index()]
if not refs:
    sys.exit("No synthetic dataset found. Run: python scripts/generate_dataset.py")
for ref in refs:
    app_id = register_synthetic(svc, ref)
    print(f"{ref} -> {app_id}")
print(f"{len(refs)} applications registered (not yet processed). Next: python scripts/process_all.py")
