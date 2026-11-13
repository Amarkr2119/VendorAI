#!/usr/bin/env python3
"""Score extraction against ground truth for every processed synthetic application.

Per field: CORRECT, FLAGGED_WRONG (wrong value but confidence LOW: safe), CONFIDENT_WRONG (must be zero),
MISSING. Writes data/extraction_report.csv.   Usage: python scripts/report_extraction.py [--db PATH]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vendorflow.config import load_settings                  # noqa: E402
from vendorflow.db.repositories import Repository            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db")
args = ap.parse_args()
settings = load_settings()
repo = Repository(args.db or settings.db_path)
truth = json.loads((ROOT / "data" / "synthetic" / "ground_truth.json").read_text())

from vendorflow.evaluation import score_against_truth   # noqa: E402

rows = score_against_truth(repo, truth)
counts = {"CORRECT": 0, "FLAGGED_WRONG": 0, "CONFIDENT_WRONG": 0, "MISSING": 0}
for r in rows:
    counts[r["verdict"]] += 1
out = ROOT / "data" / "extraction_report.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["ref"])
    w.writeheader()
    w.writerows(rows)
total = sum(counts.values())
if not total:
    sys.exit("No processed applications found. Run: python scripts/seed_db.py && python scripts/process_all.py")
print(f"Fields scored: {total}")
for k, v in counts.items():
    print(f"  {k:16} {v:5}  ({100 * v / total:.1f}%)" if total else k)
MAX_SHOWN = 30
bad = sorted((r for r in rows if r["verdict"] != "CORRECT"), key=lambda r: ["CONFIDENT_WRONG", "FLAGGED_WRONG", "MISSING"].index(r["verdict"]))
print(f"\nNon-CORRECT fields ({len(bad)}; showing up to {MAX_SHOWN}, full list in the CSV):")
for r in bad[:MAX_SHOWN]:
    if True:
        print(f"  {r['verdict']:15} {r['ref']:7} {r['document']:28} {r['field']:18} truth={r['truth']!r:34} got={r['extracted']!r} conf={r['confidence']}")
print(f"\nReport written to {out}")
sys.exit(1 if counts["CONFIDENT_WRONG"] else 0)
