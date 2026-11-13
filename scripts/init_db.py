#!/usr/bin/env python3
"""Create the SQLite database and all tables (idempotent)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendorflow.config import load_settings          # noqa: E402
from vendorflow.db.repositories import Repository    # noqa: E402

s = load_settings()
repo = Repository(s.db_path)
repo.init_schema()
print(f"Database ready at {s.db_path}: {len(repo.table_names())} tables, {len(repo.object_names('trigger'))} triggers, "
      f"{len(repo.object_names('view'))} views")
