"""Migrate an ato.db from schema v4 to v5 in place.

Thin CLI wrapper around ``ato_mcp.store.migrate.migrate_v4_to_v5``. Keeps
the script path stable for anyone following old instructions; prefer the
``ato-mcp migrate`` CLI for new callers.

Run:   python scripts/migrate_v4_to_v5.py <path/to/ato.db>
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ato_mcp.store.migrate import migrate_v4_to_v5, needs_v4_to_v5  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    db_path = Path(sys.argv[1])
    if not db_path.exists():
        raise SystemExit(f"no DB at {db_path}")
    probe = sqlite3.connect(str(db_path))
    probe.row_factory = sqlite3.Row
    try:
        if not needs_v4_to_v5(probe):
            print(f"{db_path}: already v5 — nothing to do")
            return
    finally:
        probe.close()
    print(f"{db_path}: migrating v4 → v5 ...")
    migrate_v4_to_v5(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        with_date = conn.execute("SELECT COUNT(*) AS n FROM documents WHERE date IS NOT NULL").fetchone()["n"]
        print(f"  {total} documents, date={with_date}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
