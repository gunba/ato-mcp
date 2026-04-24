"""Populate ``empty_shells`` from the scraper index.

Run once after the v4→v5 migration to record the 14,847 doc_ids that were
deleted as empty shells so they aren't lost. Going forward, build.py
populates this table incrementally.

Usage:
    python scripts/seed_empty_shells.py \
        --db ./release/ato.db \
        --index /path/to/ato_pages/index.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ato_mcp.indexer.metadata import doc_id_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True,
                    help="Path to ato_pages/index.jsonl")
    ap.add_argument("--source", default="migration",
                    help="Label written to empty_shells.source")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no DB at {args.db}")
    if not args.index.exists():
        raise SystemExit(f"no index at {args.index}")

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None

    # Ensure the table exists (older v5 DBs won't have it).
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS empty_shells (
            doc_id          TEXT PRIMARY KEY,
            first_seen_at   TEXT NOT NULL,
            last_checked_at TEXT NOT NULL,
            check_count     INTEGER NOT NULL DEFAULT 1,
            source          TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_shells_last_checked
          ON empty_shells(last_checked_at);
        """
    )

    # Doc_ids currently in the DB.
    alive = {r["doc_id"] for r in conn.execute("SELECT doc_id FROM documents").fetchall()}
    # Scraper index → canonical doc_ids.
    scraped: set[str] = set()
    with args.index.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("canonical_id") or rec.get("href")
            if not cid:
                continue
            did = doc_id_for(cid)
            if did:
                scraped.add(did)

    shells = sorted(scraped - alive)
    print(f"scraped={len(scraped):,}  alive={len(alive):,}  shells={len(shells):,}")

    now = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    conn.execute("BEGIN")
    try:
        conn.executemany(
            """
            INSERT INTO empty_shells (doc_id, first_seen_at, last_checked_at, check_count, source)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                last_checked_at = excluded.last_checked_at,
                check_count     = empty_shells.check_count + 1,
                source          = COALESCE(excluded.source, empty_shells.source)
            """,
            [(d, now, now, args.source) for d in shells],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    n = conn.execute("SELECT COUNT(*) FROM empty_shells").fetchone()[0]
    print(f"empty_shells table now holds {n:,} rows")
    conn.close()


if __name__ == "__main__":
    main()
