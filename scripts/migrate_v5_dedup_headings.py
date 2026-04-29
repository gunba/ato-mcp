"""One-shot rewrite of ``chunks.heading_path`` to drop title-echo prefixes.

Background:

    The pre-2026.04.30 chunker pushed both the composed root_title and each
    of its constituent leading h1/h2/h3 headings onto the heading stack,
    producing paths like::

        Taxation Ruling — TR 2024/3 — Subject › Taxation Ruling › TR 2024/3 › Ruling

    The body text (and therefore embeddings) are unaffected; only the
    metadata label is wrong. ``ato_mcp.indexer.chunk.strip_title_prefix``
    computes the correct form (``"Ruling"``).

This script applies that transform across the live DB. It is purely metadata:
embeddings (``chunks_vec``) are not touched, so no GPU work is required.

Steps:
1. UPDATE chunks.heading_path with the cleaned form.
2. Rebuild ``chunks_fts`` (its ``heading_path`` mirror is now stale).
3. Bump ``meta.index_version`` to today's date.
4. ANALYZE to refresh planner stats.

Idempotent: running twice is a no-op since the second pass finds no change.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ato_mcp.indexer.chunk import strip_title_prefix  # noqa: E402
from ato_mcp.store import db as store_db  # noqa: E402


def migrate(db_path: Path, *, dry_run: bool = False) -> None:
    if not db_path.exists():
        raise SystemExit(f"no DB at {db_path}")

    conn = store_db.connect(db_path, mode="rw")
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        print(f"scanning {total:,} chunks in {db_path}")

        rows = conn.execute(
            "SELECT chunk_id, heading_path FROM chunks"
        ).fetchall()

        updates: list[tuple[str, int]] = []
        unchanged = 0
        for r in rows:
            current = r["heading_path"] or ""
            cleaned = strip_title_prefix(current)
            if cleaned != current:
                updates.append((cleaned, r["chunk_id"]))
            else:
                unchanged += 1

        print(f"  {len(updates):,} chunks need rewriting; {unchanged:,} unchanged")

        if dry_run:
            for c, _cid in updates[:5]:
                print(f"  sample new heading_path: {c!r}")
            return

        if updates:
            t0 = time.monotonic()
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    "UPDATE chunks SET heading_path = ? WHERE chunk_id = ?",
                    updates,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            print(f"  UPDATE chunks: {time.monotonic() - t0:.1f}s")

        # chunks_fts mirrors heading_path; rebuild from the source-of-truth.
        print("rebuilding chunks_fts (heading_path mirror)")
        t0 = time.monotonic()
        conn.execute("BEGIN")
        try:
            conn.execute(
                "INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        print(f"  chunks_fts rebuild: {time.monotonic() - t0:.1f}s")

        # title_fts.headings is a flat space-joined string per doc; not
        # affected by heading_path's structure, so leave it alone.

        # Bump index_version so clients can detect the change.
        new_version = _dt.date.today().strftime("%Y.%m.%d")
        store_db.set_meta(conn, "index_version", new_version)
        store_db.set_meta(
            conn,
            "last_update_at",
            _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="microseconds"),
        )
        print(f"  index_version -> {new_version}")

        print("ANALYZE")
        t0 = time.monotonic()
        conn.execute("ANALYZE")
        print(f"  ANALYZE: {time.monotonic() - t0:.1f}s")
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report counts and a sample but don't UPDATE.")
    args = ap.parse_args()
    migrate(args.db, dry_run=args.dry_run)
    print("migration complete.")


if __name__ == "__main__":
    main()
