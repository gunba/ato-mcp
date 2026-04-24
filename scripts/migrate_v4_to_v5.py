"""Migrate an ato.db from schema v4 to v5 in place.

v5 collapses the document schema:
    doc_id PK | type | title | date + build-time (downloaded_at, content_hash, pack_sha8)

Drops: href, category, doc_type, human_code, human_title, pub_date,
first_published_date, effective_date, status, has_content.

Migration strategy:
    * CREATE documents_v5 with new shape.
    * INSERT SELECT rows, composing `title` from human_code + human_title
      (preferring the informative form) and `type` from category.
    * Rebuild title_fts with the new column set (drops human_code/human_title).
    * DROP old tables, RENAME, add indexes.
    * Set meta.schema_version = '5'.

Run:   python scripts/migrate_v4_to_v5.py <path/to/ato.db>
"""
from __future__ import annotations

import sys
import sqlite3
from pathlib import Path


MIGRATE_SQL = """
BEGIN;

-- 1. Create v5 documents in a side table.
CREATE TABLE documents_v5 (
    doc_id         TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    title          TEXT NOT NULL,
    date           TEXT,
    downloaded_at  TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    pack_sha8      TEXT NOT NULL
);

-- 2. Migrate. Title priority:
--      a. human_title already contains the citation + subtitle ("TR 2024/3 — ...")
--      b. Otherwise "<human_code> — <title>" when both present and distinct.
--      c. Otherwise human_code, human_title, or title — whichever is non-null.
INSERT INTO documents_v5 (doc_id, type, title, date, downloaded_at, content_hash, pack_sha8)
SELECT
    doc_id,
    category AS type,
    CASE
        WHEN human_title IS NOT NULL
         AND human_code  IS NOT NULL
         AND human_title LIKE human_code || '%'
            THEN human_title
        WHEN human_code IS NOT NULL AND human_title IS NOT NULL
            THEN human_code || ' — ' || human_title
        WHEN human_title IS NOT NULL
            THEN human_title
        WHEN human_code IS NOT NULL
            THEN human_code
        ELSE title
    END AS title,
    COALESCE(first_published_date, pub_date) AS date,
    downloaded_at,
    content_hash,
    pack_sha8
FROM documents;

-- 3. Drop old FTS and old documents table.
DROP TABLE title_fts;
DROP TABLE documents;
ALTER TABLE documents_v5 RENAME TO documents;
CREATE INDEX idx_doc_type ON documents(type);
CREATE INDEX idx_doc_date ON documents(date);

-- 4. Rebuild title_fts with the v5 column set.
CREATE VIRTUAL TABLE title_fts USING fts5(
    doc_id UNINDEXED,
    title,
    headings,
    tokenize = "porter unicode61 remove_diacritics 2"
);

-- 5. Seed title_fts.title from the new documents.title. `headings` starts
--    empty; `ato-mcp backfill` repopulates it by re-reading chunk breadcrumbs.
INSERT INTO title_fts (doc_id, title, headings)
SELECT doc_id, title, '' FROM documents;

-- 6. Bump the schema version.
INSERT INTO meta(key, value) VALUES ('schema_version', '5')
ON CONFLICT(key) DO UPDATE SET value = excluded.value;

-- 7. Delete empty-shell documents (has no chunks). v4 inserted these for
--    failed scrapes; v5 build skips them. Clears ~15k rows of noise.
INSERT INTO title_fts(title_fts, doc_id, title, headings)
  SELECT 'delete', doc_id, title, headings FROM title_fts
  WHERE doc_id IN (
    SELECT d.doc_id FROM documents d
    WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)
  );
DELETE FROM documents WHERE doc_id IN (
  SELECT d.doc_id FROM documents d
  WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)
);

COMMIT;

VACUUM;
"""


def migrate(db_path: Path) -> None:
    if not db_path.exists():
        raise SystemExit(f"no DB at {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None  # manual BEGIN/COMMIT
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "human_code" not in cols:
            print(f"{db_path}: not a v4 DB (no human_code column) — nothing to do")
            return
        print(f"{db_path}: migrating v4 → v5 ...")
        conn.executescript(MIGRATE_SQL)
        total = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        with_title = conn.execute("SELECT COUNT(*) AS n FROM documents WHERE title IS NOT NULL").fetchone()["n"]
        with_date = conn.execute("SELECT COUNT(*) AS n FROM documents WHERE date IS NOT NULL").fetchone()["n"]
        print(f"  {total} documents, title={with_title}, date={with_date}")
    finally:
        conn.close()


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    migrate(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
