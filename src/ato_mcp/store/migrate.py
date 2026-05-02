"""Maintainer schema migrations.

v4 → v5 is the only migration currently supported. Earlier versions
(v1/v2/v3) need a full rebuild from ``ato_pages/``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# v4 → v5 migration SQL. Collapses the document schema from 15 columns
# (11 user-facing) to 7 (4 user-facing). Safe to re-run — all steps are
# idempotent except the DROP/RENAME in the transaction. If the DB is
# already v5 we short-circuit in ``needs_v4_to_v5``.
_V4_TO_V5_SQL = """
BEGIN;

CREATE TABLE documents_v5 (
    doc_id         TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    title          TEXT NOT NULL,
    date           TEXT,
    downloaded_at  TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    pack_sha8      TEXT NOT NULL
);

-- Title priority: prefer human_title when non-null (already composed),
-- then "human_code — title" when both distinct, then whatever we have.
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

DROP TABLE title_fts;
DROP TABLE documents;
ALTER TABLE documents_v5 RENAME TO documents;
CREATE INDEX idx_doc_type ON documents(type);
CREATE INDEX idx_doc_date ON documents(date);

CREATE VIRTUAL TABLE title_fts USING fts5(
    doc_id UNINDEXED,
    title,
    headings,
    tokenize = "porter unicode61 remove_diacritics 2"
);
INSERT INTO title_fts (doc_id, title, headings)
SELECT doc_id, title, '' FROM documents;

-- Additive: empty_shells tracker.
CREATE TABLE IF NOT EXISTS empty_shells (
    doc_id          TEXT PRIMARY KEY,
    first_seen_at   TEXT NOT NULL,
    last_checked_at TEXT NOT NULL,
    check_count     INTEGER NOT NULL DEFAULT 1,
    source          TEXT
);
CREATE INDEX IF NOT EXISTS idx_shells_last_checked
  ON empty_shells(last_checked_at);

INSERT INTO meta(key, value) VALUES ('schema_version', '5')
ON CONFLICT(key) DO UPDATE SET value = excluded.value;

-- Drop empty shells from documents. The old delta install path inserted
-- scrape failures here; the v5 builder writes them to empty_shells
-- instead.
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


def needs_v4_to_v5(conn: sqlite3.Connection) -> bool:
    """True iff ``conn`` points at a v4-shape ``documents`` table."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if not cols:
        return False
    return any(c in cols for c in ("human_code", "category", "href"))


def migrate_v4_to_v5(db_path: Path) -> None:
    """Upgrade an ato.db from schema v4 to v5 in place.

    Raises ``RuntimeError`` if the DB is older than v4 — those require a
    full rebuild from ``ato_pages/``. No-op when the DB is already v5.
    """
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    # Use a raw connection (not store_db.connect) because v4 DBs will
    # trigger store_db._migrate's reject-path during loading.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "canonical_id" in cols or "docid_code" in cols:
            raise RuntimeError(
                "DB is pre-v4 (has canonical_id/docid_code). Run "
                "`ato-mcp build-index ...` from ato_pages/ to rebuild."
            )
        if not needs_v4_to_v5(conn):
            return  # already v5 or empty
        conn.executescript(_V4_TO_V5_SQL)
    finally:
        conn.close()
