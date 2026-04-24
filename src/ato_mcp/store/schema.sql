-- ato-mcp SQLite schema v5
-- Minimal field set: doc_id PK, type, title, date + 3 build-time columns.
--
-- Design notes:
--   doc_id   The full ATO docid path, slashes included. The canonical URL
--            is synthesised at query time as
--              https://www.ato.gov.au/law/view/document?docid={doc_id}
--            so we don't store ``href`` separately.
--   type     Top-level bucket ("Public_rulings", "Cases", ...). Finer
--            doc_type is implicit in the first segment of doc_id.
--   title    Human-readable label with citation inlined
--            ("TR 2024/3 — R&D tax incentive eligibility"). The rule
--            engine produces this; title_fts searches it directly.
--   date     Best-guess publication date (ISO yyyy-mm-dd). Used only for
--            filters and recency sort — not presented as authoritative.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    title          TEXT NOT NULL,
    date           TEXT,
    -- build-time internals, never exposed via tools:
    downloaded_at  TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    pack_sha8      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(type);
CREATE INDEX IF NOT EXISTS idx_doc_date ON documents(date);

-- Chunks: text is zstd-compressed UTF-8.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      INTEGER PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord           INTEGER NOT NULL,
    heading_path  TEXT,
    anchor        TEXT,
    text          BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- Title-level FTS — just the title plus per-doc heading text. Citations
-- like "TR 2024/3" live inside ``title`` so title_fts finds them.
CREATE VIRTUAL TABLE IF NOT EXISTS title_fts USING fts5(
    doc_id UNINDEXED,
    title,
    headings,
    tokenize = "porter unicode61 remove_diacritics 2"
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    heading_path,
    tokenize = "porter unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- chunks_vec is created at runtime by store/db.py after sqlite-vec is loaded.
