-- ato-mcp SQLite schema v4
-- Single-file layout: one database, no category shards.
--
-- Identifier model (v4):
--   doc_id      = the full docid path from the ATO URL, prefix included
--                 (e.g. "TXR/TR20133/NAT/ATO/00001"). PRIMARY KEY. Unique.
--   human_code  = short human citation ("TR 2013/3"). Nullable. Populated
--                 by the main-PC corpus parser; may be ambiguous across
--                 versions / addenda.
--
-- Dropped from v3: canonical_id (derivable from doc_id) and docid_code
-- (redundant with doc_id).

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id                TEXT PRIMARY KEY,
    href                  TEXT NOT NULL,
    category              TEXT NOT NULL,
    doc_type              TEXT,
    human_code            TEXT,
    title                 TEXT NOT NULL,
    human_title           TEXT,
    pub_date              TEXT,
    first_published_date  TEXT,
    effective_date        TEXT,
    status                TEXT,
    has_content           INTEGER NOT NULL DEFAULT 1,
    downloaded_at         TEXT NOT NULL,
    content_hash          TEXT NOT NULL,
    pack_sha8             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_category     ON documents(category);
CREATE INDEX IF NOT EXISTS idx_doc_type         ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_doc_pubdate      ON documents(pub_date);
CREATE INDEX IF NOT EXISTS idx_doc_firstpub     ON documents(first_published_date);
CREATE INDEX IF NOT EXISTS idx_doc_human_code   ON documents(human_code);

-- Chunks: text is zstd-compressed UTF-8 (shared dictionary per build).
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      INTEGER PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord           INTEGER NOT NULL,
    heading_path  TEXT,
    anchor        TEXT,
    text          BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- Title-level FTS (content-less; populated directly).
-- Indexes human_code (citation) and human_title (generated long title) plus
-- the extracted title and heading list for fuzzy title searches.
CREATE VIRTUAL TABLE IF NOT EXISTS title_fts USING fts5(
    doc_id UNINDEXED,
    human_code,
    title,
    human_title,
    headings,
    tokenize = "porter unicode61 remove_diacritics 2"
);

-- Chunk-level FTS (external-content; mirrors chunks.text, chunks.heading_path).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    heading_path,
    tokenize = "porter unicode61 remove_diacritics 2"
);

-- Metadata key/value (schema_version, index_version, embedding_model_id, last_update_at, ...).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- chunks_vec is created at runtime by store/db.py after sqlite-vec is loaded.
