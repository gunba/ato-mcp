-- ato-mcp SQLite schema v3
-- Single-file layout: one database, no category shards.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- Documents: one row per ATO document (including metadata-only rows for has_content=0).
CREATE TABLE IF NOT EXISTS documents (
    doc_id                TEXT PRIMARY KEY,
    canonical_id          TEXT NOT NULL UNIQUE,
    href                  TEXT NOT NULL,
    category              TEXT NOT NULL,
    doc_type              TEXT,
    docid_code            TEXT,
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
CREATE INDEX IF NOT EXISTS idx_doc_category    ON documents(category);
CREATE INDEX IF NOT EXISTS idx_doc_type        ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_doc_pubdate     ON documents(pub_date);
CREATE INDEX IF NOT EXISTS idx_doc_firstpub    ON documents(first_published_date);
CREATE INDEX IF NOT EXISTS idx_doc_code        ON documents(docid_code);

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
CREATE VIRTUAL TABLE IF NOT EXISTS title_fts USING fts5(
    doc_id UNINDEXED,
    docid_code,
    title,
    headings,
    tokenize = "porter unicode61 remove_diacritics 2"
);

-- Chunk-level FTS (external-content; mirrors chunks.text, chunks.heading_path).
-- Text is stored decompressed in the FTS segments for BM25 ranking.
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
