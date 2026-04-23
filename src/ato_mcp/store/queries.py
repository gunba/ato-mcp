"""Prepared SQL queries used across tools and updater."""
from __future__ import annotations

INSERT_DOCUMENT = """
INSERT OR REPLACE INTO documents
    (doc_id, canonical_id, href, category, doc_type, docid_code, title,
     human_title, pub_date, first_published_date, effective_date, status,
     has_content, downloaded_at, content_hash, pack_sha8)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

DELETE_DOCUMENT = "DELETE FROM documents WHERE doc_id = ?"

INSERT_CHUNK = """
INSERT INTO chunks (doc_id, ord, heading_path, anchor, text)
VALUES (?, ?, ?, ?, ?)
"""

INSERT_CHUNK_FTS = """
INSERT INTO chunks_fts (rowid, text, heading_path) VALUES (?, ?, ?)
"""

DELETE_CHUNK_FTS = "INSERT INTO chunks_fts (chunks_fts, rowid) VALUES ('delete', ?)"

INSERT_TITLE_FTS = """
INSERT INTO title_fts (doc_id, docid_code, title, headings) VALUES (?, ?, ?, ?)
"""

DELETE_TITLE_FTS_BY_DOC = """
INSERT INTO title_fts (title_fts, doc_id, docid_code, title, headings)
  SELECT 'delete', doc_id, docid_code, title, headings FROM title_fts WHERE doc_id = ?
"""
# Note: external-content FTS 'delete-all' form needs explicit column echo.
# Alternate: rebuild via delete+re-insert; we use INSERT...'delete' with full row.

INSERT_VEC = "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, vec_int8(?))"
DELETE_VEC = "DELETE FROM chunks_vec WHERE chunk_id = ?"

# Hybrid search helpers: we execute two separate queries (FTS + vec) and fuse
# in python via RRF. That keeps the SQL small and portable across sqlite-vec
# versions.
SELECT_CHUNK_FOR_RESULT = """
SELECT c.chunk_id, c.doc_id, c.ord, c.heading_path, c.anchor, c.text,
       d.docid_code, d.title, d.category, d.doc_type, d.href, d.pub_date
FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
WHERE c.chunk_id = ?
"""

SELECT_CHUNKS_FOR_DOC = """
SELECT chunk_id, ord, heading_path, anchor, text
FROM chunks WHERE doc_id = ? ORDER BY ord ASC
"""

SELECT_DOCUMENT = "SELECT * FROM documents WHERE doc_id = ?"

COUNT_DOCUMENTS = "SELECT COUNT(*) AS n FROM documents"
COUNT_CHUNKS = "SELECT COUNT(*) AS n FROM chunks"

LIST_CATEGORIES = """
SELECT category, COUNT(*) AS n
FROM documents GROUP BY category ORDER BY n DESC
"""

LIST_DOC_TYPES = """
SELECT doc_type, COUNT(*) AS n
FROM documents WHERE doc_type IS NOT NULL
GROUP BY doc_type ORDER BY n DESC
"""
