"""Prepared SQL queries used across tools and updater."""
from __future__ import annotations

INSERT_DOCUMENT = """
INSERT OR REPLACE INTO documents
    (doc_id, type, title, date, downloaded_at, content_hash, pack_sha8)
VALUES (?, ?, ?, ?, ?, ?, ?)
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
INSERT INTO title_fts (doc_id, title, headings) VALUES (?, ?, ?)
"""

DELETE_TITLE_FTS_BY_DOC = """
INSERT INTO title_fts (title_fts, doc_id, title, headings)
  SELECT 'delete', doc_id, title, headings FROM title_fts WHERE doc_id = ?
"""

INSERT_VEC = "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, vec_int8(?))"
DELETE_VEC = "DELETE FROM chunks_vec WHERE chunk_id = ?"

SELECT_CHUNKS_FOR_DOC = """
SELECT chunk_id, ord, heading_path, anchor, text
FROM chunks WHERE doc_id = ? ORDER BY ord ASC
"""

SELECT_DOCUMENT = "SELECT * FROM documents WHERE doc_id = ?"

COUNT_DOCUMENTS = "SELECT COUNT(*) AS n FROM documents"
COUNT_CHUNKS = "SELECT COUNT(*) AS n FROM chunks"

LIST_TYPES = """
SELECT type, COUNT(*) AS n
FROM documents GROUP BY type ORDER BY n DESC
"""
