"""DB schema + basic insert / search sanity tests.

These exercise the schema without the embedding model — we stub chunks_vec
entries with all-zero vectors so sqlite-vec accepts them, but never query it.
"""
from __future__ import annotations

from pathlib import Path

import zstandard as zstd

from ato_mcp.store import db as store_db
from ato_mcp.store.queries import (
    INSERT_CHUNK,
    INSERT_CHUNK_FTS,
    INSERT_DOCUMENT,
    INSERT_TITLE_FTS,
    INSERT_VEC,
)


def _seed_doc(conn, doc_id: str, title: str, text: str) -> int:
    conn.execute(
        INSERT_DOCUMENT,
        (
            doc_id, f"/law/view/document?docid={doc_id}", f"/law/view/document?docid={doc_id}",
            "Public_rulings", "TR", "TR 2024/3", title,
            "2024-07-01", None, "active", 1, "2026-04-18T00:00:00Z",
            "sha256:" + "0" * 64, "deadbeef",
        ),
    )
    conn.execute(INSERT_TITLE_FTS, (doc_id, "TR 2024/3", title, ""))
    compressed = zstd.ZstdCompressor(level=3).compress(text.encode("utf-8"))
    cur = conn.execute(INSERT_CHUNK, (doc_id, 0, "Root", None, compressed))
    rowid = cur.lastrowid
    conn.execute(INSERT_CHUNK_FTS, (rowid, text, "Root"))
    conn.execute(INSERT_VEC, (rowid, b"\x00" * store_db.EMBEDDING_DIM))
    return rowid


def test_schema_inserts_and_queries(tmp_path: Path) -> None:
    conn = store_db.init_db(tmp_path / "ato.db")
    _seed_doc(conn, "d1", "R&D tax incentive ruling", "Research and development activities definition.")
    _seed_doc(conn, "d2", "Capital works deductions", "Division 43 applies to buildings.")

    docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    assert docs == 2

    # FTS over titles
    rows = conn.execute(
        "SELECT doc_id FROM title_fts WHERE title_fts MATCH ?",
        ("incentive",),
    ).fetchall()
    assert [r["doc_id"] for r in rows] == ["d1"]

    # FTS over chunk text
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
        ("buildings",),
    ).fetchall()
    assert len(rows) == 1

    # Meta metadata
    assert store_db.get_meta(conn, "schema_version") == store_db.SCHEMA_VERSION
    conn.close()
