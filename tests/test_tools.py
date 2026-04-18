"""MCP tool shape tests using a small stubbed index.

We seed the DB via the same helpers the builder uses, then call each tool and
assert its return shape. The embedding model is unavailable so vector search
returns empty — keyword/hybrid degrade to FTS-only, which is exactly the
user-facing contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard as zstd

from ato_mcp import tools
from ato_mcp.store import db as store_db
from ato_mcp.store.queries import (
    INSERT_CHUNK,
    INSERT_CHUNK_FTS,
    INSERT_DOCUMENT,
    INSERT_TITLE_FTS,
    INSERT_VEC,
)


def _seed(conn, doc_id: str, code: str, title: str, category: str, doc_type: str, text: str) -> None:
    conn.execute(
        INSERT_DOCUMENT,
        (
            doc_id, f"/law/view/document?docid={doc_id}", f"/law/view/document?docid={doc_id}",
            category, doc_type, code, title, "2024-07-01", None, "active", 1,
            "2026-04-18T00:00:00Z", "sha256:" + "0" * 64, "deadbeef",
        ),
    )
    conn.execute(INSERT_TITLE_FTS, (doc_id, code, title, ""))
    blob = zstd.ZstdCompressor(level=3).compress(text.encode("utf-8"))
    cur = conn.execute(INSERT_CHUNK, (doc_id, 0, "Root", None, blob))
    rowid = cur.lastrowid
    conn.execute(INSERT_CHUNK_FTS, (rowid, text, "Root"))
    conn.execute(INSERT_VEC, (rowid, b"\x00" * store_db.EMBEDDING_DIM))


@pytest.fixture()
def seeded_db(tmp_data_dir: Path):
    conn = store_db.init_db(tmp_data_dir / "live" / "ato.db")
    _seed(conn, "tr2024_3", "TR 2024/3", "Research and development ruling",
          "Public_rulings", "TR",
          "This ruling sets out the Commissioner's view on R&D tax incentive eligibility.")
    _seed(conn, "div43", "TR 97/25", "Capital works deductions",
          "Public_rulings", "TR",
          "Division 43 deductions for capital works on buildings.")
    store_db.set_meta(conn, "index_version", "2026.04.18")
    conn.close()
    return tmp_data_dir


def test_list_categories_markdown(seeded_db: Path) -> None:
    out = tools.list_categories(format="markdown")
    assert "Public_rulings" in out
    assert "| 2 |" in out


def test_list_doc_types_json(seeded_db: Path) -> None:
    out = tools.list_doc_types(format="json")
    data = json.loads(out)
    assert {"doc_type": "TR", "count": 2} in data["doc_types"]


def test_search_keyword_mode(seeded_db: Path) -> None:
    out = tools.search("incentive eligibility", mode="keyword", format="json")
    data = json.loads(out)
    assert any(h["doc_id"] == "tr2024_3" for h in data["hits"])
    for h in data["hits"]:
        assert h["canonical_url"].startswith("https://www.ato.gov.au")


def test_search_titles_exact(seeded_db: Path) -> None:
    out = tools.search_titles("capital works", format="json")
    data = json.loads(out)
    assert any(h["doc_id"] == "div43" for h in data["hits"])


def test_resolve_by_code(seeded_db: Path) -> None:
    out = tools.resolve("TR 2024/3", format="json")
    data = json.loads(out)
    assert data["hits"][0]["doc_id"] == "tr2024_3"


def test_get_document_outline(seeded_db: Path) -> None:
    out = tools.get_document("tr2024_3", format="outline")
    assert "Research and development ruling" in out
    assert "canonical" not in out.lower() or "https://www.ato.gov.au" in out


def test_get_section_returns_content(seeded_db: Path) -> None:
    out = tools.get_section("tr2024_3", heading_path="Root", format="markdown")
    assert "Commissioner" in out


def test_whats_new(seeded_db: Path) -> None:
    out = tools.whats_new(limit=10, format="json")
    data = json.loads(out)
    assert len(data["hits"]) == 2


def test_stats_reports_counts(seeded_db: Path) -> None:
    out = tools.stats(format="json")
    data = json.loads(out)
    assert data["documents"] == 2
    assert data["chunks"] == 2
    assert data["index_version"] == "2026.04.18"
