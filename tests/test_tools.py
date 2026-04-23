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


def _seed(
    conn,
    doc_id: str,
    code: str,
    title: str,
    category: str,
    doc_type: str,
    chunks: list[tuple[str, str]],
    *,
    first_published_date: str | None = "2024-07-01",
    downloaded_at: str = "2026-04-18T00:00:00Z",
    human_title: str | None = None,
) -> None:
    """Seed a document plus its chunks. ``chunks`` is [(heading_path, text), ...]."""
    conn.execute(
        INSERT_DOCUMENT,
        (
            doc_id, f"/law/view/document?docid={doc_id}", f"/law/view/document?docid={doc_id}",
            category, doc_type, code, title, human_title,
            "2024-07-01", first_published_date, None, "active", 1,
            downloaded_at, "sha256:" + "0" * 64, "deadbeef",
        ),
    )
    conn.execute(INSERT_TITLE_FTS, (doc_id, code, title, ""))
    for i, (heading_path, text) in enumerate(chunks):
        blob = zstd.ZstdCompressor(level=3).compress(text.encode("utf-8"))
        cur = conn.execute(INSERT_CHUNK, (doc_id, i, heading_path, None, blob))
        rowid = cur.lastrowid
        conn.execute(INSERT_CHUNK_FTS, (rowid, text, heading_path))
        conn.execute(INSERT_VEC, (rowid, b"\x00" * store_db.EMBEDDING_DIM))


@pytest.fixture()
def seeded_db(tmp_data_dir: Path):
    conn = store_db.init_db(tmp_data_dir / "live" / "ato.db")
    # tr2024_3 gets three chunks — two of them mention the phrase, so without
    # dedup both should appear in the results.
    _seed(
        conn, "tr2024_3", "TR 2024/3", "Research and development ruling",
        "Public_rulings", "TR",
        chunks=[
            ("Ruling", "This ruling sets out the Commissioner's view on R&D tax incentive eligibility."),
            ("Ruling > Eligibility", "An eligible activity for the R&D tax incentive eligibility must be systematic."),
            ("Ruling > Date of effect", "The R&D tax incentive commenced in 2011 and applies from that date."),
        ],
        first_published_date="2024-07-01",
    )
    _seed(
        conn, "div43", "TR 97/25", "Capital works deductions",
        "Public_rulings", "TR",
        chunks=[
            ("Ruling", "Division 43 deductions for capital works on buildings."),
        ],
        first_published_date="1997-09-24",
    )
    _seed(
        conn, "pcg2025_d6", "PCG 2025/D6", "Draft practical compliance guideline",
        "Practical_compliance_guidelines", "PCG",
        chunks=[
            ("Guideline", "Compliance guideline for reviewing R&D claims."),
        ],
        first_published_date="2025-11-10",
    )
    store_db.set_meta(conn, "index_version", "2026.04.18")
    conn.close()
    return tmp_data_dir


def test_search_keyword_returns_multiple_chunks_per_doc(seeded_db: Path) -> None:
    out = tools.search("R&D tax incentive eligibility", mode="keyword", k=8, format="json")
    data = json.loads(out)
    tr_hits = [h for h in data["hits"] if h["doc_id"] == "tr2024_3"]
    # Dedup removed: both matching chunks from tr2024_3 should be present.
    assert len(tr_hits) >= 2, data["hits"]


def test_search_doc_scope_by_docid_code(seeded_db: Path) -> None:
    out = tools.search("eligibility", mode="keyword", doc_scope="TR 2024/3", format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == "tr2024_3" for h in data["hits"])


def test_search_doc_scope_glob_on_docid_code(seeded_db: Path) -> None:
    out = tools.search("systematic", mode="keyword", doc_scope="TR 2024/*", format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == "tr2024_3" for h in data["hits"])


def test_search_doc_scope_glob_on_slug(seeded_db: Path) -> None:
    out = tools.search("systematic", mode="keyword", doc_scope="tr2024_*", format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == "tr2024_3" for h in data["hits"])


def test_search_doc_scope_literal_underscore_escaped(seeded_db: Path) -> None:
    # With underscore escaping, "tr2024_3" should NOT match "div43" (different slug).
    out = tools.search("Division", mode="keyword", doc_scope="tr2024_3", format="json")
    data = json.loads(out)
    assert not any(h["doc_id"] == "div43" for h in data["hits"])


def test_search_category_scope_glob(seeded_db: Path) -> None:
    # "eligibility" appears only in the Public_rulings doc; "compliance" only in the PCG.
    # category_scope="Public_*" should return the former and exclude the latter.
    public = json.loads(tools.search(
        "eligibility", mode="keyword", category_scope="Public_*", format="json",
    ))
    assert public["hits"]
    assert all(h["category"].startswith("Public_") for h in public["hits"])
    assert not any(h["doc_id"] == "pcg2025_d6" for h in public["hits"])

    pcg = json.loads(tools.search(
        "compliance guideline", mode="keyword", category_scope="Practical_*", format="json",
    ))
    assert pcg["hits"]
    assert all(h["doc_id"] == "pcg2025_d6" for h in pcg["hits"])


def test_search_titles_returns_by_title(seeded_db: Path) -> None:
    out = tools.search_titles("capital works", format="json")
    data = json.loads(out)
    assert any(h["doc_id"] == "div43" for h in data["hits"])


def test_get_document_outline(seeded_db: Path) -> None:
    out = tools.get_document("tr2024_3", format="outline")
    assert "Research and development ruling" in out


def test_get_section_returns_content(seeded_db: Path) -> None:
    out = tools.get_section("tr2024_3", heading_path="Ruling", format="markdown")
    assert "Commissioner" in out


def test_whats_new_orders_by_first_published(seeded_db: Path) -> None:
    out = tools.whats_new(limit=10, format="json")
    data = json.loads(out)
    order = [h["doc_id"] for h in data["hits"]]
    # pcg2025_d6 (2025-11-10) > tr2024_3 (2024-07-01) > div43 (1997-09-24)
    assert order == ["pcg2025_d6", "tr2024_3", "div43"], order
    assert data["hits"][0]["first_published_date"] == "2025-11-10"


def test_whats_new_snippet_prefers_first_published(seeded_db: Path) -> None:
    out = tools.whats_new(limit=10, format="json")
    data = json.loads(out)
    snippets = [h["snippet"] for h in data["hits"]]
    assert all(s.startswith("published ") for s in snippets)


def test_removed_tools_are_not_registered() -> None:
    """Negative guard: deleted tools must not leak back into the surface."""
    from ato_mcp import server

    registered = {
        name
        for name in dir(server)
        if not name.startswith("_") and callable(getattr(server, name, None))
    }
    for gone in ("resolve", "stats", "list_categories", "list_doc_types"):
        assert gone not in registered, f"{gone} should have been removed"


def test_compose_human_title_joins_and_dedupes() -> None:
    from ato_mcp.indexer.metadata import compose_human_title

    assert compose_human_title([]) is None
    assert compose_human_title(None) is None
    assert compose_human_title(["Solo"]) == "Solo"
    # Consecutive duplicates (ATO templates often repeat the doc title) collapse.
    assert compose_human_title(["TR 2024/3", "TR 2024/3", "Ruling"]) == "TR 2024/3 — Ruling"
    # Whitespace is normalised inside each heading; blanks are dropped.
    assert compose_human_title(["  Hello\n  world ", "", "Next"]) == "Hello world — Next"


def test_human_title_persists_through_insert(seeded_db: Path) -> None:
    """A seeded human_title round-trips through SELECT to confirm the column is wired."""
    from ato_mcp.store import db as store_db

    conn = store_db.connect(seeded_db / "live" / "ato.db", mode="rw")
    conn.execute(
        "UPDATE documents SET human_title = ? WHERE doc_id = ?",
        ("TR 2024/3 — Research and Development — Ruling", "tr2024_3"),
    )
    row = conn.execute(
        "SELECT human_title FROM documents WHERE doc_id = ?", ("tr2024_3",)
    ).fetchone()
    conn.close()
    assert row["human_title"] == "TR 2024/3 — Research and Development — Ruling"
