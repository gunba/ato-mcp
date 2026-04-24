"""MCP tool shape tests using a small stubbed v5 index."""
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
    title: str,
    type_: str,
    chunks: list[tuple[str, str]],
    *,
    date: str | None = "2024-07-01",
    downloaded_at: str = "2026-04-18T00:00:00Z",
) -> None:
    conn.execute(
        INSERT_DOCUMENT,
        (doc_id, type_, title, date, downloaded_at, "sha256:" + "0" * 64, "deadbeef"),
    )
    conn.execute(
        INSERT_TITLE_FTS,
        (doc_id, title, ""),
    )
    for i, (heading_path, text) in enumerate(chunks):
        blob = zstd.ZstdCompressor(level=3).compress(text.encode("utf-8"))
        cur = conn.execute(INSERT_CHUNK, (doc_id, i, heading_path, None, blob))
        rowid = cur.lastrowid
        conn.execute(INSERT_CHUNK_FTS, (rowid, text, heading_path))
        conn.execute(INSERT_VEC, (rowid, b"\x00" * store_db.EMBEDDING_DIM))


DOC_TR   = "TXR/TR20243/NAT/ATO/00001"
DOC_DIV  = "TXR/TR9725/NAT/ATO/00001"
DOC_PCG  = "DPC/PCG2025D6/NAT/ATO/00001"


@pytest.fixture()
def seeded_db(tmp_data_dir: Path):
    conn = store_db.init_db(tmp_data_dir / "live" / "ato.db")
    _seed(
        conn, DOC_TR, "TR 2024/3 — R&D tax incentive", "Public_rulings",
        chunks=[
            ("Ruling", "This ruling sets out the Commissioner's view on R&D tax incentive eligibility."),
            ("Ruling > Eligibility", "An eligible activity for the R&D tax incentive eligibility must be systematic."),
            ("Ruling > Date of effect", "The R&D tax incentive commenced in 2011 and applies from that date."),
        ],
        date="2024-07-01",
    )
    _seed(
        conn, DOC_DIV, "TR 97/25 — Capital works deductions", "Public_rulings",
        chunks=[
            ("Ruling", "Division 43 deductions for capital works on buildings."),
        ],
        date="1997-09-24",
    )
    _seed(
        conn, DOC_PCG, "PCG 2025/D6 — Draft compliance guideline", "Practical_compliance_guidelines",
        chunks=[
            ("Guideline", "Compliance guideline for reviewing R&D claims."),
        ],
        date="2025-11-10",
    )
    store_db.set_meta(conn, "index_version", "2026.04.18")
    conn.close()
    return tmp_data_dir


def test_search_keyword_returns_multiple_chunks_per_doc(seeded_db: Path) -> None:
    out = tools.search("R&D tax incentive eligibility", mode="keyword", k=8, format="json")
    data = json.loads(out)
    tr_hits = [h for h in data["hits"] if h["doc_id"] == DOC_TR]
    assert len(tr_hits) >= 2, data["hits"]


def test_doc_scope_glob_matches_doc_id(seeded_db: Path) -> None:
    out = tools.search("systematic", mode="keyword", doc_scope="TXR/TR20243/*", format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == DOC_TR for h in data["hits"])


def test_doc_scope_exact_doc_id(seeded_db: Path) -> None:
    out = tools.search("systematic", mode="keyword", doc_scope=DOC_TR, format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == DOC_TR for h in data["hits"])


def test_types_filter_exact(seeded_db: Path) -> None:
    public = json.loads(tools.search(
        "eligibility", mode="keyword", types=["Public_rulings"], format="json",
    ))
    assert public["hits"]
    assert all(h["type"] == "Public_rulings" for h in public["hits"])
    assert not any(h["doc_id"] == DOC_PCG for h in public["hits"])


def test_types_filter_glob(seeded_db: Path) -> None:
    public = json.loads(tools.search(
        "eligibility", mode="keyword", types=["Public_*"], format="json",
    ))
    assert public["hits"]
    assert all(h["type"].startswith("Public_") for h in public["hits"])


def test_search_titles_returns_by_title(seeded_db: Path) -> None:
    out = tools.search_titles("capital works", format="json")
    data = json.loads(out)
    assert any(h["doc_id"] == DOC_DIV for h in data["hits"])


def test_search_titles_finds_citation(seeded_db: Path) -> None:
    out = tools.search_titles("TR 2024 3", format="json")
    data = json.loads(out)
    assert any(h["doc_id"] == DOC_TR for h in data["hits"])


def test_get_document_outline(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, format="outline")
    assert "TR 2024/3" in out


def test_get_document_section_via_heading_path(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, heading_path="Ruling", format="markdown")
    assert "Commissioner" in out


def test_get_document_from_ord_paginates(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, from_ord=1, count=1, format="json")
    data = json.loads(out)
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["ord"] == 1


def test_get_document_max_chars_truncates(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, format="json", max_chars=50)
    data = json.loads(out)
    assert len(data["chunks"]) >= 1
    total = sum(len(c["text"]) for c in data["chunks"])
    if data.get("continuation_ord") is not None:
        assert total <= 1000


def test_recency_boost_decay() -> None:
    """Recency boost monotonically prefers newer over older."""
    from ato_mcp.tools import _apply_recency_boost
    recs = [
        {"score": 1.0, "date": "2025-11-10"},  # fresh
        {"score": 1.0, "date": "2020-01-01"},  # 6 yr old
        {"score": 1.0, "date": None},          # unknown (untouched)
    ]
    _apply_recency_boost(recs, half_life_years=5.0)
    assert recs[0]["score"] > recs[2]["score"] > recs[1]["score"], [r["score"] for r in recs]


def test_search_exposes_date(seeded_db: Path) -> None:
    out = tools.search("eligibility", mode="keyword", k=3, format="json")
    data = json.loads(out)
    assert data["hits"]
    for h in data["hits"]:
        assert "date" in h


def test_whats_new_orders_by_date(seeded_db: Path) -> None:
    out = tools.whats_new(limit=10, format="json")
    data = json.loads(out)
    order = [h["doc_id"] for h in data["hits"]]
    assert order == [DOC_PCG, DOC_TR, DOC_DIV], order
    assert data["hits"][0]["date"] == "2025-11-10"


def test_whats_new_snippet_shows_date(seeded_db: Path) -> None:
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
    for gone in ("resolve", "stats", "list_categories", "list_doc_types", "get_section"):
        assert gone not in registered, f"{gone} should have been removed"


def test_canonical_url_synthesised_from_doc_id() -> None:
    from ato_mcp.formatters import canonical_url
    assert canonical_url("TXR/TR20243/NAT/ATO/00001") == (
        "https://www.ato.gov.au/law/view/document?docid=TXR/TR20243/NAT/ATO/00001"
    )
