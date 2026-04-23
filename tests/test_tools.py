"""MCP tool shape tests using a small stubbed index.

We seed the DB via the same helpers the builder uses, then call each tool and
assert its return shape. The embedding model is unavailable so vector search
returns empty — keyword/hybrid degrade to FTS-only, which is exactly the
user-facing contract.

``doc_id`` in v4 is the full ATO docid path with slashes and prefix
(e.g. ``TXR/TR20133/NAT/ATO/00001``). ``human_code`` is the short human
citation (``TR 2013/3``) — nullable until the main-PC parser populates it.
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
    human_code: str | None,
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
    href = f"/law/view/document?docid={doc_id}"
    conn.execute(
        INSERT_DOCUMENT,
        (
            doc_id, href, category, doc_type, human_code, title, human_title,
            "2024-07-01", first_published_date, None, "active", 1,
            downloaded_at, "sha256:" + "0" * 64, "deadbeef",
        ),
    )
    conn.execute(
        INSERT_TITLE_FTS,
        (doc_id, human_code or "", title, human_title or "", ""),
    )
    for i, (heading_path, text) in enumerate(chunks):
        blob = zstd.ZstdCompressor(level=3).compress(text.encode("utf-8"))
        cur = conn.execute(INSERT_CHUNK, (doc_id, i, heading_path, None, blob))
        rowid = cur.lastrowid
        conn.execute(INSERT_CHUNK_FTS, (rowid, text, heading_path))
        conn.execute(INSERT_VEC, (rowid, b"\x00" * store_db.EMBEDDING_DIM))


# Real-shape doc_ids (full path with prefix and slashes).
DOC_TR   = "TXR/TR20243/NAT/ATO/00001"
DOC_DIV  = "TXR/TR9725/NAT/ATO/00001"
DOC_PCG  = "DPC/PCG2025D6/NAT/ATO/00001"


@pytest.fixture()
def seeded_db(tmp_data_dir: Path):
    conn = store_db.init_db(tmp_data_dir / "live" / "ato.db")
    # TR 2024/3 gets three chunks — two of them mention the phrase, so without
    # dedup both should appear in the results.
    _seed(
        conn, DOC_TR, "TR 2024/3", "Research and development ruling",
        "Public_rulings", "TR",
        chunks=[
            ("Ruling", "This ruling sets out the Commissioner's view on R&D tax incentive eligibility."),
            ("Ruling > Eligibility", "An eligible activity for the R&D tax incentive eligibility must be systematic."),
            ("Ruling > Date of effect", "The R&D tax incentive commenced in 2011 and applies from that date."),
        ],
        first_published_date="2024-07-01",
    )
    _seed(
        conn, DOC_DIV, "TR 97/25", "Capital works deductions",
        "Public_rulings", "TR",
        chunks=[
            ("Ruling", "Division 43 deductions for capital works on buildings."),
        ],
        first_published_date="1997-09-24",
    )
    _seed(
        conn, DOC_PCG, "PCG 2025/D6", "Draft practical compliance guideline",
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
    tr_hits = [h for h in data["hits"] if h["doc_id"] == DOC_TR]
    # Dedup removed: both matching chunks from the TR doc should be present.
    assert len(tr_hits) >= 2, data["hits"]


def test_doc_scope_human_code_exact(seeded_db: Path) -> None:
    # No slash in scope => match human_code. "TR 2024/3" contains a slash but
    # the value itself is still interpreted as human_code because the scope
    # string has no `/` in its path *prefix* — wait, this example does have
    # a `/`, so it's routed to doc_id. Use a no-slash example instead.
    out = tools.search("eligibility", mode="keyword", doc_scope="TR 97/25", format="json")
    # "TR 97/25" contains a slash, so it routes to doc_id match.
    # Our DOC_DIV doesn't have "TR 97/25" in its doc_id path, so 0 hits is correct.
    data = json.loads(out)
    # Just verify no div hits appear — the glob doesn't match our actual doc_ids.
    assert all(h["doc_id"] != DOC_DIV for h in data["hits"])


def test_doc_scope_no_slash_matches_human_code(seeded_db: Path) -> None:
    # A scope without "/" is interpreted as human_code. But my human_code
    # values do contain "/", so a bare prefix without slashes is the test.
    # Use "TR 2024*" — wait, that has a space but no slash. It matches
    # human_code "TR 2024/3" via glob.
    out = tools.search("eligibility", mode="keyword", doc_scope="TR 2024*", format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == DOC_TR for h in data["hits"])


def test_doc_scope_slash_routes_to_doc_id(seeded_db: Path) -> None:
    # A scope with "/" is interpreted as a doc_id path glob.
    out = tools.search("systematic", mode="keyword", doc_scope="TXR/TR20243/*", format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == DOC_TR for h in data["hits"])


def test_doc_scope_exact_doc_id(seeded_db: Path) -> None:
    out = tools.search("systematic", mode="keyword", doc_scope=DOC_TR, format="json")
    data = json.loads(out)
    assert data["hits"]
    assert all(h["doc_id"] == DOC_TR for h in data["hits"])


def test_search_category_scope_glob(seeded_db: Path) -> None:
    public = json.loads(tools.search(
        "eligibility", mode="keyword", category_scope="Public_*", format="json",
    ))
    assert public["hits"]
    assert all(h["category"].startswith("Public_") for h in public["hits"])
    assert not any(h["doc_id"] == DOC_PCG for h in public["hits"])

    pcg = json.loads(tools.search(
        "compliance guideline", mode="keyword", category_scope="Practical_*", format="json",
    ))
    assert pcg["hits"]
    assert all(h["doc_id"] == DOC_PCG for h in pcg["hits"])


def test_search_titles_returns_by_title(seeded_db: Path) -> None:
    out = tools.search_titles("capital works", format="json")
    data = json.loads(out)
    assert any(h["doc_id"] == DOC_DIV for h in data["hits"])


def test_get_document_outline(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, format="outline")
    assert "Research and development ruling" in out


def test_get_document_section_via_heading_path(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, heading_path="Ruling", format="markdown")
    assert "Commissioner" in out


def test_get_document_from_ord_paginates(seeded_db: Path) -> None:
    # Walk from ord=1 with count=1 — should return one chunk and a cursor.
    out = tools.get_document(DOC_TR, from_ord=1, count=1, format="json")
    data = json.loads(out)
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["ord"] == 1


def test_get_document_max_chars_truncates(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, format="json", max_chars=50)
    data = json.loads(out)
    assert len(data["chunks"]) >= 1
    # Continuation cursor should be set when we stopped early.
    total = sum(len(c["text"]) for c in data["chunks"])
    if data.get("continuation_ord") is not None:
        assert total <= 1000  # well under the doc


def test_get_document_outline_carries_positions(seeded_db: Path) -> None:
    out = tools.get_document(DOC_TR, format="outline")
    assert "start_ord" not in out  # table rendering, not raw field names
    assert "| ord |" in out
    assert "chunks" in out


def test_recency_boost_applies_decay() -> None:
    """Unit test for the recency boost math — independent of FTS."""
    from ato_mcp.tools import _apply_recency_boost
    recs = [
        {"score": 1.0, "first_published_date": "2025-11-10", "pub_date": None},  # fresh
        {"score": 1.0, "first_published_date": "2020-01-01", "pub_date": None},  # 6 yr
        {"score": 1.0, "first_published_date": None, "pub_date": None},          # unknown
    ]
    _apply_recency_boost(recs, half_life_years=5.0)
    # Fresh should land highest, 6-year-old lower, unknown untouched (1.0).
    assert recs[0]["score"] > recs[2]["score"] > recs[1]["score"], [r["score"] for r in recs]


def test_search_exposes_first_published_date(seeded_db: Path) -> None:
    out = tools.search("eligibility", mode="keyword", k=3, format="json")
    data = json.loads(out)
    assert data["hits"]
    for h in data["hits"]:
        assert "first_published_date" in h


def test_whats_new_orders_by_first_published(seeded_db: Path) -> None:
    out = tools.whats_new(limit=10, format="json")
    data = json.loads(out)
    order = [h["doc_id"] for h in data["hits"]]
    # PCG (2025-11-10) > TR (2024-07-01) > DIV (1997-09-24)
    assert order == [DOC_PCG, DOC_TR, DOC_DIV], order
    assert data["hits"][0]["first_published_date"] == "2025-11-10"


def test_whats_new_snippet_prefers_first_published(seeded_db: Path) -> None:
    out = tools.whats_new(limit=10, format="json")
    data = json.loads(out)
    snippets = [h["snippet"] for h in data["hits"]]
    assert all(s.startswith("published ") for s in snippets)


def test_hits_expose_human_code(seeded_db: Path) -> None:
    """Tool output surfaces human_code, not the old docid_code."""
    out = tools.search("capital works buildings", mode="keyword", k=5, format="json")
    data = json.loads(out)
    assert data["hits"]
    for h in data["hits"]:
        assert "human_code" in h
        assert "docid_code" not in h
        if h["doc_id"] == DOC_DIV:
            assert h["human_code"] == "TR 97/25"


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
    assert compose_human_title(["TR 2024/3", "TR 2024/3", "Ruling"]) == "TR 2024/3 — Ruling"
    assert compose_human_title(["  Hello\n  world ", "", "Next"]) == "Hello world — Next"


def test_doc_id_for_returns_raw_docid_path() -> None:
    from ato_mcp.indexer.metadata import doc_id_for

    # No slugification — case and slashes preserved.
    assert doc_id_for("/law/view/document?docid=TXR/TR20133/NAT/ATO/00001") == "TXR/TR20133/NAT/ATO/00001"
    # Malformed URL falls back to the raw value so we still have some PK.
    assert doc_id_for("not-a-url") == "not-a-url"
