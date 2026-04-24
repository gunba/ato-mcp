"""Simulates the workflows an LLM agent would run against the v5 MCP surface.

Each scenario prints what the agent would "see" plus a sanity-check
assertion. Run with:

    ATO_MCP_DATA_DIR=/tmp/ato-mcp-ship-test python tests/ship_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make sure we pick up the repo's src/ not a stale install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ato_mcp import tools  # noqa: E402
from ato_mcp import formatters  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def call(name: str, **kwargs) -> None:
    k_for_print = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if k != "format")
    print(f"\n> {name}({k_for_print})")


def _json_search(**kwargs) -> dict:
    t0 = time.perf_counter()
    out = tools.search(format="json", **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    data = json.loads(out)
    print(f"  [{ms:.0f} ms] {len(data['hits'])} hits")
    return data


def _json_titles(**kwargs) -> dict:
    t0 = time.perf_counter()
    out = tools.search_titles(format="json", **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    data = json.loads(out)
    print(f"  [{ms:.0f} ms] {len(data['hits'])} hits")
    return data


def _json_doc(doc_id: str, **kwargs) -> dict:
    t0 = time.perf_counter()
    out = tools.get_document(doc_id, format="json", **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    data = json.loads(out)
    print(f"  [{ms:.0f} ms]")
    return data


def _json_whats_new(**kwargs) -> dict:
    t0 = time.perf_counter()
    out = tools.whats_new(format="json", **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    data = json.loads(out)
    print(f"  [{ms:.0f} ms] {len(data['hits'])} hits")
    return data


def show_hit(h: dict, prefix: str = "    ") -> None:
    title = h.get("title") or "(no title)"
    if len(title) > 90:
        title = title[:87] + "…"
    print(f"{prefix}[{h.get('type') or '?':<30}] {h.get('date') or '        '}  {title}")
    print(f"{prefix}  doc_id: {h['doc_id']}")
    if h.get("heading_path"):
        hp = h["heading_path"]
        if len(hp) > 80:
            hp = hp[:77] + "…"
        print(f"{prefix}  section: {hp}")
    if h.get("snippet"):
        snip = h["snippet"].replace("\n", " ")
        if len(snip) > 180:
            snip = snip[:177] + "…"
        print(f"{prefix}  {snip}")
    print(f"{prefix}  url: {h['canonical_url']}")


FAIL_COUNT = 0


def require(cond: bool, msg: str) -> None:
    global FAIL_COUNT
    tag = "PASS" if cond else "FAIL"
    print(f"    {tag}: {msg}")
    if not cond:
        FAIL_COUNT += 1


# ---------------------------------------------------------------------------
# WORKFLOW 1 — R&D tax incentive research
# ---------------------------------------------------------------------------


def wf1_rd_tax() -> None:
    section("WORKFLOW 1 — research R&D tax incentive in Public rulings + Legislation")

    # Without a vector model loaded, mode='hybrid' degrades to keyword-only
    # AND-join — so the query terms must actually co-occur in some chunk.
    # "software development" doesn't; widen to the corpus's real language.
    query = "R&D tax incentive research and development"
    call("search", query=query,
         types=["Public_rulings", "Practical_compliance_guidelines",
                "Legislation_and_supporting_material"], k=5)
    data = _json_search(
        query=query,
        types=["Public_rulings", "Practical_compliance_guidelines",
               "Legislation_and_supporting_material"],
        k=5,
    )
    for h in data["hits"]:
        show_hit(h)

    require(len(data["hits"]) > 0, "search returned hits")
    require(all(h.get("type") in ("Public_rulings", "Practical_compliance_guidelines",
                                   "Legislation_and_supporting_material")
                for h in data["hits"]), "all hits respected types filter")
    require(all(h.get("title") for h in data["hits"]), "all hits have a title")
    require(all(h.get("date") for h in data["hits"]), "all hits have a date")
    require(all(h.get("canonical_url", "").startswith("https://www.ato.gov.au/") for h in data["hits"]),
            "all hits have a canonical_url")
    require(all("score" in h for h in data["hits"]), "all hits carry a relevance score")

    # Scope to the first hit and re-pull outline.
    if data["hits"]:
        top = data["hits"][0]
        call("get_document", doc_id=top["doc_id"], format="outline")
        out = tools.get_document(top["doc_id"], format="outline")
        print("\n" + "\n".join(out.split("\n")[:15]))
        require(top["title"] in out, "outline header shows the same title")
        require(top["doc_id"] in out, "outline header shows doc_id")
        require("Date:" in out, "outline shows date")
        require("Source:" in out, "outline shows canonical URL")


# ---------------------------------------------------------------------------
# WORKFLOW 2 — PepsiCo case (citation lookup)
# ---------------------------------------------------------------------------


def wf2_pepsico() -> None:
    section("WORKFLOW 2 — PepsiCo case (title FTS citation lookup)")

    call("search_titles", query="PepsiCo")
    data = _json_titles(query="PepsiCo", k=10)
    for h in data["hits"][:6]:
        show_hit(h)

    require(len(data["hits"]) > 0, "PepsiCo resolves via title FTS")
    require(any("pepsico" in (h.get("title") or "").lower() for h in data["hits"]),
            "returned titles mention PepsiCo")
    require(any(h.get("type") == "Cases" for h in data["hits"]),
            "at least one Cases-type hit")


# ---------------------------------------------------------------------------
# WORKFLOW 3 — Find an Act section then subsection
# ---------------------------------------------------------------------------


def wf3_act_section() -> None:
    section("WORKFLOW 3 — Income Tax Assessment Act 1997 s 8-1 by title")

    call("search_titles", query="Income Tax Assessment Act 1997 s 8-1")
    data = _json_titles(query="Income Tax Assessment Act 1997 s 8-1", k=5)
    for h in data["hits"][:4]:
        show_hit(h)

    require(len(data["hits"]) > 0, "ITAA 1997 s 8-1 resolves")
    require(any("s 8-1" in (h.get("title") or "") for h in data["hits"]),
            "at least one hit has 's 8-1' in the title")

    if data["hits"]:
        # Find the real s 8-1 hit (not s 108-1, s 118-1 etc).
        top = next((h for h in data["hits"] if h.get("title", "").endswith("s 8-1")),
                   data["hits"][0])
        call("get_document", doc_id=top["doc_id"], format="outline")
        t0 = time.perf_counter()
        out = tools.get_document(top["doc_id"], format="outline")
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [{ms:.0f} ms]")
        head = "\n".join(out.split("\n")[:12])
        print("\n" + head)
        require("1997" in out, "outline mentions 1997")
        require(top["doc_id"] in out, "outline shows doc_id")


# ---------------------------------------------------------------------------
# WORKFLOW 4 — whats_new recency feed
# ---------------------------------------------------------------------------


def wf4_whats_new() -> None:
    section("WORKFLOW 4 — whats_new feed, last 2 years")

    # pick a "since" that's 2 years before the corpus's latest date
    call("whats_new", since="2024-01-01", limit=15,
         types=["Public_rulings", "Practical_compliance_guidelines",
                "Taxpayer_alerts", "Law_administration_practice_statements"])
    data = _json_whats_new(
        since="2024-01-01",
        limit=15,
        types=["Public_rulings", "Practical_compliance_guidelines",
               "Taxpayer_alerts", "Law_administration_practice_statements"],
    )
    for h in data["hits"][:10]:
        show_hit(h)

    require(len(data["hits"]) > 0, "whats_new returned recent rulings")
    dates = [h.get("date") for h in data["hits"] if h.get("date")]
    require(all(d >= "2024-01-01" for d in dates), "all dates ≥ since")
    require(dates == sorted(dates, reverse=True) or len(set(dates)) <= 2,
            "dates are sorted descending (ties OK)")
    require(all(h.get("type") in {
        "Public_rulings", "Practical_compliance_guidelines",
        "Taxpayer_alerts", "Law_administration_practice_statements"
    } for h in data["hits"]), "type filter respected")


# ---------------------------------------------------------------------------
# WORKFLOW 5 — cross-ruling comparison via doc_scope
# ---------------------------------------------------------------------------


def wf5_doc_scope() -> None:
    section("WORKFLOW 5 — compare TR 2024/* commentary on capital gains")

    call("search", query="capital gains tax base",
         doc_scope="TXR/TR2024*", k=5, sort_by="recency")
    data = _json_search(
        query="capital gains tax base",
        doc_scope="TXR/TR2024*",
        k=5,
        sort_by="recency",
    )
    for h in data["hits"]:
        show_hit(h)

    require(all(h["doc_id"].startswith("TXR/TR2024") for h in data["hits"]),
            "all hits respected the doc_scope glob")
    if data["hits"]:
        dates = [h.get("date", "") for h in data["hits"]]
        require(dates == sorted(dates, reverse=True), "recency sort ordered descending")


# ---------------------------------------------------------------------------
# WORKFLOW 6 — positional paging through a document
# ---------------------------------------------------------------------------


def wf6_pagination() -> None:
    section("WORKFLOW 6 — paginate through a doc with from_ord + max_chars")

    # Pick a doc that definitely has content.
    data = _json_search(query="R&D tax incentive", k=1, mode="keyword")
    if not data["hits"]:
        print("  no seed hit; skipping pagination test")
        return
    doc_id = data["hits"][0]["doc_id"]

    # Page 1
    call("get_document", doc_id=doc_id, from_ord=0, max_chars=1500)
    p1 = _json_doc(doc_id, from_ord=0, max_chars=1500)
    print(f"  page1: chunks={len(p1['chunks'])} cont_ord={p1.get('continuation_ord')}")
    require(len(p1["chunks"]) >= 1, "page 1 has at least one chunk")

    # Page 2
    if p1.get("continuation_ord") is not None:
        call("get_document", doc_id=doc_id, from_ord=p1["continuation_ord"], max_chars=1500)
        p2 = _json_doc(doc_id, from_ord=p1["continuation_ord"], max_chars=1500)
        print(f"  page2: chunks={len(p2['chunks'])} cont_ord={p2.get('continuation_ord')}")
        require(len(p2["chunks"]) >= 1, "page 2 has at least one chunk")
        require(p2["chunks"][0]["ord"] == p1["continuation_ord"],
                "page 2 resumes at continuation_ord")
    else:
        print("  doc fit in one page")


# ---------------------------------------------------------------------------
# WORKFLOW 7 — get_chunks by id (after a search)
# ---------------------------------------------------------------------------


def wf7_get_chunks() -> None:
    section("WORKFLOW 7 — get_chunks expands search hits to full text")

    data = _json_search(query="commercial debt forgiveness", k=3, mode="keyword")
    ids = [h["chunk_id"] for h in data["hits"]]
    if not ids:
        print("  no hits; skipping")
        return
    call("get_chunks", chunk_ids=ids)
    t0 = time.perf_counter()
    out = tools.get_chunks(ids, format="json")
    ms = (time.perf_counter() - t0) * 1000
    payload = json.loads(out)
    print(f"  [{ms:.0f} ms] {len(payload['chunks'])} chunks")
    for c in payload["chunks"]:
        snippet = c["text"][:120].replace("\n", " ")
        print(f"    [{c['doc_id']}] {snippet}…")
    require(len(payload["chunks"]) == len(ids), "get_chunks returned all ids")
    require(all(c.get("text") for c in payload["chunks"]),
            "every returned chunk has decompressed text")
    require(all(c.get("canonical_url", "").startswith("https://") for c in payload["chunks"]),
            "every chunk has canonical_url")


# ---------------------------------------------------------------------------
# WORKFLOW 8 — markdown rendering (what the agent actually sees)
# ---------------------------------------------------------------------------


def wf8_markdown() -> None:
    section("WORKFLOW 8 — default markdown formatter output (what the LLM sees)")

    call("search", query="R&D tax incentive eligibility", k=3)
    out = tools.search("R&D tax incentive eligibility", k=3)
    print("\n" + out[:1400])

    require("| # | Title |" in out, "markdown table header present")
    require("ato.gov.au" in out, "hits include canonical URLs")


# ---------------------------------------------------------------------------
# Default-exclude EPA behaviour + empty-shell cleanup
# ---------------------------------------------------------------------------


def wf9_default_excludes_epa() -> None:
    section("WORKFLOW 9 — search defaults exclude EPA; opt-in works")

    # Default call: no EPA should slip in. Use a query that EPAs would
    # otherwise flood (auth numbers / common private-ruling phrasing).
    call("search", query="ruling on private binding")
    data = _json_search(query="ruling on private binding", k=10, mode="keyword")
    epa_count = sum(1 for h in data["hits"] if h.get("type") == "Edited_private_advice")
    print(f"  EPA hits with default types: {epa_count} / {len(data['hits'])}")
    require(epa_count == 0, "default search excludes Edited_private_advice")

    # Explicit opt-in: EPA should appear when requested.
    call("search", query="private binding ruling", types=["Edited_private_advice"])
    opt_in = _json_search(
        query="private binding ruling",
        types=["Edited_private_advice"],
        k=5, mode="keyword",
    )
    require(len(opt_in["hits"]) > 0, "explicit types=['Edited_private_advice'] returns EPA")
    require(all(h.get("type") == "Edited_private_advice" for h in opt_in["hits"]),
            "opt-in query returns only EPA hits")

    # search_titles should also default-exclude.
    call("search_titles", query="ruling")
    titles = _json_titles(query="ruling", k=20)
    t_epa = sum(1 for h in titles["hits"] if h.get("type") == "Edited_private_advice")
    require(t_epa == 0, "search_titles default excludes EPA")


def wf10_no_empty_shells() -> None:
    section("WORKFLOW 10 — empty shells and Unknown category are gone")
    backend = tools.get_backend()
    unknown = backend.db.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE type = 'Unknown'"
    ).fetchone()["n"]
    empty = backend.db.execute(
        "SELECT COUNT(*) AS n FROM documents d "
        "WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)"
    ).fetchone()["n"]
    print(f"  Unknown-type docs: {unknown}")
    print(f"  Empty-shell docs:  {empty}")
    require(unknown == 0, "no Unknown-type documents remain")
    require(empty == 0, "no empty shells remain")


def coverage_probes() -> None:
    section("COVERAGE — quick probe across types and dates")
    backend = tools.get_backend()
    for t in ("Public_rulings", "Cases", "Edited_private_advice",
              "Legislation_and_supporting_material"):
        n = backend.db.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE type = ?", (t,)
        ).fetchone()["n"]
        has_title = backend.db.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE type = ? AND title IS NOT NULL AND title != ''",
            (t,)
        ).fetchone()["n"]
        has_date = backend.db.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE type = ? AND date IS NOT NULL",
            (t,)
        ).fetchone()["n"]
        pct_t = 100.0 * has_title / n if n else 0
        pct_d = 100.0 * has_date / n if n else 0
        print(f"  {t:<40} n={n:>6}  title={pct_t:5.1f}%  date={pct_d:5.1f}%")
        require(pct_t == 100.0, f"100% title coverage in {t}")
        require(pct_d == 100.0, f"100% date coverage in {t}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    db = os.environ.get("ATO_MCP_DATA_DIR", "")
    assert db, "Set ATO_MCP_DATA_DIR so backend sees the v5 DB"
    print(f"Testing v5 MCP surface against DB at {db}")

    wf1_rd_tax()
    wf2_pepsico()
    wf3_act_section()
    wf4_whats_new()
    wf5_doc_scope()
    wf6_pagination()
    wf7_get_chunks()
    wf8_markdown()
    wf9_default_excludes_epa()
    wf10_no_empty_shells()
    coverage_probes()

    section("RESULT")
    if FAIL_COUNT == 0:
        print("  ALL GREEN — ship it")
    else:
        print(f"  {FAIL_COUNT} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
