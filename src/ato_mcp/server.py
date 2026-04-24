"""FastMCP server wiring.

Run via ``ato-mcp serve`` (stdio transport). Each request lazily initializes
the shared :class:`ato_mcp.tools.Backend` (read-only SQLite + embedding model).
Between calls we re-check ``meta.last_update_at`` so a concurrent
``ato-mcp update`` is picked up automatically.
"""
from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from . import tools as T

_FALLBACK_INSTRUCTIONS = (
    "Searches and fetches Australian Taxation Office (ATO) legal documents. "
    "Run `ato-mcp update` to populate the corpus, then restart."
)

mcp: FastMCP = FastMCP(name="ato-mcp", instructions=_FALLBACK_INSTRUCTIONS)


@mcp.tool
def search(
    query: str,
    k: int = T.DEFAULT_K,
    types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    doc_scope: str | None = None,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    sort_by: Literal["relevance", "recency"] = "relevance",
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Hybrid BM25 + vector search across the ATO corpus.

    Returns the top ``k`` chunks ranked by RRF fusion. Multiple chunks
    from the same document are allowed.

    ``types`` filters by the ``type`` bucket ("Public_rulings",
    "Cases", ...); entries containing ``*`` are treated as globs
    ("Public_*"). ``doc_scope`` is a glob over the full ``doc_id`` path
    ("TXR/TR20243/*"). ``date_from`` / ``date_to`` filter on ``date``.

    ``sort_by='relevance'`` applies a 5-year-half-life recency boost by
    default — in tax law, later guidance typically supersedes earlier.
    ``sort_by='recency'`` sorts strictly by date descending.
    """
    return T.search(
        query,
        k=k,
        types=types,
        date_from=date_from,
        date_to=date_to,
        doc_scope=doc_scope,
        mode=mode,
        sort_by=sort_by,
        format=format,
    )


@mcp.tool
def search_titles(
    query: str,
    k: int = 20,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Fast title-only search. Use for citations (``TR 2024/3``, ``s 355-25``)
    or case names. Searches the ``title`` + per-doc headings, not bodies.
    """
    return T.search_titles(query, k=k, format=format)


@mcp.tool
def get_document(
    doc_id: str,
    format: Literal["outline", "markdown", "json"] = "outline",
    anchor: str | None = None,
    heading_path: str | None = None,
    from_ord: int | None = None,
    include_children: bool = False,
    count: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Fetch a document, or a slice of one, through a single tool.

    Modes:

    * No selector + ``format='outline'`` → table of contents.
    * No selector + ``format='markdown'`` → full document body.
      ``max_chars`` caps size.
    * ``anchor`` or ``heading_path`` → chunks at that heading.
      ``include_children=True`` rolls up the entire subtree.
    * ``from_ord=N`` → walk forward from an ordinal cursor. Pair with
      ``count`` (chunk limit) or ``max_chars`` (char budget). Truncated
      JSON carries ``continuation_ord`` for the follow-up call.
    """
    return T.get_document(
        doc_id,
        format=format,
        anchor=anchor,
        heading_path=heading_path,
        from_ord=from_ord,
        include_children=include_children,
        count=count,
        max_chars=max_chars,
    )


@mcp.tool
def get_chunks(
    chunk_ids: list[int], format: Literal["markdown", "json"] = "markdown"
) -> str:
    """Fetch specific chunks by id (from prior search results)."""
    return T.get_chunks(chunk_ids, format=format)


@mcp.tool
def whats_new(
    since: str | None = None,
    limit: int = 50,
    types: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Documents most recently published (by ``date``), optionally filtered
    by ``since`` or ``types``."""
    return T.whats_new(since=since, limit=limit, types=types, format=format)


def _build_instructions() -> str:
    try:
        backend = T.get_backend()
        conn = backend.db
        doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        meta_rows = {
            r["key"]: r["value"]
            for r in conn.execute("SELECT key, value FROM meta").fetchall()
        }
        types = [
            r["type"]
            for r in conn.execute(
                "SELECT type, COUNT(*) AS n FROM documents "
                "GROUP BY type ORDER BY n DESC"
            ).fetchall()
        ]
    except Exception:  # noqa: BLE001 — missing DB should not crash import
        return _FALLBACK_INSTRUCTIONS

    lines = [
        "ATO (Australian Taxation Office) tax-law corpus.",
        f"Corpus: {doc_count:,} documents, {chunk_count:,} chunks. "
        f"Index {meta_rows.get('index_version', '?')}, "
        f"updated {meta_rows.get('last_update_at', 'unknown')}. "
        f"Embedding: {meta_rows.get('embedding_model_id', 'unknown')}.",
        "",
        "Every document has four fields the LLM sees:",
        "  doc_id  Stable machine id, the ATO's own slashed path",
        "          (e.g. `TXR/TR20243/NAT/ATO/00001`). Pass this to `get_document`.",
        "  type    Top-level bucket. One of:",
        "            " + ", ".join(types),
        "  title   Human-readable label with citation inlined",
        "          (e.g. `TR 2024/3 — R&D tax incentive eligibility`).",
        "  date    Best-guess publication date (ISO yyyy-mm-dd). For filters",
        "          and recency sort only — NOT presented as authoritative.",
        "",
        "Retrieval flow:",
        "  1. `search(query, ...)` — hybrid BM25+vector. Scope with `types`,",
        "     `date_from`/`date_to`, or `doc_scope=\"TXR/TR20243/*\"`.",
        "  2. `search_titles(\"TR 2024/3\")` — direct citation lookup.",
        "  3. `get_document(doc_id, format=\"outline\")` — cheap TOC.",
        "  4. `get_document(doc_id, heading_path=...)` — expand a section.",
        "  5. `whats_new(since=\"2026-01-01\")` — recent publications.",
        "",
        "Every result carries a `canonical_url` suitable for citing back to the user.",
    ]
    return "\n".join(lines)


def run() -> None:
    """Entry point invoked by ``ato-mcp serve``."""
    try:
        backend = T.get_backend()
        if backend.model is not None:
            backend.model.encode(["warmup"], is_query=True)
    except Exception:  # noqa: BLE001 — opportunistic warmup
        pass
    mcp.instructions = _build_instructions()
    mcp.run()
