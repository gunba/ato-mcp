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
from .indexer import metadata as meta_mod

_FALLBACK_INSTRUCTIONS = (
    "Searches and fetches Australian Taxation Office (ATO) legal documents. "
    "Run `ato-mcp update` to populate the corpus, then restart."
)

mcp: FastMCP = FastMCP(name="ato-mcp", instructions=_FALLBACK_INSTRUCTIONS)


@mcp.tool
def search(
    query: str,
    k: int = T.DEFAULT_K,
    categories: list[str] | None = None,
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    doc_scope: str | None = None,
    category_scope: str | None = None,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    sort_by: Literal["relevance", "recency"] = "relevance",
    recency_half_life_years: float | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Hybrid BM25 + vector search across the ATO corpus.

    Returns the top ``k`` chunks ranked by RRF fusion. Multiple chunks from
    the same document are allowed — ranking decides distribution.

    ``doc_scope`` auto-detects: a value containing ``/`` matches against the
    machine ``doc_id`` path (e.g. ``"TXR/TR20133/*"``); otherwise it matches
    against the short human citation ``human_code`` (e.g. ``"TR 2013/3"``).
    ``category_scope`` is a glob over ``category`` (e.g. ``"Public_*"``).
    Both accept shell-style ``*`` wildcards.

    ``date_from`` / ``date_to`` filter on ``COALESCE(first_published_date,
    pub_date)`` so historical documents without a header-date scrape still
    reach the filter via their docid-derived year.

    ``sort_by='recency'`` re-sorts the top-k by publication date descending
    instead of relevance. ``recency_half_life_years`` (e.g. ``5.0``) boosts
    more recent hits during relevance ranking — useful for tax law where
    later guidance typically supersedes earlier.
    """
    return T.search(
        query,
        k=k,
        categories=categories,
        doc_types=doc_types,
        date_from=date_from,
        date_to=date_to,
        doc_scope=doc_scope,
        category_scope=category_scope,
        mode=mode,
        sort_by=sort_by,
        recency_half_life_years=recency_half_life_years,
        format=format,
    )


@mcp.tool
def search_titles(
    query: str,
    k: int = 20,
    doc_types: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Fast title-only search. Use for citations (``s355-25``, ``TR 2024/3``)."""
    return T.search_titles(query, k=k, doc_types=doc_types, format=format)


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

    * No selector + ``format='outline'`` → table of contents (cheap,
      includes ``start_ord``/``chunk_count``/``bytes`` per heading).
    * No selector + ``format='markdown'`` → full document body. Combine
      with ``max_chars`` to cap size.
    * ``anchor`` or ``heading_path`` → chunks at that heading.
      ``include_children=True`` rolls up the entire subtree (section +
      all nested sub-sections) until the next sibling or higher heading.
    * ``from_ord=N`` → walk forward from an ordinal cursor. Pair with
      ``count`` (chunk limit) or ``max_chars`` (char budget) to paginate
      through a long document. Truncated JSON responses carry
      ``continuation_ord`` for the follow-up call.
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
    categories: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Documents most recently published (or, failing that, ingested)."""
    return T.whats_new(since=since, limit=limit, categories=categories, format=format)


def _build_instructions() -> str:
    """Render the server preamble the MCP client sees on `initialize`.

    Pulls corpus stats, live category list, and live doc-type list from the
    open SQLite connection. Doc-type prefixes (TR, PCG, GSTR, ...) are
    cross-referenced against ``data/doc_type_map.yaml`` so the LLM sees both
    the shortcode and the human label where we have one.
    """
    try:
        backend = T.get_backend()
        conn = backend.db
        doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        meta_rows = {
            r["key"]: r["value"]
            for r in conn.execute("SELECT key, value FROM meta").fetchall()
        }
        categories = [
            r["category"]
            for r in conn.execute(
                "SELECT category, COUNT(*) AS n FROM documents "
                "GROUP BY category ORDER BY n DESC"
            ).fetchall()
        ]
        doc_types = [
            r["doc_type"]
            for r in conn.execute(
                "SELECT doc_type, COUNT(*) AS n FROM documents "
                "WHERE doc_type IS NOT NULL GROUP BY doc_type ORDER BY n DESC"
            ).fetchall()
        ]
    except Exception:  # noqa: BLE001 — a missing DB should not crash the server import path
        return _FALLBACK_INSTRUCTIONS

    type_map = meta_mod._load_doc_type_map()
    prefix_to_name = {p: entry.get("name") for p, entry in type_map.items() if entry.get("name")}
    name_to_prefix = {name: p for p, name in prefix_to_name.items()}
    labelled_types: list[str] = []
    for dt in doc_types[:40]:
        prefix = name_to_prefix.get(dt)
        labelled_types.append(f"{dt} ({prefix})" if prefix else dt)
    truncated = "..." if len(doc_types) > 40 else ""

    lines = [
        "ATO (Australian Taxation Office) tax-law corpus.",
        f"Corpus: {doc_count:,} documents, {chunk_count:,} chunks. "
        f"Index {meta_rows.get('index_version', '?')}, "
        f"updated {meta_rows.get('last_update_at', 'unknown')}. "
        f"Embedding: {meta_rows.get('embedding_model_id', 'unknown')}.",
        "",
        "Identifiers:",
        "  doc_id      Unique machine ID. The full ATO docid path with slashes,",
        "              e.g. `TXR/TR20133/NAT/ATO/00001`. Stable, use this whenever",
        "              you need to refer to a specific document.",
        "  human_code  Short human citation e.g. `TR 2013/3`. May be NULL; may be",
        "              ambiguous (versions / addenda share citations).",
        "",
        "Categories (use with `category_scope` glob or the `categories` filter):",
        "  " + ", ".join(categories),
        "",
        "Doc types (use with the `doc_types` filter; exact label required):",
        "  " + ", ".join(labelled_types) + truncated,
        "",
        "Retrieval flow:",
        "  1. `search(query, ...)` — hybrid BM25+vector. Returns chunks with",
        "     heading_path + human_code + doc_id. Scope with",
        "       `doc_scope=\"TXR/TR20133/*\"` (slash → matches doc_id path) or",
        "       `doc_scope=\"TR 2013/3\"`    (no slash → matches human_code) or",
        "       `category_scope=\"Public_*\"` (shell-style globs).",
        "  2. `search_titles(\"TR 2013/3\")` — direct citation / title lookup.",
        "  3. `get_document(doc_id, format=\"outline\")` — cheap heading-only TOC.",
        "  4. `get_section(doc_id, heading_path=...)` or `get_chunks([ids])` —",
        "     expand context once you know what you want.",
        "  5. `whats_new(since=\"2026-01-01\")` — recently-published documents.",
        "",
        "Every result carries a `canonical_url` suitable for citing back to the user.",
    ]
    return "\n".join(lines)


def run() -> None:
    """Entry point invoked by ``ato-mcp serve``."""
    # Warm the backend (DB + ONNX model + tokenizer) and run a throwaway
    # inference before yielding stdio. The first encode() call pays a
    # ~3-4 s ONNX JIT cost that we'd rather absorb during startup than on
    # the user's first search. Errors are non-fatal — if the DB or model
    # is missing we let the first call surface the real failure.
    try:
        backend = T.get_backend()
        if backend.model is not None:
            backend.model.encode(["warmup"], is_query=True)
    except Exception:  # noqa: BLE001 — opportunistic warmup, not correctness
        pass
    # Dynamic preamble: computed once here so the DB is guaranteed open
    # before we build the stats + vocabulary strings the client sees on
    # `initialize`.
    mcp.instructions = _build_instructions()
    mcp.run()
