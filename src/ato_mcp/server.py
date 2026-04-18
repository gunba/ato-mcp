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

mcp: FastMCP = FastMCP(
    name="ato-mcp",
    instructions=(
        "Searches and fetches Australian Taxation Office (ATO) legal documents:\n"
        "- Start with `search` for natural-language questions.\n"
        "- Use `search_titles` for short codes like 's355-25' or 'TR 2024/3'.\n"
        "- Use `resolve` when you have a citation like 'TR 2024/3'.\n"
        "- Pull full content with `get_document` (outline -> markdown) or a\n"
        "  single section via `get_section(doc_id, anchor=...)`.\n"
        "Every result includes a `canonical_url` you can cite back to the user."
    ),
)


@mcp.tool
def search(
    query: str,
    k: int = T.DEFAULT_K,
    categories: list[str] | None = None,
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Hybrid search across the ATO corpus.

    Returns the top matches as a compact table or JSON. Use ``categories`` or
    ``doc_types`` to narrow (see ``list_categories`` / ``list_doc_types``).
    """
    return T.search(
        query,
        k=k,
        categories=categories,
        doc_types=doc_types,
        date_from=date_from,
        date_to=date_to,
        mode=mode,
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
def resolve(citation: str, format: Literal["markdown", "json"] = "markdown") -> str:
    """Resolve a citation (e.g. ``TR 2024/3``) to specific documents."""
    return T.resolve(citation, format=format)


@mcp.tool
def get_document(
    doc_id: str, format: Literal["outline", "markdown", "json"] = "outline"
) -> str:
    """Fetch a document. ``outline`` is cheap (~2 KB); ``markdown`` returns full content."""
    return T.get_document(doc_id, format=format)


@mcp.tool
def get_section(
    doc_id: str,
    anchor: str | None = None,
    heading_path: str | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Fetch a single section of a document by anchor or heading path."""
    return T.get_section(doc_id, anchor=anchor, heading_path=heading_path, format=format)


@mcp.tool
def get_chunks(
    chunk_ids: list[int], format: Literal["markdown", "json"] = "markdown"
) -> str:
    """Fetch specific chunks by id (from prior search results)."""
    return T.get_chunks(chunk_ids, format=format)


@mcp.tool
def list_categories(format: Literal["markdown", "json"] = "markdown") -> str:
    """List document categories with counts."""
    return T.list_categories(format=format)


@mcp.tool
def list_doc_types(format: Literal["markdown", "json"] = "markdown") -> str:
    """List ATO document types (TR, GSTR, PCG, ...) with counts."""
    return T.list_doc_types(format=format)


@mcp.tool
def whats_new(
    since: str | None = None,
    limit: int = 50,
    categories: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Documents most recently ingested or refreshed."""
    return T.whats_new(since=since, limit=limit, categories=categories, format=format)


@mcp.tool
def stats(format: Literal["markdown", "json"] = "json") -> str:
    """Index version, doc/chunk counts, last update timestamp."""
    return T.stats(format=format)


def run() -> None:
    """Entry point invoked by ``ato-mcp serve``."""
    mcp.run()
