"""Markdown + JSON formatters for MCP tool outputs.

Agents get a one-line-per-hit markdown table and everything they need to
decide whether to fetch the full document.
"""
from __future__ import annotations

import orjson

BASE_URL = "https://www.ato.gov.au"


def canonical_url(doc_id: str) -> str:
    """Synthesise the ATO URL for a ``doc_id``.

    ``doc_id`` is the ATO's own slashed path (e.g. ``TXR/TR20243/NAT/ATO/00001``);
    the URL is a direct substitution so we don't store ``href`` separately.
    """
    return f"{BASE_URL}/law/view/document?docid={doc_id}"


def format_hits_markdown(hits: list[dict]) -> str:
    if not hits:
        return "_No matches._"
    lines = ["| # | Title | Type | Heading | Snippet |", "|---|---|---|---|---|"]
    for i, hit in enumerate(hits, start=1):
        title = hit.get("title") or hit["doc_id"]
        typ = hit.get("type") or ""
        heading = (hit.get("heading_path") or "").replace("|", "\\|")
        snippet = (hit.get("snippet") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | [{title}]({hit['canonical_url']}) `{hit['doc_id']}` "
            f"| {typ} | {heading} | {snippet} |"
        )
    return "\n".join(lines)


def format_document_outline_markdown(
    doc: dict,
    *,
    outline_entries: list[dict] | None = None,
) -> str:
    """Render a document outline as a markdown table."""
    header = (
        f"# {doc['title']}\n\n"
        f"- **Doc ID:** `{doc['doc_id']}`\n"
        f"- **Type:** {doc.get('type') or ''}\n"
        f"- **Date:** {doc.get('date') or 'n/a'}\n"
        f"- **Source:** {doc['canonical_url']}\n\n"
    )
    if not outline_entries:
        return header + "_No content available for this document._\n"
    lines = [
        "## Outline\n",
        "| # | Heading | Anchor | ord | chunks | bytes |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(outline_entries, start=1):
        heading = (e.get("heading_path") or "(intro)").replace("|", "\\|")
        indent = "&nbsp;&nbsp;" * max(0, (e.get("depth") or 1) - 1)
        lines.append(
            f"| {i} | {indent}{heading} | `{e.get('anchor') or ''}` "
            f"| {e.get('start_ord')} | {e.get('chunk_count')} | {e.get('bytes')} |"
        )
    return header + "\n".join(lines) + "\n"


def format_document_section_markdown(
    doc: dict,
    chunks: list[dict],
    continuation_ord: int | None = None,
) -> str:
    """Render a section (or range) of a document as markdown."""
    header = f"**{doc['title']}** — [{doc['doc_id']}]({doc['canonical_url']})\n\n"
    if not chunks:
        return header + "_No chunks returned._\n"
    body = "\n\n".join(
        f"### {c['heading_path'] or '(intro)'}\n\n{c['text']}" for c in chunks
    )
    tail = ""
    if continuation_ord is not None:
        tail = (
            f"\n\n---\n\n_Truncated. Continue with "
            f"`get_document(doc_id, from_ord={continuation_ord})`._"
        )
    return header + body + tail + "\n"


def format_document_full_markdown(doc: dict, chunks: list[dict]) -> str:
    header = (
        f"# {doc['title']}\n\n"
        f"**Source:** {doc['canonical_url']}\n"
        f"**Type:** {doc.get('type') or ''}  \n"
        f"**Date:** {doc.get('date') or 'n/a'}\n\n---\n\n"
    )
    blocks = []
    current_heading: str | None = None
    for c in chunks:
        heading = c.get("heading_path") or ""
        if heading != current_heading:
            blocks.append(f"\n## {heading or '(intro)'}\n")
            current_heading = heading
        blocks.append(c.get("text", ""))
    return header + "\n".join(blocks) + "\n"


def as_json(obj: object) -> str:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8")
