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
    # [OF-01] Direct substitution — no separate href stored, link always reflects current doc_id.
    return f"{BASE_URL}/law/view/document?docid={doc_id}"


def format_hits_markdown(
    hits: list[dict],
    *,
    previously_seen: list[dict] | None = None,
) -> str:
    if not hits and not previously_seen:
        return "_No matches._"
    parts: list[str] = []
    if hits:
        lines = ["| # | Title | Type | Heading | Snippet |", "|---|---|---|---|---|"]
        for i, hit in enumerate(hits, start=1):
            title = hit.get("title") or hit["doc_id"]
            typ = hit.get("type") or ""
            # [OF-04] Escape '|' to '\\|' and replace newlines with spaces so cells stay inside the table grid.
            heading = (hit.get("heading_path") or "").replace("|", "\\|")
            snippet = (hit.get("snippet") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i} | [{title}]({hit['canonical_url']}) `{hit['doc_id']}` "
                f"| {typ} | {heading} | {snippet} |"
            )
        parts.append("\n".join(lines))
    else:
        # [OF-02] Distinguish 'nothing matched at all' from 'all top results were suppressed by SeenTracker'.
        parts.append("_No fresh matches._")
    if previously_seen:
        # [OF-03] Suppressed-results tail: bullet list of (chunk_id, title, doc_id, heading) + ready-to-paste get_chunks([...]) for one-shot recovery.
        ids_repr = ", ".join(str(s["chunk_id"]) for s in previously_seen)
        tail = [
            "",
            "---",
            "",
            f"_{len(previously_seen)} previously surfaced result(s) hidden:_",
            "",
        ]
        for s in previously_seen:
            heading = (s.get("heading_path") or "").replace("|", "\\|") or "(intro)"
            title = s.get("title") or s["doc_id"]
            tail.append(
                f"- `{s['chunk_id']}` — [{title}]({s['canonical_url']}) "
                f"`{s['doc_id']}` — {heading}"
            )
        tail.append("")
        tail.append(f"_Re-fetch with `get_chunks([{ids_repr}])`._")
        parts.append("\n".join(tail))
    return "\n".join(parts)


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
        # [OF-05] Indent by depth using doubled non-breaking spaces ('&nbsp;&nbsp;' per level) so deeper headings nest visually.
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
    # [OF-06] orjson + OPT_INDENT_2 for human-readable output; decoded to UTF-8 so the MCP response is always str, not bytes.
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8")
