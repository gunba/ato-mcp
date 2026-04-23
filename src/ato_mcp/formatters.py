"""Markdown + JSON formatters for MCP tool outputs.

The markdown is intentionally compact — agents get a one-line-per-hit table
with everything they need to decide whether to fetch the full document.
"""
from __future__ import annotations

import orjson

BASE_URL = "https://www.ato.gov.au"


def canonical_url(href: str | None) -> str:
    if not href:
        return BASE_URL
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return BASE_URL + href


def format_hits_markdown(hits: list[dict]) -> str:
    if not hits:
        return "_No matches._"
    lines = ["| # | Doc | Category | Heading | Snippet |", "|---|---|---|---|---|"]
    for i, hit in enumerate(hits, start=1):
        code = hit.get("human_code") or hit["doc_id"]
        title = hit.get("human_title") or hit.get("title") or code
        category = hit.get("category") or ""
        heading = (hit.get("heading_path") or "").replace("|", "\\|")
        snippet = (hit.get("snippet") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | **{code}** — [{title}]({hit['canonical_url']}) `{hit['doc_id']}` "
            f"| {category} | {heading} | {snippet} |"
        )
    return "\n".join(lines)


def format_document_outline_markdown(
    doc: dict,
    chunks: list[dict] | None = None,
    *,
    outline_entries: list[dict] | None = None,
) -> str:
    """Render a document outline as a markdown table.

    Two input shapes are accepted for backward compat:

    * ``outline_entries`` (preferred): precomputed list of dicts with
      ``{heading_path, anchor, depth, start_ord, chunk_count, bytes}``.
    * ``chunks``: raw chunks list — the function derives a basic outline
      from distinct heading_paths.
    """
    header = (
        f"# {doc.get('human_title') or doc['title']}\n\n"
        f"- **Doc ID:** `{doc['doc_id']}`\n"
        f"- **Category:** {doc.get('category') or ''}\n"
        f"- **Type:** {doc.get('doc_type') or ''}\n"
        f"- **Citation:** {doc.get('human_code') or ''}\n"
        f"- **Published:** {doc.get('first_published_date') or doc.get('pub_date') or 'n/a'}\n"
        f"- **Source:** {doc['canonical_url']}\n\n"
    )
    entries = outline_entries
    if entries is None and chunks is not None:
        seen: set[str] = set()
        entries = []
        for c in chunks:
            hp = c.get("heading_path") or ""
            if hp in seen:
                continue
            seen.add(hp)
            entries.append({
                "heading_path": hp,
                "anchor": c.get("anchor"),
                "depth": hp.count(" › ") + 1 if hp else 0,
                "start_ord": c.get("ord"),
                "chunk_count": 1,
                "bytes": len((c.get("text") or "").encode("utf-8")),
            })
    if not entries:
        return header + "_No content available for this document._\n"
    lines = [
        "## Outline\n",
        "| # | Heading | Anchor | ord | chunks | bytes |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(entries, start=1):
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
    header = (
        f"**{doc.get('human_title') or doc['title']}** — "
        f"[{doc.get('human_code') or doc['doc_id']}]({doc['canonical_url']})\n\n"
    )
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
        f"# {doc.get('human_title') or doc['title']}\n\n"
        f"**Source:** {doc['canonical_url']}\n"
        f"**Type:** {doc.get('doc_type') or ''}  \n"
        f"**Citation:** {doc.get('human_code') or ''}  \n"
        f"**Published:** {doc.get('pub_date') or 'n/a'}\n\n---\n\n"
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
