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
        code = hit.get("docid_code") or hit["doc_id"]
        title = hit.get("title") or code
        category = hit.get("category") or ""
        heading = (hit.get("heading_path") or "").replace("|", "\\|")
        snippet = (hit.get("snippet") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | **{code}** — [{title}]({hit['canonical_url']}) `{hit['doc_id']}` "
            f"| {category} | {heading} | {snippet} |"
        )
    return "\n".join(lines)


def format_document_outline_markdown(doc: dict, chunks: list[dict]) -> str:
    header = (
        f"# {doc['title']}\n\n"
        f"- **Doc ID:** `{doc['doc_id']}`\n"
        f"- **Category:** {doc.get('category') or ''}\n"
        f"- **Type:** {doc.get('doc_type') or ''}\n"
        f"- **Code:** {doc.get('docid_code') or ''}\n"
        f"- **Published:** {doc.get('pub_date') or 'n/a'}\n"
        f"- **Source:** {doc['canonical_url']}\n\n"
    )
    if not chunks:
        return header + "_No content available for this document._\n"
    # Outline: take the first chunk per unique heading_path.
    seen: set[str] = set()
    body_lines: list[str] = ["## Outline\n"]
    for c in chunks:
        heading = c.get("heading_path") or ""
        if heading in seen:
            continue
        seen.add(heading)
        first_line = (c.get("text") or "").strip().splitlines()
        lead = first_line[0] if first_line else ""
        if len(lead) > 220:
            lead = lead[:217].rstrip() + "..."
        body_lines.append(f"- **{heading or '(intro)'}** — {lead}")
    return header + "\n".join(body_lines) + "\n"


def format_document_full_markdown(doc: dict, chunks: list[dict]) -> str:
    header = (
        f"# {doc['title']}\n\n"
        f"**Source:** {doc['canonical_url']}\n"
        f"**Type:** {doc.get('doc_type') or ''}  \n"
        f"**Code:** {doc.get('docid_code') or ''}  \n"
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
