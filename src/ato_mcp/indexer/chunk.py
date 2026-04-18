"""Heading-aware recursive chunker for ATO markdown.

Strategy:
1. Split on ``#``/``##``/``###`` boundaries; track heading_path.
2. For any chunk that still exceeds ``max_tokens``, re-split on blank lines and
   then sentences.
3. Stitch adjacent chunks under the same heading with a small token overlap so
   section boundaries stay soft for vector search.

Tokens are estimated with a simple whitespace heuristic — good enough for
boundary control, since the embedding tokenizer enforces the real limit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{#([^}]+)\})?\s*$")
_ANCHOR_INLINE_RE = re.compile(r"\{#([^}]+)\}")

DEFAULT_MAX_TOKENS = 900
DEFAULT_OVERLAP_TOKENS = 120


@dataclass
class Chunk:
    ord: int
    heading_path: str
    anchor: str | None
    text: str


def approx_tokens(text: str) -> int:
    """Rough token count: whitespace split + a constant factor for subwords."""
    return max(1, int(len(text.split()) * 1.3))


def _split_by_heading(md: str) -> list[dict]:
    """Return list of {'level', 'heading', 'anchor', 'body'} sections."""
    sections: list[dict] = []
    current: dict | None = None
    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if current is not None:
                sections.append(current)
            level = len(m.group(1))
            heading = m.group(2).strip()
            anchor = m.group(3)
            current = {"level": level, "heading": heading, "anchor": anchor, "body": []}
        else:
            if current is None:
                current = {"level": 0, "heading": "", "anchor": None, "body": []}
            current["body"].append(line)
    if current is not None:
        sections.append(current)
    return sections


def _body_text(body: list[str]) -> str:
    return "\n".join(body).strip()


def _path_trail(stack: list[str]) -> str:
    return " › ".join(s for s in stack if s)


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Paragraph-then-sentence split for oversize sections."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for p in paragraphs:
        p_tok = approx_tokens(p)
        if p_tok > max_tokens:
            # Further split by sentence.
            for sentence in _sentence_split(p):
                s_tok = approx_tokens(sentence)
                if buf_tokens + s_tok > max_tokens and buf:
                    out.append("\n\n".join(buf))
                    buf, buf_tokens = [], 0
                buf.append(sentence)
                buf_tokens += s_tok
            continue
        if buf_tokens + p_tok > max_tokens and buf:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(p)
        buf_tokens += p_tok
    if buf:
        out.append("\n\n".join(buf))
    return out or [text]


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def chunk_markdown(
    markdown: str,
    *,
    root_title: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Return heading-aware chunks for a single document.

    Each chunk carries a heading_path formed by joining the stack of active
    headings with ``' › '`` (note the non-ASCII separator).
    """
    if not markdown.strip():
        return []

    sections = _split_by_heading(markdown)
    heading_stack: list[str] = []
    if root_title:
        heading_stack.append(root_title)

    chunks: list[Chunk] = []
    ord_counter = 0

    for section in sections:
        level = section["level"]
        heading = section["heading"]
        anchor = section["anchor"]
        # Trim stack to parent level. level==0 means content before any heading.
        if level > 0:
            # keep root_title (if present) + up to (level-1) prior headings
            root_offset = 1 if root_title else 0
            heading_stack = heading_stack[: root_offset + max(level - 1, 0)]
            # Suppress echo: if this heading is the same text as the root_title,
            # don't append it — otherwise heading_path reads "X › X › ...".
            if not (root_title and heading.strip().lower() == root_title.strip().lower()):
                heading_stack.append(heading)

        body = _body_text(section["body"])
        if not body and level == 0:
            continue
        # Strip inline anchor markers from body; keep heading-level anchors via section[anchor].
        body = _ANCHOR_INLINE_RE.sub("", body).strip()
        if not body:
            continue

        heading_path = _path_trail(heading_stack)

        if approx_tokens(body) <= max_tokens:
            chunks.append(Chunk(ord=ord_counter, heading_path=heading_path, anchor=anchor, text=body))
            ord_counter += 1
            continue

        parts = _split_long(body, max_tokens=max_tokens)
        prev_tail: str = ""
        for i, part in enumerate(parts):
            text = (prev_tail + "\n\n" + part).strip() if prev_tail else part
            chunks.append(Chunk(ord=ord_counter, heading_path=heading_path, anchor=anchor, text=text))
            ord_counter += 1
            prev_tail = _tail_overlap(part, overlap_tokens)

    return chunks


def _tail_overlap(text: str, overlap_tokens: int) -> str:
    """Keep the last ``overlap_tokens`` (approx) of ``text`` as a bridge."""
    words = text.split()
    target = max(1, int(overlap_tokens / 1.3))
    if len(words) <= target:
        return text
    return " ".join(words[-target:])


def chunk_texts(chunks: Iterable[Chunk]) -> list[str]:
    return [c.text for c in chunks]
