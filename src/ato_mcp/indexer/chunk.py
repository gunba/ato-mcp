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
_URL_HEADING_RE = re.compile(r"^/law/view/document\?docid=", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

DEFAULT_MAX_TOKENS = 900
DEFAULT_OVERLAP_TOKENS = 120
# [IB-02] Section-aware chunking with tail-overlap bridge: 900 tokens cap, 120-token bridge between adjacent chunks under the same heading so vector search doesn't lose context at boundaries.
TITLE_SEP = " — "
PATH_SEP = " › "


def _norm_heading(text: str) -> str:
    return _WS_RE.sub(" ", text.strip()).lower()


def strip_title_prefix(heading_path: str) -> str:
    """Drop the document title's front-matter echo from a heading path.

    The chunker historically pushed the composed root_title onto the heading
    stack and then also pushed each of its constituent leading h1/h2/h3
    headings, producing paths like::

        Taxation Ruling — TR 2024/3 — Subject › Taxation Ruling › TR 2024/3 › Ruling

    This helper computes the de-duplicated form (``"Ruling"``) by:

    1. Dropping any leading ``/law/view/document?docid=…`` segment (some
       ATO pages emit a navigation anchor as a heading).
    2. Treating the first remaining segment as the root title and building
       a normalised component set from its ``" — "`` split.
    3. Dropping subsequent segments while they match a component
       (case-insensitive, whitespace-collapsed).

    Pure string transform — safe to apply at chunk emission time and as a
    one-shot rewrite over an existing ``chunks.heading_path`` column.
    """
    # [IB-04] Pure string transform — safe at chunk emission time AND as a one-shot rewrite over chunks.heading_path.
    if not heading_path:
        return ""
    parts = heading_path.split(PATH_SEP)
    while parts and _URL_HEADING_RE.match(parts[0].strip()):
        parts = parts[1:]
    if not parts:
        return ""
    root = parts[0]
    parts = parts[1:]
    components = {_norm_heading(p) for p in root.split(TITLE_SEP) if p.strip()}
    components.add(_norm_heading(root))
    while parts and _norm_heading(parts[0]) in components:
        parts = parts[1:]
    return PATH_SEP.join(parts)


@dataclass
class Chunk:
    ord: int
    heading_path: str
    anchor: str | None
    text: str


def approx_tokens(text: str) -> int:
    """Rough token count: whitespace split + a constant factor for subwords."""
    # [IB-05] Boundary-control heuristic only; the embedding tokenizer enforces the real per-batch token limit.
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
    # [IB-01] Recursive chunker: heading-boundary split first, then paragraph (blank-line) split, then sentence split — never silently truncates.
    if not markdown.strip():
        return []

    sections = _split_by_heading(markdown)
    # heading_levels parallels heading_stack so same-level siblings can pop the
    # previous entry instead of stacking under it. The old "truncate to
    # root_offset + level - 1" logic only capped depth, leaving siblings (e.g.
    # consecutive ``<h5>Note 1:</h5>`` / ``<h5>Note 2:</h5>`` blocks in
    # ITAA legislation) falsely nested under each other.
    heading_stack: list[str] = []
    heading_levels: list[int] = []
    if root_title:
        heading_stack.append(root_title)
        heading_levels.append(0)

    chunks: list[Chunk] = []
    ord_counter = 0

    for section in sections:
        level = section["level"]
        heading = section["heading"]
        anchor = section["anchor"]
        if level > 0:
            # [IB-03] Pop same-level siblings before pushing — converts consecutive <h5>Note 1:</h5> / <h5>Note 2:</h5> blocks from false 2-deep stack into siblings.
            # Pop siblings and descendants of the new heading. This is what
            # converts ``<h5>Note 1:</h5>`` followed by ``<h5>Note 2:</h5>``
            # from a false 2-deep stack into siblings.
            while heading_levels and heading_levels[-1] >= level:
                heading_stack.pop()
                heading_levels.pop()
            # Echo suppression: a leading ``<h1>`` whose text matches the
            # composed root_title is the document's own banner — don't push
            # it onto the path. The body still emits with the current stack.
            if not (root_title and heading.strip().lower() == root_title.strip().lower()):
                heading_stack.append(heading)
                heading_levels.append(level)

        body = _body_text(section["body"])
        if not body and level == 0:
            continue
        # Strip inline anchor markers from body; keep heading-level anchors via section[anchor].
        body = _ANCHOR_INLINE_RE.sub("", body).strip()
        if not body:
            continue

        heading_path = _path_trail(heading_stack)
        if root_title:
            heading_path = strip_title_prefix(heading_path)

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
