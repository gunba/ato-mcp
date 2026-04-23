"""HTML -> structured markdown for ATO documents.

Strategy:
1. Parse with selectolax (lexbor) for speed.
2. Find the content container (``lawContents`` div, falling back to ``<article>``
   or ``<main>`` / ``<body>`` with nav stripped).
3. Walk headings and collect anchor IDs onto a ``{#anchor}`` suffix.
4. Convert to markdown via ``markdownify`` with a tight tag whitelist.
5. Emit a bare markdown string; the chunker handles heading-based splits.

Output also includes a plain-text ``outline`` (heading-path tuples) used by the
metadata/chunking steps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from markdownify import markdownify
from selectolax.parser import HTMLParser, Node

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_NAV_LIKE_CLASSES = (
    "site-header",
    "global-header",
    "breadcrumb",
    "breadcrumbs",
    "site-footer",
    "page-footer",
    "navigation",
    "skip-links",
)


@dataclass
class ExtractedDoc:
    markdown: str
    title: str | None
    html_title: str | None = None  # raw <title> (browser tab text)
    headings: list[str] = field(default_factory=list)
    anchors: list[tuple[str, str]] = field(default_factory=list)  # (heading_text, anchor_id)


def extract(html: str) -> ExtractedDoc:
    if not html or not html.strip():
        return ExtractedDoc(markdown="", title=None, html_title=None)

    tree = HTMLParser(html)
    html_title = _first_text(tree, "title")

    container = _pick_container(tree)
    if container is None:
        return ExtractedDoc(markdown="", title=None, html_title=html_title)

    _strip_noise(container)
    anchors = _collect_anchors(container)

    # Capture "title headings" — consecutive leading headings before any body
    # content. On ATO rulings that gives h1=doc_type, h2=code, h3=subject.
    lead_headings = _leading_headings(container)
    title = _compose_title(lead_headings) or html_title

    _inject_anchor_suffixes(container)

    headings = [
        (h.text(deep=True, separator=" ", strip=True) or "")
        for h in container.css(",".join(_HEADING_TAGS))
    ]
    html_fragment = container.html or ""
    markdown = markdownify(
        html_fragment,
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "iframe"],
    )
    markdown = _tidy_markdown(markdown)
    return ExtractedDoc(
        markdown=markdown, title=title, html_title=html_title,
        headings=headings, anchors=anchors,
    )


def _leading_headings(container: Node) -> list[str]:
    """Return text of consecutive headings at the start of the container.

    Walk direct children; collect ``h1-h6`` text until we hit a non-heading
    element with substantial content. Wrapper divs that only contain other
    headings count as heading-bearing too (so we don't stop at a ``LawFront``
    / ``front`` / ``LawPreamble`` div wrapping the h1/h2/h3 block).

    Once we have dived into a front-matter wrapper, we do not dive into any
    subsequent wrapper — the next div is almost always the body.
    """
    out: list[str] = []
    dived = False
    for child in container.iter(include_text=False):
        tag = (child.tag or "").lower()
        if tag in _HEADING_TAGS:
            text = child.text(deep=True, separator=" ", strip=True) or ""
            if text:
                out.append(text)
            continue
        if dived:
            break
        nested_headings = child.css(",".join(_HEADING_TAGS))
        non_heading_text = child.text(deep=True, separator=" ", strip=True) or ""
        if nested_headings and len(non_heading_text) <= 800:
            for h in nested_headings:
                t = h.text(deep=True, separator=" ", strip=True) or ""
                if t:
                    out.append(t)
            dived = True
            continue
        if non_heading_text.strip():
            break
    return out[:4]


def _compose_title(headings: list[str]) -> str | None:
    """Join a small number of leading headings into a readable title."""
    cleaned = [h.strip() for h in headings if h and h.strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    out: list[str] = []
    for h in cleaned:
        if out and (out[-1].lower().startswith(h.lower()) or h.lower().startswith(out[-1].lower())):
            continue
        out.append(h)
    return " — ".join(out)


def _first_text(tree: HTMLParser, selector: str) -> str | None:
    node = tree.css_first(selector)
    if node is None:
        return None
    text = node.text(deep=True, separator=" ", strip=True)
    return text or None


def _pick_container(tree: HTMLParser) -> Node | None:
    # ATO has used several wrapper ids over the years; try each.
    for selector in ("#LawContent", "#lawContents", "#contents", "#content", "article", "main"):
        node = tree.css_first(selector)
        if node is not None:
            return node
    return tree.body or tree.root


def _strip_noise(node: Node) -> None:
    for selector in ("script", "style", "noscript", "template"):
        for el in node.css(selector):
            el.decompose()
    for cls in _NAV_LIKE_CLASSES:
        for el in node.css(f".{cls}"):
            el.decompose()
    for el in node.css("nav"):
        el.decompose()


def _inject_anchor_suffixes(node: Node) -> None:
    """Rewrite ``<h* id="foo">Title</h*>`` to append ``{#foo}`` inside the heading.

    markdownify preserves the text; we append the anchor so chunks can reference
    it directly. Same rule applied to ``<a name="foo">`` siblings.
    """
    for tag in _HEADING_TAGS:
        for heading in node.css(tag):
            anchor = heading.attributes.get("id")
            if not anchor:
                # Look for a child <a name="...">
                for a in heading.css("a"):
                    name = a.attributes.get("name") or a.attributes.get("id")
                    if name:
                        anchor = name
                        break
            if not anchor:
                continue
            # Append ' {#anchor}' to the heading text
            heading.insert_child(f" {{#{anchor}}}")


_MD_COLLAPSE = re.compile(r"\n{3,}")
_MD_TRAIL_WS = re.compile(r"[ \t]+\n")


def _tidy_markdown(md: str) -> str:
    md = _MD_TRAIL_WS.sub("\n", md)
    md = _MD_COLLAPSE.sub("\n\n", md)
    return md.strip() + "\n"


def _collect_anchors(node: Node) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for heading in node.css(",".join(_HEADING_TAGS)):
        anchor = heading.attributes.get("id")
        if not anchor:
            for a in heading.css("a"):
                name = a.attributes.get("name") or a.attributes.get("id")
                if name:
                    anchor = name
                    break
        if anchor:
            text = heading.text(deep=True, separator=" ", strip=True)
            out.append((text, anchor))
    return out


def heading_outline(headings: Iterable[str]) -> str:
    return " › ".join(h for h in headings if h)
