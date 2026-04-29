"""Chunker invariants."""
from __future__ import annotations

from ato_mcp.indexer.chunk import approx_tokens, chunk_markdown, strip_title_prefix


def test_chunk_keeps_headings_in_path() -> None:
    md = """
# ITAA 1997

Some intro.

## Division 355

Research and development tax incentive.

### Section 355-25

Core R&D activities definition.

A core R&D activity is experimental.
"""
    chunks = chunk_markdown(md, root_title="ITAA 1997")
    # Every chunk must reflect the heading it lives under.
    assert any("Division 355" in c.heading_path for c in chunks)
    assert any("Section 355-25" in c.heading_path for c in chunks)
    # No chunk exceeds the max-token budget.
    for c in chunks:
        assert approx_tokens(c.text) <= 1200


def test_chunk_splits_oversize_section() -> None:
    big = "\n\n".join(["para " + " ".join(["alpha"] * 40) for _ in range(60)])
    md = f"# Heading\n\n{big}\n"
    chunks = chunk_markdown(md, max_tokens=200, overlap_tokens=40)
    assert len(chunks) > 1
    for c in chunks:
        # 200 max + overlap bridge + small slack
        assert approx_tokens(c.text) <= 280


def test_chunk_stable_under_small_edit() -> None:
    md1 = "# Title\n\n" + "Alpha beta gamma delta. " * 40 + "\n\n## Section\n\nContent.\n"
    md2 = md1.replace("delta", "deltax", 1)  # one-word change
    c1 = [c.text for c in chunk_markdown(md1)]
    c2 = [c.text for c in chunk_markdown(md2)]
    changed = sum(1 for a, b in zip(c1, c2) if a != b)
    assert changed <= 2, "a one-word change should flip at most two chunks"


def test_chunk_empty_returns_no_chunks() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n  ") == []


def test_strip_title_prefix_drops_root_and_components() -> None:
    title = ("Taxation Ruling — TR 2024/3 — Income tax: deductibility of "
             "self-education expenses incurred by an individual")
    hp = (f"{title} › Taxation Ruling › TR 2024/3 › Ruling")
    assert strip_title_prefix(hp) == "Ruling"
    hp_with_subtree = f"{title} › Taxation Ruling › TR 2024/3 › Ruling › Footnotes"
    assert strip_title_prefix(hp_with_subtree) == "Ruling › Footnotes"


def test_strip_title_prefix_collapses_whitespace_in_components() -> None:
    # Source title sometimes has double spaces around colons; the path
    # component is single-spaced. The dedup must still treat them as equal.
    title = "Taxation Ruling — TR 2024/3 — Income tax:  deductibility"
    hp = f"{title} › Taxation Ruling › TR 2024/3 › Income tax: deductibility"
    assert strip_title_prefix(hp) == ""


def test_strip_title_prefix_drops_url_front_segment() -> None:
    hp = ("/law/view/document?docid=PAC/19970038/25-25 › "
          "Income Tax Assessment Act 1997 › Note:")
    assert strip_title_prefix(hp) == "Note:"


def test_strip_title_prefix_keeps_real_body_headings() -> None:
    hp = "Income Tax Assessment Act 1997 › Division 355 › Section 355-25"
    assert strip_title_prefix(hp) == "Division 355 › Section 355-25"


def test_chunk_emits_clean_heading_path() -> None:
    md = """
# Taxation Ruling

## TR 2024/3

### Subject heading

intro paragraph.

## Ruling

Body content for the ruling section.
"""
    title = "Taxation Ruling — TR 2024/3 — Subject heading"
    chunks = chunk_markdown(md, root_title=title)
    paths = [c.heading_path for c in chunks]
    # Front-matter title segments should not appear in any chunk's path.
    for p in paths:
        for component in ("Taxation Ruling", "TR 2024/3", "Subject heading"):
            assert not p.startswith(component), f"front-matter echo: {p!r}"
    # Real body section is preserved.
    assert "Ruling" in paths
