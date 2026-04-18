"""HTML -> markdown extraction edge cases."""
from __future__ import annotations

from ato_mcp.indexer.extract import extract


def test_extract_law_contents_div() -> None:
    html = """
    <html><body>
        <header>skip me</header>
        <div id="lawContents">
            <h1 id="top">Taxation Ruling TR 2024/3</h1>
            <p>This ruling sets out the Commissioner's view.</p>
            <h2 id="background">Background</h2>
            <p>There is a specific scheme in place.</p>
        </div>
    </body></html>
    """
    doc = extract(html)
    assert "Taxation Ruling TR 2024/3" in doc.markdown
    assert doc.title == "Taxation Ruling TR 2024/3"
    # Anchor captured on heading
    assert ("Taxation Ruling TR 2024/3", "top") in doc.anchors
    assert ("Background", "background") in doc.anchors
    # Header nav was stripped
    assert "skip me" not in doc.markdown


def test_extract_missing_lawcontents_uses_article() -> None:
    html = """
    <html><body>
        <article>
            <h1>Court Decision</h1>
            <p>Judgment text.</p>
        </article>
    </body></html>
    """
    doc = extract(html)
    assert "Court Decision" in doc.markdown
    assert "Judgment text." in doc.markdown


def test_extract_empty_returns_empty_markdown() -> None:
    doc = extract("")
    assert doc.markdown == ""
    assert doc.title is None


def test_extract_strips_scripts() -> None:
    html = """
    <div id="lawContents">
        <h1>Ruling</h1>
        <script>alert('x')</script>
        <p>Hello.</p>
    </div>
    """
    doc = extract(html)
    assert "alert" not in doc.markdown
    assert "Hello." in doc.markdown


def test_compose_title_from_leading_headings() -> None:
    """ATO rulings put h1=doc_type, h2=code, h3=subject consecutively."""
    html = """
    <div id="LawContent">
        <div id="LawFront">
            <h1>Class Ruling</h1>
            <h2>CR 2024/3</h2>
            <h3>Scrip for scrip rollover</h3>
        </div>
        <div id="LawBody">
            <p>The Commissioner rules as follows.</p>
            <h2>Background</h2>
            <p>The scheme is...</p>
        </div>
    </div>
    """
    doc = extract(html)
    assert doc.title == "Class Ruling — CR 2024/3 — Scrip for scrip rollover"
    # Background is a body section, not part of the title.
    assert "Background" not in (doc.title or "")
