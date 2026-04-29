"""Trial: strip leading-title heading echo from chunks.heading_path.

For each sample doc, we read every chunk's current ``heading_path`` and compute
a candidate ``heading_path`` by:

1. Splitting the path on the chunker's separator (' › ').
2. Dropping the leading segment if it equals the document title.
3. Dropping subsequent segments while they appear in the title's
   ' — '-split component set (case-insensitive, trimmed).
4. Re-joining what's left.

The script does NOT modify the database. It writes a side-by-side comparison
report so we can eyeball whether the rule keeps real body sections intact.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("/home/jordan/.local/share/ato-mcp/live/ato.db")
SEPARATOR = " › "
TITLE_JOIN = " — "

SAMPLE_DOC_IDS = [
    # ITAA 1997 sections (long legislation)
    "PAC/19970038/8-1",
    "PAC/19970038/6-5",
    "PAC/19970038/25-25",
    "PAC/19970038/40-25",
    "PAC/19970038/104-10",
    "PAC/19970038/152-10",
    "PAC/19970038/815-105",
    "PAC/19970038/855-10",
    # ITAA 1936 schedule section
    "PAC/19360027/Sch2F-268-10",
    # TAA 1953 schedule section
    "PAC/19530001/Sch1-128-10",
    # Public rulings
    "TXR/TR20243/NAT/ATO/00001",
    "TXR/TR20101/NAT/ATO/00001",
    "MXR/MT20082/NAT/ATO/00001",
    "GST/GSTR20063/NAT/ATO/00001",
    # Compliance guidance
    "COG/PCG20195/NAT/ATO/00001",
    "COG/PCG20242/NAT/ATO/00001",
    "PSR/PS20201/NAT/ATO/00001",
    # Taxpayer alert (often very short title)
    "TPA/TA20241/NAT/ATO/00001",
    # Case
    "JUD/2008ATC20-087/00001",
    # Edited private advice (one-off, usually minimal structure)
    "EV/ 105137529852",
]


_URL_RE = re.compile(r"^/law/view/document\?docid=", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _looks_like_url(s: str) -> bool:
    return bool(_URL_RE.match(s.strip()))


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s.strip()).lower()


def candidate_heading_path(current: str) -> str:
    """Mutex the title prefix out of a heading_path.

    The first segment of heading_path is always the chunker's ``root_title``
    (a ' — '-joined string of leading h1/h2/h3 text). Any subsequent segment
    that's a component of that root counts as front-matter echo and is
    dropped. URL-shaped leading segments are also dropped.
    """
    if not current:
        return ""
    parts = current.split(SEPARATOR)
    if not parts:
        return ""

    # Drop URL-shaped leading segments outright.
    while parts and _looks_like_url(parts[0]):
        parts = parts[1:]
    if not parts:
        return ""

    # Treat segment 0 as the root_title; build a component set from it.
    root = parts[0]
    parts = parts[1:]
    components = {_norm(p) for p in root.split(TITLE_JOIN) if p.strip()}
    components.add(_norm(root))  # also drops a re-echoed full root

    while parts and _norm(parts[0]) in components:
        parts = parts[1:]

    return SEPARATOR.join(parts)


def fetch_outline(conn: sqlite3.Connection, doc_id: str) -> tuple[str, list[tuple[str, str, int, int]]]:
    title_row = conn.execute("SELECT title FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if title_row is None:
        return ("(missing)", [])
    title = title_row[0]
    rows = conn.execute(
        """
        SELECT heading_path, anchor, MIN(ord) AS start_ord, COUNT(*) AS n
        FROM chunks WHERE doc_id = ? GROUP BY heading_path
        ORDER BY start_ord ASC
        """,
        (doc_id,),
    ).fetchall()
    return title, [(r[0] or "", r[1] or "", r[2], r[3]) for r in rows]


def _has_repeated_segments(s: str) -> bool:
    parts = [p.strip() for p in s.split(SEPARATOR) if p.strip()]
    return len(parts) != len(set(parts))


def render(out: list[str], doc_id: str, title: str, rows: list[tuple[str, str, int, int]]) -> None:
    out.append(f"## `{doc_id}`")
    out.append(f"**title:** {title}")
    if not rows:
        out.append("_no chunks_\n")
        return
    out.append("")
    out.append("| ord | n | anchor | current heading_path | candidate heading_path | flags |")
    out.append("|---:|---:|---|---|---|---|")
    for hp, anchor, ord_, n in rows:
        cand = candidate_heading_path(hp)
        flags = []
        if cand and _has_repeated_segments(cand):
            flags.append("repeat")
        if hp and _looks_like_url(hp.split(SEPARATOR)[0]):
            flags.append("url-front")
        cur_md = (hp or "(empty)").replace("|", "\\|")
        cand_md = (cand or "(intro)").replace("|", "\\|")
        flag_md = ",".join(flags)
        out.append(f"| {ord_} | {n} | `{anchor}` | {cur_md} | {cand_md} | {flag_md} |")
    out.append("")


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    out: list[str] = ["# Heading dedup trial", ""]
    for doc_id in SAMPLE_DOC_IDS:
        title, rows = fetch_outline(conn, doc_id)
        render(out, doc_id, title, rows)
    Path("scripts/heading_dedup_report.md").write_text("\n".join(out))
    print(f"wrote {Path('scripts/heading_dedup_report.md').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
