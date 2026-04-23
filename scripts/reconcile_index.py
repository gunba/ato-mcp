#!/usr/bin/env python3
"""Reconcile index.jsonl against payloads/ on disk.

Orphans are canonical_ids present in the tree-crawl output (deduped_links.jsonl)
whose payload HTML exists on disk but have no row in ato_pages/index.jsonl.
Such rows are created by earlier catch-up runs where the downloader skipped
re-download because the payload was already present but never emitted an
index record.

Usage:
    python scripts/reconcile_index.py \\
        --pages-dir /home/jordan/Desktop/Projects/ato_pages \\
        --links-file /home/jordan/Desktop/Projects/ato_snapshots/deduped_links.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def slug(text: str, fallback: str = "node") -> str:
    s = _SLUG_RE.sub("_", text.strip()).strip("_")
    return (s or fallback)[:80]


def payload_path_for(payload_root: Path, link: dict) -> Path:
    rep = link.get("representative_path") or []
    p = payload_root
    for seg in rep:
        p = p / slug(seg)
    return p / f"{slug(link['canonical_id'], 'link')}.html"


def load_index_ids(index_path: Path) -> set[str]:
    ids: set[str] = set()
    if not index_path.exists():
        return ids
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("canonical_id")
            if cid:
                ids.add(cid)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages-dir", type=Path, required=True)
    ap.add_argument("--links-file", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pages_dir: Path = args.pages_dir
    links_file: Path = args.links_file
    payload_root = pages_dir / "payloads"
    index_path = pages_dir / "index.jsonl"

    if not links_file.exists():
        raise SystemExit(f"links file not found: {links_file}")
    if not payload_root.is_dir():
        raise SystemExit(f"payloads dir not found: {payload_root}")

    existing = load_index_ids(index_path)
    print(f"existing index rows: {len(existing):,}")

    added = 0
    seen = 0
    skipped_already = 0
    skipped_no_payload = 0
    now = datetime.now(timezone.utc).isoformat()

    out_fh = None
    if not args.dry_run:
        out_fh = index_path.open("a", encoding="utf-8")

    try:
        with links_file.open("r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                rec = json.loads(line)
                cid = rec.get("canonical_id")
                if not cid:
                    continue
                if cid in existing:
                    skipped_already += 1
                    continue
                path = payload_path_for(payload_root, rec)
                if not path.exists():
                    skipped_no_payload += 1
                    continue
                row = {
                    "canonical_id": cid,
                    "href": rec.get("href") or cid,
                    "status": "success",
                    "payload_path": str(path.relative_to(pages_dir)),
                    "assets": [],
                    "error": None,
                    "http_status": None,
                    "downloaded_at": now,
                    "reconciled": True,
                }
                if out_fh is not None:
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.add(cid)
                added += 1
                if added % 5000 == 0:
                    print(f"  added {added:,} so far...")
    finally:
        if out_fh is not None:
            out_fh.close()

    print()
    print(f"links scanned:          {seen:,}")
    print(f"already in index:       {skipped_already:,}")
    print(f"no payload on disk:     {skipped_no_payload:,}")
    print(f"{'would add' if args.dry_run else 'added'} to index: {added:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
