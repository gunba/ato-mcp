"""Rewrite ``chunks.heading_path`` using the level-aware chunker.

The pre-fix chunker assumed increasing heading levels mean nesting and
never popped same-level siblings, so e.g. ITAA 1997 sections with a flat
run of ``<h5>Note 1:</h5>`` / ``<h5>Note 2:</h5>`` came out as
``Note 1: › Note 2: › Note 3:``. ``chunk.chunk_markdown`` now tracks
levels and pops correctly.

This script re-extracts each doc's HTML, re-runs ``chunk_markdown``, and
matches new chunks to existing ones by ``ord``. If chunk text matches,
we UPDATE the heading_path. If text drifts (e.g. extractor / markdownify
output changed), we skip and log — those docs need a full re-embed
which is out of scope for this metadata-only migration.

Cost: pure CPU. Embeddings (``chunks_vec``) untouched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import zstandard as zstd  # noqa: E402

from ato_mcp.indexer import chunk as chunk_mod  # noqa: E402
from ato_mcp.indexer import extract as extract_mod  # noqa: E402
from ato_mcp.store import db as store_db  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--pages-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--type-filter", default=None,
                    help="Process only documents of this type (e.g. "
                         "Legislation_and_supporting_material).")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no DB at {args.db}")

    # Build canonical_id -> payload_path map from index.jsonl.
    print("loading index.jsonl …")
    payloads: dict[str, str] = {}
    with (args.pages_dir / "index.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid = rec.get("canonical_id") or ""
            if "docid=" not in cid:
                continue
            doc_id = cid.split("docid=", 1)[1]
            pp = rec.get("payload_path")
            if pp:
                payloads[doc_id] = pp
    print(f"  {len(payloads):,} payload paths")

    conn = store_db.connect(args.db, mode="rw")
    where = "WHERE type = ?" if args.type_filter else ""
    params: tuple = (args.type_filter,) if args.type_filter else ()
    rows = conn.execute(
        f"SELECT doc_id, type, title FROM documents {where} ORDER BY doc_id",
        params,
    ).fetchall()
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"target docs: {len(rows):,}")

    dctx = zstd.ZstdDecompressor()
    updated_chunks = 0
    updated_docs = 0
    skipped_text_drift = 0
    skipped_count_drift = 0
    skipped_no_payload = 0
    examined = 0
    t0 = time.monotonic()

    conn.execute("BEGIN")
    try:
        for i, doc_row in enumerate(rows, start=1):
            doc_id = doc_row["doc_id"]
            payload_rel = payloads.get(doc_id)
            if not payload_rel:
                skipped_no_payload += 1
                continue
            payload_path = args.pages_dir / payload_rel
            try:
                html = payload_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                skipped_no_payload += 1
                continue

            extracted = extract_mod.extract(html)
            if not extracted.markdown.strip():
                continue
            new_chunks = chunk_mod.chunk_markdown(
                extracted.markdown, root_title=extracted.title
            )
            if not new_chunks:
                continue

            existing = conn.execute(
                "SELECT chunk_id, ord, heading_path, text FROM chunks "
                "WHERE doc_id = ? ORDER BY ord ASC",
                (doc_id,),
            ).fetchall()

            if len(existing) != len(new_chunks):
                skipped_count_drift += 1
                continue

            doc_updates: list[tuple[str, int]] = []
            text_match = True
            for er, nc in zip(existing, new_chunks):
                if er["ord"] != nc.ord:
                    text_match = False
                    break
                # Verify text matches before trusting the heading_path.
                old_text = dctx.decompress(er["text"]).decode("utf-8")
                if old_text != nc.text:
                    text_match = False
                    break
                if (er["heading_path"] or "") != nc.heading_path:
                    doc_updates.append((nc.heading_path, er["chunk_id"]))

            if not text_match:
                skipped_text_drift += 1
                continue

            if doc_updates:
                if not args.dry_run:
                    conn.executemany(
                        "UPDATE chunks SET heading_path = ? WHERE chunk_id = ?",
                        doc_updates,
                    )
                updated_chunks += len(doc_updates)
                updated_docs += 1

            examined += 1
            if i % 5000 == 0:
                conn.execute("COMMIT")
                conn.execute("BEGIN")
                dt = time.monotonic() - t0
                rate = i / dt if dt else 0
                print(
                    f"  {i}/{len(rows)} ({rate:.0f} docs/s) "
                    f"updated_docs={updated_docs} updated_chunks={updated_chunks} "
                    f"text_drift={skipped_text_drift} count_drift={skipped_count_drift} "
                    f"no_payload={skipped_no_payload}"
                )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    if not args.dry_run and updated_chunks:
        print("rebuilding chunks_fts …")
        t1 = time.monotonic()
        conn.execute("BEGIN")
        try:
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        print(f"  done in {time.monotonic() - t1:.1f}s")

        new_version = _dt.date.today().strftime("%Y.%m.%d")
        store_db.set_meta(conn, "index_version", new_version)
        store_db.set_meta(
            conn, "last_update_at",
            _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="microseconds"),
        )
        conn.execute("ANALYZE")
        print(f"  index_version -> {new_version}")

    print(
        f"done: examined={examined}, updated_docs={updated_docs}, "
        f"updated_chunks={updated_chunks}, "
        f"text_drift_skipped={skipped_text_drift}, "
        f"count_drift_skipped={skipped_count_drift}, "
        f"no_payload={skipped_no_payload}, "
        f"total_time={time.monotonic() - t0:.1f}s"
    )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
