"""Surgical ingest of new docs into the live DB.

build-index requires that ``previous_manifest`` content_hashes match
re-extracted ones, which fails after even small dep updates (markdownify,
selectolax). Re-embedding the entire corpus to bridge that gap is wasteful
when the actual delta is a few thousand new docs.

This script reads ``ato_pages/index.jsonl``, picks every record whose
doc_id isn't in ``documents`` or ``empty_shells`` yet, and runs the build
pipeline for that doc only:

    extract  -> chunk (clean heading_path) -> embed -> insert into
    documents + chunks + chunks_fts + chunks_vec + title_fts + empty_shells

Existing rows are untouched. The new docs use the same chunker as a fresh
build, so their heading_path is dedup-clean from the first write.

Usage:
    LD_LIBRARY_PATH="$NVIDIA_LIBS" \\
    python scripts/ingest_new_docs.py \\
        --pages-dir /path/to/ato_pages \\
        --db /path/to/ato.db \\
        --model-path /path/to/model_quantized.onnx \\
        --tokenizer-path /path/to/tokenizer.json \\
        --gpu
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
import zstandard as zstd  # noqa: E402

from ato_mcp.embed.model import EmbeddingModel, vec_to_bytes  # noqa: E402
from ato_mcp.indexer import chunk as chunk_mod  # noqa: E402
from ato_mcp.indexer import extract as extract_mod  # noqa: E402
from ato_mcp.indexer import metadata as meta_mod  # noqa: E402
from ato_mcp.indexer import rules as rules_mod  # noqa: E402
from ato_mcp.store import db as store_db  # noqa: E402
from ato_mcp.store.queries import (  # noqa: E402
    INSERT_CHUNK,
    INSERT_CHUNK_FTS,
    INSERT_DOCUMENT,
    INSERT_EMPTY_SHELL,
    INSERT_TITLE_FTS,
    INSERT_VEC,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages-dir", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--tokenizer-path", type=Path, required=True)
    ap.add_argument("--gpu", action="store_true",
                    help="Use CUDAExecutionProvider when available.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N new docs (smoke).")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no DB at {args.db}")

    conn = store_db.connect(args.db, mode="rw")

    have_docs = {r[0] for r in conn.execute("SELECT doc_id FROM documents")}
    have_shells = {r[0] for r in conn.execute("SELECT doc_id FROM empty_shells")}
    print(f"existing: {len(have_docs):,} docs, {len(have_shells):,} empty shells")

    candidates: list[tuple[str, str, Path]] = []
    with (args.pages_dir / "index.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cid = rec.get("canonical_id") or ""
            if "docid=" not in cid:
                continue
            doc_id = meta_mod.doc_id_for(cid)
            if doc_id in have_docs or doc_id in have_shells:
                continue
            if rec.get("status") != "success":
                continue
            payload = rec.get("payload_path")
            if not payload:
                continue
            candidates.append((doc_id, cid, args.pages_dir / payload))

    if args.limit is not None:
        candidates = candidates[: args.limit]
    print(f"candidates: {len(candidates):,}")

    providers: tuple[str, ...] | None = None
    if args.gpu:
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    model = EmbeddingModel(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        providers=providers,
    )

    cctx = zstd.ZstdCompressor(level=3)
    inserted = 0
    shelved = 0
    t0 = time.monotonic()
    conn.execute("BEGIN")
    try:
        for i, (doc_id, canonical_id, payload_path) in enumerate(candidates, start=1):
            try:
                html = payload_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                shelved += 1
                conn.execute(
                    INSERT_EMPTY_SHELL,
                    (doc_id, _now(), _now(), "ingest:missing-payload"),
                )
                continue

            extracted = extract_mod.extract(html)
            if not extracted.markdown.strip():
                shelved += 1
                conn.execute(
                    INSERT_EMPTY_SHELL,
                    (doc_id, _now(), _now(), "ingest:no-content"),
                )
                continue

            category = meta_mod.category_from_path(str(payload_path))
            if category == "Unknown":
                category = meta_mod.category_for_docid(canonical_id)
            pub_date = meta_mod.extract_pub_date(extracted.markdown)
            derived = rules_mod.derive_metadata(rules_mod.RuleInputs(
                doc_id=doc_id,
                title=extracted.title or doc_id,
                headings=tuple(extracted.headings),
                body_head=extracted.markdown[:3000],
                category=category,
                pub_date=pub_date,
            ))
            title = derived.title or extracted.title or doc_id
            ch = meta_mod.content_hash(extracted.markdown, {"title": title})

            chunks = chunk_mod.chunk_markdown(extracted.markdown, root_title=extracted.title)
            if not chunks:
                shelved += 1
                conn.execute(
                    INSERT_EMPTY_SHELL,
                    (doc_id, _now(), _now(), "ingest:no-chunks"),
                )
                continue

            encoded = model.encode([c.text for c in chunks], is_query=False, batch_size=64)
            vectors_i8 = encoded.vectors_int8

            conn.execute(
                INSERT_DOCUMENT,
                (
                    doc_id, category, title, derived.date,
                    _now(), ch, "INGESTED",
                ),
            )
            conn.execute(
                INSERT_TITLE_FTS,
                (doc_id, title, " ".join(extracted.headings)),
            )
            for j, c in enumerate(chunks):
                compressed = cctx.compress(c.text.encode("utf-8"))
                cur = conn.execute(
                    INSERT_CHUNK,
                    (doc_id, c.ord, c.heading_path, c.anchor, compressed),
                )
                rowid = cur.lastrowid
                conn.execute(INSERT_CHUNK_FTS, (rowid, c.text, c.heading_path))
                conn.execute(INSERT_VEC, (rowid, vec_to_bytes(vectors_i8[j])))
            inserted += 1

            if i % 250 == 0:
                conn.execute("COMMIT")
                conn.execute("BEGIN")
                dt = time.monotonic() - t0
                rate = i / dt if dt else 0
                print(f"  {i}/{len(candidates)} ({rate:.1f} docs/s) "
                      f"inserted={inserted} shelved={shelved}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    new_version = _dt.date.today().strftime("%Y.%m.%d")
    store_db.set_meta(conn, "index_version", new_version)
    store_db.set_meta(
        conn, "last_update_at",
        _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="microseconds"),
    )
    print(f"done: inserted={inserted}, shelved={shelved}, "
          f"index_version={new_version}, total_time={time.monotonic() - t0:.1f}s")
    conn.execute("ANALYZE")
    conn.close()
    return 0


def _now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="microseconds")


if __name__ == "__main__":
    sys.exit(main())
