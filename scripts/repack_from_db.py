"""Rebuild v5-shape pack files + manifest from a live v5 ato.db.

v0.1 released in April 2026 used v4-era record shapes. After the v5 schema
migration, the DB is clean but the published packs are stale. This script
reads the live DB and writes fresh pack records — no re-scrape, no GPU,
just a CPU-bound walk over the existing documents + chunks + embeddings.

Usage:
    python scripts/repack_from_db.py \\
        --db ./release/ato.db \\
        --out ./release-v5 \\
        --index-version 2026.04.24

Output tree:
    release-v5/
        manifest.json
        packs/pack-<sha8>.bin.zst  (~170 files, ~1 GB total)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import zstandard as zstd  # noqa: E402

from ato_mcp.indexer.pack import PackBuilder, encode_embedding  # noqa: E402
from ato_mcp.store import db as store_db  # noqa: E402
from ato_mcp.store.manifest import (  # noqa: E402
    DocRef,
    Manifest,
    ModelInfo,
    PackInfo,
    save_manifest,
)


def _load_embedding(conn: sqlite3.Connection, chunk_id: int) -> bytes:
    # sqlite-vec's vec0 stores int8 vectors; we pull them as raw bytes
    # via the virtual-table's rowid lookup, then base64-encode for the
    # pack record (matching build.py's encode_embedding helper).
    row = conn.execute(
        "SELECT embedding FROM chunks_vec WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"no embedding for chunk_id={chunk_id}")
    raw = row["embedding"]
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise RuntimeError(
            f"chunks_vec.embedding returned {type(raw)} for chunk_id={chunk_id}"
        )
    b = bytes(raw)
    if len(b) != store_db.EMBEDDING_DIM:
        raise RuntimeError(
            f"embedding size mismatch: got {len(b)}, expected {store_db.EMBEDDING_DIM}"
        )
    return b


def repack(db_path: Path, out_dir: Path, index_version: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    packs_dir = out_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    # Nothing should already be there; refuse to mix old + new packs.
    if any(packs_dir.glob("pack-*.bin.zst")):
        raise SystemExit(
            f"{packs_dir} is non-empty. Move it aside before repacking."
        )

    conn = store_db.connect(db_path, mode="ro")
    try:
        total_docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        print(f"repacking {total_docs:,} docs from {db_path}")

        dctx = zstd.ZstdDecompressor()
        pack_builder = PackBuilder(packs_dir)

        # Two passes so DocRefs know their final pack_sha8 + offset + length.
        # Pass 1: stream records into the pack builder; it finalises packs on
        # its own rhythm (~10 MB each). Pass 2: manifest assembly from the
        # returned per-pack ref lists.
        doc_refs_out: list[DocRef] = []
        processed = 0

        rows = conn.execute(
            "SELECT doc_id, type, title, date, downloaded_at, content_hash "
            "FROM documents ORDER BY doc_id"
        ).fetchall()

        for doc_row in rows:
            doc_id = doc_row["doc_id"]
            chunks_rows = conn.execute(
                "SELECT chunk_id, ord, heading_path, anchor, text "
                "FROM chunks WHERE doc_id = ? ORDER BY ord ASC",
                (doc_id,),
            ).fetchall()

            chunks = []
            for c in chunks_rows:
                text = dctx.decompress(c["text"]).decode("utf-8")
                emb_bytes = _load_embedding(conn, c["chunk_id"])
                chunks.append({
                    "ord": c["ord"],
                    "heading_path": c["heading_path"] or "",
                    "anchor": c["anchor"],
                    "text": text,
                    "embedding_b64": encode_embedding(emb_bytes),
                })

            record = {
                "doc_id": doc_id,
                "type": doc_row["type"],
                "title": doc_row["title"],
                "date": doc_row["date"],
                "downloaded_at": doc_row["downloaded_at"],
                "content_hash": doc_row["content_hash"],
                "chunks": chunks,
            }
            pack_builder.add(doc_id, record)
            processed += 1
            if processed % 5000 == 0:
                print(f"  processed {processed:,}/{total_docs:,}")

        finished_packs = pack_builder.close()
        print(f"  sealed {len(finished_packs)} packs")

    finally:
        conn.close()

    # Build DocRef list from the sealed packs. DocRef carries
    # content_hash + pack_sha8 + offset/length so clients can fetch
    # records directly from the manifest.
    doc_hash_type_title = {
        r["doc_id"]: (r["content_hash"], r["type"], r["title"])
        for r in rows
    }

    for pack_path, sha8, sha256, size, refs in finished_packs:
        for ref in refs:
            ch, typ, title = doc_hash_type_title[ref.doc_id]
            doc_refs_out.append(DocRef(
                doc_id=ref.doc_id,
                content_hash=ch,
                pack_sha8=sha8,
                offset=ref.offset,
                length=ref.length,
                type=typ,
                title=title,
            ))

    # Model info is a placeholder — publish-release overwrites it with the
    # real sha256/size/url when it bundles the model alongside.
    model = ModelInfo(
        id="embeddinggemma-300m-int8-256d",
        sha256="",
        size=0,
        url="embeddinggemma-bundle.tar.zst",
    )

    pack_infos = [
        PackInfo(
            sha8=sha8,
            sha256=sha256,
            size=size,
            url=f"packs/{pack_path.name}",
        )
        for pack_path, sha8, sha256, size, _refs in finished_packs
    ]

    manifest = Manifest(
        index_version=index_version,
        created_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        min_client_version="0.1.0",
        model=model,
        documents=doc_refs_out,
        packs=pack_infos,
    )
    manifest_path = out_dir / "manifest.json"
    save_manifest(manifest, manifest_path)
    print(f"manifest: {manifest_path} ({manifest_path.stat().st_size/1_000_000:.1f} MB)")
    print(f"packs:    {packs_dir} ({len(pack_infos)} files, "
          f"{sum(p.size for p in pack_infos)/1_000_000:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index-version", required=True,
                    help="String like '2026.04.24' — goes into manifest.index_version.")
    args = ap.parse_args()
    if not args.db.exists():
        raise SystemExit(f"no DB at {args.db}")
    repack(args.db, args.out, args.index_version)


if __name__ == "__main__":
    main()
