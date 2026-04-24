"""Maintainer entry point: read ato_pages/, write ato.db + packs + manifest.

Orchestrates:
1. Enumerate ``index.jsonl`` records from a scraped ``ato_pages/`` directory.
2. Load each payload HTML.
3. Extract markdown, metadata, status.
4. Chunk.
5. Embed chunks (int8 via EmbeddingGemma ONNX).
6. Write document into a sqlite ``ato.db`` and a release pack.
7. Emit a ``manifest.json`` suitable for clients.

Supports ``incremental`` rebuilds: if a prior manifest is supplied and a
document's ``content_hash`` is unchanged, the existing pack slot is reused.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import zstandard as zstd

from ..embed.model import EmbeddingModel, vec_to_bytes
from ..store import db as store_db
from ..store.manifest import DocRef, Manifest, ModelInfo, PackInfo, load_manifest
from ..store.queries import (
    INSERT_CHUNK,
    INSERT_CHUNK_FTS,
    INSERT_DOCUMENT,
    INSERT_TITLE_FTS,
    INSERT_VEC,
)
from ..util.log import get_logger
from . import chunk as chunk_mod
from . import extract as extract_mod
from . import metadata as meta_mod
from . import rules as rules_mod
from .pack import TRAILER_MAGIC, PackBuilder, PackedDocRef, encode_embedding

LOGGER = get_logger(__name__)

BASE_URL = "https://www.ato.gov.au"

# Commit the in-progress transaction every N newly-processed docs so a kill
# mid-run only loses at most this many docs of work. Packs sealed at each
# checkpoint become immutable on disk and are picked up on restart.
CHECKPOINT_EVERY = 1000


@dataclass
class BuildArgs:
    pages_dir: Path
    out_dir: Path  # receives manifest.json + packs/
    db_path: Path  # new ato.db
    model_id: str
    model_path: Path
    tokenizer_path: Path
    model_url: str | None = None
    model_sha256: str | None = None
    model_size: int | None = None
    previous_manifest: Path | None = None
    limit: int | None = None  # optional cap for testing
    encode_batch_size: int = 64
    providers: tuple[str, ...] | None = None  # ORT execution providers override


def build(args: BuildArgs) -> Manifest:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packs_dir = args.out_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale tmp pack files left by a prior crashed run.
    for stale in packs_dir.glob(".pack-writing-*.bin.zst.tmp"):
        stale.unlink()

    prev_manifest: Manifest | None = None
    prev_docs: dict[str, DocRef] = {}
    prev_pack_info: dict[str, PackInfo] = {}
    prev_packs_dir: Path | None = None
    if args.previous_manifest and args.previous_manifest.exists():
        prev_manifest = load_manifest(args.previous_manifest)
        prev_docs = prev_manifest.doc_index()
        prev_pack_info = prev_manifest.pack_index()
        prev_packs_dir = Path(args.previous_manifest).parent / "packs"
        LOGGER.info("Loaded previous manifest with %d documents", len(prev_docs))

    conn = store_db.init_db(args.db_path)
    store_db.set_meta(conn, "embedding_model_id", args.model_id)
    store_db.set_meta(conn, "index_version", _today_version())

    # Resume support: any doc_id already in documents with a sealed pack
    # (pack_sha8 != PENDING) is skipped this run. The prior commit landed its
    # rows + pack bytes atomically, so the state is safe to keep.
    resume_done = _load_resume_state(conn)
    if resume_done:
        LOGGER.info("Resuming: %d documents already committed; will skip them", len(resume_done))

    index_records = _iter_index(args.pages_dir)
    if args.limit is not None:
        index_records = _take(index_records, args.limit)

    model = EmbeddingModel(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        providers=args.providers,
    )
    pack_builder = PackBuilder(out_dir=packs_dir)

    doc_refs: list[DocRef] = []
    reused_pack_shas: set[str] = set()
    processed = 0
    reused = 0
    since_checkpoint = 0
    t0 = time.monotonic()

    conn.execute("BEGIN")
    try:
        for rec in index_records:
            canonical_id = rec["canonical_id"]
            href = rec.get("href") or canonical_id
            doc_id = meta_mod.doc_id_for(canonical_id)
            if doc_id in resume_done:
                continue
            category = meta_mod.category_from_path(rec.get("payload_path"))
            status = rec.get("status")
            has_content = status == "success"

            markdown = ""
            headings: list[str] = []
            anchors: list[tuple[str, str]] = []
            if has_content and rec.get("payload_path"):
                payload_path = args.pages_dir / rec["payload_path"]
                if payload_path.exists():
                    html = payload_path.read_text(encoding="utf-8", errors="replace")
                    extracted = extract_mod.extract(html)
                    markdown = extracted.markdown
                    headings = extracted.headings
                    anchors = extracted.anchors
                    title = extracted.title
                else:
                    has_content = False
                    title = None
            else:
                title = None

            # A payload with status=success but no extractable content (EPA
            # empty shells, broken pages) should still surface by title, but
            # shouldn't generate empty chunks.
            if has_content and not markdown.strip():
                has_content = False

            if not title:
                title = (rec.get("title") or canonical_id).strip() or canonical_id

            prefix, _ = meta_mod.parse_docid(canonical_id)
            pub_date = meta_mod.extract_pub_date(markdown) if markdown else None
            derived = rules_mod.derive_metadata(rules_mod.RuleInputs(
                doc_id=doc_id,
                title=title,
                headings=tuple(headings),
                body_head=markdown[:3000] if markdown else "",
                category=category,
                pub_date=pub_date,
            ))
            derived_title = derived.title or title
            derived_date = derived.date
            downloaded_at = rec.get("downloaded_at") or datetime.now(timezone.utc).isoformat()

            meta_fields = {
                "title": derived_title,
                "type": category,
                "date": derived_date,
            }
            ch = meta_mod.content_hash(markdown, meta_fields)

            prev_ref = prev_docs.get(doc_id)
            if prev_ref and prev_ref.content_hash == ch and prev_ref.pack_sha8 in prev_pack_info:
                # Reuse prior pack slot; skip re-embedding.
                doc_refs.append(prev_ref)
                reused_pack_shas.add(prev_ref.pack_sha8)
                reused += 1
                processed += 1
                since_checkpoint += 1
                _insert_from_previous(conn, rec, prev_ref, args.previous_manifest, prev_pack_info)
                if since_checkpoint >= CHECKPOINT_EVERY:
                    _checkpoint(conn, pack_builder, doc_refs)
                    since_checkpoint = 0
                continue

            chunks = (
                chunk_mod.chunk_markdown(markdown, root_title=title) if has_content and markdown else []
            )
            if chunks:
                texts = [c.text for c in chunks]
                encoded = model.encode(texts, is_query=False, batch_size=args.encode_batch_size)
                vectors_i8 = encoded.vectors_int8
            else:
                vectors_i8 = np.empty((0, store_db.EMBEDDING_DIM), dtype=np.int8)

            # Insert document row — v5 columns only.
            conn.execute(
                INSERT_DOCUMENT,
                (
                    doc_id, category, derived_title, derived_date,
                    downloaded_at, ch, "PENDING",  # pack_sha8 backfilled below
                ),
            )
            conn.execute(
                INSERT_TITLE_FTS,
                (doc_id, derived_title, " ".join(headings)),
            )
            for i, c in enumerate(chunks):
                compressed_text = zstd.ZstdCompressor(level=3).compress(c.text.encode("utf-8"))
                cur = conn.execute(
                    INSERT_CHUNK,
                    (doc_id, c.ord, c.heading_path, c.anchor, compressed_text),
                )
                chunk_rowid = cur.lastrowid
                conn.execute(INSERT_CHUNK_FTS, (chunk_rowid, c.text, c.heading_path))
                conn.execute(INSERT_VEC, (chunk_rowid, vec_to_bytes(vectors_i8[i])))

            # Build a pack record. pack_sha8 + offset/length filled after pack close.
            record = {
                "doc_id": doc_id,
                "type": category,
                "title": derived_title,
                "date": derived_date,
                "downloaded_at": downloaded_at,
                "content_hash": ch,
                "anchors": anchors,
                "chunks": [
                    {
                        "ord": c.ord,
                        "heading_path": c.heading_path,
                        "anchor": c.anchor,
                        "text": c.text,
                        "embedding_b64": encode_embedding(vec_to_bytes(vectors_i8[i])),
                    }
                    for i, c in enumerate(chunks)
                ],
            }
            pack_builder.add(doc_id, record)
            doc_refs.append(
                DocRef(
                    doc_id=doc_id,
                    content_hash=ch,
                    pack_sha8="PENDING",
                    offset=0,
                    length=0,
                    type=category,
                    title=derived_title,
                    has_content=has_content,
                )
            )
            processed += 1
            since_checkpoint += 1
            if processed % 500 == 0:
                LOGGER.info("processed=%d reused=%d", processed, reused)
            if since_checkpoint >= CHECKPOINT_EVERY:
                _checkpoint(conn, pack_builder, doc_refs)
                since_checkpoint = 0

        # Final checkpoint seals the last pack and commits leftover docs.
        _checkpoint(conn, pack_builder, doc_refs)
    except Exception:
        conn.execute("ROLLBACK")
        raise

    # Build the manifest from DB state so resumed runs pick up work committed
    # in prior sessions as well as this one.
    pack_search_dirs = [packs_dir]
    if prev_packs_dir is not None:
        pack_search_dirs.append(prev_packs_dir)
    doc_refs_final = _load_doc_refs_from_db(conn, pack_search_dirs)
    new_packs = _scan_packs_dir(packs_dir)
    have = {p.sha8 for p in new_packs}
    # Any pack referenced by a doc_ref but not in our new packs dir must be
    # a reused pack — pull its PackInfo from the previous manifest.
    referenced = {r.pack_sha8 for r in doc_refs_final}
    for sha8 in referenced - have:
        if sha8 in prev_pack_info:
            new_packs.append(prev_pack_info[sha8])
        else:
            raise RuntimeError(
                f"doc references pack {sha8} but it's neither in {packs_dir} "
                f"nor in the previous manifest"
            )

    manifest = Manifest(
        index_version=_today_version(),
        created_at=datetime.now(timezone.utc).isoformat(),
        model=ModelInfo(
            id=args.model_id,
            sha256=args.model_sha256 or "",
            size=args.model_size or 0,
            url=args.model_url or f"model/{args.model_id}.onnx.zst",
        ),
        documents=doc_refs_final,
        packs=new_packs,
    )
    (args.out_dir / "manifest.json").write_bytes(manifest.to_bytes())
    dt = time.monotonic() - t0
    LOGGER.info(
        "Indexed %d docs this session (%d reused); manifest has %d total in %.1fs",
        processed, reused, len(doc_refs_final), dt,
    )
    return manifest


def _backfill_pack_slots(
    doc_refs: list[DocRef],
    packs_written: list[tuple[Path, str, str, int, list[PackedDocRef]]],
    conn,
) -> None:
    # Build a lookup of (doc_id -> pack_sha8, offset, length)
    slot: dict[str, tuple[str, int, int]] = {}
    for _path, sha8, _sha256, _size, refs in packs_written:
        for r in refs:
            slot[r.doc_id] = (sha8, r.offset, r.length)
    for ref in doc_refs:
        if ref.pack_sha8 != "PENDING":
            continue  # reused from previous manifest
        found = slot.get(ref.doc_id)
        if not found:
            raise RuntimeError(f"pack slot not found for doc {ref.doc_id}")
        ref.pack_sha8, ref.offset, ref.length = found
        conn.execute(
            "UPDATE documents SET pack_sha8 = ? WHERE doc_id = ?",
            (ref.pack_sha8, ref.doc_id),
        )


def _insert_from_previous(
    conn,
    rec: dict,
    prev_ref: DocRef,
    prev_manifest_path: Path | None,
    prev_packs: dict[str, PackInfo],
) -> None:
    """When reusing a document, we still need its rows in the new DB.

    We read the document record out of the previous pack file (next to the
    previous manifest) and replay the inserts.
    """
    from .pack import read_record

    if prev_manifest_path is None:
        raise RuntimeError("cannot reuse document without previous manifest path")
    prev_root = Path(prev_manifest_path).parent
    pack_path = prev_root / "packs" / f"pack-{prev_ref.pack_sha8}.bin.zst"
    if not pack_path.exists():
        # Fallback: url relative to manifest root
        info = prev_packs.get(prev_ref.pack_sha8)
        if info:
            pack_path = prev_root / info.url
    record = read_record(pack_path, prev_ref.offset, prev_ref.length)

    conn.execute(
        INSERT_DOCUMENT,
        (
            record["doc_id"],
            record.get("type") or record.get("category") or "",
            record["title"],
            record.get("date") or record.get("first_published_date"),
            record["downloaded_at"],
            record["content_hash"],
            prev_ref.pack_sha8,
        ),
    )
    conn.execute(
        INSERT_TITLE_FTS,
        (record["doc_id"], record["title"], ""),
    )
    for c in record.get("chunks", []):
        compressed_text = zstd.ZstdCompressor(level=3).compress(c["text"].encode("utf-8"))
        cur = conn.execute(
            INSERT_CHUNK,
            (record["doc_id"], c["ord"], c.get("heading_path"), c.get("anchor"), compressed_text),
        )
        chunk_rowid = cur.lastrowid
        conn.execute(INSERT_CHUNK_FTS, (chunk_rowid, c["text"], c.get("heading_path") or ""))
        from .pack import decode_embedding as _dec
        conn.execute(INSERT_VEC, (chunk_rowid, _dec(c["embedding_b64"])))


def _iter_index(pages_dir: Path) -> Iterator[dict]:
    index_path = pages_dir / "index.jsonl"
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _take(it: Iterator[dict], n: int) -> Iterator[dict]:
    count = 0
    for rec in it:
        if count >= n:
            return
        count += 1
        yield rec


def _today_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y.%m.%d")


def _checkpoint(conn, pack_builder: PackBuilder, doc_refs: list[DocRef]) -> None:
    """Seal the current pack, backfill pack_sha8 for this session's new docs,
    and commit + reopen the transaction. Safe to call when nothing is pending.
    """
    pack_builder.flush()
    sealed = pack_builder.finalized_packs
    pending = [r for r in doc_refs if r.pack_sha8 == "PENDING"]
    if pending:
        _backfill_pack_slots(pending, sealed, conn)
    store_db.set_meta(conn, "last_update_at", datetime.now(timezone.utc).isoformat())
    conn.execute("COMMIT")
    conn.execute("BEGIN")


def _load_resume_state(conn) -> set[str]:
    """Doc IDs already sealed in a prior checkpoint of this DB.

    A row is resumable only if its pack has been sealed (pack_sha8 != PENDING).
    PENDING rows live in an uncommitted transaction anyway; a crash rolled
    them back. Returns the set of doc_ids to skip on this session.
    """
    try:
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE pack_sha8 IS NOT NULL AND pack_sha8 != 'PENDING'"
        ).fetchall()
    except Exception:
        return set()
    return {r["doc_id"] for r in rows}


def _load_doc_refs_from_db(conn, pack_search_dirs: list[Path]) -> list[DocRef]:
    """Reconstruct the manifest's document list from committed DB state."""
    rows = conn.execute(
        "SELECT doc_id, content_hash, pack_sha8, type, title "
        "FROM documents WHERE pack_sha8 != 'PENDING' ORDER BY doc_id"
    ).fetchall()
    refs: list[DocRef] = [
        DocRef(
            doc_id=row["doc_id"],
            content_hash=row["content_hash"],
            pack_sha8=row["pack_sha8"],
            offset=0,
            length=0,
            type=row["type"] or "",
            title=row["title"],
            has_content=True,
        )
        for row in rows
    ]
    # offset/length aren't stored in the DB; read them from each pack's trailer.
    _populate_offsets_from_packs(refs, pack_search_dirs)
    return refs


def _populate_offsets_from_packs(refs: list[DocRef], pack_search_dirs: list[Path]) -> None:
    """Fill offset/length on doc_refs by reading each referenced pack's trailer.

    Searches the supplied directories in order so incremental builds can find
    reused packs in the previous release directory.
    """
    import orjson

    by_pack: dict[str, list[DocRef]] = {}
    for ref in refs:
        by_pack.setdefault(ref.pack_sha8, []).append(ref)

    trailer_struct = struct.Struct("<6sIQI")
    for sha8, group in by_pack.items():
        pack_path: Path | None = None
        for search_dir in pack_search_dirs:
            candidate = search_dir / f"pack-{sha8}.bin.zst"
            if candidate.exists():
                pack_path = candidate
                break
        if pack_path is None:
            raise RuntimeError(
                f"pack {sha8} not found in any of {pack_search_dirs}"
            )
        with open(pack_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(size - trailer_struct.size)
            magic, _count, index_offset, index_len = trailer_struct.unpack(fh.read(trailer_struct.size))
            if magic != TRAILER_MAGIC:
                raise RuntimeError(f"pack {pack_path} has bad trailer magic")
            fh.seek(index_offset)
            index_blob = fh.read(index_len)
        entries = orjson.loads(zstd.ZstdDecompressor().decompress(index_blob))
        lut = {e["doc_id"]: (e["offset"], e["length"]) for e in entries}
        for ref in group:
            hit = lut.get(ref.doc_id)
            if hit is None:
                raise RuntimeError(f"doc {ref.doc_id} missing from pack {sha8}")
            ref.offset, ref.length = hit


def _scan_packs_dir(packs_dir: Path) -> list[PackInfo]:
    """List every sealed pack present in the release packs dir."""
    out: list[PackInfo] = []
    for p in sorted(packs_dir.glob("pack-*.bin.zst")):
        sha8 = p.stem.split("-", 1)[1].split(".", 1)[0]
        out.append(
            PackInfo(
                sha8=sha8,
                sha256="",  # filled in by the release step when needed
                size=p.stat().st_size,
                url=f"packs/pack-{sha8}.bin.zst",
            )
        )
    return out
