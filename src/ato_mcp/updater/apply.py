"""Apply a manifest delta to the live SQLite database.

Flow:
1. Load old + new manifests; diff to ``added / changed / removed``.
2. Group changed+added docs by pack_sha8, fetch needed byte ranges.
3. Snapshot ``ato.db`` -> ``backups/ato.db.prev`` (cheap copy; SQLite
   supports this while WAL is quiescent).
4. Mutate ``documents``, ``chunks``, ``chunks_fts``, ``chunks_vec``,
   ``title_fts`` in one transaction.
5. Touch ``meta.last_update_at`` so the live serve process notices the
   change.
6. Write ``installed_manifest.json`` LAST (pointer-after-data).

Rollback = copy the snapshot back and drop the new manifest. ``doctor
--rollback`` uses this.
"""
from __future__ import annotations

import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import zstandard as zstd

from ..indexer.pack import decode_embedding, read_record_from_bytes
from ..store import db as store_db
from ..store.manifest import (
    DocRef,
    Manifest,
    diff_manifests,
    load_manifest,
    save_manifest,
    verify_signature,
)
from ..store.queries import (
    DELETE_DOCUMENT,
    INSERT_CHUNK,
    INSERT_CHUNK_FTS,
    INSERT_DOCUMENT,
    INSERT_TITLE_FTS,
    INSERT_VEC,
)
from ..util import paths
from ..util.log import get_logger
from .fetch import fetch_url, make_client, verify_sha256
from .lock import exclusive_lock

LOGGER = get_logger(__name__)


@dataclass
class UpdateStats:
    added: int
    changed: int
    removed: int
    bytes_downloaded: int


def _db_path_exists() -> bool:
    return paths.db_path().exists()


def _load_local_manifest() -> Manifest | None:
    p = paths.installed_manifest_path()
    if not p.exists():
        return None
    return load_manifest(p)


def fetch_manifest(
    client: httpx.Client,
    manifest_url: str,
    sig_url: str | None,
    *,
    pubkey_path: Path | None,
    dest: Path,
) -> Manifest:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch_url(client, manifest_url, dest)
    if sig_url and pubkey_path and pubkey_path.exists():
        sig_dest = dest.with_suffix(dest.suffix + ".minisig")
        fetch_url(client, sig_url, sig_dest)
        verify_signature(dest, sig_dest, pubkey_path)
    return load_manifest(dest)


def apply_update(
    *,
    manifest_url: str,
    sig_url: str | None = None,
    pubkey_path: Path | None = None,
    client: httpx.Client | None = None,
) -> UpdateStats:
    owns_client = client is None
    if client is None:
        client = make_client()
    try:
        with exclusive_lock():
            return _apply_locked(
                client=client,
                manifest_url=manifest_url,
                sig_url=sig_url,
                pubkey_path=pubkey_path,
            )
    finally:
        if owns_client:
            client.close()


def _apply_locked(
    *,
    client: httpx.Client,
    manifest_url: str,
    sig_url: str | None,
    pubkey_path: Path | None,
) -> UpdateStats:
    staging = paths.staging_dir()
    new_manifest_tmp = staging / "manifest.json.new"
    new_manifest = fetch_manifest(
        client,
        manifest_url,
        sig_url,
        pubkey_path=pubkey_path,
        dest=new_manifest_tmp,
    )

    old_manifest = _load_local_manifest()

    # Ensure model present + matches. If not, download it.
    _ensure_model(client, new_manifest, staging)

    # Initialize the DB if missing.
    if not _db_path_exists():
        conn = store_db.init_db(paths.db_path())
        conn.close()

    added, changed, removed = diff_manifests(old_manifest, new_manifest)
    LOGGER.info("delta: +%d ~%d -%d", len(added), len(changed), len(removed))

    # Snapshot for rollback.
    backup = paths.backups_dir() / "ato.db.prev"
    if paths.db_path().exists():
        shutil.copy2(paths.db_path(), backup)

    # Group affected refs by pack so we can stream one pack at a time —
    # avoids holding all 145k record blobs in RAM for a fresh install.
    pack_to_refs: dict[str, list[DocRef]] = defaultdict(list)
    for ref in added + changed:
        pack_to_refs[ref.pack_sha8].append(ref)
    pack_index = new_manifest.pack_index()
    bytes_downloaded = 0

    conn = store_db.connect(paths.db_path(), mode="rw")
    try:
        conn.execute("BEGIN")
        for doc_id in removed:
            _delete_doc(conn, doc_id)
        for ref in changed:
            _delete_doc(conn, ref.doc_id)

        # Ingest one pack at a time. Download → process → remove.
        processed = 0
        total = len(added) + len(changed)
        for pack_sha8, refs in pack_to_refs.items():
            info = pack_index.get(pack_sha8)
            if info is None:
                raise ValueError(f"manifest missing pack info for {pack_sha8}")
            url = _absolute_url(new_manifest, info.url)
            pack_path = _download_pack(client, url)
            bytes_downloaded += pack_path.stat().st_size
            with open(pack_path, "rb") as fh:
                for ref in refs:
                    fh.seek(ref.offset)
                    blob = fh.read(ref.length)
                    record = read_record_from_bytes(blob)
                    _insert_record(conn, record, ref)
                    processed += 1
                    if processed % 5000 == 0:
                        LOGGER.info("ingest: %d/%d", processed, total)

        store_db.set_meta(conn, "index_version", new_manifest.index_version)
        store_db.set_meta(conn, "embedding_model_id", new_manifest.model.id)
        store_db.set_meta(conn, "last_update_at", datetime.now(timezone.utc).isoformat())
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        conn.close()
        if backup.exists():
            shutil.copy2(backup, paths.db_path())
        raise
    finally:
        conn.close()

    # Manifest-swap-last.
    final = paths.installed_manifest_path()
    save_manifest(new_manifest, final)

    # Success — drop staged pack downloads + temp manifest.
    import contextlib
    for stale in paths.staging_dir().glob("pack-download-*"):
        with contextlib.suppress(OSError):
            stale.unlink()
    with contextlib.suppress(OSError):
        new_manifest_tmp.unlink()

    return UpdateStats(
        added=len(added),
        changed=len(changed),
        removed=len(removed),
        bytes_downloaded=bytes_downloaded,
    )


def _download_pack(client: httpx.Client, url: str) -> Path:
    """Download a pack file to staging and return the local path."""
    from .fetch import fetch_url
    stage_dir = paths.staging_dir()
    asset = url.rsplit("/", 1)[-1] or "pack.bin.zst"
    dest = stage_dir / f"pack-download-{asset}"
    fetch_url(client, url, dest)
    return dest


def _ensure_model(client: httpx.Client, manifest: Manifest, staging: Path) -> None:
    """Fetch the embedding model bundle if missing or mismatched.

    The model is distributed as a single ``.tar.zst`` bundle containing the
    ONNX graph (with any external-data sibling file) plus ``tokenizer.json``.
    We verify the bundle's sha256, extract into the live directory atomically.
    """
    model_info = manifest.model
    model_live = paths.model_path()
    live_dir = model_live.parent
    installed_marker = live_dir / ".model.sha256"
    if (
        model_live.exists()
        and paths.tokenizer_path().exists()
        and installed_marker.exists()
        and installed_marker.read_text().strip() == model_info.sha256
    ):
        return

    tmp = staging / "model-bundle.tar.zst.part"
    fetch_url(client, model_info.url, tmp)
    if model_info.sha256:
        verify_sha256(tmp, model_info.sha256)

    # Extract into a staging subdir, then atomically move into place.
    extract_dir = staging / "model-bundle-extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    import tarfile
    with open(tmp, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(extract_dir, filter="data")

    for src in extract_dir.rglob("*"):
        if src.is_file():
            dst = live_dir / src.name
            shutil.move(str(src), str(dst))
    # Ensure a model.onnx pointer even if the bundle only shipped a different
    # name (onnx-community ships ``model_quantized.onnx``). Prefer a symlink
    # on Unix; fall back to a copy on Windows where unprivileged users can't
    # create symlinks. The ``.onnx_data`` sibling resolves correctly either
    # way because it lives next to the target in ``live/``.
    if not (live_dir / "model.onnx").exists():
        for candidate in ("model_quantized.onnx", "model_fp16.onnx", "model.onnx"):
            path = live_dir / candidate
            if path.exists():
                try:
                    (live_dir / "model.onnx").symlink_to(candidate)
                except OSError:
                    shutil.copy2(path, live_dir / "model.onnx")
                break
    installed_marker.write_text(model_info.sha256 or "")
    tmp.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def _absolute_url(manifest: Manifest, rel: str) -> str:
    """Resolve a pack URL. Absolute URLs are returned as-is."""
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    base = paths.releases_url().rstrip("/")
    return f"{base}/{rel.lstrip('/')}"


def _delete_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    # title_fts (content-less): delete by rowid lookup via the doc_id column.
    rows = conn.execute(
        "SELECT rowid FROM title_fts WHERE doc_id = ?",
        (doc_id,),
    ).fetchall()
    for row in rows:
        conn.execute(
            "INSERT INTO title_fts(title_fts, rowid, doc_id, title, headings) "
            "SELECT 'delete', rowid, doc_id, title, headings "
            "FROM title_fts WHERE rowid = ?",
            (row["rowid"],),
        )
    # chunks_fts (content-less here too since we store text directly): remove
    # by matching chunk rowids for the doc.
    chunk_rows = conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    for row in chunk_rows:
        conn.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path) "
            "SELECT 'delete', rowid, text, heading_path FROM chunks_fts WHERE rowid = ?",
            (row["chunk_id"],),
        )
        conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (row["chunk_id"],))
    # ON DELETE CASCADE removes chunks; we explicit-delete to keep IDs synced.
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute(DELETE_DOCUMENT, (doc_id,))


def _insert_record(conn: sqlite3.Connection, record: dict, ref: DocRef) -> None:
    conn.execute(
        INSERT_DOCUMENT,
        (
            record["doc_id"],
            record.get("type") or record.get("category") or "",
            record["title"],
            record.get("date") or record.get("first_published_date"),
            record["downloaded_at"],
            record["content_hash"],
            ref.pack_sha8,
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
        conn.execute(INSERT_VEC, (chunk_rowid, decode_embedding(c["embedding_b64"])))


def rollback() -> None:
    """Restore the previous DB snapshot, if any."""
    backup = paths.backups_dir() / "ato.db.prev"
    if not backup.exists():
        raise FileNotFoundError("no backup found at backups/ato.db.prev")
    shutil.copy2(backup, paths.db_path())
    LOGGER.info("rolled back to %s", backup)
