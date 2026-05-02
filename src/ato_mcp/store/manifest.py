"""Manifest schema + signature verification (minisign).

A manifest enumerates every document in a release and the pack-file byte range
it lives in. Clients diff content_hash to produce the delta work list.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, Field

MANIFEST_SCHEMA_VERSION = 1


class ModelInfo(BaseModel):
    id: str
    sha256: str
    size: int
    url: str


class DocRef(BaseModel):
    # The five fields the updater actually needs for diffing + fetching.
    doc_id: str
    content_hash: str
    pack_sha8: str
    offset: int
    length: int
    # Client-unused metadata. Kept on the model for build-side debugging
    # but excluded from serialization so produced manifests stay lean.
    type: str = Field(default="", exclude=True)
    title: str = Field(default="", exclude=True)


class PackInfo(BaseModel):
    sha8: str
    sha256: str
    size: int
    url: str


class Manifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    index_version: str
    created_at: str
    min_client_version: str = "0.1.0"
    model: ModelInfo
    documents: list[DocRef] = Field(default_factory=list)
    packs: list[PackInfo] = Field(default_factory=list)

    def doc_index(self) -> dict[str, DocRef]:
        return {d.doc_id: d for d in self.documents}

    def pack_index(self) -> dict[str, PackInfo]:
        return {p.sha8: p for p in self.packs}

    def to_bytes(self) -> bytes:
        return orjson.dumps(self.model_dump(), option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(Path(path).read_bytes())


def save_manifest(manifest: Manifest, path: Path) -> None:
    Path(path).write_bytes(manifest.to_bytes())


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_signature(manifest_path: Path, sig_path: Path, pubkey_path: Path) -> bool:
    """Verify the manifest.minisig signature.

    Returns True on success. Signature verification uses the ``minisign`` CLI
    so it exercises the same verifier maintainers use outside Python.
    """
    # [SL-07] Use the minisign CLI via subprocess (not a Python library) so the offline verifier path is exercised — signing-key hygiene problems surface early.
    cli = shutil.which("minisign")
    if cli is None:
        raise RuntimeError(
            "signature verification requested but the `minisign` CLI is not installed"
        )

    result = subprocess.run(
        [cli, "-V", "-m", str(manifest_path), "-x", str(sig_path), "-p", str(pubkey_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"manifest signature verification failed: {detail}")
    return True


def diff_manifests(
    old: Manifest | None, new: Manifest
) -> tuple[list[DocRef], list[DocRef], list[str]]:
    """Return (added, changed, removed_doc_ids)."""
    # [SL-08] content_hash is the only diff signal — a chunk-only edit that doesn't bump content_hash is invisible to the updater (intentional simplification).
    old_ix: dict[str, DocRef] = old.doc_index() if old else {}
    new_ix = new.doc_index()
    added: list[DocRef] = []
    changed: list[DocRef] = []
    for doc_id, ref in new_ix.items():
        if doc_id not in old_ix:
            added.append(ref)
        elif old_ix[doc_id].content_hash != ref.content_hash:
            changed.append(ref)
    removed = [doc_id for doc_id in old_ix if doc_id not in new_ix]
    return added, changed, removed


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON for hashing/signing."""
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
