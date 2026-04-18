"""Manifest + diff round-trip."""
from __future__ import annotations

from pathlib import Path

from ato_mcp.store.manifest import (
    DocRef,
    Manifest,
    ModelInfo,
    PackInfo,
    diff_manifests,
    load_manifest,
    save_manifest,
)


def _m(docs: list[DocRef]) -> Manifest:
    return Manifest(
        index_version="2026.04.18",
        created_at="2026-04-18T00:00:00+00:00",
        model=ModelInfo(id="x", sha256="0" * 64, size=1, url="model"),
        documents=docs,
        packs=[PackInfo(sha8="deadbeef", sha256="0" * 64, size=1, url="p")],
    )


def _doc(doc_id: str, content_hash: str) -> DocRef:
    return DocRef(
        doc_id=doc_id,
        content_hash=content_hash,
        pack_sha8="deadbeef",
        offset=0,
        length=10,
        category="Cases",
        title=doc_id,
    )


def test_roundtrip(tmp_path: Path) -> None:
    m = _m([_doc("a", "h1"), _doc("b", "h2")])
    path = tmp_path / "manifest.json"
    save_manifest(m, path)
    again = load_manifest(path)
    assert again.documents[0].doc_id == "a"
    assert again.packs[0].sha8 == "deadbeef"


def test_diff_added_changed_removed() -> None:
    old = _m([_doc("a", "h1"), _doc("b", "h2")])
    new = _m([_doc("a", "h1"), _doc("b", "h2b"), _doc("c", "h3")])
    added, changed, removed = diff_manifests(old, new)
    assert [r.doc_id for r in added] == ["c"]
    assert [r.doc_id for r in changed] == ["b"]
    assert removed == []
