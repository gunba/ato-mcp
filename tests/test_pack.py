"""Pack writer + reader round-trip."""
from __future__ import annotations

from pathlib import Path

from ato_mcp.indexer.pack import (
    PackBuilder,
    PackWriter,
    encode_embedding,
    read_record,
    read_record_from_bytes,
)


def _record(doc_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "href": f"/law/view/document?docid={doc_id}",
        "category": "Cases",
        "doc_type": "JUD",
        "human_code": None,
        "title": f"Document {doc_id}",
        "human_title": None,
        "pub_date": None,
        "first_published_date": None,
        "effective_date": None,
        "status": "active",
        "has_content": True,
        "downloaded_at": "2026-04-18T00:00:00Z",
        "content_hash": "sha256:" + "0" * 64,
        "chunks": [
            {
                "ord": 0,
                "heading_path": "Root › Section 1",
                "anchor": "s1",
                "text": "Hello world chunk for " + doc_id,
                "embedding_b64": encode_embedding(b"\x00" * 256),
            }
        ],
    }


def test_pack_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "pack.bin.zst"
    with PackWriter(path=path) as writer:
        writer.add("a", _record("a"))
        writer.add("b", _record("b"))
        writer.add("c", _record("c"))
        refs = list(writer.refs)
    for r in refs:
        record = read_record(path, r.offset, r.length)
        assert record["doc_id"] == r.doc_id


def test_pack_builder_splits_on_size(tmp_path: Path) -> None:
    builder = PackBuilder(out_dir=tmp_path, target_size=512)
    for i in range(20):
        builder.add(f"doc-{i}", _record(f"doc-{i}"))
    packs = builder.close()
    assert len(packs) > 1
    # every doc resolvable from its pack + range
    for path, _sha8, _sha256, _size, refs in packs:
        for r in refs:
            rec = read_record(path, r.offset, r.length)
            assert rec["doc_id"] == r.doc_id


def test_read_record_from_bytes_matches_disk(tmp_path: Path) -> None:
    path = tmp_path / "pack.bin.zst"
    with PackWriter(path=path) as writer:
        writer.add("only", _record("only"))
        refs = list(writer.refs)
    r = refs[0]
    disk = read_record(path, r.offset, r.length)
    with open(path, "rb") as fh:
        fh.seek(r.offset)
        blob = fh.read(r.length)
    assert read_record_from_bytes(blob) == disk
