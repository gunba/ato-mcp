"""End-to-end smoke test of the indexer against a sample of real ato_pages/.

Skipped when ``ato_pages/index.jsonl`` or the embedding model is absent.
Only exercises the non-embedding path: extract, chunk, manifest, pack, and
DB inserts via a monkeypatched ``EmbeddingModel``. Gives us one signal that
the pipeline wires together against real HTML.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ATO_PAGES = Path("/home/jordan/Desktop/Projects/ato_pages")


@pytest.fixture()
def sample_pages_dir(tmp_path: Path) -> Path:
    if not (ATO_PAGES / "index.jsonl").exists():
        pytest.skip("ato_pages/ not present")

    sample_dir = tmp_path / "ato_pages_sample"
    sample_dir.mkdir()
    index_lines: list[str] = []
    count = 0
    with open(ATO_PAGES / "index.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("status") != "success":
                continue
            payload_rel = rec.get("payload_path")
            if not payload_rel:
                continue
            src = ATO_PAGES / payload_rel
            if not src.exists():
                continue
            dest = sample_dir / payload_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            index_lines.append(json.dumps(rec))
            count += 1
            if count >= 5:
                break
    (sample_dir / "index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return sample_dir


def test_build_small_index(sample_pages_dir: Path, tmp_path: Path, monkeypatch) -> None:
    from ato_mcp.indexer import build as build_mod  # noqa: F401 — keep module imported
    import ato_mcp.indexer.build as build_module
    from ato_mcp.store import db as store_db

    class StubModel:
        def __init__(self, *a, **kw) -> None:
            pass

        def encode(self, texts, *, is_query, batch_size: int = 16):
            from ato_mcp.embed.model import EncodedBatch
            n = len(list(texts))
            return EncodedBatch(
                vectors_int8=np.zeros((n, store_db.EMBEDDING_DIM), dtype=np.int8),
                tokens_seen=0,
            )

    monkeypatch.setattr(build_module, "EmbeddingModel", StubModel)

    out_dir = tmp_path / "release"
    db_path = out_dir / "ato.db"
    args = build_module.BuildArgs(
        pages_dir=sample_pages_dir,
        out_dir=out_dir,
        db_path=db_path,
        model_id="stub",
        model_path=Path("/dev/null"),
        tokenizer_path=Path("/dev/null"),
        model_url="stub",
        model_sha256="0" * 64,
        model_size=0,
    )
    manifest = build_module.build(args)

    assert (out_dir / "manifest.json").exists()
    assert len(manifest.documents) >= 1
    assert len(manifest.packs) >= 1
    assert db_path.exists()

    conn = store_db.connect(db_path, mode="ro")
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        assert row["n"] == len(manifest.documents)
        row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        assert row["n"] >= 0
    finally:
        conn.close()
