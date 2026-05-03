"""End-to-end smoke test of the indexer against a sample of real ato_pages/.

Skipped when ``ato_pages/index.jsonl`` or the embedding model is absent.
Only exercises the non-embedding path: extract, chunk, manifest, pack, and
DB inserts via a monkeypatched ``EmbeddingModel``. Gives us one signal that
the pipeline wires together against real HTML.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

ATO_PAGES = Path(os.environ.get("ATO_MCP_TEST_PAGES_DIR", "ato_pages"))


def test_embedding_input_includes_heading_between_title_and_text() -> None:
    from ato_mcp.indexer.build import _embedding_input

    assert (
        _embedding_input("Example title", "Section 8-1 > Reasons", "Body text")
        == "Example title\nSection 8-1 > Reasons\nBody text"
    )
    assert _embedding_input("", "", "Body text") == "Body text"


def test_length_bucketed_encoder_reports_batch_telemetry(monkeypatch) -> None:
    import ato_mcp.indexer.build as build_module
    from ato_mcp.embed.model import EncodedBatch
    from ato_mcp.store import db as store_db

    token_counts = {"a": 10, "b": 20, "c": 30, "d": 200}
    monkeypatch.setattr(
        build_module.chunk_mod,
        "approx_tokens",
        lambda text: token_counts[text],
    )

    class StubModel:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def encode(self, texts, *, is_query, batch_size: int):
            batch = list(texts)
            self.batches.append(batch)
            return EncodedBatch(
                vectors_int8=np.zeros((len(batch), store_db.EMBEDDING_DIM), dtype=np.int8),
                tokens_seen=sum(token_counts[text] for text in batch),
            )

    model = StubModel()
    encoded = build_module._encode_length_bucketed(
        model,
        ["d", "a", "c", "b"],
        batch_size=4,
        max_batch_tokens=100,
    )

    assert model.batches == [["a", "b"], ["c"], ["d"]]
    assert encoded.vectors_int8.shape == (4, store_db.EMBEDDING_DIM)
    assert encoded.tokens_seen == 260
    assert encoded.encode_calls == 3
    assert encoded.max_batch_size == 2
    assert encoded.max_padded_tokens == 216
    assert encoded.approx_padded_tokens == 334


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

    captured_texts: list[str] = []

    class StubModel:
        def __init__(self, *a, **kw) -> None:
            pass

        def encode(self, texts, *, is_query, batch_size: int = 16):
            from ato_mcp.embed.model import EncodedBatch
            texts_list = list(texts)
            captured_texts.extend(texts_list)
            n = len(texts_list)
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

        # W2.1: the embedder must receive title + heading_path + text, not
        # bare chunk text. We inspect at least one captured input string and
        # check that it carries the document title (which is also stored in
        # documents.title) somewhere in its prefix.
        if captured_texts:
            titles = {row[0] for row in conn.execute("SELECT title FROM documents").fetchall()}
            hits = sum(
                1
                for txt in captured_texts
                for title in titles
                if title and txt.startswith(title)
            )
            assert hits > 0, (
                "expected at least one embedder input to start with a stored document title; "
                f"first 3 captured: {captured_texts[:3]!r}"
            )
            headed_rows = conn.execute(
                """
                SELECT d.title, c.heading_path, c.text
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.heading_path <> ''
                LIMIT 20
                """
            ).fetchall()
            if headed_rows:
                expected = {
                    build_module._embedding_input(row["title"], row["heading_path"], row["text"])
                    for row in headed_rows
                }
                assert expected.intersection(captured_texts), (
                    "expected at least one embedder input to exactly preserve "
                    "title\\nheading_path\\ntext; "
                    f"expected sample: {list(expected)[:1]!r}, first 3 captured: {captured_texts[:3]!r}"
                )
    finally:
        conn.close()
