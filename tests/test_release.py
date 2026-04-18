"""Release helpers — URL rewrite only (gh CLI calls not exercised)."""
from __future__ import annotations

from pathlib import Path

from ato_mcp.indexer.release import rewrite_manifest_urls
from ato_mcp.store.manifest import (
    DocRef,
    Manifest,
    ModelInfo,
    PackInfo,
    load_manifest,
    save_manifest,
)


def test_rewrite_manifest_urls_flattens_asset_names(tmp_path: Path) -> None:
    manifest = Manifest(
        index_version="2026.04.18",
        created_at="2026-04-18T00:00:00+00:00",
        model=ModelInfo(id="m", sha256="0" * 64, size=1,
                        url="model/placeholder.onnx.zst"),
        documents=[DocRef(doc_id="d", content_hash="sha256:abc", pack_sha8="deadbeef",
                          offset=0, length=1, category="c", title="T")],
        packs=[PackInfo(sha8="deadbeef", sha256="0" * 64, size=1,
                       url="packs/pack-deadbeef.bin.zst")],
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    rewrite_manifest_urls(path, repo="gunba/ato-mcp", tag="index-2026.04.18")

    out = load_manifest(path)
    # Model URL is managed by publish() when bundling, not by this helper.
    assert out.model.url == "model/placeholder.onnx.zst"
    assert out.packs[0].url == (
        "https://github.com/gunba/ato-mcp/releases/download/"
        "index-2026.04.18/pack-deadbeef.bin.zst"
    )


def test_bundle_model_round_trip(tmp_path: Path) -> None:
    """bundle_model + tarfile extract restores the original files byte-for-byte."""
    import tarfile
    import zstandard as zstd

    from ato_mcp.indexer.release import bundle_model

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_quantized.onnx").write_bytes(b"ONNX\x00" * 50)
    (model_dir / "model_quantized.onnx_data").write_bytes(b"\x01\x02\x03" * 200)
    (model_dir / "tokenizer.json").write_text('{"tok":"json"}')

    bundle = tmp_path / "bundle.tar.zst"
    sha256, size = bundle_model(model_dir, bundle)
    assert size == bundle.stat().st_size
    assert len(sha256) == 64

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with open(bundle, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(extract_dir, filter="data")

    for name in ("model_quantized.onnx", "model_quantized.onnx_data", "tokenizer.json"):
        assert (extract_dir / name).read_bytes() == (model_dir / name).read_bytes()
