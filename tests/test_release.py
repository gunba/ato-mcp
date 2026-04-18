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
                        url="model/embeddinggemma-300m-<sha>.onnx.zst"),
        documents=[DocRef(doc_id="d", content_hash="sha256:abc", pack_sha8="deadbeef",
                          offset=0, length=1, category="c", title="T")],
        packs=[PackInfo(sha8="deadbeef", sha256="0" * 64, size=1,
                       url="packs/pack-deadbeef.bin.zst")],
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    rewrite_manifest_urls(path, repo="sensis/ato-mcp", tag="index-2026.04.18")

    out = load_manifest(path)
    assert out.model.url == (
        "https://github.com/sensis/ato-mcp/releases/download/"
        "index-2026.04.18/embeddinggemma-300m-<sha>.onnx.zst"
    )
    assert out.packs[0].url == (
        "https://github.com/sensis/ato-mcp/releases/download/"
        "index-2026.04.18/pack-deadbeef.bin.zst"
    )
