# Maintainer Runbook

End users install the Rust binary and never run the Python maintainer
commands. This file is for corpus publication only.

## What Runs Where

- Rust binary: end-user CLI, updater, MCP server, search/fetch tools.
- Python maintainer package: scrape, extract, chunk, GPU-backed embedding
  build, pack generation, GitHub release upload.
- GitHub Actions: cheap CI, cross-platform Rust binary release assets, and
  an optional self-hosted GPU release workflow.

The corpus build must use GPU-backed embeddings. If there is no suitable
GitHub GPU runner, keep the build local. Do not silently fall back to a CPU
or keyword-only release build.

## Weekly Local Release

```bash
cd /path/to/ato-mcp

python -m venv .venv
.venv/bin/pip install -e '.[dev]'
# Swap CPU onnxruntime for the GPU build (end-users never load this Python
# package — they ship with the Rust binary — so the heavier wheel is
# isolated to the maintainer venv). Both packages provide the same
# ``onnxruntime`` module name, so the CPU one must come off first or it
# clobbers the GPU build.
.venv/bin/pip uninstall -y onnxruntime
.venv/bin/pip install -e '.[gpu]'

ATO_MCP_MODE=catch_up \
ATO_MCP_REPO_DIR="$PWD" \
ATO_MCP_PAGES_DIR="/path/to/ato_pages" \
ATO_MCP_MODEL_DIR="$PWD/models/embeddinggemma" \
ATO_MCP_GH_REPO=gunba/ato-mcp \
scripts/maintainer-sync.sh
```

`scripts/maintainer-sync.sh` will:

1. Refresh `ato_pages` in the requested mode.
2. Build `release/<tag>/ato.db`, packs, and `manifest.json`.
3. Write the pinned Hugging Face EmbeddingGemma source into the manifest,
   unless `ATO_MCP_MODEL_URL` points at an approved mirror.
4. Upload the corpus assets with `.venv/bin/ato-mcp release`.
5. Mark the release latest.

The script requires `nvidia-smi`/CUDA to be available through the local
Python ONNX Runtime install. If CUDA is unavailable, fix the environment
instead of publishing a degraded corpus.

Manifest signing with `--sign-key` requires the `minisign` CLI on `PATH`.

## Optional GPU Workflow

`.github/workflows/corpus-release-gpu.yml` targets:

```yaml
runs-on: [self-hosted, linux, x64, gpu]
```

It fails before scraping if `nvidia-smi` or ONNX Runtime's
`CUDAExecutionProvider` is missing. It is manual-only by default to avoid
hosted GPU spend.

## Binary Release Assets

`.github/workflows/release-binaries.yml` builds and uploads:

- `ato-mcp-x86_64-unknown-linux-gnu.tar.gz`
- `ato-mcp-aarch64-apple-darwin.tar.gz`
- `ato-mcp-x86_64-pc-windows-msvc.zip`

Run it by pushing a `v*` tag or via `workflow_dispatch`.

## Manual Corpus Publication

After a local `build-index`:

```bash
jq '.packs | length' release/manifest.json
scripts/publish-release.sh v0.3.0 gunba/ato-mcp
```

Set `ATO_MCP_MODEL_URL` only when publishing against an approved model mirror.
By default the manifest points at pinned Hugging Face EmbeddingGemma files.
This uploads manifest and packs to GitHub Releases; it does not upload the
model to GitHub, build Python wheels, or duplicate the corpus into an offline
bundle by default. Do not publish DB-derived repacks.

For an explicit air-gapped install package:

```bash
scripts/make-offline-bundle.sh release/ato-mcp-offline-v0.3.0.tar.zst
```

The offline bundle script runs the Rust installer against a local mirror of
the manifest, packs, and model bundle, then packages the resulting data
directory. Do not build offline bundles by copying `release/ato.db` directly.

## Health Checks

```bash
ato-mcp stats
ato-mcp doctor
cargo test --locked
.venv/bin/pytest -q
```

Watch for:

- Zero new rows for several weekly catch-up runs.
- Growing failed rows in `ato_pages/index.jsonl`.
- `ato-mcp doctor` failures after update.
- Missing `CUDAExecutionProvider` before a release build.

## Do Not

- Do not hand-edit `index.jsonl`.
- Do not delete published packs referenced by a manifest.
- Do not publish a corpus built without GPU-backed embeddings.
- Do not paste or print local tokens in logs or release notes.
