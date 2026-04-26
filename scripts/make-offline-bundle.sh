#!/usr/bin/env bash
# Build a single pre-populated data directory as a tar.zst, so a locked-down
# machine can install ato-mcp by placing the Rust binary on PATH and extracting
# this tarball over $XDG_DATA_HOME/ato-mcp/.
#
# Usage (from repo root):
#   scripts/make-offline-bundle.sh                    # defaults below
#   scripts/make-offline-bundle.sh ./out.tar.zst
#
# Env overrides:
#   ATO_MCP_REPO_DIR     (default: $(pwd))
#   ATO_MCP_RELEASE_DIR  (default: $REPO_DIR/release)       — must contain ato.db + packs/ + manifest.json
#   ATO_MCP_MODEL_DIR    (default: $REPO_DIR/models/embeddinggemma)  — model_quantized.onnx + tokenizer.json
set -euo pipefail

REPO_DIR="${ATO_MCP_REPO_DIR:-$(pwd)}"
RELEASE_DIR="${ATO_MCP_RELEASE_DIR:-$REPO_DIR/release}"
MODEL_DIR="${ATO_MCP_MODEL_DIR:-$REPO_DIR/models/embeddinggemma}"
OUT="${1:-$RELEASE_DIR/ato-mcp-offline-bundle.tar.zst}"

for f in "$RELEASE_DIR/ato.db" "$RELEASE_DIR/manifest.json" \
         "$MODEL_DIR/onnx/model_quantized.onnx" "$MODEL_DIR/tokenizer.json"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/live/packs"
cp "$RELEASE_DIR/ato.db" "$STAGE/live/ato.db"
cp "$RELEASE_DIR"/packs/pack-*.bin.zst "$STAGE/live/packs/"
cp "$MODEL_DIR/onnx/model_quantized.onnx" "$STAGE/live/"
[ -e "$MODEL_DIR/onnx/model_quantized.onnx_data" ] && \
  cp "$MODEL_DIR/onnx/model_quantized.onnx_data" "$STAGE/live/"
cp "$MODEL_DIR/tokenizer.json" "$STAGE/live/"
# Regular copy, not a symlink — tarballs cross-filesystem cleanly this way.
cp "$MODEL_DIR/onnx/model_quantized.onnx" "$STAGE/live/model.onnx"
cp "$RELEASE_DIR/manifest.json" "$STAGE/installed_manifest.json"

# Deterministic ordering keeps the tarball byte-identical across re-runs.
tar --sort=name --mtime='2026-01-01 UTC' -C "$STAGE" -cf - . | zstd -T0 -10 -o "$OUT"

# GitHub caps single release assets at 2 GiB. Split anything larger so each
# part uploads individually; the install command reassembles them with `cat`.
SPLIT_THRESHOLD=$((1900 * 1024 * 1024))
SIZE=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
if [ "$SIZE" -gt "$SPLIT_THRESHOLD" ]; then
  echo "bundle > 1.9 GiB; splitting into parts"
  split --bytes=1900M --numeric-suffixes=1 --additional-suffix=.bin "$OUT" "${OUT}.part"
  rm -f "$OUT"
  ls -lh "${OUT}.part"*
else
  echo "bundle: $OUT"
  du -h "$OUT"
fi
