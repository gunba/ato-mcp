#!/usr/bin/env bash
# One-shot: build wheel + offline bundle, upload to an existing release,
# mark it --latest so /releases/latest/download/* works.
#
# Prereqs:
#   - build-index has finished; release/ contains ato.db, packs/, manifest.json
#   - model at models/embeddinggemma/
#   - gh authenticated
#
# Usage (from repo root):
#   scripts/publish-release.sh v0.1             # defaults below
#   scripts/publish-release.sh v0.1 gunba/ato-mcp
set -euo pipefail

TAG="${1:?usage: publish-release.sh <tag> [owner/repo]}"
REPO="${2:-${ATO_MCP_GH_REPO:-gunba/ato-mcp}}"
REPO_DIR="${ATO_MCP_REPO_DIR:-$(pwd)}"
VENV="${ATO_MCP_VENV:-$REPO_DIR/.venv}"
RELEASE_DIR="${ATO_MCP_RELEASE_DIR:-$REPO_DIR/release}"
MODEL_DIR="${ATO_MCP_MODEL_DIR:-$REPO_DIR/models/embeddinggemma}"

echo "=> building wheel"
"$VENV/bin/python" -m build --wheel --outdir "$REPO_DIR/dist"
WHEEL=$(ls -t "$REPO_DIR/dist"/ato_mcp-*.whl | head -1)
echo "   $WHEEL"

echo "=> uploading manifest + packs via ato-mcp release"
"$VENV/bin/ato-mcp" release \
  --out-dir   "$RELEASE_DIR" \
  --tag       "$TAG" \
  --repo      "$REPO" \
  --model-dir "$MODEL_DIR" \
  --overwrite

echo "=> building offline bundle"
BUNDLE="$RELEASE_DIR/ato-mcp-offline-$TAG.tar.zst"
"$REPO_DIR/scripts/make-offline-bundle.sh" "$BUNDLE"

# The bundle script splits into .part01.bin, .part02.bin, ... if > 1.9 GiB.
# Collect the asset list (single bundle OR parts, whichever exists).
if compgen -G "${BUNDLE}.part*.bin" > /dev/null; then
  BUNDLE_ASSETS=("${BUNDLE}".part*.bin)
else
  BUNDLE_ASSETS=("$BUNDLE")
fi

echo "=> uploading wheel + offline bundle as assets"
gh release upload "$TAG" "$WHEEL" "${BUNDLE_ASSETS[@]}" --repo "$REPO" --clobber

echo "=> promoting $TAG to latest"
gh release edit "$TAG" --repo "$REPO" --latest --prerelease=false

echo
echo "Done. Install on an offline/work machine:"
echo "  1. In browser at https://github.com/$REPO/releases/$TAG :"
echo "     - download $(basename "$WHEEL")"
for f in "${BUNDLE_ASSETS[@]}"; do
  echo "     - download $(basename "$f")"
done
echo "  2. Transfer the files to the machine."
echo "  3. On the machine:"
echo "       pipx install ~/Downloads/$(basename "$WHEEL")"
echo "       mkdir -p ~/.local/share/ato-mcp"
if [ "${#BUNDLE_ASSETS[@]}" -gt 1 ]; then
  echo "       cat ~/Downloads/$(basename "$BUNDLE").part*.bin | tar -I zstd -x -C ~/.local/share/ato-mcp"
else
  echo "       tar -I zstd -xf ~/Downloads/$(basename "$BUNDLE") -C ~/.local/share/ato-mcp"
fi
echo "       ato-mcp doctor"
echo "       claude mcp add --scope user ato -- ato-mcp serve"
