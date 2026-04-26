#!/usr/bin/env bash
# One-shot corpus publication. Binary assets are built by
# .github/workflows/release-binaries.yml; this script uploads the corpus
# manifest, packs, model bundle, and optional offline data bundle.
#
# Prereqs:
#   - build-index has finished; release/ contains ato.db, packs/, manifest.json
#   - model at models/embeddinggemma/
#   - gh authenticated for the maintainer account
#
# Usage:
#   scripts/publish-release.sh v0.3.0
#   scripts/publish-release.sh v0.3.0 gunba/ato-mcp
set -euo pipefail

TAG="${1:?usage: publish-release.sh <tag> [owner/repo]}"
REPO="${2:-${ATO_MCP_GH_REPO:-gunba/ato-mcp}}"
REPO_DIR="${ATO_MCP_REPO_DIR:-$(pwd)}"
VENV="${ATO_MCP_VENV:-$REPO_DIR/.venv}"
RELEASE_DIR="${ATO_MCP_RELEASE_DIR:-$REPO_DIR/release}"
MODEL_DIR="${ATO_MCP_MODEL_DIR:-$REPO_DIR/models/embeddinggemma}"

echo "=> uploading manifest, packs, and model bundle"
"$VENV/bin/ato-mcp" release \
  --out-dir   "$RELEASE_DIR" \
  --tag       "$TAG" \
  --repo      "$REPO" \
  --model-dir "$MODEL_DIR" \
  --overwrite

echo "=> building offline data bundle"
BUNDLE="$RELEASE_DIR/ato-mcp-offline-$TAG.tar.zst"
"$REPO_DIR/scripts/make-offline-bundle.sh" "$BUNDLE"

if compgen -G "${BUNDLE}.part*.bin" > /dev/null; then
  BUNDLE_ASSETS=("${BUNDLE}".part*.bin)
else
  BUNDLE_ASSETS=("$BUNDLE")
fi

echo "=> uploading offline data bundle"
gh release upload "$TAG" "${BUNDLE_ASSETS[@]}" --repo "$REPO" --clobber

echo "=> promoting $TAG to latest"
gh release edit "$TAG" --repo "$REPO" --latest --prerelease=false

echo
echo "Done. End-user install requires a Rust binary asset plus these corpus assets."
echo "Release: https://github.com/$REPO/releases/tag/$TAG"
