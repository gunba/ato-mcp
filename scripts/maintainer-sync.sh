#!/usr/bin/env bash
# Maintainer steady-state: refresh ato_pages, rebuild index, publish release.
#
# Expects these env vars (set in the systemd unit or your shell):
#   ATO_MCP_REPO_DIR   absolute path to this repo checkout
#   ATO_MCP_PAGES_DIR  absolute path to ato_pages/ (default: $ATO_MCP_REPO_DIR/../ato_pages)
#   ATO_MCP_MODEL_DIR  absolute path to a dir holding model_quantized.onnx,
#                      model_quantized.onnx_data, tokenizer.json
#   ATO_MCP_MODEL_URL  optional approved model mirror URL
#   ATO_MCP_RELEASE_TAG  tag prefix (default: index)
#   ATO_MCP_GH_REPO    owner/name (default: gunba/ato-mcp)
#   ATO_MCP_MODE       incremental | catch_up | full (default: incremental)
#
# Flow:
#   1. refresh-source in the requested mode (incremental by default)
#   2. If ato_pages/index.jsonl actually grew (or catch-up wrote new rows),
#      re-run build-index --incremental against the previous release manifest.
#   3. Publish a new release under tag $ATO_MCP_RELEASE_TAG-YYYY.MM.DD and
#      mark it latest. GitHub's "download latest" URL then points at it,
#      so end-users' `ato-mcp update` picks it up on their next run.

set -euo pipefail

REPO_DIR="${ATO_MCP_REPO_DIR:?set ATO_MCP_REPO_DIR}"
PAGES_DIR="${ATO_MCP_PAGES_DIR:-$REPO_DIR/../ato_pages}"
MODEL_DIR="${ATO_MCP_MODEL_DIR:?set ATO_MCP_MODEL_DIR}"
MODEL_URL="${ATO_MCP_MODEL_URL:-}"
MODEL_URL_ARG=()
if [ -n "$MODEL_URL" ]; then
    MODEL_URL_ARG=(--model-url "$MODEL_URL")
fi
GH_REPO="${ATO_MCP_GH_REPO:-gunba/ato-mcp}"
MODE="${ATO_MCP_MODE:-incremental}"
TAG_PREFIX="${ATO_MCP_RELEASE_TAG:-index}"

cd "$REPO_DIR"
VENV="$REPO_DIR/.venv"
ATO_MCP="$VENV/bin/ato-mcp"
if [[ ! -x "$ATO_MCP" ]]; then
    echo "no venv at $VENV — run: python -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 2
fi

# nvidia libs for GPU build (harmless if absent)
LIBS=$(find "$VENV"/lib*/python*/site-packages/nvidia/ -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="${LIBS:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LOG="$REPO_DIR/logs/maintainer-sync-$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "== $(date -u +%FT%TZ) maintainer sync ($MODE) =="

BEFORE_COUNT=$(wc -l < "$PAGES_DIR/index.jsonl" 2>/dev/null || echo 0)

case "$MODE" in
    incremental)
        "$ATO_MCP" refresh-source --mode incremental --output-dir "$PAGES_DIR"
        ;;
    catch_up)
        "$ATO_MCP" catch-up --output-dir "$PAGES_DIR"
        ;;
    full)
        "$ATO_MCP" refresh-source --mode full --output-dir "$PAGES_DIR"
        ;;
    *)
        echo "unknown MODE=$MODE (incremental|catch_up|full)" >&2
        exit 2
        ;;
esac

AFTER_COUNT=$(wc -l < "$PAGES_DIR/index.jsonl" 2>/dev/null || echo 0)
echo "index.jsonl rows: $BEFORE_COUNT -> $AFTER_COUNT"

if (( AFTER_COUNT == BEFORE_COUNT )); then
    echo "no new rows; skipping rebuild+release"
    exit 0
fi

TAG="$TAG_PREFIX-$(date -u +%Y.%m.%d)"
RELEASE_DIR="$REPO_DIR/release/$TAG"
PREV_MANIFEST="$REPO_DIR/release/.latest/manifest.json"
mkdir -p "$RELEASE_DIR"

PREV_ARG=()
if [[ -f "$PREV_MANIFEST" ]]; then
    PREV_ARG=(--previous-manifest "$PREV_MANIFEST")
fi

"$ATO_MCP" build-index \
    --pages-dir "$PAGES_DIR" \
    --out-dir "$RELEASE_DIR" \
    --db-path "$RELEASE_DIR/ato.db" \
    --model-path "$MODEL_DIR/model_quantized.onnx" \
    --tokenizer-path "$MODEL_DIR/tokenizer.json" \
    --gpu \
    "${PREV_ARG[@]}"

"$ATO_MCP" release \
    --out-dir "$RELEASE_DIR" \
    --tag "$TAG" \
    --repo "$GH_REPO" \
    --model-dir "$MODEL_DIR" \
    "${MODEL_URL_ARG[@]}" \
    --overwrite

# Promote to "latest" so /releases/latest/download resolves to this tag.
gh release edit "$TAG" --repo "$GH_REPO" --latest --prerelease=false

# Remember this manifest so the next incremental build can reuse packs.
mkdir -p "$REPO_DIR/release/.latest"
cp "$RELEASE_DIR/manifest.json" "$REPO_DIR/release/.latest/manifest.json"

echo "== done: released $TAG ($(( AFTER_COUNT - BEFORE_COUNT )) new rows) =="
