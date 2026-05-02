# MCP Improvement Plan

Saved from the MCP server review on 2026-05-02.

## Priorities

1. Make the Rust binary the only installed MCP/CLI runtime surface.
2. Preserve pack embeddings during Rust install/update; packs already carry
   `embedding_b64`.
3. Fix `get_document(count=...)` continuation handling.
4. Add a cheap document/card description layer between search and full
   hydration.
5. Make search results directly hydrateable by surfacing `chunk_id`, `ord`,
   source URL, and next-call hints.
6. Make `get_chunks` deterministic, include `ord`, and support contextual
   before/after neighbor expansion.
7. Expose ranking diagnostics and provenance in JSON outputs.
8. Add budget-aware response metadata and truncation contracts.
9. Improve tool schemas and keep the top-level tool surface small.
10. Add better diagnostics and explicit offline operation.
11. Delete duplicate Python MCP runtime code; keep Python only for corpus
    production tooling until that tooling is replaced.

## Implemented In Current Branch

- Rust preserves pack embeddings and performs hybrid/vector semantic search
  with EmbeddingGemma ONNX/tokenizer.
- Rust install/update creates `chunk_embeddings` from pack `embedding_b64`
  and rehydrates old installs when the live DB has chunks but no embeddings.
- Explicit `mode=keyword` remains available; hybrid/vector fail instead of
  silently downgrading.
- Python MCP runtime files, user-install updater commands, and runtime tests
  are removed.
- Search/document/chunk/recent JSON responses include truncation and retrieval
  metadata.
- `get_chunks` preserves requested order, includes `ord`, and supports
  before/after neighbor context.
- `get_document(format="card")` exposes cheap hydration metadata.
- Release tooling defaults to pinned Hugging Face EmbeddingGemma files and
  still refuses GitHub-hosted model bundles.
- `scripts/smoke-rust-install.sh` verifies a published manifest install using
  the Rust binary.

## Deferred

- Citation resolver for deterministic citation/section lookups.
- Local reranker for top-N hybrid candidates.

## Avoid

- Arbitrary code-mode execution for legal retrieval.
- One tool per document type.
- Default-on telemetry.
- TTL caches, debounced watchers, or fixed timer/polling control flow.
- External vector-server dependencies for end-user installs.

## External Repos Reviewed

- `cloudflare/mcp`
- `lyonzin/knowledge-rag`
- `shinpr/mcp-local-rag`
- `qdrant/mcp-server-qdrant`
- `github/github-mcp-server`
- `microsoft/playwright-mcp`
- `upstash/context7`
