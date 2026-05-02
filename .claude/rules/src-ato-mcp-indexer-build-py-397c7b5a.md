---
paths:
  - "src/ato_mcp/indexer/build.py"
---

# src/ato_mcp/indexer/build.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Index Builder
Build orchestration, heading-aware chunking with overlap, token estimation, packing, manifest, release.

- [IB-13 L61] Windowed processing groups records into args.window_docs (default 20,000); CHECKPOINT_EVERY=1000 commits the in-progress SQLite transaction and flushes the in-flight pack so a kill mid-run loses at most that many docs.
- [IB-12 L147] Two build paths: _build_fresh_windowed (no prior manifest, full re-embed) vs incremental build() (when previous_manifest matches, reuse pack slot + skip re-embed for unchanged content_hash). Incremental requires embeddinggemma — the lexical fallback intentionally only runs in fresh mode.
- [IB-14 L325] Resume support: on incremental restart, doc_ids already in documents with a sealed pack_sha8 (not 'PENDING') are skipped — the prior commit landed rows + pack atomically, so the on-disk state is safe to keep.
- [IB-15 L393] Empty shells (status=success but no extractable body — broken pages, EPA stubs) are written to empty_shells with first_seen_at/last_checked_at; they don't enter documents/title_fts/chunks_fts so they can't pollute search.
- [IB-16 L619] Window preparation parallelises HTML extract + chunking via ProcessPoolExecutor with workers = max(1, cpu_count - 1); only the embed + DB-write phases stay single-threaded since they hold the SQLite transaction.
