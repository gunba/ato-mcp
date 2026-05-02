---
paths:
  - "src/ato_mcp/tools.py"
---

# src/ato_mcp/tools.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## MCP Tools (search / get_chunks / get_document)
Hybrid BM25+vector search, slim hits, RRF fusion, recency boost, session-scoped seen tracker.

- [MT-07 L59] Per-thread read-only SQLite connections via threading.local; the connection is reopened when the db file's mtime advances, so a concurrent 'ato-mcp update' is picked up automatically without a restart.
- [MT-01 L95] SeenTracker is a process-scoped, thread-safe set of chunk_ids; MCP stdio runs one session per process so module-level state is naturally session-scoped.
- [MT-13 L183] types and doc_scope accept shell glob patterns: '*' is translated to SQL LIKE '%', and '\\', '%', '_' are escaped via _glob_to_like + ESCAPE clause.
- [MT-10 L187] Defaults exclude Edited_private_advice (DEFAULT_EXCLUDED_TYPES); content dated before 2000 is also excluded unless include_old=True or types matches DEFAULT_OLD_CONTENT_EXCEPTION_TYPES (legislation).
- [MT-08 L289] FTS query construction: tokens joined with implicit AND, single-char tokens dropped (so R&D doesn't degenerate to zero results), hyphenated tokens preserved as quoted phrases ('s 8-1', '355-25').
- [MT-09 L297] Query embeddings encoded with is_query=True (applies the EmbeddingGemma query prefix); falls back to query_lexical_hash when meta.embedding_model_id starts with 'lexical-hash-rust'.
- [MT-05 L342] Hybrid mode fuses BM25 and vector results via Reciprocal Rank Fusion with K=60: each result contributes 1/(K+rank+1) per ranker, scores summed across rankers.
- [MT-02 L402] search filters the fused ranking against the SeenTracker before materialization; internal_k is bumped by len(seen) so the filter can hide that many chunks without starving the frontier.
- [MT-03 L422] previously_seen echo is capped at MAX_SEEN_ECHO=10 entries, taken in fused-rank order so the most relevant suppressed chunk for this query is first.
- [MT-06 L467] Recency boost is multiplicative in (0.5, 1.5] with a 5-year half-life via RECENCY_HALF_LIFE_YEARS; applied only when sort_by='relevance' and date parses as YYYY.
- [MT-04 L484] search returns slim hits only (chunk_id, doc_id, title, type, date, heading_path, anchor, snippet, canonical_url, score) — never the full chunk body; bodies materialize via get_chunks (progressive disclosure).
- [MT-14 L534] search_titles bm25-ranks against title_fts (title + collected headings) — independent of chunks and the SeenTracker; the default exclusions for EPA and old non-legislation match search.
- [MT-11 L580] get_document supports three retrieval modes through one tool: format='outline' returns the TOC; anchor/heading_path returns a section (include_children rolls up the subtree); from_ord paginates with count or max_chars and emits continuation_ord.
- [MT-12 L616] get_document non-outline paths register materialized chunks with the SeenTracker; the outline path does not (no chunk content surfaces, only headings).
- [MT-15 L813] whats_new sorts by COALESCE(date, downloaded_at) DESC; the synthesised snippet says 'published <date>' when date is present, otherwise 'ingested <downloaded_at>'.
