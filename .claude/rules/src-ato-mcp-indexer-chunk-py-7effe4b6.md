---
paths:
  - "src/ato_mcp/indexer/chunk.py"
---

# src/ato_mcp/indexer/chunk.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Index Extraction And Chunking

- [IB-02 L26] DEFAULT_MAX_TOKENS=900 with DEFAULT_OVERLAP_TOKENS=120; adjacent chunks under the same heading are stitched with a tail-overlap bridge so vector search doesn't lose context at section boundaries.
- [IB-04 L56] strip_title_prefix de-duplicates the document title's front-matter echo from heading_path; e.g. 'Taxation Ruling — TR 2024/3 — Subject › Taxation Ruling › TR 2024/3 › Ruling' collapses to 'Ruling'. Pure string transform — safe to apply at chunk emission and as a one-shot rewrite.
- [IB-05 L83] Token count is estimated via len(text.split()) * 1.3 — boundary-control only. The real per-batch token limit is enforced by the embedding tokenizer, so this estimate just needs to be safely conservative.
- [IB-01 L164] Chunker is recursive: split markdown on #/##/### heading boundaries first; if a section still exceeds max_tokens, fall back to paragraph (blank-line) split, then sentence split — never silently truncates.
- [IB-03 L188] Heading stack pops same-level siblings before pushing, so consecutive <h5>Note 1:</h5> / <h5>Note 2:</h5> blocks (common in ITAA legislation) become siblings rather than falsely nesting under each other.
