---
paths:
  - "src/ato_mcp/store/manifest.py"
---

# src/ato_mcp/store/manifest.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Storage Layer
SQLite schema, sqlite-vec virtual table, FTS5 with Porter stemmer, WAL+mmap, prepared queries, migration.

- [SL-07 L88] Manifest signature verification calls the minisign CLI via subprocess rather than a Python library; the choice exercises the same verifier maintainers use offline so signing-key hygiene problems surface early.
- [SL-08 L111] diff_manifests compares content_hash to produce (added, changed, removed_doc_ids) — content_hash is the only diff signal, so a chunk-only edit that doesn't change content_hash is invisible to the updater (an intentional simplification, not a bug).
