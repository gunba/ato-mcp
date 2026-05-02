---
paths:
  - "src/ato_mcp/scraper/snapshot.py"
---

# src/ato_mcp/scraper/snapshot.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Source Scraper
Incremental/full/catch-up scraping from ato.gov.au, threadpool, snapshot, tree crawler.

- [SS-05 L63] SnapshotWriter emits nodes.jsonl + meta.json under a timestamped directory; diff_snapshots compares two snapshots by canonical_id (or 'uid:<uid>' fallback) for added/removed/changed sets.
