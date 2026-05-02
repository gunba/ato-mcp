---
paths:
  - "src/ato_mcp/scraper/reducer.py"
---

# src/ato_mcp/scraper/reducer.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Source Scraper
Incremental/full/catch-up scraping from ato.gov.au, threadpool, snapshot, tree crawler.

- [SS-07 L37] SnapshotReducer dedupes canonical_ids across folders, picks a representative_path per canonical_id, and flags redundant folder paths; titles in constants.EXCLUDED_TITLES (and their descendants) are filtered out before reduction.
