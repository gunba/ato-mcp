---
paths:
  - "src/ato_mcp/scraper/client.py"
---

# src/ato_mcp/scraper/client.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Source Scraper
Incremental/full/catch-up scraping from ato.gov.au, threadpool, snapshot, tree crawler.

- [SS-02 L36] AtoBrowseClient calls https://www.ato.gov.au/API/v1/law/lawservices/browse-content/ as the single source of truth; thin wrapper around requests.Session with no retry policy because the rate-limiter already smooths the load.
- [SS-03 L71] Global rate limiting: AtoBrowseClient._acquire_request_slot uses a threading.Lock + time.monotonic() to enforce request_interval seconds between any two outgoing calls — the tree crawler issues thousands per run.
