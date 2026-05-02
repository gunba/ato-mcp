---
paths:
  - "src/ato_mcp/scraper/threadpool.py"
---

# src/ato_mcp/scraper/threadpool.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Source Scraper
Incremental/full/catch-up scraping from ato.gov.au, threadpool, snapshot, tree crawler.

- [SS-08 L8] scraper.threadpool exposes a 4-worker default ThreadPoolExecutor for fan-out; concurrent fetches still serialise on AtoBrowseClient's rate lock so worker count caps parsing parallelism, not network throughput.
