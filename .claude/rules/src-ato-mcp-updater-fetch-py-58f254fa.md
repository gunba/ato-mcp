---
paths:
  - "src/ato_mcp/updater/fetch.py"
---

# src/ato_mcp/updater/fetch.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Update Mechanism
End-user update flow: manifest diff, pack fetch, in-place SQLite patch application, lock.

- [UM-04 L20] fetch helpers intentionally don't read GitHub token env vars and don't shell out to gh — private release assets must be exposed through an approved mirror or installed from a local/offline bundle. This keeps the end-user runtime credential-free.
- [UM-03 L46] fetch_url uses httpx with http/2 enabled, streams in 64 KB chunks (CHUNK_BYTES), and verifies sha256 against the manifest entry before any file is considered downloaded; bounded timeouts (connect=10s, read=60s, write=60s, pool=60s).
