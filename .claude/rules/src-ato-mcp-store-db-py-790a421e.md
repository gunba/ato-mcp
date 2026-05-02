---
paths:
  - "src/ato_mcp/store/db.py"
---

# src/ato_mcp/store/db.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Storage Layer
SQLite schema, sqlite-vec virtual table, FTS5 with Porter stemmer, WAL+mmap, prepared queries, migration.

- [SL-06 L31] On open, sqlite-vec is loaded via enable_load_extension(True) → sqlite_vec.load(conn) → enable_load_extension(False) — extension loading is re-disabled immediately so application queries can't smuggle their own extensions in.
- [SL-05 L60] Connections enable WAL+synchronous=NORMAL for write modes only; read-only handles skip those pragmas. mmap_size=256 MB and temp_store=MEMORY are set unconditionally for both modes.
