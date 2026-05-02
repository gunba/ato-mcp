---
paths:
  - "src/ato_mcp/server.py"
---

# src/ato_mcp/server.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Server Wiring
FastMCP tool registration, instructions builder from corpus stats, opportunistic warmup.

- [SW-01 L24] Five tools are registered with FastMCP: search, search_titles, get_document, get_chunks, whats_new — kept minimal so the agent has a small, predictable surface.
- [SW-02 L149] Server instructions are built dynamically at start time from corpus stats (doc count, chunk count, type breakdown, meta keys), so the agent sees up-to-date corpus shape without restart-time configuration.
- [SW-03 L167] When the DB is missing or unreadable at instructions-build time, the server falls back to _FALLBACK_INSTRUCTIONS (a static one-line message) instead of crashing on import — works around fresh installs that haven't run 'ato-mcp update' yet.
- [SW-04 L208] On startup, run() opportunistically calls model.encode(['warmup'], is_query=True) inside a try/except so the ONNX model is paged into memory before the first real tool call; failure is swallowed (the server still starts even without a model).
