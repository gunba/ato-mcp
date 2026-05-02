---
paths:
  - "src/ato_mcp/formatters.py"
---

# src/ato_mcp/formatters.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Output Formatters
Markdown table for hits, previously_seen tail, document outline + section + full renderers, JSON output.

- [OF-01 L19] canonical_url is synthesised from doc_id by direct substitution into the ATO URL pattern; href is not stored separately so the link always reflects the current doc_id.
- [OF-04 L36] Markdown table cells escape '|' to '\\|' and replace newlines with spaces so snippets and heading_paths can't break out of the table grid.
- [OF-02 L45] format_hits_markdown returns '_No matches._' when both hits and previously_seen are empty; switches to '_No fresh matches._' when only previously_seen has content (i.e. all top results were suppressed by the SeenTracker).
- [OF-03 L48] previously_seen tail rendering: dashed separator + bullet list of (chunk_id, title, doc_id, heading_path), ending with a ready-to-paste 'Re-fetch with get_chunks([...])' invocation so the agent has a one-shot recovery path.
- [OF-05 L92] Outline rows indent by heading depth using doubled non-breaking spaces ('&nbsp;&nbsp;' per level), so deeper headings sit visually nested under shallower ones.
- [OF-06 L141] as_json serialises with orjson + OPT_INDENT_2 (2-space indent); decoded to UTF-8 before return so the MCP response is always a string, not bytes.
