# ato-mcp

Local MCP server giving AI clients (Claude Code, etc.) search and fetch access to the full Australian Taxation Office legal database — legislation, cases, public rulings, private advice, PCGs, TAs, ATO IDs, PS LAs, and more.

## Install

```
pipx install ato-mcp
ato-mcp init                 # first-run download (~710 MB: model + index packs)
claude mcp add ato -- ato-mcp serve
```

Run `ato-mcp update` weekly for deltas (typically 2-5 MB transferred).

## Data location

`$XDG_DATA_HOME/ato-mcp/` (defaults to `~/.local/share/ato-mcp/`). Override with `ATO_MCP_DATA_DIR`.

## Tools exposed

`search`, `search_titles`, `resolve`, `get_document`, `get_section`, `get_chunks`, `list_categories`, `list_doc_types`, `whats_new`, `stats`.

See `/home/jordan/.claude/plans/it-s-been-a-long-dreamy-hare.md` for the full design document.
