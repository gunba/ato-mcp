# AGENTS.md

Instructions for an automated coding/support agent (Claude Code, Cursor,
Continue, etc.) tasked with installing or operating `ato-mcp` on the
user's machine. Read this first; fall back to [README.md](README.md) for
design detail.

## Install path

Pick the first option that works:

1. **`pipx` installed?** Run:
   ```bash
   pipx install git+https://github.com/gunba/ato-mcp.git
   ```

2. **`uv` installed?** Run:
   ```bash
   uv tool install --from git+https://github.com/gunba/ato-mcp.git ato-mcp
   ```

3. Else `pip install --user`:
   ```bash
   pip install --user git+https://github.com/gunba/ato-mcp.git
   ```

Verify: `ato-mcp --help` should print a subcommand list.

## First-run data download

```bash
ato-mcp init
```

This downloads the embedding model and prebuilt index (~700 MB) into
`$XDG_DATA_HOME/ato-mcp/` (Linux/macOS) or `%LOCALAPPDATA%\ato-mcp\`
(Windows). Override with `ATO_MCP_DATA_DIR`.

If `init` fails with a signature error, the release probably has no
`.minisig` yet (v0.1 ships unsigned). Re-run with signatures disabled:

```bash
ATO_MCP_INSECURE=1 ato-mcp init
```

## Register with the MCP host

**Claude Code:**

```bash
claude mcp add ato -- ato-mcp serve
```

**Claude Desktop** — edit the client config and restart:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ato": { "command": "ato-mcp", "args": ["serve"] }
  }
}
```

**Cursor / Continue / other stdio MCP clients** — register a new MCP
server with `command: ato-mcp`, `args: ["serve"]`. Consult the client's
MCP docs for the exact file.

## Verify

```bash
ato-mcp stats       # should print non-zero documents + chunks
ato-mcp doctor      # verifies the DB opens
```

Then inside the host client, invoke the `ato.search` tool with a plain
English query like `"research and development tax incentive eligibility"`
and confirm you get back a table of hits with `canonical_url` links.

## Routine maintenance

Weekly:

```bash
ato-mcp update      # typical: 2-5 MB transferred
```

If something breaks after an update:

```bash
ato-mcp doctor --rollback
```

## Don'ts

- **Don't** edit files under `$XDG_DATA_HOME/ato-mcp/live/` manually.
  The updater expects specific content-hash invariants.
- **Don't** run two `ato-mcp update` processes simultaneously — the
  advisory lock rejects the second, but you'd still be burning bandwidth.
- **Don't** use the `build-index` / `release` / `refresh-source`
  subcommands on a user install. Those are for the maintainer only and
  require GPU + a working ato_pages scrape.

## Troubleshooting hints

| Symptom | Fix |
|---|---|
| `ato-mcp: command not found` | Ensure `~/.local/bin` (or `pipx`'s bin dir) is on `PATH`. `pipx ensurepath` and open a new shell. |
| `init` hangs at 0% | The GitHub release may be empty. Check https://github.com/gunba/ato-mcp/releases |
| `doctor` reports 0 documents | `init` didn't complete; rerun. |
| `search` returns no hits | Confirm `stats` shows `chunks > 0`. If it shows 0, the index download was truncated — delete the data dir and re-init. |
| Slow responses | First call per process loads the ONNX model; subsequent calls are fast. Check `stats.data_dir` is on a local disk, not a network share. |
| "embedding model unavailable" in logs | The model file is missing or mismatched; rerun `ato-mcp init`. |
