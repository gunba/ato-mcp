---
paths:
  - "src/ato_mcp/cli.py"
---

# src/ato_mcp/cli.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## CLI Commands
Typer command surface, end-user vs maintainer split, defaults and global excludes.

- [CC-01 L31] Two-tier command surface: end-user commands (serve, init, update, doctor, stats) ship in the wheel; maintainer commands (refresh-source, build-index, release) require a repo checkout. The split keeps end-user installs minimal.
- [CC-06 L32] typer.Typer is configured with no_args_is_help=True and add_completion=False — the agent surface is intentionally small, no shell-completion magic, no implicit subcommands.
- [CC-02 L38] serve auto-applies a pending update on startup, but falls back to the installed corpus if update fails — a stale-but-working install always serves; only a missing DB is fatal.
- [CC-03 L48] init / update / serve all run apply_update with manifest_url + signature; the minisig signature is verified only when the bundled pubkey exists. Opt-out is structural (don't bundle the key) — there's no flag to disable verification.
- [CC-04 L155] _maybe_migrate_v4_to_v5 runs on init/update/serve before apply_update; pre-v5 DBs are upgraded in place. Pre-v4 raises and demands a rebuild from ato_pages/ — the schema is too different for an in-place patch.
- [CC-05 L233] refresh-source defaults to --output-dir ./ato_pages; build-index requires --pages-dir pointing at a populated ato_pages/. The split keeps the scrape and index stages independently re-runnable — same pages dir can feed multiple builds.
