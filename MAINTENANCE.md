# Maintainer runbook

Day-to-day operation of the `ato-mcp` release pipeline. End users never read
this file — they follow the top of the main `README.md`.

## Automated pipeline (set-and-forget)

Three systemd **user** timers do all the routine work. They live in
`systemd/` and install under `~/.config/systemd/user/`. No sudo required;
`loginctl enable-linger $USER` keeps them running across logouts.

| Timer | Cadence | Command behind it | Typical runtime |
|---|---|---|---|
| `ato-mcp-update` | Daily | `ato-mcp update` (end-user side — pulls published manifest, applies per-doc delta) | seconds, transfer usually 2-5 MB |
| `ato-mcp-maintainer-daily` | Daily ~02:37 | `refresh-source --mode incremental` → `build-index --incremental` → `release --overwrite` | minutes |
| `ato-mcp-maintainer-weekly` | Sun ~03:43 | `refresh-source --mode catch_up` (tree diff) → build → release | hours |

Install instructions live in `systemd/README.md`. This machine is already
set up.

### Health checks

```bash
systemctl --user list-timers --all | grep ato-mcp
systemctl --user status ato-mcp-maintainer-daily
journalctl --user -u ato-mcp-maintainer-daily -n 100 --no-pager
journalctl --user -u ato-mcp-maintainer-weekly -n 200 --no-pager
ato-mcp stats            # sanity-check the installed index locally
```

Symptoms to watch for:
- Several daily runs in a row showing zero new docs → ATO may have changed
  their What's New feed / tree API.
- Growing `status=failed` counts in `index.jsonl` → network or remote-side
  regressions.
- `ato-mcp doctor` failing → corrupt DB; roll back with
  `ato-mcp doctor --rollback`.

## Manual, occasional

### Monthly (optional) — full rebuild as drift guard

```bash
cd ~/Desktop/Projects/ato-mcp
.venv/bin/ato-mcp refresh-source --mode full --output-dir ~/Desktop/Projects/ato_pages
# then build + release as shown below (or let the next maintainer timer run).
```

Typical runtime: scrape several hours at 4 req/s, build ~1-2 h on GPU.

### Full manual release (after a one-off structural change)

```bash
cd ~/Desktop/Projects/ato-mcp

LD_LIBRARY_PATH="$(find .venv/lib*/python3.*/site-packages/nvidia/ -maxdepth 2 -name 'lib' -type d | tr '\n' ':')$LD_LIBRARY_PATH" \
  .venv/bin/ato-mcp build-index \
  --pages-dir /home/jordan/Desktop/Projects/ato_pages \
  --out-dir   ./release \
  --db-path   ./release/ato.db \
  --model-path     ./models/embeddinggemma/onnx/model_quantized.onnx \
  --tokenizer-path ./models/embeddinggemma/tokenizer.json \
  --gpu

.venv/bin/ato-mcp release \
  --out-dir ./release \
  --tag v0.1 \
  --repo gunba/ato-mcp \
  --model-dir ./models/embeddinggemma \
  --overwrite
```

Drop `--previous-manifest` unless the referenced packs are actually on the
local filesystem (remote URLs in the manifest do not resolve — the
`_insert_from_previous` path reads packs locally).

### Reconciling orphan payloads

If `wc -l ato_pages/index.jsonl` is smaller than the count of HTML files
under `payloads/`, the scraper wrote files to disk without emitting index
rows (historical bug — now guarded by the synthetic-success path in
`_download_link`). Rebuild the index:

```bash
.venv/bin/python scripts/reconcile_index.py \
  --pages-dir /home/jordan/Desktop/Projects/ato_pages \
  --links-file /home/jordan/Desktop/Projects/ato_snapshots/deduped_links.jsonl
```

The script is idempotent and backs up nothing — take a copy of
`index.jsonl` yourself if paranoid (`cp index.jsonl index.jsonl.bak`).

### Kernel-level scraping manners

Scraper defaults are **1 worker, 0.5 s interval = ~2 req/s**. The global
rate lock in `AtoBrowseClient._acquire_request_slot` and
`LinkDownloader._acquire_request_slot` enforces it across threads. If you
ever bump `--max-workers`, do not touch the interval — the lock converts
parallelism into burstiness, not aggression.

Sleep/lid behaviour on KDE/Plasma: use `systemd-inhibit` around long
overnight scrapes, e.g.

```bash
systemd-inhibit --what=idle:sleep:shutdown:handle-lid-switch \
  --who=ato-mcp --why="overnight catch-up" \
  .venv/bin/ato-mcp catch-up --output-dir ~/Desktop/Projects/ato_pages
```

### Dependency bumps

```bash
cd ~/Desktop/Projects/ato-mcp
pip install -U pip
.venv/bin/pip install -e '.[verify]' --upgrade
pytest -q
```

On any other machine where you use the server:

```bash
pipx upgrade ato-mcp
ato-mcp doctor
```

## Release tags

- `v0.1` — current line; bumped in place by `release --overwrite`.
- Use `--tag vX.Y` for breaking schema changes. The client refuses
  manifests whose `min_client_version` is newer than the installed CLI.

## What you do not do

- **Never** hand-edit `index.jsonl`. Use the reconcile script or re-run
  `refresh-source`.
- **Never** delete packs from a published release — they are
  content-addressable and manifest-referenced. Old installs still resolve
  to them until they update.
- **Never** push directly into `~/.local/share/ato-mcp/live/` — treat it
  as read-only outside of `ato-mcp update` / `doctor`.

## Design references

- Full design doc: `/home/jordan/.claude/plans/it-s-been-a-long-dreamy-hare.md`
- End-user docs: `README.md` (main)
- systemd units: `systemd/*.{service,timer}` + `systemd/README.md`
- Scraper internals: `src/ato_mcp/scraper/`
- Indexer internals: `src/ato_mcp/indexer/`
- Updater internals: `src/ato_mcp/updater/`
