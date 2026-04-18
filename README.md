# ato-mcp

Local [MCP](https://modelcontextprotocol.io/) server giving AI clients (Claude Code, Claude Desktop, Cursor, any other MCP client) search and fetch access to the **full Australian Taxation Office (ATO) legal corpus** — legislation, cases, public rulings, private advice, PCGs, TAs, ATO IDs, PS LAs, decision impact statements, and more.

Runs entirely on your machine. No API keys. No hosted backend. A ~300 MB embedding model and a pre-built SQLite index are downloaded from GitHub Releases on first run. Updates are per-document deltas so weekly refreshes transfer single-digit MB.

---

## What you get

Ten MCP tools exposed over stdio:

| Tool | Purpose |
|---|---|
| `search` | Hybrid search (BM25 + vector) across ~86k documents. Filter by category, doc type, date. |
| `search_titles` | Fast title-only search. Use for citation lookups like `s355-25` or `TR 2024/3`. |
| `resolve` | Parse a citation (`TR 2024/3`, `CR 2025/62`, `ATO ID 2001/332`) and return exact matches. |
| `get_document` | Full document content (markdown) or a compact outline. |
| `get_section` | A single section by anchor or heading path. Cheap, precise. |
| `get_chunks` | Batched chunk fetch for reranker / context-building flows. |
| `list_categories` | 13 categories with document counts. |
| `list_doc_types` | 30+ ATO document types with counts. |
| `whats_new` | Recently updated documents. |
| `stats` | Index version, document/chunk counts, last update. |

Every hit includes a `canonical_url` pointing to the authoritative ATO page.

### Example agent interaction

```
> What does the ATO say about R&D tax incentive eligibility for software?

[agent calls search("R&D tax incentive eligibility software")]
  → top hits: TR 2013/3, PCG 2025/D6, CR 2025/73
[agent calls get_section("tr_2013_3", heading_path="Ruling › Eligibility")]
  → returns the exact section
[agent cites https://www.ato.gov.au/law/view/document?docid=TXR/TR20133/NAT/ATO/00001]
```

---

## Install

**Requirements:** Python 3.11+ (3.14 tested), `pipx`, ~1 GB disk (model + index), one-time 700 MB download.

```bash
pipx install git+https://github.com/gunba/ato-mcp.git
ato-mcp init                       # downloads model + index (~700 MB)
```

On Windows, substitute `%LOCALAPPDATA%\ato-mcp\` for the data directory.

### Wire up Claude Code

```bash
claude mcp add --scope user ato -- ato-mcp serve
```

Use `--scope project` if you want it in `.claude/settings.json` for a
specific repo instead of user-wide.

Confirm it's registered:

```bash
claude mcp list
claude mcp get ato
```

Then inside Claude Code:

```
/mcp
```

You should see `ato` listed. Test with a search:

```
Ask ato: search "capital gains tax small business concessions"
```

### Wire up Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "ato": {
      "command": "ato-mcp",
      "args": ["serve"]
    }
  }
}
```

Restart Claude Desktop.

### Wire up Cursor, Continue, etc.

Any MCP stdio client works — point it at the `ato-mcp serve` command. See
the [MCP client docs](https://modelcontextprotocol.io/clients) for
client-specific config.

---

## Keep it up to date

Weekly:

```bash
ato-mcp update
```

Typical transfer: 2–5 MB. The updater downloads only the byte ranges for
changed documents, applies them to the local SQLite in one transaction,
and atomically swaps the installed manifest. Crash-safe: the live DB is
either the pre-update or post-update state, never mixed. On a bad update,
roll back:

```bash
ato-mcp doctor --rollback
```

### Verify install

```bash
ato-mcp stats
ato-mcp doctor
```

`stats` shows index version + counts; `doctor` verifies the DB opens.

---

## For AI agents (Claude Code, Cursor, etc.) setting this up

If you are an AI agent tasked with installing this server for the user, run these steps end to end:

```bash
# 1. Install the CLI (prefer pipx; fall back to `uv tool install` or `pip install --user`).
pipx install git+https://github.com/gunba/ato-mcp.git

# 2. Download the prebuilt index + embedding model (~700 MB one-time).
ato-mcp init

# 3. Register the server with the host MCP client. For Claude Code:
claude mcp add ato -- ato-mcp serve

# 4. Sanity check.
ato-mcp stats
```

If the host is Claude Desktop or a config-file-based client, edit the
client's MCP servers JSON as shown above instead of step 3.

If `init` fails because signatures are missing (our repo is private /
unsigned), set `ATO_MCP_INSECURE=1` before running it — we skip
verification when no public key is bundled and no `.minisig` is
published. (This is the default in v0.1.)

If the machine is offline, copy a pre-populated
`$XDG_DATA_HOME/ato-mcp/` from another install and run
`ato-mcp doctor` to verify.

---

## Data location

All persistent state lives under the data directory:

```
$XDG_DATA_HOME/ato-mcp/                  (Linux default: ~/.local/share/ato-mcp/)
├── live/
│   ├── ato.db                           # SQLite: documents + chunks + FTS + vectors
│   ├── model.onnx -> model_quantized.onnx
│   ├── model_quantized.onnx             # 309 MB
│   ├── model_quantized.onnx_data
│   ├── tokenizer.json
│   └── packs/pack-<sha8>.bin.zst        # immutable, content-addressable
├── installed_manifest.json              # last verified manifest
├── backups/ato.db.prev                  # 1 generation retained for rollback
├── staging/                             # transient during update
└── LOCK
```

Override the location with `ATO_MCP_DATA_DIR=/path/to/dir`. Windows
defaults to `%LOCALAPPDATA%\ato-mcp\`.

---

## Architecture at a glance

```
┌──────────────────┐   stdio    ┌────────────────────────────┐
│  Claude Code /   │◀──────────▶│  ato-mcp serve             │
│  Desktop / etc.  │   MCP      │  (FastMCP, 10 tools)       │
└──────────────────┘            └──────────────┬─────────────┘
                                               │
                                      ┌────────▼────────┐
                                      │  ato.db         │  single SQLite file:
                                      │  ├─ documents   │  86k rows
                                      │  ├─ chunks      │  ~700k zstd blobs
                                      │  ├─ chunks_fts  │  FTS5 (BM25)
                                      │  ├─ chunks_vec  │  sqlite-vec int8×256
                                      │  └─ title_fts   │  FTS5 over titles
                                      └─────────────────┘
                                               ▲
                                      EmbeddingGemma 300M
                                      ONNX int8, Matryoshka 256
```

### Hybrid search

Each `search` call:
1. FTS5 top-100 by BM25.
2. sqlite-vec top-100 by cosine against the EmbeddingGemma query embedding.
3. Reciprocal Rank Fusion (k=60) merges the two lists.
4. Optional recency boost on `pub_date`.
5. Deduplicate by document, return the top k.

### Updates

Release artifacts:

```
manifest.json                           # signed; lists every doc + pack
packs/pack-<sha8>.bin.zst                # immutable content-addressable packs
model/embeddinggemma-<sha8>.onnx.zst     # rarely changes
```

`ato-mcp update`:
1. Fetch + verify new manifest.
2. Diff `content_hash` per `doc_id` against the installed manifest.
3. Group changed doc_ids by pack; issue HTTP/2 range requests for just
   the needed bytes.
4. In one SQLite transaction: delete removed + changed docs, insert new
   versions. FTS5 and sqlite-vec rows propagate in the same transaction.
5. Write `installed_manifest.json` last. A crash before this step leaves
   the transaction-complete DB ready for the next `update` to re-apply
   idempotently.

---

## Maintainer workflow (updating the release)

Only needed if you are publishing new index releases. End users never run
these.

```bash
# 1. Scrape. Incremental mode pulls the ATO "What's New" feed + refreshes
#    matching payloads. Full mode re-crawls everything (hours).
ato-mcp refresh-source --mode incremental --output-dir ./ato_pages

# 2. Build the index. GPU recommended for full rebuilds.
LD_LIBRARY_PATH=$(find .venv/lib64/python3.14/site-packages/nvidia/ -maxdepth 2 -name "lib" -type d | tr '\n' ':') \
ato-mcp build-index \
  --pages-dir ./ato_pages \
  --out-dir   ./release \
  --db-path   ./release/ato.db \
  --model-path ./models/embeddinggemma/onnx/model_quantized.onnx \
  --tokenizer-path ./models/embeddinggemma/tokenizer.json \
  --previous-manifest ./release-prev/manifest.json \
  --gpu

# 3. Publish to a GitHub release.
ato-mcp release \
  --out-dir ./release \
  --tag index-2026.04.18 \
  --repo gunba/ato-mcp \
  --sign-key ~/.minisign/ato-mcp.key   # optional
```

Release artifacts are uploaded via `gh release upload`. Manifest URLs are
rewritten to absolute `https://github.com/.../releases/download/<tag>/<file>`
so clients can resolve them.

### Hardware notes

Built and tested on an RTX 4070 Ti + 24-core CPU. Full 86k-doc build
takes ~8 hours CPU-only, well under an hour on a mid-range modern GPU.
Per-doc cost is dominated by HTML parsing + DB inserts once embeddings
are parallelised.

---

## Status

**v0.1 (initial release)** — 10 MCP tools, hybrid search, document-level
delta updates, crash-safe apply, full maintainer pipeline. Known
trade-offs:

- Edited Private Advice payloads in the current scrape are empty shells
  (source-side). Docs still index by title but carry `has_content=false`
  and no chunks. Will improve once the scraper learns to follow the
  JS-rendered EPA pages.
- sqlite-vec is still brute-force; fine at 86k × 256-dim but plan to
  switch on the forthcoming ANN index at >500k chunks.
- Signature verification is opt-in (`pip install 'ato-mcp[verify]'`).
  v0.2 will bundle a maintainer public key and sign every release.

See [the plan](/home/jordan/.claude/plans/it-s-been-a-long-dreamy-hare.md)
for the full design document.

---

## License

MIT — see [LICENSE](LICENSE).

The EmbeddingGemma 300M weights are redistributed under the Gemma Terms
of Use. ATO content is Crown copyright, reused here for factual research
purposes consistent with the ATO's publication terms.
