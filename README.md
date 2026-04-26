# ato-mcp

Standalone MCP server for local search and retrieval over the Australian
Taxation Office legal corpus.

The installed server is a Rust binary. End users do not need Python, pip,
pipx, uv, a compiler, `gh`, or an API key. The corpus is shipped as GitHub
release assets and installed into the user's local data directory.

## Tools

| Tool | Purpose |
|---|---|
| `search` | BM25 search over the GPU-built corpus. Defaults exclude Edited Private Advice and very old non-legislation content. |
| `search_titles` | Fast citation/title lookup, for example `TR 2024/3` or `Income Tax Assessment Act 1997 s 8-1`. |
| `get_document` | Fetch an outline, a full document, a section, or an ordinal range. |
| `get_chunks` | Fetch exact chunks returned by `search`. |
| `whats_new` | Recent documents by corpus date. |
| `stats` | Index version, counts, and default search policy. |

Every result includes the ATO `canonical_url`.

## Install

Download the binary for your platform from the latest release:

- Linux x64: `ato-mcp-x86_64-unknown-linux-gnu.tar.gz`
- macOS Apple Silicon: `ato-mcp-aarch64-apple-darwin.tar.gz`
- Windows x64: `ato-mcp-x86_64-pc-windows-msvc.zip`

Linux example:

```bash
mkdir -p ~/.local/bin
tar -xzf ato-mcp-x86_64-unknown-linux-gnu.tar.gz -C ~/.local/bin ato-mcp
ato-mcp init
ato-mcp doctor
ato-mcp stats
```

Windows: unzip `ato-mcp.exe` into a directory on `%PATH%`, then run:

```powershell
ato-mcp.exe init
ato-mcp.exe doctor
ato-mcp.exe stats
```

`init` downloads `manifest.json`, the embedding model bundle, and the
document packs from the configured release URL. By default that is:

```text
https://github.com/gunba/ato-mcp/releases/latest/download
```

Override with `ATO_MCP_RELEASES_URL` for staging or an internal corporate
mirror. The Rust client intentionally does not read GitHub token
environment variables and does not shell out to `gh`. If release assets are
private, publish them to an authenticated mirror or install from an offline
bundle.

## Wire Into MCP Clients

Claude Code:

```bash
claude mcp add --scope user ato -- ato-mcp serve
claude mcp list
```

Claude Desktop:

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

Cursor, Continue, and other stdio MCP clients use the same command:

```text
ato-mcp serve
```

## Search Defaults

Default search is tuned for current public tax-law work:

- `Edited_private_advice` is excluded unless `types` explicitly includes it.
- Non-legislation documents dated before `2000-01-01` are excluded unless
  `include_old=true`.
- Legislation is not excluded by the old-content rule because current Acts
  often have old commencement dates.

Examples:

```bash
ato-mcp search "R&D tax incentive eligibility" --k 5
ato-mcp search-titles "TR 2024 3"
ato-mcp search "royalties withholding old cases" --include-old --types Cases
```

## Updates

Run weekly, or whenever you want the latest published corpus:

```bash
ato-mcp update
ato-mcp doctor
```

The update path diffs the installed manifest against the new manifest,
downloads only changed pack assets, mutates SQLite in one transaction, and
writes `installed_manifest.json` last. If an update fails, the previous
database snapshot is retained:

```bash
ato-mcp doctor --rollback
```

## Data Directory

Override the install location with `ATO_MCP_DATA_DIR`.

```text
Linux:   ~/.local/share/ato-mcp
macOS:   ~/Library/Application Support/ato-mcp
Windows: %APPDATA%\ato-mcp or the platform data directory
```

Layout:

```text
ato-mcp/
├── live/
│   ├── ato.db
│   ├── model.onnx -> model_quantized.onnx
│   ├── model_quantized.onnx
│   ├── model_quantized.onnx_data
│   └── tokenizer.json
├── installed_manifest.json
├── backups/ato.db.prev
├── staging/
└── LOCK
```

## Maintainer Workflow

The Rust binary is the end-user product. Python remains maintainer tooling
for scraping, metadata extraction, GPU-backed embedding generation, pack
building, and release publication.

Local GPU release build:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

LD_LIBRARY_PATH="$(find .venv/lib*/python3.*/site-packages/nvidia/ -maxdepth 2 -name lib -type d | tr '\n' ':')$LD_LIBRARY_PATH" \
  .venv/bin/ato-mcp build-index \
  --pages-dir /home/jordan/Desktop/Projects/ato_pages \
  --out-dir ./release \
  --db-path ./release/ato.db \
  --model-path ./models/embeddinggemma/onnx/model_quantized.onnx \
  --tokenizer-path ./models/embeddinggemma/tokenizer.json \
  --gpu

.venv/bin/ato-mcp release \
  --out-dir ./release \
  --tag v0.3.0 \
  --repo gunba/ato-mcp \
  --model-dir ./models/embeddinggemma \
  --overwrite
```

The maintainer build must use GPU-backed embeddings. The optional
`corpus release (gpu)` workflow targets a self-hosted runner labelled
`gpu` and fails if `nvidia-smi` or ONNX Runtime's `CUDAExecutionProvider`
is unavailable. It is not scheduled by default, so it does not spend hosted
GPU minutes.

## Development

```bash
cargo test --locked
.venv/bin/pytest -q
```

CI runs both the Rust binary checks and the Python maintainer test suite.
Release binary assets are produced by `.github/workflows/release-binaries.yml`.

## License

MIT. ATO content remains subject to the ATO's publication terms. The
EmbeddingGemma model is redistributed under the Gemma Terms of Use.
