# Main PC runbook — backfilling `first_published_date` and `human_title`

This runbook is for the agent running on the main PC (the one that holds the
source `ato_pages/` corpus and the 4070 GPU used for embeddings). It covers
the two data-quality gaps the upstream rework introduced and left as
follow-up work on this machine:

1. **`first_published_date`** — new column, populated by a best-effort
   heuristic. Needs to be filled in for the existing corpus.
2. **`human_title`** — new column, seeded with a trivial heading
   concatenation. The long-term algorithm lives on the main PC and will be
   iterated from here; the schema + pipeline hooks are ready.

Both columns are exposed by the existing tools (search, whats_new,
get_document). Neither tool breaks if the column is `NULL`; downstream is
already `NULL`-safe.

## What pulling the update gets you for free

- The DB schema is now `v3`. `store.db.init_db` runs `migrate()` on every
  open, which `ALTER TABLE`-adds `first_published_date` and `human_title`
  to existing `documents` tables. **No manual migration step is required** —
  the next time any `ato-mcp` command touches the DB, the columns appear.
- New documents ingested by `ato-mcp refresh` + `ato-mcp build-index` will
  have both columns populated automatically.
- Existing documents stay `NULL` until you backfill them.

## Scope of work on the main PC

### 1. Regenerate the corpus (preferred path)

If a full rebuild is already on the roadmap (which it is — for human-title
iteration), just re-run the normal pipeline against the live corpus. Both
columns populate as a side effect:

```bash
ato-mcp refresh --mode full          # optional if ato_pages/ is fresh
ato-mcp build-index \
    --pages-dir ./ato_pages \
    --out-dir ./release \
    --db-path /path/to/live/ato.db \
    --model-id <current-model> \
    --model-path <path/to/model.onnx> \
    --tokenizer-path <path/to/tokenizer.json>
```

This goes through `indexer/build.py`, which calls:

- `metadata.extract_first_published_date(markdown, canonical_id, pub_date)`
  — layered heuristic (title-band date → body `pub_date` → year from docid
  → modal year across body → `None`).
- `metadata.compose_human_title(headings)` — joins the extracted
  `<h1>/<h2>/…` headings with ` — `, de-dupes consecutive repeats, strips
  internal whitespace. **This is the seed algorithm; iterate here.**

Both live in `src/ato_mcp/indexer/metadata.py`. Change them in-place, re-run
the build, and every row gets the new values.

### 2. Iterating the human-title algorithm

The placeholder algorithm is deliberately dumb. The main PC holds the
full corpus and is the natural place to try richer variants:

- Weighted selection (top-N headings, skipping boilerplate).
- Combining with `docid_code` (e.g. `"TR 2024/3 — <headings>"`).
- Ruling-type prefixes (EM, Judgment, PCG draft marker, …).
- Truncation policies for very long docs.

Workflow for each iteration:

1. Edit `compose_human_title` in `src/ato_mcp/indexer/metadata.py` (or
   factor it out into a dedicated module when it outgrows one function).
2. Re-run `ato-mcp build-index` against the existing `ato_pages/`.
3. Spot-check a sample: `ato-mcp serve`, then call `get_document` or
   inspect `SELECT doc_id, human_title FROM documents LIMIT 50`.
4. Commit and publish the updated release from the main PC as usual.

The `title` column is unchanged by this work — it still holds the
extractor's single-`<title>` output. `human_title` is the new
consumer-facing field. Feel free to add FTS indexing on it when the
algorithm stabilises (add it to the `title_fts` virtual table column list
and repopulate — a schema v4 when that happens).

### 3. Cheap backfill without re-embedding (if needed)

If a full rebuild is ever too costly (unchanged embeddings, just wanting
to populate the new columns), a standalone one-pass UPDATE script is
viable and fast (minutes, not hours):

```python
# sketch — not yet wired as an `ato-mcp` subcommand. Add if useful.
import sqlite3, zstandard
from ato_mcp.store import db as store_db
from ato_mcp.indexer import metadata as meta

conn = store_db.connect("/path/to/ato.db", mode="rw")
dctx = zstandard.ZstdDecompressor()
for row in conn.execute(
    "SELECT doc_id, canonical_id, pub_date "
    "FROM documents "
    "WHERE first_published_date IS NULL OR human_title IS NULL"
).fetchall():
    doc_id = row["doc_id"]
    chunks = conn.execute(
        "SELECT heading_path, text FROM chunks WHERE doc_id = ? ORDER BY ord",
        (doc_id,),
    ).fetchall()
    markdown = "\n\n".join(dctx.decompress(c["text"]).decode("utf-8") for c in chunks)
    # Reconstruct an approximate headings list from stored heading_paths.
    headings = list(dict.fromkeys(
        (c["heading_path"] or "").split(" › ")[-1]
        for c in chunks if c["heading_path"]
    ))
    first_pub = meta.extract_first_published_date(markdown, row["canonical_id"], row["pub_date"])
    human = meta.compose_human_title(headings)
    conn.execute(
        "UPDATE documents SET first_published_date = COALESCE(?, first_published_date), "
        "human_title = COALESCE(?, human_title) WHERE doc_id = ?",
        (first_pub, human, doc_id),
    )
conn.commit()
```

Promote this to `ato-mcp backfill --fields date,human_title` when it earns
its keep.

## Contract with the dev-environment agent

The dev-environment agent (on the Windows laptop) owns:

- The `ato-mcp` tool surface (search, whats_new, get_document, …).
- The heuristic in `extract_first_published_date`.
- The seed in `compose_human_title`.
- Schema migrations and SQL-level plumbing for new columns.

The main PC agent owns:

- The source corpus on disk (`ato_pages/` and prior releases).
- The actual content of `human_title` (iterating the algorithm).
- Build invocations (`ato-mcp build-index`) and release publication.
- Any standalone backfill / one-off SQL that doesn't belong as a
  long-lived subcommand.

When the human-title algorithm stabilises into something shaped like a
real module (data files, learned rules, LLM calls on the 4070, etc.),
push a short PR that folds it back into the shared codebase so the dev
agent can reason about it. Until then, changes here are expected to stay
on the main PC.

## Verification

After a re-run, sanity-check with:

```sql
SELECT
  SUM(first_published_date IS NOT NULL) AS has_date,
  SUM(human_title          IS NOT NULL) AS has_title,
  COUNT(*)                              AS total
FROM documents;
```

All three numbers should be close. Non-`has_content` rows legitimately
have neither — that's expected.

Spot-check via the serve tools:

```bash
ato-mcp serve   # in one terminal
# then from an MCP client:
#   whats_new(limit=10)        -> newest published, snippets show the date
#   get_document(<doc_id>)     -> outline/markdown; human_title visible
#   search("...",
#          category_scope="Public_*")
```

The `whats_new` result order is the cheapest smoke test: post-backfill,
ordering should reflect publication dates (2025 > 2024 > …), not ingest
timestamps.
