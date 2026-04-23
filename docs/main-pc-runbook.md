# Main PC runbook — backfilling `first_published_date`, `human_title`, `human_code`

This runbook is for the agent running on the main PC (the one that holds the
source `ato_pages/` corpus and the 4070 GPU used for embeddings). It covers
the three data-quality gaps the v4 rework introduced and left as follow-up
work on this machine:

1. **`first_published_date`** — new column, populated by a best-effort
   heuristic. Needs to be filled in for the existing corpus.
2. **`human_title`** — new column, seeded with a heading concatenation.
   Long-term algorithm lives here and will be iterated.
3. **`human_code`** — new column, **empty until a main-PC parser derives
   the short human citation** (`TR 2013/3` etc.) from the corpus. This is
   where the "human code parser" you planned actually lives.

All three are exposed by the existing tools. Tools don't break if a column
is `NULL`; the downstream layer already tolerates it.

## Identifier model (v4)

Only **two** identifiers for a document, one machine and one human:

- **`doc_id`** — full ATO docid path, prefix and slashes preserved
  (`TXR/TR20133/NAT/ATO/00001`). PRIMARY KEY. Unique. Always present. The
  agent uses this whenever precision matters.
- **`human_code`** — short human citation (`TR 2013/3`). Nullable. May be
  ambiguous across versions / addenda / errata (that's acceptable —
  `doc_id` is the unique handle).

The old `canonical_id` and `docid_code` columns are gone; both were
derivable from `doc_id`.

## What pulling the update gets you for free

- The DB schema is now `v4`.
- **A v3 or older DB will refuse to open**: `init_db()` raises with a clear
  rebuild instruction rather than silently migrating (the `doc_id` format
  changed and the old slug is lossy — we can't reconstruct). Pulling is
  not enough; you must rebuild.
- New documents ingested by `ato-mcp build-index` will have
  `first_published_date` and `human_title` populated automatically.
  `human_code` stays `NULL` until the parser you build here writes to it.

## Scope of work on the main PC

### 1. Rebuild the corpus once to land v4

This is unavoidable. The normal pipeline handles two of the three gaps:

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

Side effects:
- Every row gets `doc_id` in the new path form (`TXR/...`).
- `metadata.extract_first_published_date` populates `first_published_date`.
- `metadata.compose_human_title` populates `human_title` from `<h1>/<h2>/…`.
- `human_code` stays `NULL`.

### 2. The `human_code` parser — the real work

Your plan: "for the DOCID `TR20243` find the human header `TR 2024/3`".
This is a docid-prefix-aware regex pass over the path string, not an LLM
job. The parser lives on the main PC because you'll iterate on it as new
ATO docid formats surface.

Rough shape:

```python
# Pseudo-code. Edit and grow on the main PC.
import re, sqlite3
from ato_mcp.store import db as store_db

RULES = [
    # (prefix regex matching the second path segment, substitution giving
    #  the human citation). Applied in order, first match wins.
    (re.compile(r"^TR(\d{4})(\d+)$"),       r"TR \1/\2"),
    (re.compile(r"^GSTR(\d{4})(\d+)$"),     r"GSTR \1/\2"),
    (re.compile(r"^PCG(\d{4})D?(\d+)$"),    r"PCG \1/D\2"),  # distinguish drafts
    (re.compile(r"^CR(\d{4})(\d+)$"),       r"CR \1/\2"),
    # ... iterate as you find ATO format variants in the corpus
]

def human_code_for_doc_id(doc_id: str) -> str | None:
    segments = doc_id.split("/")
    if len(segments) < 2:
        return None
    body = segments[1]  # e.g. "TR20243" from "TXR/TR20243/NAT/ATO/00001"
    for pat, repl in RULES:
        m = pat.match(body)
        if m:
            return pat.sub(repl, body)
    return None

conn = store_db.connect("/path/to/ato.db", mode="rw")
conn.execute("BEGIN")
for row in conn.execute(
    "SELECT doc_id FROM documents WHERE human_code IS NULL"
).fetchall():
    hc = human_code_for_doc_id(row["doc_id"])
    if hc:
        conn.execute(
            "UPDATE documents SET human_code = ? WHERE doc_id = ?",
            (hc, row["doc_id"]),
        )
conn.execute("COMMIT")
```

This is the algorithm you'll iterate. Each pass refines the rules, walk
through the `WHERE human_code IS NULL` residual to find missed formats,
add a rule, rerun.

Promote it to `ato-mcp backfill-human-codes --rules path/to/rules.py`
when it stabilises.

### 3. Iterating the `human_title` heuristic

Currently: `compose_human_title` in `src/ato_mcp/indexer/metadata.py` joins
extracted `<h1>/<h2>/…` headings with ` — ` and dedupes consecutive repeats.

Room for improvement on the main PC:
- Weighted selection (top-N headings, skipping boilerplate).
- Combining with `human_code` once it's populated
  (e.g. `"TR 2024/3 — <headings>"`).
- Doc-type-specific templates (EM, Judgment, PCG draft markers, …).

Iteration loop:
1. Edit `compose_human_title`.
2. Re-run `ato-mcp build-index` (full rebuild, since content hash may not
   flip just because the heuristic changed).
3. Spot-check: `SELECT doc_id, human_title FROM documents LIMIT 50`.

### 4. Cheap backfill without re-embedding (date / human_title only)

If a full rebuild feels heavy for iterating on dates / human titles alone,
a standalone UPDATE pass is fast (minutes). Don't use it for the v3→v4
rebuild — only for follow-on tweaks.

```python
# sketch — promote to `ato-mcp backfill --fields date,human_title` if useful.
import sqlite3, zstandard
from ato_mcp.store import db as store_db
from ato_mcp.indexer import metadata as meta

conn = store_db.connect("/path/to/ato.db", mode="rw")
dctx = zstandard.ZstdDecompressor()
for row in conn.execute(
    "SELECT doc_id, pub_date "
    "FROM documents "
    "WHERE first_published_date IS NULL OR human_title IS NULL"
).fetchall():
    doc_id = row["doc_id"]
    chunks = conn.execute(
        "SELECT heading_path, text FROM chunks WHERE doc_id = ? ORDER BY ord",
        (doc_id,),
    ).fetchall()
    markdown = "\n\n".join(dctx.decompress(c["text"]).decode("utf-8") for c in chunks)
    headings = list(dict.fromkeys(
        (c["heading_path"] or "").split(" › ")[-1]
        for c in chunks if c["heading_path"]
    ))
    canonical = f"/law/view/document?docid={doc_id}"
    first_pub = meta.extract_first_published_date(markdown, canonical, row["pub_date"])
    human = meta.compose_human_title(headings)
    conn.execute(
        "UPDATE documents SET first_published_date = COALESCE(?, first_published_date), "
        "human_title = COALESCE(?, human_title) WHERE doc_id = ?",
        (first_pub, human, doc_id),
    )
conn.commit()
```

## Contract with the dev-environment agent

The dev-environment agent (on the Windows laptop) owns:
- The `ato-mcp` tool surface (search, whats_new, get_document, …).
- Schema + migrations + SQL plumbing.
- Seed heuristics in `metadata.py` (date, human_title).

The main PC agent owns:
- The source corpus (`ato_pages/` and prior releases).
- The `human_code` parser rules (new work).
- Iteration on `compose_human_title`.
- Build invocations (`ato-mcp build-index`) and release publication.
- Standalone backfills / one-off SQL.

When the human_code parser stabilises — or the title algorithm gets
non-trivial — push a short PR so the shared codebase can reason about it.
Until then, changes stay on the main PC.

## Verification

```sql
SELECT
  SUM(first_published_date IS NOT NULL) AS has_date,
  SUM(human_title          IS NOT NULL) AS has_title,
  SUM(human_code           IS NOT NULL) AS has_code,
  COUNT(*)                              AS total
FROM documents;
```

Expected post-rebuild: `has_date` and `has_title` close to `total`;
`has_code` starts near zero and grows as you iterate the parser. Non-
`has_content` rows legitimately have none of the three.

Smoke test via tools:

```bash
ato-mcp serve   # in one terminal, from an MCP client:
#   whats_new(limit=10)                      -> ordered by first_published_date
#   search("eligibility",
#          doc_scope="TR 2024/3")            -> scopes via human_code (no slash)
#   search("eligibility",
#          doc_scope="TXR/TR20243/*")        -> scopes via doc_id path (has slash)
#   get_document("TXR/TR20243/NAT/ATO/00001") -> full document
```

Post-backfill `whats_new` ordering should reflect publication dates, not
ingest timestamps. Post-`human_code` backfill, results should show
citation strings in the `human_code` field; before then they'll be `NULL`
and the tools fall back to rendering `doc_id`.
