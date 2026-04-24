"""``ato-mcp`` CLI entry point.

End-user commands:
    ato-mcp serve                 MCP stdio server (spawned by Claude Code).
    ato-mcp init                  First-run: download manifest + model + packs.
    ato-mcp update                Pull delta against current install.
    ato-mcp doctor                Verify DB + file hashes.
    ato-mcp doctor --rollback     Restore previous snapshot.
    ato-mcp stats                 Current index version + counts.

Maintainer commands:
    ato-mcp refresh-source ...    Scrape (incremental | full) into ato_pages/.
    ato-mcp build-index ...       Produce ato.db + packs + manifest.json.
    ato-mcp release ...           Upload release artifacts (stub).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import tools as tool_module
from .util import paths
from .util.log import get_logger

LOGGER = get_logger("ato_mcp.cli")

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.command()
def serve() -> None:
    """Run the MCP stdio server."""
    from .server import run
    run()


@app.command()
def init(
    manifest_url: Optional[str] = typer.Option(
        None, help="Manifest URL. Defaults to $ATO_MCP_RELEASES_URL/manifest.json."
    ),
) -> None:
    """First-run: download manifest + model + required packs."""
    from .updater.apply import apply_update

    _maybe_migrate_v4_to_v5()

    manifest_url = manifest_url or f"{paths.releases_url().rstrip('/')}/manifest.json"
    sig_url = manifest_url + ".minisig"
    pubkey = _bundled_pubkey_path()
    stats = apply_update(
        manifest_url=manifest_url,
        sig_url=sig_url if pubkey and pubkey.exists() else None,
        pubkey_path=pubkey,
    )
    typer.echo(
        f"init complete: +{stats.added} ~{stats.changed} -{stats.removed} "
        f"({stats.bytes_downloaded / 1_000_000:.1f} MB downloaded)"
    )


@app.command()
def update(
    manifest_url: Optional[str] = typer.Option(None),
) -> None:
    """Apply an incremental delta from a new release.

    If the live DB is still on schema v4 (pre-v5), migrates it in place
    first so the delta applies against the current schema.
    """
    from .updater.apply import apply_update

    _maybe_migrate_v4_to_v5()

    manifest_url = manifest_url or f"{paths.releases_url().rstrip('/')}/manifest.json"
    sig_url = manifest_url + ".minisig"
    pubkey = _bundled_pubkey_path()
    stats = apply_update(
        manifest_url=manifest_url,
        sig_url=sig_url if pubkey and pubkey.exists() else None,
        pubkey_path=pubkey,
    )
    typer.echo(
        f"update complete: +{stats.added} ~{stats.changed} -{stats.removed} "
        f"({stats.bytes_downloaded / 1_000_000:.2f} MB downloaded)"
    )


@app.command()
def migrate(
    db_path: Optional[Path] = typer.Option(
        None, help="Path to ato.db. Defaults to the live install path."
    ),
) -> None:
    """Upgrade an ato.db from schema v4 to v5 in place.

    Safe to run on an already-v5 DB — it's a no-op. For pre-v4 DBs
    (``canonical_id``/``docid_code`` present) this fails: those need a
    full rebuild from ``ato_pages/``.
    """
    import sqlite3
    from .store.migrate import migrate_v4_to_v5, needs_v4_to_v5

    target = db_path or paths.db_path()
    if not target.exists():
        typer.echo(f"no DB at {target}", err=True)
        raise typer.Exit(code=1)
    probe = sqlite3.connect(str(target))
    probe.row_factory = sqlite3.Row
    try:
        if not needs_v4_to_v5(probe):
            typer.echo(f"{target}: already v5 — nothing to do")
            return
    finally:
        probe.close()
    typer.echo(f"{target}: migrating v4 → v5 ...")
    migrate_v4_to_v5(target)
    typer.echo("migrate: done")


def _maybe_migrate_v4_to_v5() -> None:
    """Auto-migrate the live DB if it's still on v4 before an update fires."""
    import sqlite3
    from .store.migrate import migrate_v4_to_v5, needs_v4_to_v5

    live = paths.db_path()
    if not live.exists():
        return
    probe = sqlite3.connect(str(live))
    probe.row_factory = sqlite3.Row
    try:
        if not needs_v4_to_v5(probe):
            return
    finally:
        probe.close()
    typer.echo(f"detected pre-v5 DB at {live}; migrating in place ...")
    migrate_v4_to_v5(live)
    typer.echo("migrate: done (continuing with update)")


@app.command()
def doctor(
    rollback: bool = typer.Option(False, help="Restore previous DB snapshot."),
) -> None:
    """Verify installed artifacts, optionally rolling back a bad update."""
    from .updater.apply import rollback as do_rollback

    if rollback:
        do_rollback()
        typer.echo("rollback complete.")
        return

    live_db = paths.db_path()
    if not live_db.exists():
        typer.echo("no live DB found; run `ato-mcp init` first.", err=True)
        raise typer.Exit(code=1)
    from .store import db as store_db
    conn = store_db.connect(live_db, mode="ro")
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        typer.echo(f"documents: {row['n']}")
        row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        typer.echo(f"chunks: {row['n']}")
    finally:
        conn.close()


@app.command()
def stats() -> None:
    """Print index version and counts."""
    typer.echo(tool_module.stats(format="markdown"))


@app.command("refresh-source")
def refresh_source(
    mode: str = typer.Option("incremental", help="incremental | full | catch_up"),
    output_dir: Path = typer.Option(Path("./ato_pages"), help="Destination for payloads/."),
    links_file: Optional[Path] = typer.Option(None, help="deduped_links.jsonl for incremental mode."),
    max_workers: int = typer.Option(1, help="Parallel request workers. Keep low to be polite."),
    request_interval: float = typer.Option(
        0.5,
        help="Minimum seconds between HTTP request starts, globally across workers. "
             "Default 0.5 s = ~2 req/sec. Drop to 1.0 for a slower/safer rate.",
    ),
    verbose: bool = typer.Option(False, help="Emit downloader status snapshots."),
    root_query: str = typer.Option(
        "Mode=type&Action=initialise",
        help="Tree root. Override to scope catch_up to a subtree.",
    ),
    max_nodes: Optional[int] = typer.Option(None, help="Cap for debugging."),
) -> None:
    """Maintainer: scrape the ATO site into ``ato_pages/``."""
    from .scraper import refresh_source as run_refresh

    result = run_refresh(
        mode=mode,  # type: ignore[arg-type]
        output_dir=output_dir,
        links_file=links_file,
        max_workers=max_workers,
        request_interval=request_interval,
        verbose_progress=verbose,
        root_query=root_query,
        max_nodes=max_nodes,
    )
    typer.echo(f"refresh-source complete: mode={result.mode} output={result.output_dir}")
    if result.catch_up_summary is not None:
        s = result.catch_up_summary
        typer.echo(
            f"catch-up: {s.missing} missing of {s.total_current_links} current "
            f"(existing={s.existing_canonical_ids}); downloaded={s.downloaded}"
        )
        for cat, n in s.by_category.items():
            typer.echo(f"  {n:6d}  {cat}")


@app.command("catch-up")
def catch_up(
    output_dir: Path = typer.Option(..., help="Existing ato_pages/ directory (must contain index.jsonl)."),
    max_workers: int = typer.Option(1, help="Parallel request workers. Keep low to be polite."),
    request_interval: float = typer.Option(
        0.5,
        help="Minimum seconds between HTTP request starts, globally across workers. "
             "Default 0.5 s = ~2 req/sec. Drop to 1.0 for a slower/safer rate.",
    ),
    verbose: bool = typer.Option(False, help="Print downloader status snapshots."),
    root_query: str = typer.Option(
        "Mode=type&Action=initialise",
        help="Tree root. Scope to a subtree for faster runs "
             "(e.g. 'Mode=type&Action=inject&TOC=01%3A%23002%23Public%20rulings').",
    ),
    path_prefix: Optional[str] = typer.Option(
        None,
        help="REQUIRED when --root-query is scoped. Slash-separated ancestor "
             "folders from the absolute root down to the scope, e.g. "
             "'Public_rulings/Rulings/Class'. Omit for a full-tree crawl.",
    ),
    max_nodes: Optional[int] = typer.Option(None, help="Cap nodes crawled (debugging)."),
) -> None:
    """Crawl the ATO tree, diff against the existing index, and download only the
    missing documents. Each new doc is placed into its proper category folder
    automatically via the reducer's representative_path.

    Defaults are polite (1 worker, 1.0 s between requests = ~1 req/sec).
    A full-tree catch-up takes hours at these rates — scope with
    ``--root-query`` + ``--path-prefix`` when you only need recent docs.

    Full catch-up:
        ato-mcp catch-up --output-dir ./ato_pages

    Scoped catch-up — must supply path_prefix so paths line up:
        ato-mcp catch-up --output-dir ./ato_pages \\
          --root-query 'Mode=type&Action=inject&TOC=03%3APublic%20rulings%3ARulings%3A%23011%23Class' \\
          --path-prefix 'Public_rulings/Rulings/Class'
    """
    from .scraper import refresh_source as run_refresh

    prefix = [p for p in (path_prefix or "").split("/") if p] or None
    result = run_refresh(
        mode="catch_up",
        output_dir=output_dir,
        max_workers=max_workers,
        request_interval=request_interval,
        verbose_progress=verbose,
        root_query=root_query,
        max_nodes=max_nodes,
        path_prefix=prefix,
    )
    s = result.catch_up_summary
    assert s is not None
    typer.echo(
        f"catch-up: {s.missing} missing of {s.total_current_links} current "
        f"(existing={s.existing_canonical_ids}); downloaded={s.downloaded}"
    )
    for cat, n in s.by_category.items():
        typer.echo(f"  {n:6d}  {cat}")
    typer.echo(f"diff_file: {s.diff_file}")


@app.command("build-index")
def build_index(
    pages_dir: Path = typer.Option(..., help="Directory produced by refresh-source (contains index.jsonl)."),
    out_dir: Path = typer.Option(Path("./release"), help="Where to write manifest.json + packs/."),
    db_path: Path = typer.Option(Path("./release/ato.db")),
    model_path: Path = typer.Option(..., help="Path to embeddinggemma ONNX file."),
    tokenizer_path: Path = typer.Option(..., help="Path to tokenizer.json."),
    model_id: str = typer.Option("embeddinggemma-300m-int8-256d"),
    model_url: Optional[str] = typer.Option(None),
    previous_manifest: Optional[Path] = typer.Option(None, help="Previous manifest for incremental reuse."),
    limit: Optional[int] = typer.Option(None, help="Cap documents processed (for testing)."),
    encode_batch_size: int = typer.Option(64, help="Embedding batch size. Bump for GPU."),
    gpu: bool = typer.Option(False, "--gpu/--cpu", help="Use CUDAExecutionProvider when available."),
) -> None:
    """Maintainer: build a fresh index + packs + manifest from ``ato_pages/``."""
    from .indexer.build import BuildArgs
    from .indexer.build import build as run_build
    from .store.manifest import sha256_file

    model_sha = sha256_file(model_path)
    model_size = model_path.stat().st_size
    providers: tuple[str, ...] | None = None
    if gpu:
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    args = BuildArgs(
        pages_dir=pages_dir,
        out_dir=out_dir,
        db_path=db_path,
        model_id=model_id,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        model_url=model_url,
        model_sha256=model_sha,
        model_size=model_size,
        previous_manifest=previous_manifest,
        limit=limit,
        encode_batch_size=encode_batch_size,
        providers=providers,
    )
    manifest = run_build(args)
    typer.echo(
        f"build-index complete: {len(manifest.documents)} docs, {len(manifest.packs)} packs"
    )


@app.command()
def release(
    out_dir: Path = typer.Option(..., help="Directory produced by build-index."),
    tag: str = typer.Option(..., help="GitHub release tag, e.g. index-2026.04.18."),
    repo: Optional[str] = typer.Option(None, help="owner/repo; defaults to gh's default."),
    title: Optional[str] = typer.Option(None),
    notes: Optional[str] = typer.Option(None, help="Release notes body; omit for auto-generated."),
    draft: bool = typer.Option(False, help="Create as a draft release."),
    prerelease: bool = typer.Option(False, help="Mark as prerelease."),
    sign_key: Optional[Path] = typer.Option(None, help="minisign secret-key file for signing the manifest."),
    overwrite: bool = typer.Option(False, help="Replace existing assets on the release (gh release upload --clobber)."),
    model_dir: Optional[Path] = typer.Option(
        None, help="Directory holding the embedding ONNX + tokenizer; creates a bundled tar.zst."
    ),
) -> None:
    """Maintainer: upload the build artifacts to a GitHub release.

    Shells out to the local ``gh`` CLI; use ``gh auth login`` beforehand.
    """
    from .indexer.release import ReleaseArgs, publish

    publish(ReleaseArgs(
        out_dir=out_dir,
        tag=tag,
        repo=repo,
        title=title,
        notes=notes,
        draft=draft,
        prerelease=prerelease,
        sign_key=sign_key,
        overwrite=overwrite,
        model_dir=model_dir,
    ))
    typer.echo(f"release {tag} published with manifest + packs")


@app.command("backfill")
def backfill(
    db_path: Optional[Path] = typer.Option(
        None,
        help="Path to the ato.db to update. Defaults to the live install path.",
    ),
    limit: Optional[int] = typer.Option(
        None, help="Process at most N documents (for smoke tests)."
    ),
) -> None:
    """Re-run the rule engine across every document in the live DB.

    Reads each document's existing title + chunk headings + first chunk's
    text (no HTML re-fetch, no GPU), runs
    ``ato_mcp.indexer.rules.derive_metadata``, and writes ``title`` +
    ``date`` back to ``documents`` + ``title_fts``. Iterating rules and
    re-running this takes minutes, not hours.
    """
    from .indexer import rules as rules_mod
    from .store import db as store_db

    target = db_path or paths.db_path()
    if not target.exists():
        typer.echo(f"no DB at {target}", err=True)
        raise typer.Exit(code=1)

    conn = store_db.connect(target, mode="rw")
    try:
        sql = "SELECT doc_id, title, type, date, downloaded_at FROM documents ORDER BY doc_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
        total = len(rows)
        typer.echo(f"backfill: {total} rows")

        # doc_id -> title_fts.rowid map so per-row FTS updates use the integer
        # PK (O(1)) instead of scanning the UNINDEXED doc_id column.
        fts_rowid = {
            r["doc_id"]: r["rowid"]
            for r in conn.execute("SELECT rowid, doc_id FROM title_fts").fetchall()
        }

        import zstandard as zstd_mod
        dctx = zstd_mod.ZstdDecompressor()

        title_updates = 0
        date_updates = 0
        conn.execute("BEGIN")
        try:
            for i, row in enumerate(rows, start=1):
                doc_id = row["doc_id"]
                chunk_rows = conn.execute(
                    "SELECT ord, heading_path, text FROM chunks "
                    "WHERE doc_id = ? ORDER BY ord ASC LIMIT 3",
                    (doc_id,),
                ).fetchall()
                headings: list[str] = []
                body_head = ""
                seen_headings: set[str] = set()
                for cr in chunk_rows:
                    hp = (cr["heading_path"] or "").strip()
                    if hp and hp not in seen_headings:
                        seen_headings.add(hp)
                        # Decompose on both " › " (hierarchy) and " — " (flat
                        # concat in h1) so embedded citations surface.
                        for top in hp.split(" › "):
                            for seg in top.split(" — "):
                                s = seg.strip()
                                if not s or s.startswith("/law/view/"):
                                    continue
                                if s not in headings:
                                    headings.append(s)
                    if not body_head:
                        body_head = dctx.decompress(cr["text"]).decode(
                            "utf-8", errors="replace"
                        )[:3000]

                derived = rules_mod.derive_metadata(rules_mod.RuleInputs(
                    doc_id=doc_id,
                    title=row["title"],
                    headings=tuple(headings),
                    body_head=body_head,
                    category=row["type"],
                    pub_date=row["date"],
                ))

                new_title = derived.title or row["title"]
                new_date = derived.date
                # Last-resort date: downloaded_at truncated (upper bound).
                if not new_date:
                    dl = row["downloaded_at"] or ""
                    if len(dl) >= 10 and dl[:10].count("-") == 2:
                        new_date = dl[:10]

                changes: list[str] = []
                params: list = []
                if new_title and new_title != row["title"]:
                    changes.append("title = ?")
                    params.append(new_title)
                    title_updates += 1
                if new_date and new_date != row["date"]:
                    changes.append("date = ?")
                    params.append(new_date)
                    date_updates += 1
                if changes:
                    params.append(doc_id)
                    conn.execute(
                        f"UPDATE documents SET {', '.join(changes)} WHERE doc_id = ?",
                        params,
                    )
                    rowid = fts_rowid.get(doc_id)
                    if rowid is not None and any(c.startswith("title =") for c in changes):
                        conn.execute(
                            "UPDATE title_fts SET title = ? WHERE rowid = ?",
                            (new_title, rowid),
                        )
                if i % 20000 == 0:
                    typer.echo(f"  processed {i}/{total}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        null_titles = conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE title IS NULL OR title = ''"
        ).fetchone()["n"]
        null_dates = conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE date IS NULL"
        ).fetchone()["n"]
    finally:
        conn.close()

    typer.echo(f"  title: updated={title_updates}, still NULL={null_titles}")
    typer.echo(f"  date:  updated={date_updates},  still NULL={null_dates}")
    typer.echo("backfill: done")


def _bundled_pubkey_path() -> Path | None:
    candidate = Path(__file__).parent / "keys" / "maintainer.pub"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Empty-shell inspection.


shells_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect the empty_shells log (docs the scraper fetched but which "
         "yielded no extractable content).",
)
app.add_typer(shells_app, name="empty-shells")


@shells_app.command("count")
def shells_count(
    db_path: Optional[Path] = typer.Option(
        None, help="Path to ato.db. Defaults to the live install path."
    ),
) -> None:
    """Print total + top-20 doc_id-prefix breakdown."""
    from .store import db as store_db
    target = db_path or paths.db_path()
    conn = store_db.connect(target, mode="ro")
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM empty_shells").fetchone()["n"]
        typer.echo(f"total shells: {total:,}")
        rows = conn.execute(
            """
            SELECT substr(doc_id, 1, instr(doc_id||'/', '/')-1) AS prefix,
                   COUNT(*) AS n
            FROM empty_shells GROUP BY prefix ORDER BY n DESC LIMIT 20
            """
        ).fetchall()
        for r in rows:
            typer.echo(f"  {r['prefix']:<8} {r['n']:>7,}")
    finally:
        conn.close()


@shells_app.command("list")
def shells_list(
    limit: int = typer.Option(20, help="How many shells to show."),
    prefix: Optional[str] = typer.Option(
        None, help="Filter by doc_id prefix (e.g. 'EV', 'JUD')."
    ),
    db_path: Optional[Path] = typer.Option(None),
) -> None:
    """Show the N oldest-unchecked shells (re-probe candidates)."""
    from .store import db as store_db
    target = db_path or paths.db_path()
    conn = store_db.connect(target, mode="ro")
    try:
        sql = (
            "SELECT doc_id, first_seen_at, last_checked_at, check_count, source "
            "FROM empty_shells"
        )
        params: list = []
        if prefix:
            sql += " WHERE doc_id LIKE ?"
            params.append(f"{prefix.rstrip('/')}/%")
        sql += " ORDER BY last_checked_at ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            typer.echo(
                f"{r['doc_id']:<50}  first={r['first_seen_at']}  "
                f"last={r['last_checked_at']}  n={r['check_count']}  "
                f"src={r['source'] or ''}"
            )
    finally:
        conn.close()


@shells_app.command("export")
def shells_export(
    out: Path = typer.Argument(..., help="Destination CSV path."),
    db_path: Optional[Path] = typer.Option(None),
) -> None:
    """Dump the whole table to CSV (doc_id, canonical_url, first_seen_at, last_checked_at, check_count, source)."""
    import csv
    from .store import db as store_db
    from .formatters import canonical_url
    target = db_path or paths.db_path()
    conn = store_db.connect(target, mode="ro")
    try:
        rows = conn.execute(
            "SELECT doc_id, first_seen_at, last_checked_at, check_count, source "
            "FROM empty_shells ORDER BY doc_id"
        ).fetchall()
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["doc_id", "canonical_url", "first_seen_at",
                        "last_checked_at", "check_count", "source"])
            for r in rows:
                w.writerow([
                    r["doc_id"], canonical_url(r["doc_id"]),
                    r["first_seen_at"], r["last_checked_at"],
                    r["check_count"], r["source"] or "",
                ])
        typer.echo(f"wrote {len(rows):,} rows to {out}")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
