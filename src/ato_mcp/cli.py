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
    """Apply an incremental delta from a new release."""
    from .updater.apply import apply_update

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


def _bundled_pubkey_path() -> Path | None:
    candidate = Path(__file__).parent / "keys" / "maintainer.pub"
    return candidate if candidate.exists() else None
