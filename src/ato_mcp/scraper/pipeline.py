"""Orchestrator for the ATO scrape pipeline.

Collapses taxiv's ``download_pages`` + ``reduce_snapshot`` + ``run_pipeline`` +
``whatsnew_update`` into a single ``refresh_source(mode, output_dir)`` entry
point. Produces (or updates) the ``output_dir/index.jsonl`` +
``output_dir/payloads/`` layout that the indexer consumes.

Two modes:

- ``incremental`` — pulls the ATO ``What's new`` feed, refreshes matching
  payloads, and drops any new/pending documents under
  ``payloads/<pending_folder>/``.
- ``full`` — runs the whole crawl + reduce + download pipeline. Takes hours;
  intended for monthly full rebuilds.
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from .client import AtoBrowseClient
from .downloader import LinkDownloader
from .reducer import SnapshotReducer
from .snapshot import SnapshotWriter
from .tree_crawler import AtoTreeCrawler
from .whats_new import DedupedLinkIndex, WhatsNewFetcher, build_pending_record

LOGGER = logging.getLogger(__name__)

Mode = Literal["incremental", "full"]


@dataclass
class RefreshResult:
    mode: Mode
    output_dir: Path
    whats_new_summary: dict[str, Any] | None = None
    snapshot_dir: Path | None = None


def refresh_source(
    *,
    mode: Mode = "incremental",
    output_dir: Path | str,
    links_file: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
    base_url: str = "https://www.ato.gov.au",
    whats_new_url: str = "https://www.ato.gov.au/law/view/whatsnew.htm?fid=whatsnew",
    pending_folder: str = "whats_new",
    parser_run_date: str | None = None,
    max_workers: int = 2,
    request_interval: float = 0.1,
    verbose_progress: bool = False,
    force: bool = True,
    root_query: str = "Mode=type&Action=initialise",
    max_nodes: int | None = None,
) -> RefreshResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parser_run_date = parser_run_date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if mode == "incremental":
        if links_file is None:
            links_file = output_dir.parent / "ato_snapshots" / "deduped_links.jsonl"
        links_file = Path(links_file)
        if not links_file.exists():
            raise FileNotFoundError(
                f"deduped_links.jsonl not found at {links_file}. Run a full crawl first."
            )
        summary = _run_whats_new(
            links_file=links_file,
            output_dir=output_dir,
            whats_new_url=whats_new_url,
            base_url=base_url,
            parser_run_date=parser_run_date,
            pending_folder=pending_folder,
            max_workers=max_workers,
            request_interval=request_interval,
            verbose_progress=verbose_progress,
            force=force,
        )
        return RefreshResult(mode="incremental", output_dir=output_dir, whats_new_summary=summary)

    # full mode
    snapshot_base = Path(snapshot_dir) if snapshot_dir else output_dir.parent / "ato_snapshots"
    snapshot_base.mkdir(parents=True, exist_ok=True)

    client = AtoBrowseClient()
    crawler = AtoTreeCrawler(client)
    nodes = crawler.crawl(root_query=root_query, max_nodes=max_nodes)
    writer = SnapshotWriter(base_dir=snapshot_base)
    snap_dir, meta = writer.write(nodes, root_query=root_query, output_dir=snapshot_base)
    LOGGER.info("Crawl complete: %s nodes (%s links)", meta.node_count, meta.link_count)

    reducer = SnapshotReducer(snap_dir / "nodes.jsonl")
    outputs = reducer.run(output_dir=snap_dir)
    links_path = outputs["deduped_links"]

    downloader = LinkDownloader(
        deduped_links_path=links_path,
        output_dir=output_dir,
        base_url=base_url,
        parser_run_date=parser_run_date,
        request_delay=request_interval,
        verbose_progress=verbose_progress,
    )
    downloader.download_all(force=force, max_workers=max_workers)

    summary = _run_whats_new(
        links_file=links_path,
        output_dir=output_dir,
        whats_new_url=whats_new_url,
        base_url=base_url,
        parser_run_date=parser_run_date,
        pending_folder=pending_folder,
        max_workers=max_workers,
        request_interval=request_interval,
        verbose_progress=verbose_progress,
        force=True,
    )
    return RefreshResult(
        mode="full",
        output_dir=output_dir,
        whats_new_summary=summary,
        snapshot_dir=snap_dir,
    )


def _run_whats_new(
    *,
    links_file: Path,
    output_dir: Path,
    whats_new_url: str,
    base_url: str,
    parser_run_date: str,
    pending_folder: str,
    max_workers: int,
    request_interval: float,
    verbose_progress: bool,
    force: bool,
    html_fetcher: Optional[Callable[[str], str]] = None,
    page_fetcher: Optional[Callable[[str], tuple[int, str]]] = None,
    asset_fetcher: Optional[Callable[[str], bytes]] = None,
) -> dict[str, Any]:
    fetcher = WhatsNewFetcher(whats_new_url, base_url=base_url, fetcher=html_fetcher)
    entries = fetcher.fetch_entries()
    dedup_index = DedupedLinkIndex(links_file)

    known, pending = [], []
    for entry in entries:
        match = dedup_index.find(entry.href)
        if match:
            known.append(match)
        else:
            pending.append(build_pending_record(entry, pending_folder))

    summary = {
        "whats_new_url": whats_new_url,
        "total_links": len(entries),
        "refreshed_links": len(known),
        "pending_links": len(pending),
        "run_started_at": datetime.now(timezone.utc).isoformat(),
    }

    if known:
        LOGGER.info("Refreshing %s existing payload(s)", len(known))
        _download_records(
            records=known,
            output_dir=output_dir,
            base_url=base_url,
            parser_run_date=parser_run_date,
            max_workers=max_workers,
            request_interval=request_interval,
            verbose_progress=verbose_progress,
            force=force,
            page_fetcher=page_fetcher,
            asset_fetcher=asset_fetcher,
        )
    if pending:
        LOGGER.info("Writing %s pending document(s) under payloads/%s", len(pending), pending_folder)
        _download_records(
            records=pending,
            output_dir=output_dir,
            base_url=base_url,
            parser_run_date=parser_run_date,
            max_workers=max_workers,
            request_interval=request_interval,
            verbose_progress=verbose_progress,
            force=True,
            page_fetcher=page_fetcher,
            asset_fetcher=asset_fetcher,
        )

    summary["run_completed_at"] = datetime.now(timezone.utc).isoformat()
    summary["processed_ids"] = sorted(
        {record["canonical_id"] for record in (*known, *pending) if record.get("canonical_id")}
    )
    (output_dir / "whats_new_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _download_records(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    base_url: str,
    parser_run_date: str,
    max_workers: int,
    request_interval: float,
    verbose_progress: bool,
    force: bool,
    page_fetcher: Optional[Callable[[str], tuple[int, str]]],
    asset_fetcher: Optional[Callable[[str], bytes]],
) -> None:
    if not records:
        return
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        temp_path = Path(handle.name)
    try:
        downloader = LinkDownloader(
            deduped_links_path=temp_path,
            output_dir=output_dir,
            base_url=base_url,
            parser_run_date=parser_run_date,
            request_delay=request_interval,
            verbose_progress=verbose_progress,
            fetcher=page_fetcher,
            asset_fetcher=asset_fetcher,
        )
        downloader.download_all(force=force, max_workers=max_workers)
    finally:
        temp_path.unlink(missing_ok=True)
