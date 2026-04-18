from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .tree_crawler import SnapshotNode


@dataclass
class SnapshotMeta:
	generated_at: str
	node_count: int
	folder_count: int
	link_count: int
	root_query: str


@dataclass
class SnapshotDiff:
	added: set[str]
	removed: set[str]
	changed: set[str]


class SnapshotWriter:
	def __init__(self, base_dir: Path | str = Path("ingest/output/ato_snapshots")) -> None:
		self.base_dir = Path(base_dir)

	def write(
		self,
		nodes: Sequence[SnapshotNode],
		root_query: str,
		output_dir: Path | str | None = None,
	) -> tuple[Path, SnapshotMeta]:
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
		snapshot_dir = Path(output_dir) if output_dir else self.base_dir
		snapshot_dir.mkdir(parents=True, exist_ok=True)

		nodes_path = snapshot_dir / "nodes.jsonl"
		with nodes_path.open("w", encoding="utf-8") as fh:
			for node in nodes:
				json.dump(node.to_dict(), fh, ensure_ascii=False)
				fh.write("\n")

		meta = SnapshotMeta(
			generated_at=timestamp,
			node_count=len(nodes),
			folder_count=sum(1 for n in nodes if "folder" in n.node_type),
			link_count=sum(1 for n in nodes if "link" in n.node_type),
			root_query=root_query,
		)

		meta_path = snapshot_dir / "meta.json"
		meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

		return snapshot_dir, meta


def diff_snapshots(old_file: Path | str, new_file: Path | str) -> SnapshotDiff:
	old_nodes = _load_node_map(Path(old_file))
	new_nodes = _load_node_map(Path(new_file))

	added_keys = set(new_nodes.keys()) - set(old_nodes.keys())
	removed_keys = set(old_nodes.keys()) - set(new_nodes.keys())
	changed_keys = {
		key
		for key in old_nodes.keys() & new_nodes.keys()
		if old_nodes[key]["title"] != new_nodes[key]["title"]
	}

	return SnapshotDiff(added=added_keys, removed=removed_keys, changed=changed_keys)


def _load_node_map(file_path: Path) -> dict[str, Mapping[str, str]]:
	result: dict[str, Mapping[str, str]] = {}
	with file_path.open("r", encoding="utf-8") as fh:
		for line in fh:
			record = json.loads(line)
			key = record.get("canonical_id") or f"uid:{record['uid']}"
			result[key] = {
				"title": record.get("title", ""),
				"canonical_id": key,
			}
	return result
