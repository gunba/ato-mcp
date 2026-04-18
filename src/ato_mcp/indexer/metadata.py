"""Metadata extraction from ATO canonical IDs + payload HTML.

The canonical_id for every ATO document is a URL fragment of the form
``/law/view/document?docid=<PREFIX>/<CODE>/.../<VERSION>`` where PREFIX is one
of ~40 known document-type codes (TR, GSTR, ATOID, PCG, TA, LCR, PS LA, ...).

We use the prefix as the primary doc_type signal; titles + content provide
human-readable ``docid_code`` ("TR 2024/3") and publication dates when
present.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import yaml

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")
_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_STATUS_BANNER_RE = re.compile(r"\b(withdrawn|superseded|replaced by)\b", re.IGNORECASE)

_DOC_TYPE_MAP: dict[str, dict[str, str]] | None = None


def _load_doc_type_map() -> dict[str, dict[str, str]]:
    global _DOC_TYPE_MAP
    if _DOC_TYPE_MAP is not None:
        return _DOC_TYPE_MAP
    # Try package-resource first; fall back to repo-root data/ during dev.
    data_text: str | None = None
    try:
        files = resources.files("ato_mcp").joinpath("_data/doc_type_map.yaml")
        if files.is_file():
            data_text = files.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        pass
    if data_text is None:
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / "data" / "doc_type_map.yaml"
        if candidate.exists():
            data_text = candidate.read_text(encoding="utf-8")
    if data_text is None:
        _DOC_TYPE_MAP = {}
        return _DOC_TYPE_MAP
    loaded = yaml.safe_load(data_text) or {}
    _DOC_TYPE_MAP = {str(k).upper(): v for k, v in loaded.items()}
    return _DOC_TYPE_MAP


@dataclass
class DocMetadata:
    doc_id: str
    canonical_id: str
    href: str
    category: str
    doc_type: str | None
    docid_code: str | None
    title: str
    pub_date: str | None
    effective_date: str | None
    status: str | None
    has_content: bool
    content_hash: str


def slugify(text: str, fallback: str = "doc", max_len: int = 80) -> str:
    cleaned = _SLUG_RE.sub("_", text.strip()).strip("_").lower()
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


def doc_id_for(canonical_id: str) -> str:
    parsed = urlparse(canonical_id)
    docid_values = parse_qs(parsed.query).get("docid")
    seed = unquote(docid_values[0]) if docid_values else canonical_id
    return slugify(seed)


def category_from_path(payload_path: str | None) -> str:
    if not payload_path:
        return "Unknown"
    parts = Path(payload_path).parts
    if parts and parts[0].lower() in ("payloads",):
        parts = parts[1:]
    return parts[0] if parts else "Unknown"


def parse_docid(canonical_id: str) -> tuple[str | None, str | None]:
    """Return ``(prefix, doc_type_name)``. Prefix is uppercased first segment
    of the docid, e.g. ``TR`` from ``TR/TR20243/NAT/ATO/00001``.
    """
    parsed = urlparse(canonical_id)
    docid_values = parse_qs(parsed.query).get("docid")
    if not docid_values:
        return None, None
    docid = unquote(docid_values[0])
    segments = [s for s in docid.split("/") if s]
    if not segments:
        return None, None
    prefix = segments[0].upper()
    name = _load_doc_type_map().get(prefix, {}).get("name")
    return prefix, name


def extract_pub_date(markdown: str) -> str | None:
    """Best-effort publication-date scrape. Returns ISO yyyy-mm-dd or None."""
    match = _DATE_RE.search(markdown[:2000])
    if not match:
        return None
    day, month_name, year = match.groups()
    month = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }[month_name.lower()]
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def extract_status(markdown: str) -> str | None:
    if _STATUS_BANNER_RE.search(markdown[:1500]):
        match = _STATUS_BANNER_RE.search(markdown[:1500])
        return match.group(1).lower() if match else None
    return None


def content_hash(markdown: str, metadata: dict[str, Any]) -> str:
    """Stable hash of (cleaned markdown + key metadata). Used for delta diffing."""
    h = hashlib.sha256()
    h.update(markdown.encode("utf-8", errors="replace"))
    for key in ("title", "doc_type", "docid_code", "pub_date", "status"):
        value = metadata.get(key)
        if value:
            h.update(b"\0")
            h.update(key.encode("ascii"))
            h.update(b"=")
            h.update(str(value).encode("utf-8"))
    return "sha256:" + h.hexdigest()


_CODE_PATTERNS = (
    re.compile(r"\b((?:TR|GSTR|LCR|SGR|FTR|PR|CR|TD|MT|IT|FBT|SCD|SMSFRB|PSLA|PS LA|PCG|TA|ATOID|LCG|LG|LI)\s?[0-9]{1,4}/[0-9A-Za-z]+)\b"),
)


def extract_docid_code(title: str | None, markdown: str) -> str | None:
    for src in (title or "", markdown[:500]):
        for pat in _CODE_PATTERNS:
            m = pat.search(src)
            if m:
                return m.group(1).upper().replace("  ", " ")
    return None
