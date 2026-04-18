"""HTTP fetch helpers with range-read + resume + sha256 verification."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

from ..util.log import get_logger

LOGGER = get_logger(__name__)

CHUNK_BYTES = 1 << 16  # 64 KB


@dataclass
class RangeSpec:
    """Byte range within a pack file."""
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length - 1


def fetch_url(client: httpx.Client, url: str, dest: Path) -> None:
    """Download ``url`` into ``dest`` with resume support. Streams to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume_from = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    mode = "ab" if resume_from else "wb"
    with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
        if response.status_code == 416:
            # range not satisfiable -> file is already complete
            return
        response.raise_for_status()
        with open(dest, mode) as fh:
            for data in response.iter_bytes(CHUNK_BYTES):
                fh.write(data)


def verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while data := fh.read(CHUNK_BYTES):
            h.update(data)
    if h.hexdigest() != expected:
        raise ValueError(f"sha256 mismatch for {path}: got {h.hexdigest()} expected {expected}")


def fetch_ranges(
    client: httpx.Client,
    url: str,
    ranges: Iterable[RangeSpec],
) -> dict[tuple[int, int], bytes]:
    """Fetch multiple byte ranges from a single URL.

    Uses a single HTTP/1.1 multipart request when the server supports it.
    Falls back to one request per range. Returns a mapping from
    ``(start, length)`` to the raw bytes for that range.

    Many static hosts (including GitHub release downloads) do **not** honour
    multipart byte ranges — they only return the first range. Our default path
    issues one HTTP/2-multiplexed request per range, which is cheap and
    reliable.
    """
    spec_list = list(ranges)
    out: dict[tuple[int, int], bytes] = {}
    for rng in spec_list:
        headers = {"Range": f"bytes={rng.start}-{rng.end}"}
        response = client.get(url, headers=headers, follow_redirects=True)
        if response.status_code not in (200, 206):
            response.raise_for_status()
        data = response.content
        if len(data) != rng.length and response.status_code == 206:
            raise ValueError(
                f"range response length mismatch: expected {rng.length} got {len(data)}"
            )
        out[(rng.start, rng.length)] = data[: rng.length]
    return out


def make_client() -> httpx.Client:
    """Return a reusable httpx client with http/2 + sane timeouts."""
    return httpx.Client(
        http2=True,
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0),
        follow_redirects=True,
    )
