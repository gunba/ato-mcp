"""HTTP fetch helpers with streaming downloads and sha256 verification."""
from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

CHUNK_BYTES = 1 << 16  # 64 KB


def fetch_url(client: httpx.Client, url: str, dest: Path) -> None:
    """Download ``url`` into ``dest`` using plain HTTP(S).

    This helper intentionally does not read GitHub token environment
    variables and does not shell out to ``gh``. Private release assets should
    be exposed through an approved mirror or installed from a local/offline
    bundle.
    """
    # [UM-04] No GitHub token env vars, no `gh` shell-out — keeps end-user runtime credential-free; private assets must use an approved mirror or local bundle.
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    _httpx_stream_download(client, url, dest)


def _httpx_stream_download(client: httpx.Client, url: str, dest: Path) -> None:
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        with open(dest, "wb") as fh:
            for data in response.iter_bytes(CHUNK_BYTES):
                fh.write(data)


def verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while data := fh.read(CHUNK_BYTES):
            h.update(data)
    if h.hexdigest() != expected:
        raise ValueError(f"sha256 mismatch for {path}: got {h.hexdigest()} expected {expected}")


def make_client() -> httpx.Client:
    """Return a reusable httpx client with http/2 + bounded timeouts."""
    # [UM-03] httpx + http/2 + 64 KB streaming chunks; bounded timeouts (connect=10s, read=60s, write=60s, pool=60s); sha256 verify in verify_sha256 before any file is considered downloaded.
    return httpx.Client(
        http2=True,
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0),
        follow_redirects=True,
    )
