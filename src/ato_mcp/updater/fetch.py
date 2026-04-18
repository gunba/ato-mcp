"""HTTP fetch helpers with range-read + resume + sha256 verification."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

from ..util.log import get_logger

LOGGER = get_logger(__name__)

CHUNK_BYTES = 1 << 16  # 64 KB

_GH_RELEASE_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/releases/download/(?P<tag>[^/]+)/(?P<asset>[^/?#]+)"
)
_GH_LATEST_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/releases/latest/download/(?P<asset>[^/?#]+)"
)


@dataclass
class RangeSpec:
    """Byte range within a pack file."""
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length - 1


def _gh_release_match(url: str) -> re.Match[str] | None:
    m = _GH_RELEASE_RE.match(url)
    if m:
        return m
    latest = _GH_LATEST_RE.match(url)
    if latest:
        if not shutil.which("gh"):
            return None
        tag = _gh_latest_tag(latest["owner"], latest["repo"])
        concrete = (
            f"https://github.com/{latest['owner']}/{latest['repo']}"
            f"/releases/download/{tag}/{latest['asset']}"
        )
        return _GH_RELEASE_RE.match(concrete)
    return None


def _gh_latest_tag(owner: str, repo: str) -> str:
    res = subprocess.run(
        ["gh", "release", "view", "--repo", f"{owner}/{repo}", "--json", "tagName",
         "-q", ".tagName"],
        capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


def fetch_url(client: httpx.Client, url: str, dest: Path) -> None:
    """Download ``url`` into ``dest`` with resume support.

    For ``https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>``
    URLs we prefer ``gh release download`` so private repos work under the
    user's existing ``gh auth login``. Falls back to httpx streaming for any
    other URL or if ``gh`` isn't available.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    m = _gh_release_match(url)
    if m and shutil.which("gh"):
        _gh_download(
            owner=m["owner"], repo=m["repo"], tag=m["tag"], asset=m["asset"],
            dest=dest,
        )
        return
    _httpx_stream_download(client, url, dest)


def _gh_download(*, owner: str, repo: str, tag: str, asset: str, dest: Path) -> None:
    LOGGER.debug("gh release download %s %s -> %s", tag, asset, dest)
    cmd = [
        "gh", "release", "download", tag,
        "--repo", f"{owner}/{repo}",
        "--pattern", asset,
        "--output", str(dest),
        "--clobber",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _httpx_stream_download(client: httpx.Client, url: str, dest: Path) -> None:
    resume_from = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    mode = "ab" if resume_from else "wb"
    with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
        if response.status_code == 416:
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

    For github.com release URLs on a private repo, we cannot issue a direct
    Range request — gh CLI doesn't support partial downloads. In that case we
    download the full asset once (cached next to `dest` based on asset name)
    and slice locally. For any other URL (or public github), we use httpx
    Range requests directly.
    """
    spec_list = list(ranges)
    out: dict[tuple[int, int], bytes] = {}
    m = _gh_release_match(url)
    if m and shutil.which("gh"):
        # Download the full asset into the updater staging cache once.
        from . import fetch as self_mod  # noqa: F401 — keep module import
        from ..util import paths as _paths
        cache = _paths.staging_dir() / f"pack-cache-{m['asset']}"
        if not cache.exists() or cache.stat().st_size == 0:
            _gh_download(
                owner=m["owner"], repo=m["repo"], tag=m["tag"], asset=m["asset"],
                dest=cache,
            )
        with open(cache, "rb") as fh:
            for rng in spec_list:
                fh.seek(rng.start)
                out[(rng.start, rng.length)] = fh.read(rng.length)
        return out

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
