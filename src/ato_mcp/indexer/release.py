"""Release helpers: sign the manifest and upload artifacts to GitHub Releases.

We shell out to ``gh`` rather than use the GitHub API directly so uploads
run under the maintainer's existing GitHub CLI authentication. This keeps
the package dependency-free from API client libraries.

Signing is optional. Pass ``--sign-key`` to produce a ``manifest.json.minisig``
alongside the manifest. Signing requires the ``minisign`` CLI.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd

from ..store.manifest import load_manifest, save_manifest
from ..util.log import get_logger

LOGGER = get_logger(__name__)


class ReleaseError(RuntimeError):
    pass


@dataclass
class ReleaseArgs:
    out_dir: Path                # from build-index
    tag: str                     # e.g. "index-2026.04.18"
    repo: str | None = None      # "owner/repo"; defaults to the repo gh sees
    title: str | None = None     # release title; defaults to tag
    notes: str | None = None
    draft: bool = False
    prerelease: bool = False
    sign_key: Path | None = None
    overwrite: bool = False      # replace existing assets on the release
    model_dir: Path | None = None  # maintainer's ONNX + tokenizer dir
    model_bundle_name: str = "embeddinggemma-bundle.tar.zst"


def bundle_model(
    model_dir: Path,
    out_path: Path,
    *,
    include: tuple[str, ...] = (
        "model_quantized.onnx",
        "model_quantized.onnx_data",
        "tokenizer.json",
    ),
    level: int = 3,
) -> tuple[str, int]:
    """Pack the embedding model + tokenizer into a single ``.tar.zst`` bundle.

    Returns ``(sha256, size_bytes)`` of the produced bundle, which callers
    plug into the manifest's ``ModelInfo`` so clients can verify the
    download.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    # Build the uncompressed tar in memory, then stream through zstd while
    # hashing the result. Bundle is ~310 MB uncompressed — fits easily.
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for name in include:
            # Search both model_dir and model_dir/onnx (onnx-community models
            # ship the weights in an onnx/ subdir while the tokenizer sits at
            # the top level).
            for candidate in (model_dir / name, model_dir / "onnx" / name):
                if candidate.exists():
                    tar.add(str(candidate), arcname=name)
                    break
            else:
                raise FileNotFoundError(f"model bundle missing {name} under {model_dir}")
    tar_buffer.seek(0)
    cctx = zstd.ZstdCompressor(level=level)
    with open(out_path, "wb") as fh:
        with cctx.stream_writer(fh) as writer:
            while chunk := tar_buffer.read(1 << 20):
                writer.write(chunk)
    with open(out_path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest(), out_path.stat().st_size


def _release_asset_url(repo: str, tag: str, filename: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{filename}"


def rewrite_manifest_urls(manifest_path: Path, repo: str, tag: str) -> None:
    """Rewrite pack URLs in the manifest to absolute GH-Release URLs.

    The model URL is set directly by :func:`publish` once the bundle is
    produced, so we don't touch it here.

    ``gh release upload`` flattens assets into a single namespace, so URLs
    we emit must be the absolute download URL for the flattened asset
    name.
    """
    manifest = load_manifest(manifest_path)
    for idx, pack in enumerate(manifest.packs):
        manifest.packs[idx].url = _release_asset_url(repo, tag, Path(pack.url).name)
    save_manifest(manifest, manifest_path)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def _gh_default_repo() -> str:
    """Return ``owner/repo`` reported by ``gh repo view``."""
    res = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


def sign_manifest(manifest_path: Path, sign_key: Path) -> Path:
    """Produce ``manifest.json.minisig`` next to ``manifest_path``.

    Uses the ``minisign`` CLI, which supports secret-key files produced by
    ``minisign -G``.
    """
    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".minisig")
    cli = shutil.which("minisign")
    if cli is None:
        raise ReleaseError("manifest signing requires the `minisign` CLI")
    subprocess.run(
        [cli, "-S", "-s", str(sign_key), "-m", str(manifest_path), "-x", str(sig_path)],
        check=True,
    )
    return sig_path


def publish(args: ReleaseArgs) -> None:
    """Upload manifest + packs (+ optional signature) to a GitHub Release.

    Creates the release if it doesn't already exist; otherwise uploads assets
    onto the existing release. Pass ``overwrite=True`` to replace assets.
    """
    manifest = args.out_dir / "manifest.json"
    packs_dir = args.out_dir / "packs"
    if not manifest.exists():
        raise ReleaseError(f"no manifest at {manifest}")
    if not packs_dir.exists():
        raise ReleaseError(f"no packs/ dir at {packs_dir}")

    pack_files = sorted(packs_dir.glob("pack-*.bin.zst"))
    if not pack_files:
        raise ReleaseError("no pack files found to upload")

    repo = args.repo or _gh_default_repo()

    # Bundle the embedding model + tokenizer if a model_dir was supplied and
    # we haven't already produced the bundle.
    model_bundle_path: Path | None = None
    if args.model_dir is not None:
        model_bundle_path = args.out_dir / args.model_bundle_name
        if not model_bundle_path.exists():
            LOGGER.info("bundling embedding model from %s", args.model_dir)
            sha256, size = bundle_model(args.model_dir, model_bundle_path)
        else:
            sha256 = _file_sha256(model_bundle_path)
            size = model_bundle_path.stat().st_size
        # Patch the manifest with the real bundle metadata.
        current = load_manifest(manifest)
        current.model.sha256 = sha256
        current.model.size = size
        current.model.url = _release_asset_url(repo, args.tag, args.model_bundle_name)
        save_manifest(current, manifest)

    rewrite_manifest_urls(manifest, repo, args.tag)

    artifacts: list[Path] = [manifest, *pack_files]
    if model_bundle_path is not None:
        artifacts.append(model_bundle_path)
    if args.sign_key:
        sig = sign_manifest(manifest, args.sign_key)
        artifacts.insert(1, sig)

    # Ensure the release exists.
    gh_base = ["gh", "release"]
    if args.repo:
        gh_base.extend(["--repo", args.repo])
    view = subprocess.run(
        [*gh_base, "view", args.tag],
        capture_output=True,
        text=True,
    )
    if view.returncode != 0:
        LOGGER.info("creating release %s", args.tag)
        create_cmd = [*gh_base, "create", args.tag, "--title", args.title or args.tag]
        if args.notes is not None:
            create_cmd.extend(["--notes", args.notes])
        else:
            create_cmd.append("--generate-notes")
        if args.draft:
            create_cmd.append("--draft")
        if args.prerelease:
            create_cmd.append("--prerelease")
        subprocess.run(create_cmd, check=True)

    LOGGER.info("uploading %d artifacts to %s", len(artifacts), args.tag)
    upload_cmd = [*gh_base, "upload", args.tag, *[str(p) for p in artifacts]]
    if args.overwrite:
        upload_cmd.append("--clobber")
    subprocess.run(upload_cmd, check=True)
    LOGGER.info("release %s updated (%d assets)", args.tag, len(artifacts))
