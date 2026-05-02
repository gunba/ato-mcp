"""Release helpers: sign the manifest and upload corpus artifacts to GitHub Releases.

We shell out to ``gh`` rather than use the GitHub API directly so uploads
run under the maintainer's existing GitHub CLI authentication. This keeps
the package dependency-free from API client libraries.

EmbeddingGemma is not uploaded to GitHub. By default the manifest records a
pinned Hugging Face model source plus fingerprint/size; maintainers can point
at an approved mirror with ``--model-url``.

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

EMBEDDINGGEMMA_HF_URL = (
    "hf://onnx-community/embeddinggemma-300m-ONNX@"
    "5090578d9565bb06545b4552f76e6bc2c93e4a66"
)
EMBEDDINGGEMMA_HF_FINGERPRINT = (
    "5d4d31914cdb65cd84d3248390946461efdd4ec4f99afd13d23218cd4060d706"
)
EMBEDDINGGEMMA_HF_SIZE = 329_781_810


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
    model_dir: Path | None = None  # maintainer's ONNX + tokenizer dir; bundle is not uploaded to GitHub
    model_url: str | None = None   # external URL for the model bundle
    model_sha256: str | None = None
    model_size: int | None = None
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


def _is_placeholder_model_url(url: str) -> bool:
    return not url or url.startswith("model/") or url.endswith(".onnx.zst")


def _is_github_url(url: str) -> bool:
    return "://github.com/" in url or "://raw.githubusercontent.com/" in url


def _is_hf_url(url: str) -> bool:
    return url.startswith("hf://")


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

    current = load_manifest(manifest)
    if current.model.id.startswith("embeddinggemma"):
        explicit_model_url = args.model_url is not None
        model_url = args.model_url or current.model.url
        if _is_placeholder_model_url(model_url) or (
            _is_github_url(model_url) and not explicit_model_url
        ):
            model_url = EMBEDDINGGEMMA_HF_URL
            sha256 = EMBEDDINGGEMMA_HF_FINGERPRINT
            size = EMBEDDINGGEMMA_HF_SIZE
        elif _is_hf_url(model_url):
            sha256 = args.model_sha256 or current.model.sha256 or EMBEDDINGGEMMA_HF_FINGERPRINT
            size = args.model_size or current.model.size or EMBEDDINGGEMMA_HF_SIZE
        else:
            if _is_github_url(model_url):
                raise ReleaseError("EmbeddingGemma model bundles must not be hosted on GitHub")

            if args.model_dir is not None:
                model_bundle_path = args.out_dir / args.model_bundle_name
                if not model_bundle_path.exists():
                    LOGGER.info("bundling embedding model from %s", args.model_dir)
                    sha256, size = bundle_model(args.model_dir, model_bundle_path)
                else:
                    sha256 = _file_sha256(model_bundle_path)
                    size = model_bundle_path.stat().st_size
                LOGGER.info(
                    "model bundle prepared at %s; upload it to %s before publishing",
                    model_bundle_path,
                    model_url,
                )
            else:
                sha256 = args.model_sha256 or current.model.sha256
                size = args.model_size or current.model.size

        if _is_github_url(model_url):
            raise ReleaseError("EmbeddingGemma model bundles must not be hosted on GitHub")

        if not sha256 or not size:
            raise ReleaseError(
                "EmbeddingGemma releases require model sha256 and size"
            )
        current.model.sha256 = sha256
        current.model.size = size
        current.model.url = model_url
        save_manifest(current, manifest)

    rewrite_manifest_urls(manifest, repo, args.tag)

    artifacts: list[Path] = [manifest, *pack_files]
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
