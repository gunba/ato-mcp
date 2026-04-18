"""Release helpers: sign the manifest and upload artifacts to GitHub Releases.

We shell out to ``gh`` rather than use the GitHub API directly so uploads
run under the maintainer's existing gh auth / GITHUB_TOKEN. This keeps the
package dependency-free from API client libraries.

Signing is optional. Pass ``--sign-key`` to produce a ``manifest.json.minisig``
alongside the manifest. If ``minisign`` (the CLI tool) is installed we use it;
otherwise we fall back to the python ``minisign`` package if available.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


def _release_asset_url(repo: str, tag: str, filename: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{filename}"


def rewrite_manifest_urls(manifest_path: Path, repo: str, tag: str) -> None:
    """Rewrite model + pack URLs in the manifest to absolute GH-Release URLs.

    The build step writes relative paths (``packs/pack-<sha8>.bin.zst``) which
    are convenient locally. ``gh release upload`` flattens assets into a single
    namespace, so URLs we emit must be the absolute download URL for the
    flattened asset name.
    """
    manifest = load_manifest(manifest_path)
    # Model
    model_url = manifest.model.url
    manifest.model.url = _release_asset_url(repo, tag, Path(model_url).name)
    # Packs
    for pack in manifest.packs:
        manifest.packs[manifest.packs.index(pack)].url = _release_asset_url(
            repo, tag, Path(pack.url).name
        )
    save_manifest(manifest, manifest_path)


def _gh_default_repo() -> str:
    """Return ``owner/repo`` reported by ``gh repo view``."""
    res = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


def sign_manifest(manifest_path: Path, sign_key: Path) -> Path:
    """Produce ``manifest.json.minisig`` next to ``manifest_path``.

    Tries the ``minisign`` CLI first (supports secret-key files produced by
    ``minisign -G``); falls back to the python ``minisign`` package.
    """
    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".minisig")
    cli = shutil.which("minisign")
    if cli:
        subprocess.run(
            [cli, "-S", "-s", str(sign_key), "-m", str(manifest_path), "-x", str(sig_path)],
            check=True,
        )
        return sig_path
    try:
        import minisign  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ReleaseError(
            "no minisign CLI and the `minisign` python package is not installed"
        ) from exc
    sk = minisign.SecretKey.from_file(str(sign_key))
    sig = sk.sign_file(str(manifest_path))
    sig_path.write_bytes(bytes(sig))
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
