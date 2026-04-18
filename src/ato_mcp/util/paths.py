"""XDG-aware path resolution for the ato-mcp data directory."""
from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "ato-mcp"


def data_dir() -> Path:
    """Return the configured data directory, creating it if needed.

    Resolution order: $ATO_MCP_DATA_DIR, $XDG_DATA_HOME/ato-mcp, platformdirs default.
    """
    override = os.environ.get("ATO_MCP_DATA_DIR")
    if override:
        path = Path(override).expanduser()
    else:
        path = Path(user_data_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_dir() -> Path:
    p = data_dir() / "live"
    p.mkdir(parents=True, exist_ok=True)
    return p


def packs_dir() -> Path:
    p = live_dir() / "packs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def staging_dir() -> Path:
    p = data_dir() / "staging"
    p.mkdir(parents=True, exist_ok=True)
    return p


def backups_dir() -> Path:
    p = data_dir() / "backups"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return live_dir() / "ato.db"


def model_path() -> Path:
    return live_dir() / "model.onnx"


def tokenizer_path() -> Path:
    return live_dir() / "tokenizer.json"


def installed_manifest_path() -> Path:
    return data_dir() / "installed_manifest.json"


def local_manifest_path() -> Path:
    return data_dir() / "manifest.json.local"


def lock_path() -> Path:
    return data_dir() / "LOCK"


def releases_url() -> str:
    """Base URL for release artifacts. Override via ATO_MCP_RELEASES_URL for staging."""
    return os.environ.get(
        "ATO_MCP_RELEASES_URL",
        "https://github.com/gunba/ato-mcp/releases/latest/download",
    )
