"""Shared test fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ATO_MCP_DATA_DIR", str(tmp_path))
    yield tmp_path
