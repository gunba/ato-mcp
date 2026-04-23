"""SQLite connection helpers.

Loads sqlite-vec on every connection, applies schema.sql on first open,
creates the vec0 virtual table once the extension is available.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

import sqlite_vec

from ..util import paths

SCHEMA_VERSION = "3"
EMBEDDING_DIM = 256
EMBEDDING_DTYPE = "int8"

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_VEC_TABLE_DDL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding {EMBEDDING_DTYPE}[{EMBEDDING_DIM}] distance_metric=cosine
);
"""


def _load_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def connect(
    path: Path | None = None,
    *,
    mode: Literal["ro", "rw", "rwc"] = "rwc",
    mmap_bytes: int = 256 * 1024 * 1024,
) -> sqlite3.Connection:
    """Open an ato.db connection with sqlite-vec loaded.

    mode=ro gives a read-only handle (safe for serve). mmap raises page cache hits.
    """
    if path is None:
        path = paths.db_path()
    path = Path(path)
    if mode == "rwc":
        path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{path}?mode={mode}"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    _load_vec(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA mmap_size = {mmap_bytes}")
    conn.execute("PRAGMA temp_store = MEMORY")
    if mode != "ro":
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(path: Path | None = None) -> sqlite3.Connection:
    """Create the DB file (if missing), apply schema, and create the vec0 table."""
    conn = connect(path, mode="rwc")
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(_VEC_TABLE_DDL)
    _migrate(conn)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("schema_version", SCHEMA_VERSION),
    )
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply in-place column additions for pre-existing databases.

    ``CREATE TABLE IF NOT EXISTS`` in schema.sql is a no-op once the table
    exists, so new columns have to be ALTER-ed in. We detect older schemas
    by inspecting ``pragma_table_info`` rather than trusting the stored
    ``schema_version`` — a DB shipped before the meta key existed reports
    no version, but its column set is unambiguous.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "first_published_date" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN first_published_date TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_firstpub ON documents(first_published_date)"
        )
    if "human_title" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN human_title TEXT")


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
