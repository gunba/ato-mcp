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

SCHEMA_VERSION = "4"
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
    """Apply in-place schema changes for pre-existing databases.

    ``CREATE TABLE IF NOT EXISTS`` in schema.sql is a no-op once the table
    exists, so structural changes have to be applied here. We detect older
    schemas by inspecting ``pragma_table_info`` rather than trusting the
    stored ``schema_version`` — a DB shipped before the meta key existed
    reports no version, but its column set is unambiguous.

    v4 rewrites the identifier model: ``doc_id`` switches from a slug to
    the full docid path, and ``canonical_id`` + ``docid_code`` are dropped.
    Row content cannot be reconstructed from v3 data (the slug is lossy),
    so v3 → v4 demands a full rebuild. Earlier versions (v1, v2) only
    needed ALTER-TABLE column additions and are handled in-place for
    completeness.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}

    legacy_v1_or_v2 = "canonical_id" in cols or "docid_code" in cols
    if legacy_v1_or_v2:
        raise RuntimeError(
            "This database is pre-v4 (it still has canonical_id / docid_code columns).\n"
            "v4 changed the identifier model and needs a fresh build. Run\n"
            "  ato-mcp build-index ...\n"
            "to rebuild from ato_pages/, or delete ato.db and run `ato-mcp init`."
        )

    # Additive columns on a v4-shaped DB: these are here for forward compatibility
    # if we ever need to extend again without a full rebuild.
    if "first_published_date" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN first_published_date TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_firstpub ON documents(first_published_date)"
        )
    if "human_title" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN human_title TEXT")
    if "human_code" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN human_code TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_human_code ON documents(human_code)"
        )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
