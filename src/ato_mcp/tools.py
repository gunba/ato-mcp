"""MCP tool implementations for ato-mcp.

All tools accept ``format`` = ``"markdown"`` (default) or ``"json"``.
Every hit carries ``canonical_url``. Tools are intentionally small and
composable so agents can pipe ``search`` -> ``get_section`` with minimal
token cost.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import zstandard as zstd

from . import formatters
from .embed.model import EmbeddingModel, vec_to_bytes
from .store import db as store_db
from .store.queries import (
    SELECT_CHUNKS_FOR_DOC,
    SELECT_DOCUMENT,
)
from .util import paths
from .util.log import get_logger

LOGGER = get_logger(__name__)

DEFAULT_K = 8
MAX_K = 50
RRF_K = 60
SNIPPET_CHARS = 280


@dataclass
class Backend:
    """Shared embedding model plus per-thread read-only SQLite connections.

    FastMCP dispatches synchronous tool handlers on a thread pool, and
    ``sqlite3`` connections are thread-affine by default (``check_same_thread``).
    A single shared connection would raise ``ProgrammingError`` when the pool
    rotates threads. We instead give each worker thread its own connection via
    ``threading.local``; SQLite handles concurrent readers cleanly. The
    embedding model (ONNX Runtime session + Rust tokenizer) is thread-safe and
    stays shared.
    """
    model: EmbeddingModel | None
    _tls: threading.local

    @classmethod
    def open(cls) -> "Backend":
        model: EmbeddingModel | None = None
        try:
            model = EmbeddingModel()
        except FileNotFoundError:
            LOGGER.warning("embedding model unavailable; vector search disabled")
        return cls(model=model, _tls=threading.local())

    @property
    def db(self) -> sqlite3.Connection:
        tls = self._tls
        conn: sqlite3.Connection | None = getattr(tls, "conn", None)
        try:
            current_mtime = paths.db_path().stat().st_mtime
        except FileNotFoundError:
            current_mtime = 0.0
        # If a concurrent `ato-mcp update` replaced the DB since we last saw it,
        # drop this thread's connection so we reopen against the new file.
        if conn is not None and getattr(tls, "mtime", 0.0) + 0.001 < current_mtime:
            conn.close()
            conn = None
        if conn is None:
            conn = store_db.connect(paths.db_path(), mode="ro")
            tls.conn = conn
            tls.mtime = current_mtime
        return conn

    def close(self) -> None:
        """Close this thread's connection if any. Other threads' connections
        stay open until their thread exits — the stdlib thread-local teardown
        runs the sqlite3 destructor for us."""
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None


_BACKEND: Backend | None = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> Backend:
    """Return the process-wide backend, creating it once under a lock.

    Double-checked locking keeps the fast path lock-free once ``_BACKEND`` is
    set, and prevents the thread-pool TOCTOU race that would otherwise have
    every concurrent first-call load its own copy of the 300 MB ONNX model.
    """
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = Backend.open()
    return _BACKEND


# ---------------------------------------------------------------------------
# Helpers


def _decompress(blob: bytes) -> str:
    return zstd.ZstdDecompressor().decompress(blob).decode("utf-8")


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _highlight_snippet(text: str, query: str, max_chars: int = SNIPPET_CHARS) -> str:
    words = {w.lower() for w in _WORD_RE.findall(query)}
    if not words or not text:
        return text[:max_chars].strip()
    cleaned = re.sub(r"\s+", " ", text)
    lower = cleaned.lower()
    best = -1
    for w in words:
        idx = lower.find(w)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    if best == -1:
        snippet = cleaned[:max_chars]
    else:
        start = max(0, best - max_chars // 3)
        end = start + max_chars
        snippet = cleaned[start:end]
        if start > 0:
            snippet = "…" + snippet
    # Bold query terms
    def _bold(m: re.Match) -> str:
        token = m.group(0)
        if token.lower() in words:
            return f"**{token}**"
        return token
    return _WORD_RE.sub(_bold, snippet).strip()


def _build_sql_filter(
    categories: list[str] | None,
    doc_types: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    doc_scope: str | None = None,
    category_scope: str | None = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if categories:
        placeholders = ",".join("?" * len(categories))
        clauses.append(f"d.category IN ({placeholders})")
        params.extend(categories)
    if doc_types:
        placeholders = ",".join("?" * len(doc_types))
        clauses.append(f"d.doc_type IN ({placeholders})")
        params.extend(doc_types)
    if date_from:
        clauses.append("d.pub_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("d.pub_date <= ?")
        params.append(date_to)
    if doc_scope:
        pattern = _glob_to_like(doc_scope)
        # docid_code is the human form ("TR 2024/3"); doc_id is the slug ("tr_2024_3").
        # Match either so the agent can paste whichever it has.
        clauses.append(r"(d.doc_id LIKE ? ESCAPE '\' OR d.docid_code LIKE ? ESCAPE '\')")
        params.extend([pattern, pattern])
    if category_scope:
        clauses.append(r"d.category LIKE ? ESCAPE '\'")
        params.append(_glob_to_like(category_scope))
    return (" AND ".join(clauses), params) if clauses else ("", params)


_GLOB_ESCAPE = str.maketrans({"\\": r"\\", "%": r"\%", "_": r"\_"})


def _glob_to_like(pattern: str) -> str:
    """Turn a shell glob into a SQL LIKE pattern (use with ``ESCAPE '\\'``).

    ``*`` is the only wildcard we accept. Literal ``%``, ``_`` and ``\\``
    in the input are escaped so ``tr_2024_3`` stays a specific slug rather
    than collapsing into the LIKE single-char wildcard.
    """
    return pattern.translate(_GLOB_ESCAPE).replace("*", "%")


# ---------------------------------------------------------------------------
# Search primitives


def _fts_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    filter_sql: str,
    filter_params: list[Any],
) -> list[tuple[int, float]]:
    """Return [(chunk_id, bm25_score), ...]. Lower bm25 is better."""
    where = f"AND {filter_sql}" if filter_sql else ""
    sql = f"""
        SELECT f.rowid AS chunk_id, bm25(chunks_fts) AS score
        FROM chunks_fts f
        JOIN chunks c ON c.chunk_id = f.rowid
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE chunks_fts MATCH ? {where}
        ORDER BY score ASC LIMIT ?
    """
    try:
        rows = conn.execute(sql, [_fts_query(query), *filter_params, limit]).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(row["chunk_id"], row["score"]) for row in rows]


def _fts_query(query: str) -> str:
    """Turn a free-text query into an FTS5-safe MATCH expression.

    Tokens are space-joined, which FTS5 interprets as implicit AND — every
    returned chunk must contain every query term. Previous OR-join behaviour
    made common words like "tax" match ~500K chunks and then BM25-rank all
    of them, producing >30 s tails on natural-language queries. Single-char
    tokens (e.g. ``R``/``D`` from ``R&D``) are dropped so they don't turn
    queries into zero-result searches for punctuation artefacts.
    """
    tokens = [t for t in _WORD_RE.findall(query) if len(t) >= 2]
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


@lru_cache(maxsize=128)
def _encode_query_cached(query: str) -> bytes:
    """Embed ``query`` once and reuse the int8 vector for repeated calls.

    Keyed on the raw query string. A typical agent session asks the same
    question a handful of times (retry on format, follow-up with narrower k,
    etc.); caching skips ~500-800 ms of ONNX encoding per hit. The cache is
    process-local and uses ~32 KB at maxsize (128 × 256 bytes)."""
    backend = get_backend()
    if backend.model is None:
        return b""
    encoded = backend.model.encode([query], is_query=True)
    return vec_to_bytes(encoded.vectors_int8[0])


def _vec_search(
    conn: sqlite3.Connection,
    model: EmbeddingModel,
    query: str,
    *,
    limit: int,
    filter_sql: str,
    filter_params: list[Any],
) -> list[tuple[int, float]]:
    q_vec = _encode_query_cached(query)
    if not q_vec:
        return []
    where = f"AND {filter_sql}" if filter_sql else ""
    sql = f"""
        SELECT v.chunk_id AS chunk_id, v.distance AS score
        FROM chunks_vec v
        JOIN chunks c ON c.chunk_id = v.chunk_id
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE v.embedding MATCH vec_int8(?) AND k = ? {where}
        ORDER BY v.distance ASC LIMIT ?
    """
    try:
        rows = conn.execute(
            sql,
            [q_vec, limit, *filter_params, limit],
        ).fetchall()
    except sqlite3.OperationalError as exc:
        LOGGER.warning("vector search failed: %s", exc)
        return []
    return [(row["chunk_id"], row["score"]) for row in rows]


def _rrf_fuse(
    fts: list[tuple[int, float]],
    vec: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (chunk_id, _score) in enumerate(fts):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _score) in enumerate(vec):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _load_hit(conn: sqlite3.Connection, chunk_id: int) -> dict | None:
    row = conn.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.ord, c.heading_path, c.anchor, c.text,
               d.docid_code, d.title, d.category, d.doc_type, d.href, d.pub_date
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    text = _decompress(row["text"])
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "ord": row["ord"],
        "heading_path": row["heading_path"] or "",
        "anchor": row["anchor"],
        "text": text,
        "docid_code": row["docid_code"],
        "title": row["title"],
        "category": row["category"],
        "doc_type": row["doc_type"],
        "href": row["href"],
        "pub_date": row["pub_date"],
        "canonical_url": formatters.canonical_url(row["href"]),
    }


# ---------------------------------------------------------------------------
# Public tool handlers


def search(
    query: str,
    *,
    k: int = DEFAULT_K,
    categories: list[str] | None = None,
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    doc_scope: str | None = None,
    category_scope: str | None = None,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    backend = get_backend()
    k = max(1, min(k, MAX_K))
    filter_sql, filter_params = _build_sql_filter(
        categories, doc_types, date_from, date_to,
        doc_scope=doc_scope, category_scope=category_scope,
    )

    internal_k = max(k * 5, 50)
    fts_hits: list[tuple[int, float]] = []
    vec_hits: list[tuple[int, float]] = []
    if mode in ("hybrid", "keyword"):
        fts_hits = _fts_search(
            backend.db, query, limit=internal_k, filter_sql=filter_sql, filter_params=filter_params
        )
    if mode in ("hybrid", "vector") and backend.model is not None:
        vec_hits = _vec_search(
            backend.db, backend.model, query, limit=internal_k,
            filter_sql=filter_sql, filter_params=filter_params,
        )
    if mode == "hybrid":
        fused = _rrf_fuse(fts_hits, vec_hits)
    elif mode == "keyword":
        fused = [(cid, -score) for cid, score in fts_hits]
    else:
        fused = [(cid, -score) for cid, score in vec_hits]

    records: list[dict] = []
    for chunk_id, score in fused[:k]:
        hit = _load_hit(backend.db, chunk_id)
        if hit is None:
            continue
        hit["score"] = score
        hit["snippet"] = _highlight_snippet(hit["text"], query)
        records.append(hit)
    payload = [_slim_hit(r) for r in records]
    if format == "json":
        return formatters.as_json({"query": query, "mode": mode, "hits": payload})
    return formatters.format_hits_markdown(payload)


def _slim_hit(hit: dict) -> dict:
    keys = ("doc_id", "docid_code", "title", "category", "doc_type",
            "heading_path", "snippet", "canonical_url", "score", "chunk_id")
    return {k: hit.get(k) for k in keys}


def search_titles(
    query: str,
    *,
    k: int = 20,
    doc_types: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    backend = get_backend()
    k = max(1, min(k, 100))
    where = ""
    params: list[Any] = [_fts_query(query)]
    if doc_types:
        placeholders = ",".join("?" * len(doc_types))
        where = f"AND d.doc_type IN ({placeholders})"
        params.extend(doc_types)
    params.append(k)
    sql = f"""
        SELECT t.doc_id AS doc_id, bm25(title_fts) AS score,
               d.docid_code, d.title, d.category, d.doc_type, d.href, d.pub_date
        FROM title_fts t
        JOIN documents d ON d.doc_id = t.doc_id
        WHERE title_fts MATCH ? {where}
        ORDER BY score ASC LIMIT ?
    """
    try:
        rows = backend.db.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        rows = []
    hits = []
    for row in rows:
        hits.append({
            "doc_id": row["doc_id"],
            "docid_code": row["docid_code"],
            "title": row["title"],
            "category": row["category"],
            "doc_type": row["doc_type"],
            "heading_path": "",
            "snippet": row["title"],
            "canonical_url": formatters.canonical_url(row["href"]),
            "pub_date": row["pub_date"],
            "score": row["score"],
        })
    if format == "json":
        return formatters.as_json({"query": query, "hits": hits})
    return formatters.format_hits_markdown(hits)


def get_document(
    doc_id: str,
    *,
    format: Literal["outline", "markdown", "json"] = "outline",
) -> str:
    backend = get_backend()
    row = backend.db.execute(SELECT_DOCUMENT, (doc_id,)).fetchone()
    if row is None:
        return f"_Document not found: `{doc_id}`_"
    doc = dict(row)
    doc["canonical_url"] = formatters.canonical_url(row["href"])
    chunks_rows = backend.db.execute(SELECT_CHUNKS_FOR_DOC, (doc_id,)).fetchall()
    chunks = [
        {
            "chunk_id": r["chunk_id"],
            "ord": r["ord"],
            "heading_path": r["heading_path"] or "",
            "anchor": r["anchor"],
            "text": _decompress(r["text"]),
        }
        for r in chunks_rows
    ]
    if format == "json":
        return formatters.as_json({"document": doc, "chunks": chunks})
    if format == "markdown":
        return formatters.format_document_full_markdown(doc, chunks)
    return formatters.format_document_outline_markdown(doc, chunks)


def get_section(
    doc_id: str,
    *,
    anchor: str | None = None,
    heading_path: str | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    backend = get_backend()
    doc_row = backend.db.execute(SELECT_DOCUMENT, (doc_id,)).fetchone()
    if doc_row is None:
        return f"_Document not found: `{doc_id}`_"
    chunks_rows = backend.db.execute(SELECT_CHUNKS_FOR_DOC, (doc_id,)).fetchall()
    selected = []
    for row in chunks_rows:
        if anchor and row["anchor"] == anchor:
            selected.append(row)
        elif heading_path and row["heading_path"] == heading_path:
            selected.append(row)
    if not selected and not (anchor or heading_path):
        # With no selector, return the first heading block.
        selected = chunks_rows[:1]
    if not selected:
        return f"_Section not found in {doc_id} (anchor={anchor}, heading_path={heading_path!r})._"
    payload = [
        {
            "chunk_id": r["chunk_id"],
            "ord": r["ord"],
            "heading_path": r["heading_path"] or "",
            "anchor": r["anchor"],
            "text": _decompress(r["text"]),
        }
        for r in selected
    ]
    if format == "json":
        return formatters.as_json({
            "doc_id": doc_id,
            "canonical_url": formatters.canonical_url(doc_row["href"]),
            "chunks": payload,
        })
    header = f"**{doc_row['title']}** — [{doc_row['docid_code'] or doc_row['doc_id']}]({formatters.canonical_url(doc_row['href'])})\n\n"
    body = "\n\n".join(f"### {c['heading_path'] or '(intro)'}\n\n{c['text']}" for c in payload)
    return header + body + "\n"


def get_chunks(
    chunk_ids: list[int],
    *,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    backend = get_backend()
    if not chunk_ids:
        return "_No chunk ids provided._"
    placeholders = ",".join("?" * len(chunk_ids))
    rows = backend.db.execute(
        f"""
        SELECT c.chunk_id, c.doc_id, c.ord, c.heading_path, c.anchor, c.text,
               d.docid_code, d.title, d.href
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({placeholders})
        """,
        list(chunk_ids),
    ).fetchall()
    records = []
    for row in rows:
        records.append({
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "title": row["title"],
            "docid_code": row["docid_code"],
            "heading_path": row["heading_path"] or "",
            "anchor": row["anchor"],
            "canonical_url": formatters.canonical_url(row["href"]),
            "text": _decompress(row["text"]),
        })
    if format == "json":
        return formatters.as_json({"chunks": records})
    lines = []
    for r in records:
        lines.append(
            f"**{r['docid_code'] or r['doc_id']}** — [{r['title']}]({r['canonical_url']}) — "
            f"{r['heading_path']}\n\n{r['text']}\n\n---"
        )
    return "\n".join(lines)


def whats_new(
    *,
    since: str | None = None,
    limit: int = 50,
    categories: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    backend = get_backend()
    # Historical publication date when the heuristic produced one, otherwise
    # fall back to our ingest timestamp so newly-crawled docs still appear.
    sort_expr = "COALESCE(first_published_date, downloaded_at)"
    clauses: list[str] = []
    params: list[Any] = []
    if since:
        clauses.append(f"{sort_expr} >= ?")
        params.append(since)
    if categories:
        placeholders = ",".join("?" * len(categories))
        clauses.append(f"category IN ({placeholders})")
        params.extend(categories)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    sql = f"""
        SELECT doc_id, docid_code, title, category, doc_type, href,
               pub_date, first_published_date, downloaded_at
        FROM documents {where}
        ORDER BY {sort_expr} DESC LIMIT ?
    """
    rows = backend.db.execute(sql, params).fetchall()
    hits = [{
        "doc_id": r["doc_id"],
        "docid_code": r["docid_code"],
        "title": r["title"],
        "category": r["category"],
        "doc_type": r["doc_type"],
        "heading_path": "",
        "snippet": _whats_new_snippet(r),
        "canonical_url": formatters.canonical_url(r["href"]),
        "pub_date": r["pub_date"],
        "first_published_date": r["first_published_date"],
    } for r in rows]
    if format == "json":
        return formatters.as_json({"since": since, "hits": hits})
    return formatters.format_hits_markdown(hits)


def _whats_new_snippet(row: sqlite3.Row) -> str:
    if row["first_published_date"]:
        return f"published {row['first_published_date']}"
    return f"ingested {row['downloaded_at']}"
