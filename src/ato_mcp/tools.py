"""MCP tool implementations for ato-mcp (v5 schema).

5 tools. Every hit carries a ``canonical_url``; agents compose
``search -> get_document`` with minimal token cost.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Literal

import zstandard as zstd

from . import formatters
from .embed.lexical import query_lexical_hash
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
# Recency half-life (years) applied when ``sort_by='relevance'`` with a
# date-bearing query. Picked from tax-law practitioner intuition: a doc
# 5 years old is roughly half as likely to be current.
RECENCY_HALF_LIFE_YEARS = 5.0
# Cap on per-search ``previously_seen`` echo. Bounded so a long session
# doesn't drown the agent in suppressed-result noise.
MAX_SEEN_ECHO = 10


@dataclass
class Backend:
    """Shared embedding model plus per-thread read-only SQLite connections."""
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
        if conn is not None and getattr(tls, "mtime", 0.0) + 0.001 < current_mtime:
            conn.close()
            conn = None
        if conn is None:
            conn = store_db.connect(paths.db_path(), mode="ro")
            tls.conn = conn
            tls.mtime = current_mtime
        return conn

    def close(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None


_BACKEND: Backend | None = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> Backend:
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = Backend.open()
    return _BACKEND


class SeenTracker:
    """Per-process record of chunk_ids surfaced to the agent.

    MCP stdio sessions are one-per-process, so module-level state is
    naturally session-scoped. ``search`` filters chunks already in the
    set and echoes their handles in ``previously_seen``; ``get_chunks``
    and ``get_document`` register chunks the moment their bodies enter
    the agent's context.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: set[int] = set()

    def __contains__(self, chunk_id: int) -> bool:
        with self._lock:
            return chunk_id in self._chunks

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    def add(self, chunk_ids) -> None:
        ids = [int(c) for c in chunk_ids]
        if not ids:
            return
        with self._lock:
            self._chunks.update(ids)


_SEEN: SeenTracker | None = None
_SEEN_LOCK = threading.Lock()


def get_seen() -> SeenTracker:
    global _SEEN
    if _SEEN is None:
        with _SEEN_LOCK:
            if _SEEN is None:
                _SEEN = SeenTracker()
    return _SEEN


# ---------------------------------------------------------------------------
# Helpers


def _decompress(blob: bytes) -> str:
    return zstd.ZstdDecompressor().decompress(blob).decode("utf-8")


_WORD_RE = re.compile(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*")


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

    def _bold(m: re.Match) -> str:
        token = m.group(0)
        if token.lower() in words:
            return f"**{token}**"
        return token
    return _WORD_RE.sub(_bold, snippet).strip()


_GLOB_ESCAPE = str.maketrans({"\\": r"\\", "%": r"\%", "_": r"\_"})


def _glob_to_like(pattern: str) -> str:
    """Turn a shell glob into a SQL LIKE pattern (use with ``ESCAPE '\\'``)."""
    return pattern.translate(_GLOB_ESCAPE).replace("*", "%")


# Types excluded from search / search_titles / whats_new by default.
# Agents still reach them by passing the type name explicitly in ``types``.
# Edited private advice is one-off, individual-taxpayer rulings the ATO
# publishes in a redacted form — informative if you know you want EPA
# ("EV 1051375298526"), rarely useful when the agent is answering a
# public tax-law question.
DEFAULT_EXCLUDED_TYPES = ("Edited_private_advice",)
DEFAULT_OLD_CONTENT_CUTOFF = "2000-01-01"
DEFAULT_OLD_CONTENT_EXCEPTION_TYPES = ("Legislation_and_supporting_material",)


def _build_sql_filter(
    types: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    doc_scope: str | None,
    include_old: bool = False,
) -> tuple[str, list[Any]]:
    """Build a WHERE fragment for the documents table.

    ``types`` semantics:
      - ``None`` → exclude the ``DEFAULT_EXCLUDED_TYPES`` buckets (EPA etc.).
      - a list → positive filter. Entries containing ``*`` are globs;
        other entries are exact matches. Pass the excluded type explicitly
        (e.g. ``types=["Edited_private_advice"]``) to opt back in.

    ``doc_scope`` is a glob over the ``doc_id`` path.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if types is None:
        if DEFAULT_EXCLUDED_TYPES:
            placeholders = ",".join("?" * len(DEFAULT_EXCLUDED_TYPES))
            clauses.append(f"d.type NOT IN ({placeholders})")
            params.extend(DEFAULT_EXCLUDED_TYPES)
    else:
        ors: list[str] = []
        for t in types:
            if "*" in t:
                ors.append("d.type LIKE ? ESCAPE '\\'")
                params.append(_glob_to_like(t))
            else:
                ors.append("d.type = ?")
                params.append(t)
        clauses.append("(" + " OR ".join(ors) + ")")
    if date_from:
        clauses.append("d.date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("d.date <= ?")
        params.append(date_to)
    if doc_scope:
        clauses.append("d.doc_id LIKE ? ESCAPE '\\'")
        params.append(_glob_to_like(doc_scope))
    if not include_old and not date_from:
        placeholders = ",".join("?" * len(DEFAULT_OLD_CONTENT_EXCEPTION_TYPES))
        clauses.append(
            f"(d.date IS NULL OR d.date >= ? OR d.type IN ({placeholders}))"
        )
        params.append(DEFAULT_OLD_CONTENT_CUTOFF)
        params.extend(DEFAULT_OLD_CONTENT_EXCEPTION_TYPES)
    return (" AND ".join(clauses), params) if clauses else ("", params)


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
    returned chunk must contain every query term. Single-char tokens (e.g.
    the ``R``/``D`` from ``R&D``) are dropped so they don't turn queries
    into zero-result noise searches. Hyphenated tokens (``"s 8-1"``,
    ``"355-25"``) are preserved as phrases so section-number lookups keep
    working even though FTS5 indexes them as separate tokens internally.
    """
    tokens = [t for t in _WORD_RE.findall(query) if len(t) >= 2]
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


def _encode_query(conn: sqlite3.Connection, query: str) -> bytes:
    backend = get_backend()
    model_id = store_db.get_meta(conn, "embedding_model_id") or ""
    if model_id.startswith("lexical-hash-rust"):
        return query_lexical_hash(query)
    if backend.model is None:
        return b""
    encoded = backend.model.encode([query], is_query=True)
    return vec_to_bytes(encoded.vectors_int8[0])


def _vec_search(
    conn: sqlite3.Connection,
    model: EmbeddingModel | None,
    query: str,
    *,
    limit: int,
    filter_sql: str,
    filter_params: list[Any],
) -> list[tuple[int, float]]:
    q_vec = _encode_query(conn, query)
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
        rows = conn.execute(sql, [q_vec, limit, *filter_params, limit]).fetchall()
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
               d.type, d.title, d.date
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
        "type": row["type"],
        "title": row["title"],
        "date": row["date"],
        "canonical_url": formatters.canonical_url(row["doc_id"]),
    }


# ---------------------------------------------------------------------------
# Public tool handlers


def search(
    query: str,
    *,
    k: int = DEFAULT_K,
    types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    doc_scope: str | None = None,
    include_old: bool = False,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    sort_by: Literal["relevance", "recency"] = "relevance",
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    backend = get_backend()
    seen = get_seen()
    k = max(1, min(k, MAX_K))
    filter_sql, filter_params = _build_sql_filter(
        types, date_from, date_to, doc_scope, include_old=include_old
    )

    # Bump internal_k by |seen| so the seen filter can hide that many
    # without starving the frontier.
    internal_k = max(k * 5, 50) + len(seen)
    fts_hits: list[tuple[int, float]] = []
    vec_hits: list[tuple[int, float]] = []
    if mode in ("hybrid", "keyword"):
        fts_hits = _fts_search(
            backend.db, query, limit=internal_k, filter_sql=filter_sql, filter_params=filter_params
        )
    if mode in ("hybrid", "vector"):
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

    # Split fused ranking into fresh candidates and a capped echo of the
    # most relevant chunks the agent has already seen this session.
    fresh: list[tuple[int, float]] = []
    seen_ids: list[int] = []
    for chunk_id, score in fused:
        if chunk_id in seen:
            if len(seen_ids) < MAX_SEEN_ECHO:
                seen_ids.append(chunk_id)
        else:
            fresh.append((chunk_id, score))

    # Over-fetch so the recency boost can re-order the frontier before we cut.
    frontier = max(k * 2, 20) if sort_by == "relevance" else k * 3
    records: list[dict] = []
    for chunk_id, score in fresh[:frontier]:
        hit = _load_hit(backend.db, chunk_id)
        if hit is None:
            continue
        hit["score"] = score
        hit["snippet"] = _highlight_snippet(hit["text"], query)
        records.append(hit)

    if sort_by == "relevance":
        _apply_recency_boost(records, half_life_years=RECENCY_HALF_LIFE_YEARS)
        records.sort(key=lambda r: r["score"], reverse=True)
    else:  # recency
        records.sort(key=lambda r: r.get("date") or "", reverse=True)
    records = records[:k]

    seen.add(r["chunk_id"] for r in records)
    previously_seen = _load_seen_handles(backend.db, seen_ids)

    payload = [_slim_hit(r) for r in records]
    if format == "json":
        return formatters.as_json({
            "query": query,
            "mode": mode,
            "hits": payload,
            "previously_seen": previously_seen,
        })
    return formatters.format_hits_markdown(payload, previously_seen=previously_seen)


def _apply_recency_boost(records: list[dict], *, half_life_years: float) -> None:
    """Multiply each record's ``score`` by a recency factor in (0.5, 1.5]."""
    import datetime as _dt
    import math
    now_year = _dt.datetime.now(tz=_dt.timezone.utc).year
    decay = math.log(2) / half_life_years
    for r in records:
        date = r.get("date")
        year = None
        if date and len(date) >= 4 and date[:4].isdigit():
            year = int(date[:4])
        if year is None:
            continue
        age = max(0, now_year - year)
        r["score"] = r["score"] * (0.5 + math.exp(-age * decay))


def _slim_hit(hit: dict) -> dict:
    keys = (
        "doc_id", "title", "type", "date",
        "heading_path", "anchor", "snippet",
        "canonical_url", "score", "chunk_id",
    )
    return {k: hit.get(k) for k in keys}


def _load_seen_handles(conn: sqlite3.Connection, chunk_ids: list[int]) -> list[dict]:
    """Minimal handles for chunks the agent has already seen.

    No body, no snippet — just enough metadata for the agent to recognise
    them and re-fetch via ``get_chunks`` if needed. Order preserved from
    ``chunk_ids`` so the most fused-relevant suppressed chunk is first.
    """
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT c.chunk_id, c.doc_id, c.heading_path, d.title, d.type
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({placeholders})
        """,
        list(chunk_ids),
    ).fetchall()
    by_id = {row["chunk_id"]: row for row in rows}
    return [
        {
            "chunk_id": cid,
            "doc_id": by_id[cid]["doc_id"],
            "title": by_id[cid]["title"],
            "type": by_id[cid]["type"],
            "heading_path": by_id[cid]["heading_path"] or "",
            "canonical_url": formatters.canonical_url(by_id[cid]["doc_id"]),
        }
        for cid in chunk_ids
        if cid in by_id
    ]


def search_titles(
    query: str,
    *,
    k: int = 20,
    types: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
    include_old: bool = False,
) -> str:
    backend = get_backend()
    k = max(1, min(k, 100))
    where, params = _build_sql_filter(types, None, None, None, include_old=include_old)
    # _build_sql_filter prefixes ``d.``; strip it since title_fts joins ``d``.
    sql = f"""
        SELECT t.doc_id AS doc_id, bm25(title_fts) AS score,
               d.type, d.title, d.date
        FROM title_fts t
        JOIN documents d ON d.doc_id = t.doc_id
        WHERE title_fts MATCH ?
        {'AND ' + where if where else ''}
        ORDER BY score ASC LIMIT ?
    """
    try:
        rows = backend.db.execute(sql, [_fts_query(query), *params, k]).fetchall()
    except sqlite3.OperationalError:
        rows = []
    hits = [{
        "doc_id": row["doc_id"],
        "title": row["title"],
        "type": row["type"],
        "date": row["date"],
        "heading_path": "",
        "snippet": row["title"],
        "canonical_url": formatters.canonical_url(row["doc_id"]),
        "score": row["score"],
    } for row in rows]
    if format == "json":
        return formatters.as_json({"query": query, "hits": hits})
    return formatters.format_hits_markdown(hits)


def get_document(
    doc_id: str,
    *,
    format: Literal["outline", "markdown", "json"] = "outline",
    anchor: str | None = None,
    heading_path: str | None = None,
    from_ord: int | None = None,
    include_children: bool = False,
    count: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Return a document or a slice of it.

    Three retrieval modes, one tool:

    * **Whole document** (no selector) — ``format='outline'`` for the TOC,
      ``format='markdown'`` for the full body. ``max_chars`` caps either.
    * **A specific section** (``anchor=`` or ``heading_path=``) — returns
      the chunk(s) at that heading. ``include_children=True`` rolls up
      the entire subtree until the next sibling or higher-level heading.
    * **An ordinal range** (``from_ord=N``) — walk forward from a cursor.
      Combine with ``count`` or ``max_chars`` to paginate. The JSON
      payload carries ``continuation_ord`` when truncation happened.
    """
    backend = get_backend()
    doc_row = backend.db.execute(SELECT_DOCUMENT, (doc_id,)).fetchone()
    if doc_row is None:
        return f"_Document not found: `{doc_id}`_"
    doc = dict(doc_row)
    doc["canonical_url"] = formatters.canonical_url(doc_id)

    if format == "outline":
        outline = _outline_for_doc(backend.db, doc_id, anchor=anchor,
                                   heading_path=heading_path, from_ord=from_ord)
        return formatters.format_document_outline_markdown(doc, outline_entries=outline)

    chunks = _select_chunks(
        backend.db, doc_id,
        anchor=anchor, heading_path=heading_path, from_ord=from_ord,
        include_children=include_children,
        count=count, max_chars=max_chars,
    )
    if chunks is None:
        return (
            f"_Section not found in {doc_id} "
            f"(anchor={anchor!r}, heading_path={heading_path!r}, from_ord={from_ord})._"
        )
    selected, continuation_ord = chunks

    # Materialised chunks now sit in the agent's context; mark them as
    # seen so subsequent searches don't re-surface them.
    get_seen().add(c["chunk_id"] for c in selected)

    if format == "json":
        return formatters.as_json({
            "document": doc,
            "chunks": selected,
            "continuation_ord": continuation_ord,
        })
    return formatters.format_document_section_markdown(doc, selected, continuation_ord)


def _outline_for_doc(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    anchor: str | None,
    heading_path: str | None,
    from_ord: int | None,
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT heading_path, anchor, MIN(ord) AS start_ord,
               COUNT(*) AS chunk_count, SUM(LENGTH(text)) AS bytes
        FROM chunks
        WHERE doc_id = ?
        GROUP BY heading_path
        ORDER BY start_ord ASC
        """,
        (doc_id,),
    ).fetchall()
    entries = [
        {
            "heading_path": r["heading_path"] or "",
            "anchor": r["anchor"],
            "depth": (r["heading_path"] or "").count(" › ") + 1 if r["heading_path"] else 0,
            "start_ord": r["start_ord"],
            "chunk_count": r["chunk_count"],
            "bytes": r["bytes"],
        }
        for r in rows
    ]

    if anchor is None and heading_path is None and from_ord is None:
        return entries

    start_idx: int | None = None
    for i, e in enumerate(entries):
        if anchor and e["anchor"] == anchor:
            start_idx = i
            break
        if heading_path and e["heading_path"] == heading_path:
            start_idx = i
            break
        if from_ord is not None and e["start_ord"] >= from_ord:
            start_idx = i
            break
    if start_idx is None:
        return []
    start = entries[start_idx]
    start_path = start["heading_path"]
    out = [start]
    for e in entries[start_idx + 1:]:
        hp = e["heading_path"]
        if start_path and (hp == start_path or hp.startswith(start_path + " › ")):
            out.append(e)
        elif not start_path:
            out.append(e)
        else:
            break
    return out


def _select_chunks(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    anchor: str | None,
    heading_path: str | None,
    from_ord: int | None,
    include_children: bool,
    count: int | None,
    max_chars: int | None,
) -> tuple[list[dict], int | None] | None:
    rows = conn.execute(SELECT_CHUNKS_FOR_DOC, (doc_id,)).fetchall()
    if not rows:
        return ([], None)

    if anchor is None and heading_path is None and from_ord is None:
        candidates = list(rows)
        start_path = None
    else:
        start_idx: int | None = None
        for i, r in enumerate(rows):
            if anchor and r["anchor"] == anchor:
                start_idx = i
                break
            if heading_path and r["heading_path"] == heading_path:
                start_idx = i
                break
            if from_ord is not None and r["ord"] >= from_ord:
                start_idx = i
                break
        if start_idx is None:
            return None
        candidates = list(rows[start_idx:])
        start_path = candidates[0]["heading_path"] or ""

    if (anchor or heading_path) and not include_children:
        if anchor:
            candidates = [r for r in candidates if r["anchor"] == anchor]
        elif heading_path:
            candidates = [r for r in candidates if r["heading_path"] == heading_path]
    elif (anchor or heading_path) and include_children:
        subtree: list = []
        for r in candidates:
            hp = r["heading_path"] or ""
            if start_path == "" or hp == start_path or hp.startswith(start_path + " › "):
                subtree.append(r)
            else:
                break
        candidates = subtree

    out: list[dict] = []
    total_chars = 0
    continuation_ord: int | None = None
    for i, r in enumerate(candidates):
        text = _decompress(r["text"])
        if max_chars is not None and out and total_chars + len(text) > max_chars:
            continuation_ord = r["ord"]
            break
        out.append({
            "chunk_id": r["chunk_id"],
            "ord": r["ord"],
            "heading_path": r["heading_path"] or "",
            "anchor": r["anchor"],
            "text": text,
        })
        total_chars += len(text)
        if count is not None and len(out) >= count:
            if i + 1 < len(candidates):
                continuation_ord = candidates[i + 1]["ord"]
            break
    return (out, continuation_ord)


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
               d.type, d.title, d.date
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
            "type": row["type"],
            "title": row["title"],
            "date": row["date"],
            "heading_path": row["heading_path"] or "",
            "anchor": row["anchor"],
            "canonical_url": formatters.canonical_url(row["doc_id"]),
            "text": _decompress(row["text"]),
        })
    get_seen().add(r["chunk_id"] for r in records)
    if format == "json":
        return formatters.as_json({"chunks": records})
    lines = []
    for r in records:
        lines.append(
            f"**{r['title']}** ([{r['doc_id']}]({r['canonical_url']})) — "
            f"{r['heading_path']}\n\n{r['text']}\n\n---"
        )
    return "\n".join(lines)


def whats_new(
    *,
    since: str | None = None,
    limit: int = 50,
    types: list[str] | None = None,
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    backend = get_backend()
    sort_expr = "COALESCE(date, downloaded_at)"
    clauses: list[str] = []
    params: list[Any] = []
    if since:
        clauses.append(f"{sort_expr} >= ?")
        params.append(since)
    if types is None:
        if DEFAULT_EXCLUDED_TYPES:
            placeholders = ",".join("?" * len(DEFAULT_EXCLUDED_TYPES))
            clauses.append(f"type NOT IN ({placeholders})")
            params.extend(DEFAULT_EXCLUDED_TYPES)
    else:
        ors: list[str] = []
        for t in types:
            if "*" in t:
                ors.append("type LIKE ? ESCAPE '\\'")
                params.append(_glob_to_like(t))
            else:
                ors.append("type = ?")
                params.append(t)
        clauses.append("(" + " OR ".join(ors) + ")")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    sql = f"""
        SELECT doc_id, type, title, date, downloaded_at
        FROM documents {where}
        ORDER BY {sort_expr} DESC LIMIT ?
    """
    rows = backend.db.execute(sql, params).fetchall()
    hits = [{
        "doc_id": r["doc_id"],
        "title": r["title"],
        "type": r["type"],
        "date": r["date"],
        "heading_path": "",
        "snippet": f"published {r['date']}" if r["date"] else f"ingested {r['downloaded_at']}",
        "canonical_url": formatters.canonical_url(r["doc_id"]),
    } for r in rows]
    if format == "json":
        return formatters.as_json({"since": since, "hits": hits})
    return formatters.format_hits_markdown(hits)
