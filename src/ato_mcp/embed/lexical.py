"""Deterministic lexical vectors for fast maintainer builds."""
from __future__ import annotations

import hashlib
import struct
import subprocess
from pathlib import Path

import numpy as np

from ..store.db import EMBEDDING_DIM

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
_SRC = Path(__file__).with_name("rust_lexical_hash.rs")


def rust_lexical_hash(texts: list[str], *, binary_dir: Path) -> np.ndarray:
    binary = _ensure_rust_vectorizer(binary_dir)
    payload = bytearray()
    payload.extend(struct.pack("<I", len(texts)))
    for text in texts:
        data = text.encode("utf-8", errors="ignore")
        payload.extend(struct.pack("<I", len(data)))
        payload.extend(data)
    proc = subprocess.run(
        [str(binary)],
        input=bytes(payload),
        stdout=subprocess.PIPE,
        check=True,
    )
    expected = len(texts) * EMBEDDING_DIM
    if len(proc.stdout) != expected:
        raise RuntimeError(f"rust vectorizer returned {len(proc.stdout)} bytes, expected {expected}")
    return np.frombuffer(proc.stdout, dtype=np.int8).reshape((len(texts), EMBEDDING_DIM)).copy()


def query_lexical_hash(query: str) -> bytes:
    return _lexical_hash_one(query).tobytes()


def _ensure_rust_vectorizer(binary_dir: Path) -> Path:
    binary_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:12]
    binary = binary_dir / f"rust_lexical_hash-{digest}"
    if binary.exists():
        return binary
    tmp = binary.with_suffix(".tmp")
    subprocess.run(
        [
            "rustc",
            "--edition=2021",
            "-C",
            "opt-level=3",
            "-C",
            "target-cpu=native",
            str(_SRC),
            "-o",
            str(tmp),
        ],
        check=True,
    )
    tmp.rename(binary)
    return binary


def _lexical_hash_one(text: str) -> np.ndarray:
    data = text.encode("utf-8", errors="ignore")
    acc = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tokens: list[bytes] = []
    i = 0
    while i < len(data):
        if not _is_token_start(data[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < len(data) and _is_token_continue(data[i]):
            i += 1
        tok = data[start:i].lower()
        _add_feature(acc, tok, 1.0)
        if len(tok) >= 8:
            for gram_start in range(0, len(tok) - 3):
                _add_feature(acc, tok[gram_start : gram_start + 4], 0.25)
        tokens.append(tok)

    for a, b in zip(tokens, tokens[1:]):
        _add_feature(acc, a + b" " + b, 0.5)

    norm = float(np.linalg.norm(acc))
    if norm <= 1e-12:
        return np.zeros(EMBEDDING_DIM, dtype=np.int8)
    return np.round(np.clip(acc / norm, -1.0, 1.0) * 127.0).astype(np.int8)


def _is_token_start(byte: int) -> bool:
    return 48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122


def _is_token_continue(byte: int) -> bool:
    return _is_token_start(byte) or byte in (95, 46, 47, 45)


def _add_feature(acc: np.ndarray, feature: bytes, weight: float) -> None:
    h = FNV_OFFSET
    for byte in feature:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    idx = h & (EMBEDDING_DIM - 1)
    sign = 1.0 if (h & 0x100) else -1.0
    acc[idx] += sign * weight
