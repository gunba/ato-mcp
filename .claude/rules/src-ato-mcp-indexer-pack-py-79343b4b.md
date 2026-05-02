---
paths:
  - "src/ato_mcp/indexer/pack.py"
---

# src/ato_mcp/indexer/pack.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Index Builder
Build orchestration, heading-aware chunking with overlap, token estimation, packing, manifest, release.

- [IB-10 L47] PACK_TARGET_SIZE = 64 MB uncompressed record payload before zstd level 3; PackBuilder seals when offset crosses target and opens the next writer — keeps individual pack downloads tractable on slow links.
- [IB-11 L51] Embeddings travel through the pack as base64-encoded raw int8 bytes (encode_embedding/decode_embedding); both sides length-check against EMBEDDING_DIM so a wrong-shape embedding can't slip through.
- [IB-09 L80] Pack record format: length:uint32 (LE) | zstd(orjson(record)); pack is content-addressable via sha256[:8] (pack_sha8). Trailer at end of pack has MAGIC + count + index_offset + index_blob (a reverse index of (doc_id, offset, length)) so packs are self-describing for offline verification.
