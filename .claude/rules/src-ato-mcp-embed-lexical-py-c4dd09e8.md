---
paths:
  - "src/ato_mcp/embed/lexical.py"
---

# src/ato_mcp/embed/lexical.py

Tag line: `L<n>`; code usually starts at `L<n+1>`.

## Embedding Model
EmbeddingGemma ONNX, Matryoshka 256-dim, int8 quantization, query/passage prefixes, lexical-hash fallback.

- [EM-07 L44] Lexical fallback vectorizer: rust_lexical_hash.rs is compiled ad-hoc by rustc with -C opt-level=3 -C target-cpu=native; binary is cached in binary_dir keyed by sha256[:12] of the source so a source edit triggers an automatic rebuild.
- [EM-08 L69] Lexical hash features: per-token + 4-grams (only when token length >= 8) + adjacent-token bigrams, hashed via FNV-1a, indexed by h & (EMBEDDING_DIM-1), summed with sign drawn from h & 0x100, then L2-normalized and int8-quantized to match the model's output format so queries can flow through the same sqlite-vec path.
