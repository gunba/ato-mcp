"""EmbeddingGemma ONNX loader + batched encoder.

The model is shipped as an ``int8``-quantized ONNX graph exported such that its
output is the final (mean- or CLS-pooled) sentence embedding, truncated to
``EMBEDDING_DIM`` via Matryoshka representation learning. We l2-normalize the
output and cast to int8 for storage in sqlite-vec.

Query-time embedding is computed locally; we do not call any remote service.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..store.db import EMBEDDING_DIM
from ..util import paths
from ..util.log import get_logger

LOGGER = get_logger(__name__)

MAX_TOKENS = 1024
# Query prefix used during training per EmbeddingGemma docs. Applied at encode.
QUERY_PREFIX = "task: search result | query: "
PASSAGE_PREFIX = "title: none | text: "


@dataclass(frozen=True)
class EncodedBatch:
    """Result of encoding: int8 vectors plus the number of tokens observed."""

    vectors_int8: np.ndarray  # shape: (N, EMBEDDING_DIM), dtype int8
    tokens_seen: int


class EmbeddingModel:
    """Wraps an ONNX Runtime session around the EmbeddingGemma export."""

    def __init__(
        self,
        model_path: Path | None = None,
        tokenizer_path: Path | None = None,
        providers: tuple[str, ...] | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_path = Path(model_path or paths.model_path())
        self.tokenizer_path = Path(tokenizer_path or paths.tokenizer_path())
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {self.model_path}. Run `ato-mcp init`."
            )
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {self.tokenizer_path}. Run `ato-mcp init`."
            )

        so = ort.SessionOptions()
        so.intra_op_num_threads = 0  # let ORT decide based on core count
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        avail = set(ort.get_available_providers())
        default_providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail:
            default_providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=so,
            providers=list(providers) if providers else default_providers,
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]
        # Prefer a pooled "sentence_embedding" output if the graph provides one.
        self._pooled_output_name: str | None = (
            "sentence_embedding" if "sentence_embedding" in self.output_names else None
        )
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(length=None)  # pad to batch max
        LOGGER.info(
            "Loaded embedding model %s (providers=%s, pooled_output=%s)",
            self.model_path.name,
            self.session.get_providers(),
            self._pooled_output_name,
        )

    def encode(
        self,
        texts: Iterable[str],
        *,
        is_query: bool,
        batch_size: int = 16,
    ) -> EncodedBatch:
        texts_list = list(texts)
        if not texts_list:
            return EncodedBatch(
                vectors_int8=np.empty((0, EMBEDDING_DIM), dtype=np.int8),
                tokens_seen=0,
            )
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts_list]

        vecs: list[np.ndarray] = []
        tokens_seen = 0
        for start in range(0, len(prefixed), batch_size):
            chunk = prefixed[start : start + batch_size]
            encs = self.tokenizer.encode_batch(chunk)
            input_ids = np.stack([np.asarray(e.ids, dtype=np.int64) for e in encs])
            attention_mask = np.stack(
                [np.asarray(e.attention_mask, dtype=np.int64) for e in encs]
            )
            tokens_seen += int(attention_mask.sum())
            feed = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            if self._pooled_output_name is not None:
                (emb,) = self.session.run([self._pooled_output_name], feed)
            else:
                outputs = self.session.run(None, feed)
                emb = outputs[0]
                if emb.ndim == 3:
                    # token embeddings returned -> mean-pool with attention mask
                    mask = attention_mask.astype(np.float32)[:, :, None]
                    emb = (emb * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-6, None)
            # Matryoshka truncation
            emb = emb[:, :EMBEDDING_DIM]
            # L2 normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)
            vecs.append(_f32_to_i8(emb))
        return EncodedBatch(vectors_int8=np.concatenate(vecs, axis=0), tokens_seen=tokens_seen)


def _f32_to_i8(vectors: np.ndarray) -> np.ndarray:
    """Linearly map [-1, 1] -> [-127, 127] and round. Saturates outliers."""
    scaled = np.clip(vectors, -1.0, 1.0) * 127.0
    return np.round(scaled).astype(np.int8)


def vec_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize a single int8 embedding as raw bytes for sqlite-vec."""
    if vec.dtype != np.int8:
        vec = vec.astype(np.int8)
    if vec.shape != (EMBEDDING_DIM,):
        raise ValueError(f"expected shape ({EMBEDDING_DIM},), got {vec.shape}")
    return vec.tobytes()
