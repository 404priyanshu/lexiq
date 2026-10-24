"""bge-small-en-v1.5 sentence embeddings on ONNX Runtime (CPU)."""

from collections.abc import Sequence

import numpy as np

from app.ml.registry import EMBEDDER, ModelSpec
from app.ml.runtime import MAX_LENGTH, bert_feeds, load_tokenizer, onnx_session

# bge v1.5 recommends this prefix on queries (not passages) for retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, spec: ModelSpec = EMBEDDER, batch_size: int = 32):
        self.batch_size = batch_size
        self.tokenizer = load_tokenizer(spec)
        self.tokenizer.enable_truncation(MAX_LENGTH)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.session = onnx_session(spec)
        self.dim = self.session.get_outputs()[0].shape[-1]

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([QUERY_INSTRUCTION + query])[0]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        # Batch similar lengths together so padding stays small.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            out[idx] = self._embed_batch([texts[i] for i in idx])
        return out

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        hidden = self.session.run(None, bert_feeds(enc, self.session))
        # bge uses the [CLS] token as the sentence embedding, then L2-normalizes.
        cls = hidden[0][:, 0]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)
