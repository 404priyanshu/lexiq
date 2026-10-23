"""bge-small-en-v1.5 sentence embeddings on ONNX Runtime (CPU)."""

from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.ml.registry import EMBEDDER, ModelSpec

# bge v1.5 recommends this prefix on queries (not passages) for retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 512


def load_tokenizer(spec: ModelSpec = EMBEDDER) -> Tokenizer:
    """Raw tokenizer (no padding/truncation), shared with the chunker for token counts."""
    path = spec.path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run python -m app.ml.download`")
    tok = Tokenizer.from_file(str(path))
    tok.no_padding()
    tok.no_truncation()
    return tok


class Embedder:
    def __init__(self, spec: ModelSpec = EMBEDDER, batch_size: int = 32):
        self.batch_size = batch_size
        self.tokenizer = load_tokenizer(spec)
        self.tokenizer.enable_truncation(MAX_LENGTH)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(spec.path / spec.onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
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
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        hidden = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})
        # bge uses the [CLS] token as the sentence embedding, then L2-normalizes.
        cls = hidden[0][:, 0]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)


def model_path_exists(spec: ModelSpec = EMBEDDER) -> bool:
    return all((spec.path / f).exists() for f in spec.files)
