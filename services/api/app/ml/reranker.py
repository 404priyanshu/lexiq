"""ms-marco-MiniLM-L6-v2 cross-encoder on ONNX Runtime (CPU).

Scores (query, passage) pairs jointly. Logits are unbounded; `sigmoid(logit)` gives a 0..1
relevance used for the abstain threshold.
"""

from collections.abc import Sequence

import numpy as np

from app.ml.registry import RERANKER, ModelSpec
from app.ml.runtime import MAX_LENGTH, bert_feeds, load_tokenizer, onnx_session


class Reranker:
    def __init__(self, spec: ModelSpec = RERANKER, batch_size: int = 16):
        self.batch_size = batch_size
        self.tokenizer = load_tokenizer(spec)
        # Never cut the query; trim the passage if the pair overflows.
        self.tokenizer.enable_truncation(MAX_LENGTH, strategy="only_second")
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.session = onnx_session(spec)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Raw logits, one per passage, in input order."""
        out = np.empty(len(passages), dtype=np.float32)
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer.encode_batch([(query, passages[i]) for i in idx])
            logits = self.session.run(None, bert_feeds(enc, self.session))[0]
            out[idx] = logits[:, 0]
        return out


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))
