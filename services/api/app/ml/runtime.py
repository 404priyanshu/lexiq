"""Shared ONNX Runtime + tokenizer plumbing for the local BERT-family models."""

import numpy as np
import onnxruntime as ort
from tokenizers import Encoding, Tokenizer

from app.ml.registry import ModelSpec

MAX_LENGTH = 512


def model_path_exists(spec: ModelSpec) -> bool:
    return all((spec.path / f).exists() for f in spec.files)


def load_tokenizer(spec: ModelSpec) -> Tokenizer:
    """Raw tokenizer with no padding or truncation; callers enable what they need."""
    path = spec.path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run python -m app.ml.download`")
    tok = Tokenizer.from_file(str(path))
    tok.no_padding()
    tok.no_truncation()
    return tok


def onnx_session(spec: ModelSpec) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(spec.path / spec.onnx_file), opts, providers=["CPUExecutionProvider"]
    )


def bert_feeds(enc: list[Encoding], session: ort.InferenceSession) -> dict[str, np.ndarray]:
    feeds = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
    }
    names = {i.name for i in session.get_inputs()}
    return {k: v for k, v in feeds.items() if k in names}
