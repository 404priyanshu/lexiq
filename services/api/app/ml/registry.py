"""ONNX models LexIQ runs locally, pinned to exact Hugging Face revisions."""

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    files: tuple[str, ...]
    onnx_file: str

    @property
    def path(self) -> Path:
        return get_settings().models_dir / self.name


EMBEDDER = ModelSpec(
    name="bge-small-en-v1.5",
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    files=("onnx/model.onnx", "tokenizer.json", "config.json"),
    onnx_file="onnx/model.onnx",
)

RERANKER = ModelSpec(
    name="ms-marco-MiniLM-L6-v2",
    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    files=("onnx/model.onnx", "tokenizer.json", "config.json"),
    onnx_file="onnx/model.onnx",
)

ALL_MODELS = (EMBEDDER, RERANKER)
