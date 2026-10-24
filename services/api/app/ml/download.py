"""Download pinned ONNX models into services/api/models, and build int8 copies where a model
spec asks for one. Usage: uv run python -m app.ml.download"""

from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic

from app.ml.registry import ALL_MODELS


def main() -> None:
    for spec in ALL_MODELS:
        for filename in spec.files:
            hf_hub_download(spec.repo_id, filename, revision=spec.revision, local_dir=spec.path)
        print(f"{spec.name} @ {spec.revision[:8]} -> {spec.path}")
        if spec.int8_file and not (spec.path / spec.int8_file).exists():
            quantize_dynamic(
                spec.path / spec.onnx_file, spec.path / spec.int8_file, weight_type=QuantType.QInt8
            )
            print(f"{spec.name} int8 -> {spec.path / spec.int8_file}")


if __name__ == "__main__":
    main()
