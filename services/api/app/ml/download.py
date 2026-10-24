"""Download pinned ONNX models into services/api/models. Usage: uv run python -m app.ml.download"""

from huggingface_hub import hf_hub_download

from app.ml.registry import ALL_MODELS


def main() -> None:
    for spec in ALL_MODELS:
        for filename in spec.files:
            hf_hub_download(spec.repo_id, filename, revision=spec.revision, local_dir=spec.path)
        print(f"{spec.name} @ {spec.revision[:8]} -> {spec.path}")


if __name__ == "__main__":
    main()
