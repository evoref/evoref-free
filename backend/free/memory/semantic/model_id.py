"""埋め込みモデル名の正規化 (``embeddings/<model_id>/`` のディレクトリ名)。

ストアのメタ情報そのものは
:class:`~backend.free.rag.evidence.EvidenceManifest` (c_16 §5.1) が持つ。
ここに残っているのは ``config.yaml::embedding.model_name`` (ファイル名や
大文字混じりで来る) を、比較とディレクトリ名に使える形へ揃える 1 関数だけ。
"""

from __future__ import annotations

from pathlib import Path

#: 剥がす既知のモデルファイル拡張子 (末尾から 1 回だけ)。
EMBEDDING_MODEL_SUFFIXES: tuple[str, ...] = (
    ".gguf", ".bin", ".safetensors", ".onnx", ".pt", ".pth",
)


def normalize_embedding_model_id(model_name: str) -> str:
    """``embedding.model_name`` を ``model_id`` に正規化する。

    例:
    - ``"bge-m3-q8_0.gguf"`` → ``"bge-m3-q8_0"``
    - ``"models/qwen3-embedding-4b.onnx"`` → ``"qwen3-embedding-4b"``
    - ``"BGE-M3"`` → ``"bge-m3"``

    basename を取り出し (``models/`` 等のディレクトリ部を除去)、既知の拡張子を
    末尾から 1 回だけ剥がし、ASCII 小文字化する (比較しやすさ優先)。
    """
    name = (model_name or "").strip()
    if not name:
        raise ValueError("embedding model_name must be non-empty")
    base = Path(name).name.lower()
    for suffix in EMBEDDING_MODEL_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or name.lower()


__all__ = ["EMBEDDING_MODEL_SUFFIXES", "normalize_embedding_model_id"]
