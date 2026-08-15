"""モデル配置チェックスクリプト

自動ダウンロードは廃止した。GGUF モデルはユーザーが ``models/`` 配下へ
手動で配置する。本スクリプトは想定モデルファイルと配置状況を表示し、
未配置のものを警告するだけのヘルパーとして残す。

想定ファイル名は ``config.yaml.example`` の ``model_paths`` と整合させること。
"""

import sys
from pathlib import Path


# ── 既定モデルファイル (config.yaml.example の model_paths と整合) ──

EXPECTED_MODELS = {
    "base": "models/gemma-4-12b-it-qat-q4_0.gguf",
    "embed": "models/Qwen3-Embedding-0.6B-Q8_0.gguf",
}


def main() -> int:
    """想定モデルの配置状況を表示する（ダウンロードはしない）"""
    project_root = Path(__file__).parent.parent

    print("Automatic model download is disabled. Place GGUF files under models/ manually.")
    print("Expected models (filenames must match config.yaml model_paths):")
    print()

    missing = 0
    for role, rel in EXPECTED_MODELS.items():
        path = project_root / rel
        exists = path.exists()
        if not exists:
            missing += 1
        status = "found  " if exists else "MISSING"
        print(f"  [{status}] {role:6s} - {rel}")

    print()
    if missing:
        print(f"{missing} model file(s) missing. Place them at the paths above")
        print("(or update model_paths in config.yaml to point at your files).")
        return 1

    print("All expected model files are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
