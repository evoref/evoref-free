"""config.yaml の model_paths を更新するヘルパースクリプト。

setup.sh / setup.bat から呼び出される。
YAML コメントを保持するため、文字列置換方式で更新する。
"""
import sys
import pathlib


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: configure_shared_path.py <shared_path> <gguf_path>")
        print("  shared_path: モデルパスのルート")
        print("  gguf_path:   GGUF モデルファイルのフルパス")
        return 1

    sys.argv[1].replace("\\", "/")
    gguf = sys.argv[2].replace("\\", "/")

    config_path = pathlib.Path("config.yaml")
    if not config_path.exists():
        print("  ERROR: config.yaml not found")
        return 1

    content = config_path.read_text(encoding="utf-8")
    updated = False

    # model_paths.base_model の更新
    old_model = "base_model: models/gemma-4-12b-it-qat-q4_0.gguf"
    if old_model in content:
        content = content.replace(old_model, f"base_model: {gguf}")
        updated = True
    else:
        print("  WARNING: base_model default value not found in config.yaml")
        print("           Please update model_paths.base_model manually.")

    if updated:
        config_path.write_text(content, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
