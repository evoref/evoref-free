"""config.yaml の値を更新するヘルパースクリプト

YAML コメントを保持するため、文字列置換で更新する。

Usage:
    python scripts/update_config_value.py <dotted_key> <value>

Example:
    python scripts/update_config_value.py embedding.backend llama-cpp
"""

import re
import sys
from pathlib import Path


def update_config(config_path: Path, dotted_key: str, value: str) -> None:
    """config.yaml のキーを文字列置換で更新（コメント保持）"""
    text = config_path.read_text(encoding="utf-8")

    # dotted_key の最後のキーを取得（例: "embedding.backend" → "backend"）
    key = dotted_key.split(".")[-1]

    # "  key: old_value" のパターンを検索して置換
    pattern = rf"^(\s*{re.escape(key)}:\s*)(\S+)(.*)$"
    new_text, count = re.subn(pattern, rf"\g<1>{value}\g<3>", text, count=1, flags=re.MULTILINE)

    if count == 0:
        print(f"WARNING: Key '{dotted_key}' not found in {config_path}", file=sys.stderr)
        sys.exit(1)

    config_path.write_text(new_text, encoding="utf-8")
    print(f"  Updated {dotted_key} = {value} in config.yaml")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dotted_key> <value>", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    update_config(config_path, sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
