"""プロンプト履歴 + 本文 / メタストア共通ヘルパ

`SystemPromptManager` と `AuxPromptManager` で重複していた
履歴 / 本文 / メタファイルの読み書き処理を集約する。

両マネージャはキー命名規則だけが異なる:
- システムプロンプト: `{mode}.md` / `{mode}.meta.json` / `history/{mode}_v{NNN}.md`
  → key_prefix = mode (例: "chat", "create")
- 補助タスクプロンプト: `aux_{task}.md` / `aux_{task}.meta.json` /
  `history/aux_{task}_v{NNN}.md`
  → key_prefix = f"aux_{task}" (例: "aux_note_evolve")

ここで定義する関数はすべて key_prefix を引数に取る純粋関数または薄い I/O 委譲。
副作用は引数で受け取った Path 配下のファイル I/O のみで、ドメインロジック
(スコア計算 / 進化判定 / プロンプト保護セクション処理) は一切含まない。
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("agent.prompt_store")

__all__ = [
    "parse_version_from_filename",
    "list_history_entries",
    "archive_to_history",
    "read_history_version",
    "history_file_path",
    "body_file_path",
    "read_body",
    "write_body",
    "body_exists",
    "meta_file_path",
    "read_meta_dict",
    "write_meta_dict",
]


def parse_version_from_filename(filename: str, key_prefix: str) -> int | None:
    """履歴ファイル名からバージョン番号を抽出する。

    `{key_prefix}_v{NNN}.md` 形式のファイル名から `NNN` を整数で取り出す。
    形式が一致しない場合は `None` を返す。

    Args:
        filename: ファイル名 (ディレクトリ部分は含まない)
        key_prefix: モード名または補助タスクタスクのフルプレフィックス

    Returns:
        バージョン番号、または None (パース失敗時)
    """
    full_prefix = f"{key_prefix}_v"
    if not filename.startswith(full_prefix):
        return None
    if not filename.endswith(".md"):
        return None
    try:
        return int(filename[len(full_prefix):-3])
    except ValueError:
        return None


def history_file_path(prompt_dir: Path, key_prefix: str, version: int) -> Path:
    """履歴ファイルのフルパスを返す (存在チェックは行わない)"""
    return prompt_dir / "history" / f"{key_prefix}_v{version:03d}.md"


def list_history_entries(prompt_dir: Path, key_prefix: str) -> list[dict]:
    """指定 key_prefix の履歴エントリ一覧を取得する。

    `prompt_dir/history/{key_prefix}_v*.md` を glob し、ファイル名昇順で
    `[{"version": int, "file": str}, ...]` を返す。`history` ディレクトリが
    存在しない場合は空リスト。

    Args:
        prompt_dir: プロンプトディレクトリ (history サブディレクトリの親)
        key_prefix: 履歴ファイルのプレフィックス

    Returns:
        version / file キーを持つ dict のリスト
    """
    history_dir = prompt_dir / "history"
    if not history_dir.exists():
        return []
    result: list[dict] = []
    for p in sorted(history_dir.glob(f"{key_prefix}_v*.md")):
        version = parse_version_from_filename(p.name, key_prefix)
        if version is not None:
            result.append({"version": version, "file": p.name})
    return result


def archive_to_history(
    prompt_dir: Path,
    key_prefix: str,
    version: int,
    content: str,
) -> None:
    """指定バージョンの履歴ファイルに content を書き込む。

    `prompt_dir/history/` が存在しない場合は作成する。
    既存の同バージョンファイルがあれば上書きする。

    Args:
        prompt_dir: プロンプトディレクトリ
        key_prefix: 履歴ファイルのプレフィックス
        version: アーカイブ対象のバージョン番号
        content: 書き込む本文
    """
    history_dir = prompt_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    dst = history_file_path(prompt_dir, key_prefix, version)
    dst.write_text(content, encoding="utf-8")


def read_history_version(
    prompt_dir: Path,
    key_prefix: str,
    version: int,
) -> str:
    """指定バージョンの履歴ファイル本文を読み込む。

    Args:
        prompt_dir: プロンプトディレクトリ
        key_prefix: 履歴ファイルのプレフィックス
        version: 読み込むバージョン番号

    Returns:
        履歴ファイルの本文

    Raises:
        FileNotFoundError: 該当バージョンのファイルが存在しない場合
    """
    src = history_file_path(prompt_dir, key_prefix, version)
    if not src.exists():
        raise FileNotFoundError(f"Version not found: {src}")
    return src.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# 本文 (.md) ファイル I/O
# ──────────────────────────────────────────────────────────────────────────


def body_file_path(prompt_dir: Path, key_prefix: str) -> Path:
    """本文ファイル `{prompt_dir}/{key_prefix}.md` のフルパスを返す。"""
    return prompt_dir / f"{key_prefix}.md"


def body_exists(prompt_dir: Path, key_prefix: str) -> bool:
    """本文ファイルが存在するか判定する。"""
    return body_file_path(prompt_dir, key_prefix).exists()


def read_body(prompt_dir: Path, key_prefix: str) -> str:
    """本文ファイルを UTF-8 で読み込む。

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    path = body_file_path(prompt_dir, key_prefix)
    return path.read_text(encoding="utf-8")


def write_body(prompt_dir: Path, key_prefix: str, content: str) -> None:
    """本文ファイルへ UTF-8 で書き込む。親ディレクトリは自動作成。"""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = body_file_path(prompt_dir, key_prefix)
    path.write_text(content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# メタ (.meta.json) ファイル I/O — dict ベースで dataclass 非依存
# ──────────────────────────────────────────────────────────────────────────


def meta_file_path(prompt_dir: Path, key_prefix: str) -> Path:
    """メタファイル `{prompt_dir}/{key_prefix}.meta.json` のフルパスを返す。"""
    return prompt_dir / f"{key_prefix}.meta.json"


def read_meta_dict(prompt_dir: Path, key_prefix: str) -> dict | None:
    """メタファイルを JSON として読み込み dict を返す。

    - ファイルが存在しない場合: `None`
    - パース失敗時: 警告ログを出して `None`

    呼び出し側 (各 PromptManager) は受け取った dict を自身の dataclass に
    ハイドレートする責務を負う (この関数は dataclass を知らない)。
    """
    path = meta_file_path(prompt_dir, key_prefix)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load prompt meta from %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Prompt meta at %s is not a JSON object, ignoring", path)
        return None
    return data


def write_meta_dict(prompt_dir: Path, key_prefix: str, data: dict) -> None:
    """メタ dict を JSON として書き込む。親ディレクトリは自動作成。"""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = meta_file_path(prompt_dir, key_prefix)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
