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

from pathlib import Path
from typing import Any, TypeVar

from backend.io import atomic_write_text
from backend.io.readonly import is_readonly
from backend.io.codec import codec_for
from backend.io.format_registry import FormatSpec
from backend.io.versioned import VersionedPayloadFile
from backend.log_config import get_logger

logger = get_logger("agent.prompt_store")

T = TypeVar("T")

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
    "read_meta",
    "write_meta",
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

    データ根が readonly なら書かない (``save_ledger`` と同じ。起動時の既定の生成を
    落とさない、c_05 §0.4.2)。
    """
    if is_readonly():
        return
    history_dir = prompt_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    dst = history_file_path(prompt_dir, key_prefix, version)
    atomic_write_text(dst, content, encoding="utf-8")


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
    """本文ファイルへ UTF-8 で書き込む。親ディレクトリは自動作成。readonly なら書かない。"""
    if is_readonly():
        return
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = body_file_path(prompt_dir, key_prefix)
    atomic_write_text(path, content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# メタ (.meta.json) ファイル I/O — 永続 dataclass (:mod:`backend.io.codec`) で読み書き
# ──────────────────────────────────────────────────────────────────────────


def meta_file_path(prompt_dir: Path, key_prefix: str) -> Path:
    """メタファイル `{prompt_dir}/{key_prefix}.meta.json` のフルパスを返す。"""
    return prompt_dir / f"{key_prefix}.meta.json"


def _meta_file(prompt_dir: Path, key_prefix: str, record: type[Any], spec: FormatSpec) -> VersionedPayloadFile:
    """ペイロードを ``record`` (永続 dataclass) で読み書きするメタファイル。

    型の合わないペイロードは読み込みで ``corrupt`` (退避) になる。未知キーは各階層の
    ``_extra`` に残り、書き戻しで元の位置へ戻る。
    """
    codec = codec_for(record)
    return VersionedPayloadFile(
        spec, meta_file_path(prompt_dir, key_prefix), component="prompt_store",
        state_logger=logger, decode=codec.decode, encode=codec.encode,
    )


def read_meta(prompt_dir: Path, key_prefix: str, record: type[T], *, spec: FormatSpec) -> T | None:
    """メタファイル (``spec`` の版付き封筒) を ``record`` の値として読む。

    ファイルが無い / 読めない (G1 の封筒でない・版が新しい・壊れている・型が合わない)
    場合は ``None``。readonly / 退避の扱いは :class:`VersionedPayloadFile` に従う。
    """
    f = _meta_file(prompt_dir, key_prefix, record, spec)
    if not f.load():
        return None
    return f.payload


def write_meta(prompt_dir: Path, key_prefix: str, meta: Any, *, spec: FormatSpec) -> None:
    """``meta`` (永続 dataclass) を ``spec`` の版付き封筒で書き込む。親ディレクトリは自動作成。

    書く前にディスク上のファイルを分類する — G1 の封筒でない / 版が新しいファイルは
    上書きせず WARNING を出して見送り、壊れたファイルは退避してから書く。
    書き込みの失敗は従来どおり送出する。データ根が readonly なら書かない。
    """
    if is_readonly():
        return
    prompt_dir.mkdir(parents=True, exist_ok=True)
    f = _meta_file(prompt_dir, key_prefix, type(meta), spec)
    f.RAISE_ON_SAVE_ERROR = True
    f.load()
    f.payload = meta
    f.save()
