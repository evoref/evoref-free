"""旧モード名 ``"coding"`` → 現行名 ``"create"`` の一度きり移行。

コーディングモードのクリエイトモードへの改名に伴い、既存インストールの ``local/``
に残る旧名を現行名へ揃える。マーカー ``local/.mode_renamed_create_v1`` で一度だけ
実行し、以降の起動では即 no-op になる。

設計上の約束:

- **構造フィールドだけを書き換える**。会話本文やファクト本文にも "coding" は普通に
  現れる (改名前の設計書名 ``f_10_staged_coding_pipeline.md`` への言及など) ため、テキストの
  一括置換は内容を壊す。書き換えるのは次の 3 種のみ:

  1. mode 値を保持する既知フィールド (:data:`_MODE_VALUE_FIELDS`) の値
  2. ``{"chat": ..., "coding": ...}`` 形のモードキー辞書のキー
  3. SemMem の ``subject`` / ``type`` セグメント

- ディレクトリ / プロンプトファイル名の ``coding`` セグメントを ``create`` へ改名する。
- staged パイプラインの一時ワークスペース (``create_workspace_dir``) は中間成果物
  置き場なので移行しない。旧 config で ``local/coding/`` を指したままでも動作する。
- best-effort。個別ファイルの失敗は WARNING を出して続行し、マーカーは完了時のみ書く。
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.config import PathResolver

logger = get_logger("core.mode_rename_migrator")

MARKER_NAME = ".mode_renamed_create_v1"

_OLD_MODE = "coding"
_NEW_MODE = "create"

#: mode 値を保持する既知のフィールド名。ここに載っている名前の文字列値だけを
#: mode として読み替える (本文フィールドを巻き込まないための allowlist)。
_MODE_VALUE_FIELDS: frozenset[str] = frozenset({
    "mode",
    "mode_origin",
    "mode_key",
    "last_mode",
    "session_mode",
    "active_mode",
})

#: FactType 値の読み替え。
_TYPE_RENAMES: dict[str, str] = {
    "coding": "create",
    "coding_task": "create_task",
}

#: SemMem ``mem.*`` subject の kind セグメント読み替え (長い方を先に判定する)。
_SUBJECT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("mem.coding_task.", "mem.create_task."),
    ("mem.coding.", "mem.create."),
)

#: JSON 走査から除外するディレクトリ名 (パスのどのセグメントでも一致で除外)。
#: 埋め込み・ログ・バイナリ版管理は mode を構造として持たず、巨大なため。
_SKIP_DIRS: frozenset[str] = frozenset({
    "logs", "vectors", "cache", "models",
    "lora_versions", "lora_archive", "cvector", "migration_archive",
})

#: 走査対象の上限サイズ (これを超える JSON は構造データではないとみなす)。
_MAX_JSON_BYTES = 64 * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────
# 値の読み替え
# ──────────────────────────────────────────────────────────────────────────


def migrate_subject(subject: str) -> str:
    """SemMem subject の mode / FactType セグメントを現行名へ読み替える。

    ``mem.*`` は kind セグメント (``mem.coding.*`` / ``mem.coding_task.*``) のみ、
    ``learn.*`` は mode 次元がどの位置にあっても (``learn.<kind>.<mode>.*`` /
    ``learn.<kind>.<model>.<mode>.*`` の両形式が存在する) 対象にする。

    ``mem.preference.coding_style`` のような領域語は対象外 (改名していない)。
    """
    for old, new in _SUBJECT_PREFIXES:
        if subject.startswith(old):
            return new + subject[len(old):]
    if subject.startswith("learn."):
        segs = subject.split(".")
        if _OLD_MODE in segs:
            return ".".join(_NEW_MODE if s == _OLD_MODE else s for s in segs)
    return subject


def migrate_node(node: Any, field: str | None = None) -> tuple[Any, int]:
    """構造フィールドだけを読み替えた値と、書き換え件数を返す。"""
    if isinstance(node, dict):
        # {"chat": ..., "coding": ...} 形だけをモードキー辞書とみなす。
        mode_keyed = "chat" in node and _OLD_MODE in node
        out: dict[Any, Any] = {}
        count = 0
        for key, value in node.items():
            new_key = key
            if mode_keyed and key == _OLD_MODE:
                new_key = _NEW_MODE
                count += 1
            child, n = migrate_node(value, key if isinstance(key, str) else None)
            out[new_key] = child
            count += n
        return out, count
    if isinstance(node, list):
        out_list = []
        count = 0
        for value in node:
            child, n = migrate_node(value, field)
            out_list.append(child)
            count += n
        return out_list, count
    if isinstance(node, str):
        if field in _MODE_VALUE_FIELDS and node == _OLD_MODE:
            return _NEW_MODE, 1
        if field == "type" and node in _TYPE_RENAMES:
            return _TYPE_RENAMES[node], 1
        if field == "subject":
            migrated = migrate_subject(node)
            if migrated != node:
                return migrated, 1
    return node, 0


# ──────────────────────────────────────────────────────────────────────────
# 移行本体
# ──────────────────────────────────────────────────────────────────────────


class ModeRenameMigrator:
    """``"coding"`` → ``"create"`` の一度きり移行。"""

    def __init__(self, resolver: "PathResolver") -> None:
        self._resolver = resolver

    # ── マーカー ─────────────────────────────────────────

    def _local_root(self) -> Path:
        return self._resolver.resolve_local("local_state_file").parent

    def _marker_path(self) -> Path:
        return self._local_root() / MARKER_NAME

    def already_migrated(self) -> bool:
        return self._marker_path().exists()

    # ── メイン ───────────────────────────────────────────

    def migrate_if_needed(self) -> bool:
        """必要なら移行を実行する。実行したら ``True``、済 / 不要なら ``False``。"""
        if self.already_migrated():
            return False
        root = self._local_root()
        if not root.is_dir():
            return False

        renamed = self._rename_paths()
        rewritten, files = self._migrate_json_tree(root)

        marker = self._marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "version": 1,
                    "migrated_at": utc_now(),
                    "renamed_paths": renamed,
                    "rewritten_fields": rewritten,
                    "rewritten_files": files,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        if renamed or rewritten:
            logger.info(
                "Mode rename migration done: %d paths renamed, "
                "%d fields rewritten across %d files (coding -> create)",
                renamed, rewritten, files,
            )
        return True

    # ── パス改名 ─────────────────────────────────────────

    def _rename_paths(self) -> int:
        """学習パーティションとプロンプトの ``coding`` セグメントを改名する。"""
        targets: list[Path] = []
        try:
            learning_dir = self._resolver.resolve_local("learning_dir")
        except KeyError:  # pragma: no cover - LOCAL_DEFAULTS 由来なので通常起きない
            learning_dir = None
        if learning_dir and learning_dir.is_dir():
            # <stem>/coding/ と assist/<...>/coding/ (mode パーティション)
            targets.extend(learning_dir.glob("*/coding"))
            targets.extend(learning_dir.glob("*/*/coding"))
            # <stem>/prompts/coding.md, coding.meta.json, history/coding_v*.md
            targets.extend(learning_dir.glob("*/prompts/coding.*"))
            targets.extend(learning_dir.glob("*/prompts/history/coding_v*.md"))
        prompts_dir = self._local_root() / "prompts"
        if prompts_dir.is_dir():
            targets.extend(prompts_dir.glob("coding.*"))
            targets.extend(prompts_dir.glob("history/coding_v*.md"))

        count = 0
        for src in targets:
            dst = src.with_name(src.name.replace(_OLD_MODE, _NEW_MODE, 1))
            if dst.exists():
                logger.warning(
                    "Mode rename: target already exists, leaving %s as is", src,
                )
                continue
            try:
                src.rename(dst)
            except OSError as exc:
                logger.warning("Mode rename: failed to rename %s: %s", src, exc)
                continue
            count += 1
        return count

    # ── JSON / JSONL 書き換え ────────────────────────────

    def _migrate_json_tree(self, root: Path) -> tuple[int, int]:
        """``local/`` 配下の JSON / JSONL の構造フィールドを読み替える。"""
        fields = 0
        files = 0
        for path in root.rglob("*"):
            if path.suffix not in (".json", ".jsonl"):
                continue
            if not path.is_file():
                continue
            if _SKIP_DIRS & set(path.relative_to(root).parts[:-1]):
                continue
            try:
                if path.stat().st_size > _MAX_JSON_BYTES:
                    continue
                n = (
                    self._migrate_jsonl_file(path) if path.suffix == ".jsonl"
                    else self._migrate_json_file(path)
                )
            except (OSError, ValueError) as exc:
                logger.warning("Mode rename: skipped %s (%s)", path, exc)
                continue
            if n:
                fields += n
                files += 1
        return fields, files

    def _migrate_json_file(self, path: Path) -> int:
        data = json.loads(path.read_text(encoding="utf-8"))
        migrated, count = migrate_node(data)
        if count:
            path.write_text(
                json.dumps(migrated, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return count

    def _migrate_jsonl_file(self, path: Path) -> int:
        lines = path.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        total = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append(line)
                continue
            migrated, count = migrate_node(json.loads(stripped))
            total += count
            out.append(json.dumps(migrated, ensure_ascii=False) if count else line)
        if total:
            path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return total


__all__ = [
    "MARKER_NAME",
    "ModeRenameMigrator",
    "migrate_node",
    "migrate_subject",
]
