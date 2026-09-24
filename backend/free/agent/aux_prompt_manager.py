"""補助タスク タスク別プロンプト管理（§7.1.2）

補助タスクごとに専用プロンプトを持ち、読込み・更新・履歴管理を行う。
SystemPromptManager の補助タスク版。全エディション共通。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.free.agent._prompt_store_helpers import (
    archive_to_history,
    body_exists,
    list_history_entries,
    meta_file_path,
    read_body,
    read_history_version,
    read_meta,
    write_body,
    write_meta,
)
from backend.free.agent.prompt_utils import (
    dedupe_paragraphs,
    restore_protected_sections,
    validate_protected_sections,
)
from backend.io.codec import persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.log_config import get_logger
from backend.utils import utc_now as _now

logger = get_logger("agent.aux_prompt_manager")


@persisted()
@dataclass
class AuxPromptMeta:
    """補助タスクプロンプトメタ情報 (``aux_{task}.meta.json`` のペイロード。未知キーは ``_extra``)"""
    task: str
    version: int = 1
    updated_at: str = ""
    source: str = "default"  # "default" | "manual" | "evolution"
    fitness_score: float = 0.0
    _extra: dict[str, Any] | None = None


#: ``aux_{task}.meta.json`` (``AuxPromptMeta``) の形式。本文 ``.md`` と ``history/`` は封筒を持たない。
AUX_PROMPT_META_FORMAT = register_format(FormatSpec(
    format_id="learning.aux_prompts",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/aux_prompts/aux_<task>.meta.json",
    retention="one per task (history/ keeps every body version)",
    export=True,
    records=(AuxPromptMeta,),
))
#: 本文 ``aux_<task>.md`` (封筒を持たない)。
AUX_PROMPT_BODY_FORMAT = register_format(FormatSpec(
    format_id="learning.aux_prompt_body",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/aux_prompts/aux_<task>.md",
    retention="one per task",
    export=True,
    encodings=("md",),
))
#: 本文の過去版 (``history/aux_<task>_v<NNN>.md``)。
AUX_PROMPT_HISTORY_FORMAT = register_format(FormatSpec(
    format_id="learning.aux_prompt_history",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/aux_prompts/history/aux_<task>_v<n>.md",
    retention="unbounded (text only)",
    export=True,
    encodings=("md",),
))


# デフォルトプロンプト
DEFAULT_AUX_PROMPTS: dict[str, str] = {
    "note_evolve": """\
# メモリノート文脈説明生成

以下のメモリノートの暗黙的な意味・トピック・重要性を捉えた簡潔な文脈説明を生成してください。
この説明は将来の検索精度向上に使用されます。

## 要件
- 1〜2文で簡潔に記述
- ノートの核心的な情報を要約
- 検索キーワードとして機能する語彙を含める

<!-- PROTECTED -->
## 出力形式
文脈説明のみを出力してください。前置きや説明は不要です。
<!-- /PROTECTED -->
""",
}


class AuxPromptManager:
    """補助タスク タスク別プロンプト管理

    タスクごとに .md（本文）+ .meta.json（メタ情報）を管理する。
    SystemPromptManager と同じパターンだが、モードではなくタスク単位。
    """

    TASKS = ["note_evolve"]

    def __init__(self, prompt_dir: Path) -> None:
        """
        Args:
            prompt_dir: プロンプトディレクトリ（local/prompts/）
        """
        self.prompt_dir = prompt_dir
        self.contents: dict[str, str] = {}
        self.metas: dict[str, AuxPromptMeta] = {}
        self._load_all()

    def _load_all(self) -> None:
        """起動時に全タスクのプロンプトとメタ情報をロード"""
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        for task in self.TASKS:
            key = f"aux_{task}"
            if body_exists(self.prompt_dir, key):
                self.contents[task] = read_body(self.prompt_dir, key)
                meta = read_meta(self.prompt_dir, key, AuxPromptMeta, spec=AUX_PROMPT_META_FORMAT)
                # 読めないまま残ったメタ (G1 の封筒でない / 版が新しい) は source が
                # 分からないので、本文を既定値と見なして書き直さない。
                meta_unreadable = (
                    meta is None and meta_file_path(self.prompt_dir, key).exists()
                )
                if meta is not None:
                    self.metas[task] = meta
                else:
                    self.metas[task] = AuxPromptMeta(task=task)
                    self._save_meta(task)
                # 自己修復: 未編集 (source=default) のままコード側デフォルトが
                # 更新された場合、ディスクの旧デフォルトを現行デフォルトに書き直し、
                # デフォルト更新を既存インストールへ伝播させる。手動編集 (manual) /
                # 進化 (evolution) で更新されたプロンプトは保護する。
                if not meta_unreadable:
                    self._refresh_default_if_stale(task)
            else:
                self._create_default(task)

    def get_aux_prompt(self, task: str) -> str:
        """タスク別プロンプト本文を取得

        Args:
            task: タスク名 ("note_evolve")

        Returns:
            プロンプト本文

        Raises:
            ValueError: 不明なタスク名
        """
        if task not in self.contents:
            raise ValueError(f"Unknown aux task: {task}")
        return self.contents[task]

    def get_meta(self, task: str) -> AuxPromptMeta:
        """メタ情報を取得"""
        if task not in self.metas:
            raise ValueError(f"Unknown aux task: {task}")
        return self.metas[task]

    def update_aux_prompt(
        self,
        task: str,
        content: str,
        fitness: float,
    ) -> None:
        """Level 1 進化: 最良候補を採用

        保護セクション（<!-- PROTECTED --> マーカー）が現在のプロンプトに含まれている場合、
        進化候補がそれを維持しているか検証し、欠落時は強制復元する。

        Args:
            task: タスク名
            content: 新しいプロンプト本文
            fitness: 適応度スコア
        """
        if task not in self.TASKS:
            raise ValueError(f"Unknown aux task: {task}")

        # 保護セクション最終ゲート
        from backend.free.agent.prompt_manager import _normalized_equal
        current = self.contents.get(task, "")

        # 段落レベル重複を最終正規化
        content = dedupe_paragraphs(content)

        if not validate_protected_sections(current, content):
            logger.warning(
                "Evolved aux prompt for %s lost protected sections, force-restoring",
                task,
            )
            content = restore_protected_sections(current, content)
            content = dedupe_paragraphs(content)

        # 意味的同一性ガード - 正規化後 current と同じなら no-op
        if _normalized_equal(current, content):
            logger.warning(
                "Evolved aux prompt for %s is semantically identical to current "
                "(fitness=%.3f), skipping update",
                task, fitness,
            )
            return

        self._archive_current(task)
        write_body(self.prompt_dir, f"aux_{task}", content)
        self.contents[task] = content
        meta = self.metas[task]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "evolution"
        meta.fitness_score = fitness
        self._save_meta(task)
        logger.info(
            "Aux prompt evolved: task=%s, version=%d, fitness=%.3f",
            task, meta.version, fitness,
        )

    def update_manual(self, task: str, content: str) -> None:
        """手動編集によるプロンプト更新"""
        if task not in self.TASKS:
            raise ValueError(f"Unknown aux task: {task}")
        self._archive_current(task)
        write_body(self.prompt_dir, f"aux_{task}", content)
        self.contents[task] = content
        meta = self.metas[task]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "manual"
        self._save_meta(task)
        logger.info("Aux prompt manual update: task=%s, version=%d", task, meta.version)

    def get_history(self, task: str) -> list[dict]:
        """タスクの履歴一覧を取得"""
        if task not in self.TASKS:
            raise ValueError(f"Unknown aux task: {task}")
        return list_history_entries(self.prompt_dir, f"aux_{task}")

    def rollback(self, task: str, version: int) -> None:
        """特定バージョンにロールバック"""
        if task not in self.TASKS:
            raise ValueError(f"Unknown aux task: {task}")
        content = read_history_version(self.prompt_dir, f"aux_{task}", version)
        self._archive_current(task)
        write_body(self.prompt_dir, f"aux_{task}", content)
        self.contents[task] = content
        meta = self.metas[task]
        meta.version += 1
        meta.updated_at = _now()
        meta.source = "manual"
        self._save_meta(task)
        logger.info(
            "Aux prompt rollback: task=%s to v%03d, new version=%d",
            task, version, meta.version,
        )

    def _archive_current(self, task: str) -> None:
        """現在のプロンプトを history/ に退避"""
        if task not in self.contents:
            return
        archive_to_history(
            self.prompt_dir,
            f"aux_{task}",
            self.metas[task].version,
            self.contents[task],
        )

    def _refresh_default_if_stale(self, task: str) -> None:
        """source=default のプロンプトがコード側デフォルトと乖離していれば再生成する。

        デフォルト更新を既存インストールへ伝播させるための起動時自己修復。
        ``source`` が ``manual`` (手動編集) / ``evolution`` (進化採用) のものは
        ユーザー資産として保護し、書き換えない。``source=default`` かつディスク
        内容が現行 ``DEFAULT_AUX_PROMPTS[task]`` と異なる場合のみ書き直す。
        一致していれば no-op (毎回起動で再書き込みしない / 冪等)。
        """
        meta = self.metas.get(task)
        if meta is None or meta.source != "default":
            return
        default = DEFAULT_AUX_PROMPTS.get(task)
        if default is None or self.contents.get(task) == default:
            return
        write_body(self.prompt_dir, f"aux_{task}", default)
        self.contents[task] = default
        meta.version = 1
        meta.updated_at = _now()
        meta.source = "default"
        meta.fitness_score = 0.0
        self._save_meta(task)
        logger.info(
            "Refreshed stale default aux prompt: task=%s "
            "(on-disk default differed from current code default)",
            task,
        )

    def _create_default(self, task: str) -> None:
        """デフォルトプロンプトを生成"""
        content = DEFAULT_AUX_PROMPTS.get(
            task, f"# aux_{task}\nDefault aux prompt.\n",
        )
        write_body(self.prompt_dir, f"aux_{task}", content)
        self.contents[task] = content
        self.metas[task] = AuxPromptMeta(
            task=task, version=1, updated_at=_now(), source="default",
        )
        self._save_meta(task)
        logger.info("Created default aux prompt: task=%s", task)

    def _save_meta(self, task: str) -> None:
        """メタ情報を JSON ファイルに保存 (infra 層 `_prompt_store_helpers` に委譲)"""
        write_meta(self.prompt_dir, f"aux_{task}", self.metas[task], spec=AUX_PROMPT_META_FORMAT)
