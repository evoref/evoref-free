"""補助タスク経験バッファ（§7.5.2）

補助タスク (RAG 必要性判定 / RAG 品質判定 / ツール判定 / ノート進化) の出力と
その成否を記録し、Level 1 の補助プロンプト進化の学習信号として使う。

補助タスク撤去前は Pro 限定 (``backend/pro/aux_experience.py``) だったが、
判定自体がベースモデルへ移り Free でも同じ信号が取れるようになったため Free 側へ
移設した。永続化先は ``local_paths.aux_experience_file``。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from backend.io import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("learning.aux_experience")

MAX_ENTRIES = 1000


@dataclass
class AuxTaskExperience:
    """補助タスクの出力と成否を記録"""

    action_type: str          # "rag_necessity" | "rag_quality" | "tool_call" | "note_evolve"
    input_context: str        # 入力コンテキスト（クエリ・メモリ状態等）
    output: str               # 補助タスクの出力
    outcome: float            # 成功度（0.0〜1.0）
    cartridge_ids: list[str]  # カートリッジフィルタ用
    timestamp: float          # Unix timestamp
    # 発生元セッションのモード ("chat"/"create")。ExperienceEntry.mode
    # (backend/free/learning/level0_instant.py) と同じ非検証ポリシー
    # (自由文字列、action_type のような ValueError 検証はしない)。
    mode: str = "chat"


class AuxExperienceBuffer:
    """補助タスク経験バッファ: 処理結果を記録・永続化する

    - append() は同期メソッド（asyncio 安全）
    - save() は非同期永続化（アトミックリネーム）
    - load() は起動時読込み
    - get_filtered() はカートリッジフィルタ適用済みリストを返す
    """

    VALID_ACTION_TYPES = frozenset({
        "rag_necessity", "rag_quality", "tool_call", "note_evolve",
    })

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self.entries: list[AuxTaskExperience] = []
        self._lock = asyncio.Lock()

    def append(self, exp: AuxTaskExperience) -> None:
        """経験エントリを追加（同期・スレッドセーフ）

        Args:
            exp: 補助タスクの経験エントリ

        Raises:
            ValueError: 不正な action_type または outcome 値
        """
        if exp.action_type not in self.VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action_type: {exp.action_type}. "
                f"Must be one of {self.VALID_ACTION_TYPES}"
            )
        if not (0.0 <= exp.outcome <= 1.0):
            raise ValueError(
                f"outcome must be between 0.0 and 1.0, got {exp.outcome}"
            )

        if exp.timestamp == 0.0:
            exp.timestamp = time.time()

        self.entries.append(exp)

        # ローテーション
        if len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            self.entries = self.entries[overflow:]
            logger.info("Rotated %d old aux experience entries", overflow)

    def record(
        self,
        action_type: str,
        input_context: str,
        output: str,
        outcome: float,
        cartridge_ids: list[str] | None = None,
        mode: str = "chat",
    ) -> None:
        """primitive 値から AuxTaskExperience を構築して append する簡易記録 API。

        composition / factory 層が dataclass を import せずに記録できるよう
        buffer 側で組み立てる。outcome は
        ``[0.0, 1.0]`` にクランプする。action_type は append() が検証する。
        ``mode`` は呼び出し元 (chat.py の ``ChatRequest.mode`` に由来する明示的な
        引数リレー、``ExperienceEntry.mode`` と同じ思想) から渡される。
        """
        self.append(AuxTaskExperience(
            action_type=action_type,
            input_context=input_context,
            output=output,
            outcome=max(0.0, min(1.0, outcome)),
            cartridge_ids=list(cartridge_ids or []),
            timestamp=0.0,
            mode=mode,
        ))

    async def save(self, path: Path) -> None:
        """非同期永続化（アトミックリネーム）

        一時ファイルに書き込み後リネームすることで、
        書き込み中のクラッシュでもデータが破損しない。
        """
        async with self._lock:
            path = Path(path)
            data = [asdict(e) for e in self.entries]
            content = json.dumps(data, ensure_ascii=False, indent=2)

            # ファイル I/O をスレッドプールで実行 (AtomicWriter が tmp 書込 →
            # Windows retry 付き os.replace + 並行プロセス間の tmp 名衝突回避を担う)。
            await asyncio.to_thread(atomic_write_text, path, content)
            logger.info(
                "Saved %d aux experience entries to %s",
                len(self.entries), path,
            )

    def load(self, path: Path) -> None:
        """起動時読込み（同期）

        ファイルが存在しない場合は何もしない。
        """
        path = Path(path)
        if not path.exists():
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.entries.clear()
        for d in data:
            try:
                entry = AuxTaskExperience(
                    action_type=d["action_type"],
                    input_context=d.get("input_context", ""),
                    output=d.get("output", ""),
                    outcome=float(d.get("outcome", 0.0)),
                    cartridge_ids=d.get("cartridge_ids", []),
                    timestamp=float(d.get("timestamp", 0.0)),
                    mode=d.get("mode", "chat"),
                )
                self.entries.append(entry)
            except (KeyError, ValueError) as e:
                logger.warning("Skipped invalid aux experience entry: %s", e)

        logger.info(
            "Loaded %d aux experience entries from %s",
            len(self.entries), path,
        )

    def get_filtered(
        self,
        cartridge_ids: frozenset[str] | None = None,
        mode: str | None = None,
    ) -> list[AuxTaskExperience]:
        """カートリッジ/モードフィルタ適用済みリストを返す

        Args:
            cartridge_ids: 現在装着中のカートリッジ ID セット。
                None の場合はフィルタなし（全件返却）。
            mode: 指定時はそのモード ("chat"/"create") のエントリのみに絞る。
                None (省略、既定) の場合は全モード横断 (後方互換)。

        Returns:
            フィルタ適用済みのエントリリスト
        """
        result = list(self.entries)
        if cartridge_ids is not None:
            result = [
                e for e in result
                if not e.cartridge_ids or set(e.cartridge_ids) <= cartridge_ids
            ]
        if mode is not None:
            result = [e for e in result if e.mode == mode]
        return result

    @property
    def count(self) -> int:
        """エントリ数"""
        return len(self.entries)


__all__ = ["AuxExperienceBuffer", "AuxTaskExperience"]
