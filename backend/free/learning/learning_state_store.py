"""LearningScheduler の learning_state.json 永続化

`backend.free.learning.scheduler.LearningScheduler` からドメインロジックを
分離するための infra 層。`LearningStateStore` は scheduler 状態
(last_level1_run / fitness_history / priority_queue 等) のシリアライズ /
デシリアライズと JSON ファイル I/O のみを担い、Level1/Level2 実行ロジックや
優先キュー処理等のドメインルールは持たない。

レイヤー責務:
- `LearningScheduler`     — ドメイン (Level1/Level2 実行 / 優先キュー / fitness 算出)
- `LearningStateStore`    — インフラ (learning_state.json 永続化、ファイル I/O)

このため `LearningStateStore` は import 時に `LearningScheduler` を参照せず、
`PriorityRequest` dataclass のみに依存する (循環依存防止 + 単体テスト可能性確保)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.learning.level1_session import PriorityRequest
from backend.log_config import get_logger

logger = get_logger("learning.learning_state_store")


@dataclass
class LearningState:
    """`LearningScheduler` の永続化対象状態をまとめた dataclass。

    `LearningStateStore.save` / `load` の入出力型として使い、scheduler 側の
    フィールドを直接 dict 化するパターンを廃止する。
    """

    last_level1_run: float = 0.0
    last_level2_run: float = 0.0
    level1_run_count: int = 0
    last_level1_results: dict[str, Any] = field(default_factory=dict)
    fitness_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prev_correction_rate: float | None = None
    prev_rag_usage_rate: float | None = None
    priority_queue: list[PriorityRequest] = field(default_factory=list)


class LearningStateStore:
    """LearningScheduler の `learning_state.json` 純粋永続化担当

    全メソッドが副作用ゼロ (ファイル I/O 以外) かつ I/O は static method として
    実装。インスタンス状態を持たないため、テストで stub 化が容易。
    """

    @staticmethod
    def serialize(state: LearningState) -> dict[str, Any]:
        """`LearningState` を JSON-serializable な dict に変換する純粋関数。

        `priority_queue` 内の `PriorityRequest` は `to_dict()` で展開する。
        """
        return {
            "last_level1_run": state.last_level1_run,
            "last_level2_run": state.last_level2_run,
            "level1_run_count": state.level1_run_count,
            "last_level1_results": state.last_level1_results,
            "fitness_history": state.fitness_history,
            "prev_correction_rate": state.prev_correction_rate,
            "prev_rag_usage_rate": state.prev_rag_usage_rate,
            "priority_queue": [r.to_dict() for r in state.priority_queue],
        }

    @staticmethod
    def deserialize(data: dict[str, Any]) -> LearningState:
        """raw JSON dict から `LearningState` を再構築する純粋関数。

        欠損フィールドは既定値で埋め、後方互換 (古い JSON フォーマット) を
        維持する。`priority_queue` の非 dict 要素はスキップする。
        """
        if not isinstance(data, dict):
            return LearningState()

        raw_queue = data.get("priority_queue", []) or []
        priority_queue: list[PriorityRequest] = []
        if isinstance(raw_queue, list):
            for item in raw_queue:
                if not isinstance(item, dict):
                    continue
                try:
                    priority_queue.append(PriorityRequest.from_dict(item))
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        "Skipping malformed priority_queue entry: %r (%s)",
                        item, e,
                    )

        return LearningState(
            last_level1_run=float(data.get("last_level1_run", 0.0) or 0.0),
            last_level2_run=float(data.get("last_level2_run", 0.0) or 0.0),
            level1_run_count=int(data.get("level1_run_count", 0) or 0),
            last_level1_results=dict(data.get("last_level1_results", {}) or {}),
            fitness_history=dict(data.get("fitness_history", {}) or {}),
            prev_correction_rate=data.get("prev_correction_rate"),
            prev_rag_usage_rate=data.get("prev_rag_usage_rate"),
            priority_queue=priority_queue,
        )

    @staticmethod
    def save(state: LearningState, path: str | Path) -> None:
        """`state` を JSON ファイルに書き出す。親ディレクトリは自動作成。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = LearningStateStore.serialize(state)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved learning state to %s", path)

    @staticmethod
    def load(path: str | Path) -> LearningState | None:
        """JSON ファイルから `LearningState` を読み込む。

        ファイルが存在しない場合は `None` を返す (空 state とは区別する)。
        パース失敗時も `None` を返し、警告ログを出力する。
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load learning state from %s: %s", path, e)
            return None
        state = LearningStateStore.deserialize(data)
        logger.info("Loaded learning state from %s", path)
        return state
