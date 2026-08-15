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
from backend.io import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("learning.learning_state_store")


@dataclass
class LearningState:
    """`LearningScheduler` の永続化対象状態をまとめた dataclass。

    `LearningStateStore.save` / `load` の入出力型として使い、scheduler 側の
    フィールドを直接 dict 化するパターンを廃止する。
    """

    last_level1_run: float = 0.0
    #: target ("base"/"aux") ごとの最終 Level 2 実行時刻。base の失敗が
    #: aux の overdue 判定まで巻き込んで 24h ブロックしていた回帰
    #: (2026-07-18) の修正で、単一 float から target 別 dict へ分離した。
    last_level2_run: dict[str, float] = field(default_factory=dict)
    #: target ごとの「連続で改善が採用されなかった回数」。Level 2 は 1 サイクル
    #: 1 時間規模の実推論最適化なので、探索が空振りし続ける局面でそのまま
    #: 24h 間隔を回し続けるとリソースを浪費する。連続無改善が続いた target を
    #: 延長クールダウンへ落とすためのカウンタ (採用に成功したら 0 へ戻す)。
    level2_no_improve_streak: dict[str, int] = field(default_factory=dict)
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
            "level2_no_improve_streak": state.level2_no_improve_streak,
            "level1_run_count": state.level1_run_count,
            "last_level1_results": state.last_level1_results,
            "fitness_history": state.fitness_history,
            "prev_correction_rate": state.prev_correction_rate,
            "prev_rag_usage_rate": state.prev_rag_usage_rate,
            "priority_queue": [r.to_dict() for r in state.priority_queue],
        }

    @staticmethod
    def _deserialize_last_level2_run(raw: Any) -> dict[str, float]:
        """``last_level2_run`` を target 別 dict へ正規化する。

        旧フォーマット (単一 float、base/aux 共有) との後方互換: float の
        場合は base/aux 両方に同じ値を適用する (旧仕様では両ターゲットの
        実行がこの単一値を共有更新していたため、片方だけ「未実行」扱いに
        してしまうと移行直後に不要な overdue 発火を招く)。
        """
        if isinstance(raw, dict):
            result: dict[str, float] = {}
            for k, v in raw.items():
                if not isinstance(k, str):
                    continue
                try:
                    result[k] = float(v or 0.0)
                except (TypeError, ValueError):
                    logger.warning(
                        "Skipping malformed last_level2_run entry: %r=%r", k, v,
                    )
            return result
        if isinstance(raw, (int, float)) and raw:
            return {"base": float(raw), "aux": float(raw)}
        return {}

    @staticmethod
    def _deserialize_no_improve_streak(raw: Any) -> dict[str, int]:
        """``level2_no_improve_streak`` を target 別 dict へ正規化する。

        欠損 (旧フォーマット) は空 dict = 全 target ストリーク 0 として扱う。
        """
        if not isinstance(raw, dict):
            return {}
        result: dict[str, int] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            try:
                result[k] = max(0, int(v or 0))
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping malformed level2_no_improve_streak entry: %r=%r", k, v,
                )
        return result

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
            last_level2_run=LearningStateStore._deserialize_last_level2_run(
                data.get("last_level2_run"),
            ),
            level2_no_improve_streak=(
                LearningStateStore._deserialize_no_improve_streak(
                    data.get("level2_no_improve_streak"),
                )
            ),
            level1_run_count=int(data.get("level1_run_count", 0) or 0),
            last_level1_results=dict(data.get("last_level1_results", {}) or {}),
            fitness_history=dict(data.get("fitness_history", {}) or {}),
            prev_correction_rate=data.get("prev_correction_rate"),
            prev_rag_usage_rate=data.get("prev_rag_usage_rate"),
            priority_queue=priority_queue,
        )

    @staticmethod
    def save(state: LearningState, path: str | Path) -> None:
        """`state` を JSON ファイルに書き出す。親ディレクトリは自動作成。

        Level 1 (`_save_state`) と Level 2 (`record_level2_run`) がそれぞれ独立に
        同一ファイルへ書き戻すため、書込み途中のクラッシュや同時読み出しで壊れた
        (truncate された) ファイルを見せないよう原子的に書き込む。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = LearningStateStore.serialize(state)
        atomic_write_text(
            path,
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
