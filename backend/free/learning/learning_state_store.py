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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.learning.level1_session import PriorityRequest, PriorityRequestRecord
from backend.io.codec import codec_for, decode_skipping, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedPayloadFile
from backend.log_config import get_logger
from backend.utils import epoch_to_utc, utc_to_epoch

logger = get_logger("learning.learning_state_store")


@persisted()
@dataclass
class LearningStateFile:
    """``learning_state.json`` のペイロード (時刻は ISO 8601 UTC μs ``Z``、未実行は null)。"""

    last_level1_run: str | None = None
    last_level2_run: dict[str, str | None] = field(default_factory=dict)
    level2_no_improve_streak: dict[str, int] = field(default_factory=dict)
    level1_run_count: int = 0
    last_level1_results: dict[str, Any] = field(default_factory=dict)
    fitness_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prev_correction_rate: float | None = None
    prev_rag_usage_rate: float | None = None
    priority_queue: list[PriorityRequestRecord] = field(default_factory=list)
    prompt_adoptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _extra: dict[str, Any] | None = None


#: ``learning_state.json`` の形式。ペイロードは :class:`LearningStateFile`。
LEARNING_STATE_FORMAT = register_format(FormatSpec(
    format_id="learning.learning_state",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/prompts/learning_state.json",
    retention="one per partition",
    export=True,
    records=(LearningStateFile,),
))


def _state_file(path: str | Path) -> VersionedPayloadFile:
    """ペイロードを :class:`LearningState` で読み書きする (型の合わないファイルは退避)。"""
    return VersionedPayloadFile(
        LEARNING_STATE_FORMAT, path,
        component="LearningStateStore", state_logger=logger,
        decode=LearningStateStore.deserialize, encode=LearningStateStore.serialize,
    )


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
    #: mode → 採用直後のプロンプトを自動ロールバックで監視するための基準
    #: (``rollback_to`` / ``baseline`` / ``adopted_at`` / ``windows``)。
    #: 監視が完了 (合格 or rollback) したら mode ごと削除される。
    prompt_adoptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 読んだファイルのトップの未知キー (書き戻しで元の位置へ戻す、c_05 §0.5.2)。
    _extra: dict[str, Any] | None = None


class LearningStateStore:
    """LearningScheduler の `learning_state.json` 純粋永続化担当

    全メソッドが副作用ゼロ (ファイル I/O 以外) かつ I/O は static method として
    実装。インスタンス状態を持たないため、テストで stub 化が容易。
    """

    @staticmethod
    def serialize(state: LearningState) -> dict[str, Any]:
        """`LearningState` を永続形 (:class:`LearningStateFile`) の dict にする純粋関数。"""
        return codec_for(LearningStateFile).encode(LearningStateFile(
            # epoch を永続化しない (c_05 §0.5.4)。0.0 (未実行) は null。
            last_level1_run=epoch_to_utc(state.last_level1_run),
            last_level2_run={k: epoch_to_utc(v) for k, v in state.last_level2_run.items()},
            level2_no_improve_streak=state.level2_no_improve_streak,
            level1_run_count=state.level1_run_count,
            last_level1_results=state.last_level1_results,
            fitness_history=state.fitness_history,
            prev_correction_rate=state.prev_correction_rate,
            prev_rag_usage_rate=state.prev_rag_usage_rate,
            priority_queue=[r.to_record() for r in state.priority_queue],
            prompt_adoptions=state.prompt_adoptions,
            _extra=state._extra,
        ))

    @staticmethod
    def deserialize(data: Any) -> LearningState:
        """永続形の dict から `LearningState` を再構築する純粋関数。

        読めなければ :class:`CodecError`。読めない ``priority_queue`` の要素は
        それだけ飛ばす (c_05 §0.5.2)。
        """
        record, skipped = decode_skipping(LearningStateFile, data, each=("priority_queue",))
        if skipped:
            logger.warning("Skipped %d malformed priority_queue entr(ies)", skipped)
        return LearningState(
            last_level1_run=utc_to_epoch(record.last_level1_run, 0.0),
            last_level2_run={k: utc_to_epoch(v, 0.0) for k, v in record.last_level2_run.items()},
            level2_no_improve_streak=record.level2_no_improve_streak,
            level1_run_count=record.level1_run_count,
            last_level1_results=record.last_level1_results,
            fitness_history=record.fitness_history,
            prev_correction_rate=record.prev_correction_rate,
            prev_rag_usage_rate=record.prev_rag_usage_rate,
            priority_queue=[PriorityRequest.from_record(r) for r in record.priority_queue],
            prompt_adoptions=record.prompt_adoptions,
            _extra=record._extra,
        )

    @staticmethod
    def save(state: LearningState, path: str | Path) -> None:
        """`state` を JSON ファイルに書き出す。親ディレクトリは自動作成。

        Level 1 (`_save_state`) と Level 2 (`record_level2_run`) がそれぞれ独立に
        同一ファイルへ書き戻すため、書込み途中のクラッシュや同時読み出しで壊れた
        (truncate された) ファイルを見せないよう原子的に書き込む。ディスク上の
        ファイルが G1 の封筒でない / 版が新しいなら上書きしない (WARNING)。
        書き込みの失敗は従来どおり送出する。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = _state_file(path)
        f.RAISE_ON_SAVE_ERROR = True
        f.load()
        f.payload = state
        if f.save():
            logger.info("Saved learning state to %s", path)

    @staticmethod
    def load(path: str | Path) -> LearningState | None:
        """版付き封筒から `LearningState` を読み込む。

        ファイルが存在しない場合は `None` を返す (空 state とは区別する)。
        読めない場合 (G1 の封筒でない / 版が新しい / 壊れている) も `None`。
        """
        path = Path(path)
        f = _state_file(path)
        if not f.load():
            return None
        logger.info("Loaded learning state from %s", path)
        return f.payload
