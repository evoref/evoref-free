"""LearnedPattern の JSON 永続化

`backend.free.agent.learned_patterns.LearnedPatternStore` からドメインロジックを
分離するための infra 層。`LearnedPatternRepository` は `LearnedPattern` の
シリアライズ / デシリアライズと JSON ファイル I/O のみを担い、
ドメインルール (重み付け、スコアリング、ストップワード判定、抽出判定) は
持たない。

レイヤー責務:
- `LearnedPatternStore`       — ドメイン (重み更新、マッチング、抽出、上限制御)
- `LearnedPatternRepository`  — インフラ (JSON 永続化、ファイル I/O)

このため `LearnedPatternRepository` は import 時に `LearnedPatternStore` を
参照せず、`LearnedPattern` dataclass のみに依存する (循環依存防止 +
単体テスト可能性確保)。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from backend.free.agent.learned_patterns_types import LearnedPattern
from backend.io.codec import codec_for, decode_skipping, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedPayloadFile
from backend.log_config import get_logger
from backend.utils import epoch_to_utc, utc_to_epoch

logger = get_logger("agent.learned_pattern_store")


@persisted()
@dataclass
class LearnedPatternRecord:
    """``LearnedPattern`` の永続形 (時刻は ISO 8601 UTC μs ``Z``、epoch を永続化しない)。"""

    keyword: str
    category: str = "correction"
    weight: float = 0.5
    hit_count: int = 0
    source_count: int = 1
    first_seen: str | None = None
    last_seen: str | None = None
    last_hit: str | None = None
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class LearnedPatternsFile:
    """``learned_patterns.json`` のペイロード。"""

    patterns: list[LearnedPatternRecord] = field(default_factory=list)
    last_decay_at: str | None = None
    _extra: dict[str, Any] | None = None


#: ``learned_patterns.json`` の形式。ペイロードは :class:`LearnedPatternsFile`
#: (時刻は ISO 8601 UTC μs ``Z``。epoch を永続化しない、c_05 §0.5.4)。
LEARNED_PATTERNS_FORMAT = register_format(FormatSpec(
    format_id="learning.learned_patterns",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/shared/learned_patterns.json",
    retention="capped by LearnedPatternStore (weight decay + max patterns)",
    export=True,
    records=(LearnedPatternsFile,),
))


def _decode_file(payload: Any) -> LearnedPatternsFile:
    """ペイロードを読む。読めないパターンはそれだけ飛ばして数える (c_05 §0.5.2)。"""
    data, skipped = decode_skipping(LearnedPatternsFile, payload, each=("patterns",))
    if skipped:
        logger.warning("Skipped %d unreadable learned pattern(s)", skipped)
    return data


def _patterns_file(path: str | Path) -> VersionedPayloadFile:
    """型の合わないファイルは読み込みで退避する (SoT)。"""
    return VersionedPayloadFile(
        LEARNED_PATTERNS_FORMAT, path,
        component="LearnedPatternRepository", state_logger=logger,
        decode=_decode_file, encode=codec_for(LearnedPatternsFile).encode,
    )


class LearnedPatternRepository:
    """LearnedPattern の純粋な永続化担当

    全メソッドが副作用ゼロ (ファイル I/O 以外) かつ I/O は static method として
    実装。インスタンス状態を持たないため、テストで stub 化が容易。
    """

    @staticmethod
    def save(
        patterns: dict[str, LearnedPattern],
        path: str | Path,
        *,
        last_decay_at: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """`patterns` を版付き封筒で書き出す。親ディレクトリは自動作成。

        ``last_decay_at`` は 6h 減衰間隔の基準時刻を再起動越しに残す。``extra`` は
        読んだファイルのトップの未知キー (:meth:`load_with_meta` の ``_extra``) で、
        そのまま書き戻す。ディスク上のファイルが G1 の封筒でない / 版が新しいなら
        上書きしない (WARNING)。書き込みの失敗は従来どおり送出する。
        """
        f = _patterns_file(path)
        f.RAISE_ON_SAVE_ERROR = True
        f.load()
        f.payload = LearnedPatternsFile(
            patterns=[_pattern_to_record(p) for p in patterns.values()],
            last_decay_at=epoch_to_utc(last_decay_at),
            _extra=extra,
        )
        if f.save():
            logger.info("Saved %d learned patterns to %s", len(patterns), path)

    @staticmethod
    def load(path: str | Path) -> dict[str, LearnedPattern] | None:
        """JSON ファイルから `LearnedPattern` 辞書を読み込む。

        ファイルが存在しない場合、または読めない場合 (G1 の封筒でない / 版が
        新しい / 壊れている) は `None` を返す (空辞書とは区別する)。呼び出し側は
        `None` を「ファイル未存在 / 破損 = 既存状態を保持」と解釈できる。
        """
        loaded = LearnedPatternRepository.load_with_meta(path)
        return None if loaded is None else loaded[0]

    @staticmethod
    def load_with_meta(
        path: str | Path,
    ) -> tuple[dict[str, LearnedPattern], dict[str, Any]] | None:
        """`load` に加えてペイロードのメタ (``last_decay_at`` と未知キーの ``_extra``) も返す。

        キーはパターンの keyword を lower-case 化したもの (空 keyword のエントリは
        無視)。読めないパターンはそれだけ飛ばして数える (c_05 §0.5.2)。ドメイン
        ルール (ストップワード除外等) は呼び出し側で適用する。
        """
        path = Path(path)
        f = _patterns_file(path)
        if not f.load():
            return None
        data: LearnedPatternsFile = f.payload
        patterns: dict[str, LearnedPattern] = {}
        for record in data.patterns:
            key = record.keyword.lower()
            if key:
                patterns[key] = _pattern_from_record(record)
        logger.info("Loaded %d learned patterns from %s", len(patterns), path)
        return patterns, {"last_decay_at": data.last_decay_at, "_extra": data._extra}


# ──────────────────────────────────────────────────────────────────────────
# private 変換 (作業型 ⇔ 永続形、純粋関数)
# ──────────────────────────────────────────────────────────────────────────


#: 作業型では epoch 秒、永続形では ISO 8601 UTC μs ``Z`` のフィールド (c_05 §0.5.4)。
_TIME_FIELDS: tuple[str, ...] = ("first_seen", "last_seen", "last_hit")
_RECORD_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LearnedPatternRecord))


def _pattern_to_record(pattern: LearnedPattern) -> LearnedPatternRecord:
    values = {name: getattr(pattern, name) for name in _RECORD_FIELDS}
    for name in _TIME_FIELDS:
        values[name] = epoch_to_utc(values[name])
    return LearnedPatternRecord(**values)


def _pattern_from_record(record: LearnedPatternRecord) -> LearnedPattern:
    values = {name: getattr(record, name) for name in _RECORD_FIELDS}
    for name in _TIME_FIELDS:
        values[name] = utc_to_epoch(values[name], 0.0)
    return LearnedPattern(**values)
