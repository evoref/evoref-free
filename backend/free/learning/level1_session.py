"""Level 1 セッション + 優先キューのデータクラスと永続化ヘルパー

f_04 §4.2 / §7.1 の設計に従い、Level 1 の中断・再開を可能にする
セッション単位の進捗管理と、アイドル判定をバイパスして Level 1 を要求する
優先キューを提供する。

このモジュールはデータ構造と JSON I/O のみを担当し、スケジューラの
実行ループからは `LearningScheduler` 経由で利用される
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.learning.level0_instant import FeedbackSignals, GenerationConfigRef
from backend.io.codec import codec_for, decode_skipping, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedPayloadFile
from backend.log_config import get_logger
from backend.utils import epoch_to_utc, utc_now, utc_to_epoch

logger = get_logger("learning.level1_session")

#: ``level1_history/`` に残す完了 session の上限 (古いものから削除)。
HISTORY_KEEP = 20


@persisted()
@dataclass
class SnapshotExperience:
    """session の ``experience_snapshot`` の 1 件 (:func:`compact_experience` の射影)。

    ``gen_config`` / ``signals`` は経験 (``learning.experience``) と同じ型で、未知キーは
    各階層の ``_extra`` に残る。メモリ上の session は dict のまま持つ (Level 1 は dict を読む)。
    """

    id: str = ""
    session_id: str = ""
    timestamp: str = ""
    mode: str = ""
    query: str = ""
    base_model: str = ""
    model_key: str | None = None
    gen_config: GenerationConfigRef = field(default_factory=GenerationConfigRef)
    signals: FeedbackSignals = field(default_factory=FeedbackSignals)
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass
class Level1SessionFile:
    """``level1_session_active.json`` (と ``level1_history/``) のペイロード。

    時刻は ISO 8601 UTC μs ``Z`` (epoch を永続化しない、c_05 §0.5.4)。
    """

    session_id: str
    started_at: str | None = None
    cartridge_snapshot: list[str] = field(default_factory=list)
    experience_snapshot: list[SnapshotExperience] = field(default_factory=list)
    completed_phases: list[str] = field(default_factory=list)
    phase_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    reason: str = "idle"
    yield_count: int = 0
    #: 初回 (cutoff 0.0) は null。
    experience_cutoff: str | None = None
    _extra: dict[str, Any] | None = None


#: 中断中の session (``level1_session_active.json``)。利用者のクエリを含む。
LEVEL1_SESSION_FORMAT = register_format(FormatSpec(
    format_id="learning.level1_session",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/prompts/level1_session_active.json",
    retention="at most one; removed when the session completes or is discarded",
    records=(Level1SessionFile,),
))

#: 完了した session の控え (書き手だけで読み手が無い、監査用)。
LEVEL1_HISTORY_FORMAT = register_format(FormatSpec(
    format_id="learning.level1_history",
    version=1,
    klass="volatile",
    writers=frozenset({"free"}),
    path_key="store/learning/<mk>/prompts/level1_history/<session>.json",
    retention=f"latest {HISTORY_KEEP} by mtime",
))

#: ``experience_snapshot`` に残すクエリ本文の最大文字数。
SNAPSHOT_QUERY_CHARS = 200


def compact_experience(exp: dict) -> dict:
    """session へ保存する経験の圧縮射影。

    Level 1 が session snapshot から読むのは id / session_id / timestamp / mode /
    signals / query (失敗語抽出) / gen_config / base_model / model_key だけ。``response_full``
    (応答全文) を 1000 件 × 全モード分そのまま JSON へ書くと active session が
    数 MB になり yield ごとの保存が重くなる (L-D3)。few-shot プール補充は live
    バッファから行うので、応答本文は snapshot に要らない。

    ``id`` / ``session_id`` は落とさない (2026-09-08、F-03 追補)。採用ゲートの
    ケース選定は訂正の宛先を ``signals.corrected_entry_id`` から引くが、
    **参照先の ``id`` が snapshot に無いと解決が丸ごと無効化** され、位置での
    代用 = 別会話のターンとのペアに戻る。2 つとも短い文字列なので L-D3 の
    サイズ懸念には当たらない。
    """
    query = str(exp.get("query", "") or "")
    return {
        "id": exp.get("id", ""),
        "session_id": exp.get("session_id", ""),
        "timestamp": exp.get("timestamp", ""),
        "mode": exp.get("mode", ""),
        "query": query[:SNAPSHOT_QUERY_CHARS],
        "base_model": exp.get("base_model", ""),
        "model_key": exp.get("model_key"),
        # ``rag_usage_rate`` (c_16 §5.5) が ``evidence_ids`` の ``corpus:``
        # を数えるので、gen_config も snapshot に残す。
        "gen_config": dict(exp.get("gen_config", {}) or {}),
        "signals": dict(exp.get("signals", {}) or {}),
    }


# ── PriorityRequest（f_04 §4.2）─────────────────────────────


@dataclass
class PriorityRequest:
    """通常のアイドル判定をバイパスして Level 1 を要求するエントリ

    `reason` ごとに最新値で上書きされる（同 reason の重複は許容しない）。
    """

    reason: str
    requested_at: float
    relax_ratio: float = 1.0  # 経験数閾値の緩和率（0.5 = 半分）
    payload: dict | None = None
    #: 永続形の未知キー (書き戻しで元の位置へ戻す)。
    _extra: dict[str, Any] | None = None

    def to_record(self) -> "PriorityRequestRecord":
        """永続形へ (epoch を永続化しない、c_05 §0.5.4)。"""
        return PriorityRequestRecord(
            reason=self.reason, requested_at=epoch_to_utc(self.requested_at),
            relax_ratio=self.relax_ratio, payload=self.payload, _extra=self._extra,
        )

    @classmethod
    def from_record(cls, record: "PriorityRequestRecord") -> "PriorityRequest":
        return cls(
            reason=record.reason,
            requested_at=utc_to_epoch(record.requested_at, time.time()),
            relax_ratio=record.relax_ratio,
            payload=record.payload,
            _extra=record._extra,
        )


@persisted()
@dataclass
class PriorityRequestRecord:
    """:class:`PriorityRequest` の永続形 (``learning_state.json`` の ``priority_queue`` の要素)。"""

    reason: str
    requested_at: str | None = None
    relax_ratio: float = 1.0
    payload: dict[str, Any] | None = None
    _extra: dict[str, Any] | None = None


# ── Level1Session（f_04 §7.1）───────────────────────────────


@dataclass
class Level1Session:
    """Level 1 の 1 セッション分の進捗

    1 セッションは「カートリッジ snapshot + 経験 snapshot + フェーズ進捗」を持つ。
    yield された場合は `level1_session_active.json` に保存され、再開時に
    そのまま続きから実行できる。
    """

    session_id: str
    started_at: float
    cartridge_snapshot: list[str]      # JSON シリアライズ可能形式
    experience_snapshot: list[dict]    # 開始時に固定された経験リスト (compact_experience 射影)
    completed_phases: list[str] = field(default_factory=list)
    #: mode → yield 時点の進化進捗 (``PromptEvolver._snapshot_progress``:
    #: population / generation / initial_fitness / best)。完了したモードは削除。
    phase_state: dict[str, dict] = field(default_factory=dict)
    reason: str = "idle"
    yield_count: int = 0
    #: session 開始時点の ``LearningScheduler._last_run`` (epoch 秒)。phase7 が
    #: 「前回実行以降の新規経験だけ」を選ぶための固定カットオフ。yield 後の
    #: 再開でも開始時の値を使う (完了して進んだ ``_last_run`` を使うと全件が
    #: 除外される)。0.0 = 初回扱いで全件対象。
    experience_cutoff: float = 0.0
    #: 永続形のトップの未知キー (書き戻しで元の位置へ戻す)。
    _extra: dict[str, Any] | None = None

    @classmethod
    def new(
        cls,
        cartridge_ids: list[str] | set[str] | frozenset[str],
        experiences: list[dict],
        reason: str = "idle",
        experience_cutoff: float = 0.0,
    ) -> "Level1Session":
        return cls(
            session_id=str(uuid.uuid4()),
            # 永続形 (μs) と同じ精度で持つ — 保存 → 読み戻しで値が変わらない
            started_at=utc_to_epoch(utc_now(), time.time()),
            cartridge_snapshot=sorted(cartridge_ids),
            experience_snapshot=[compact_experience(e) for e in experiences],
            completed_phases=[],
            phase_state={},
            reason=reason,
            yield_count=0,
            experience_cutoff=experience_cutoff,
        )

    def to_dict(self) -> dict:
        """永続形 (:class:`Level1SessionFile`) の dict。読めない snapshot の行は飛ばす。

        epoch を永続化しない (c_05 §0.5.4)。cutoff 0.0 (初回) は null。
        """
        snapshot, skipped = codec_for(SnapshotExperience).decode_many(self.experience_snapshot)
        if skipped:
            logger.warning("Dropped %d unreadable experience snapshot row(s) of session %s",
                           skipped, self.session_id)
        return codec_for(Level1SessionFile).encode(Level1SessionFile(
            session_id=self.session_id,
            started_at=epoch_to_utc(self.started_at),
            cartridge_snapshot=self.cartridge_snapshot,
            experience_snapshot=snapshot,
            completed_phases=self.completed_phases,
            phase_state=self.phase_state,
            reason=self.reason,
            yield_count=self.yield_count,
            experience_cutoff=epoch_to_utc(self.experience_cutoff),
            _extra=self._extra,
        ))

    @classmethod
    def from_dict(cls, data: Any) -> "Level1Session":
        """永続形から読む。読めなければ :class:`CodecError` (snapshot の行は 1 行ずつ飛ばす)。"""
        record, skipped = decode_skipping(Level1SessionFile, data, each=("experience_snapshot",))
        if skipped:
            logger.warning("Skipped %d unreadable experience snapshot row(s) of session %s",
                           skipped, record.session_id)
        snapshot = codec_for(SnapshotExperience)
        return cls(
            session_id=record.session_id,
            started_at=utc_to_epoch(record.started_at, time.time()),
            cartridge_snapshot=record.cartridge_snapshot,
            experience_snapshot=[snapshot.encode(row) for row in record.experience_snapshot],
            completed_phases=record.completed_phases,
            phase_state=record.phase_state,
            reason=record.reason,
            yield_count=record.yield_count,
            experience_cutoff=utc_to_epoch(record.experience_cutoff, 0.0),
            _extra=record._extra,
        )


# ── 永続化ヘルパー ────────────────────────────────────────────


def _active_file(path: Path) -> VersionedPayloadFile:
    """ペイロードを :class:`Level1Session` で読み書きする (型の合わないファイルは退避)。"""
    return VersionedPayloadFile(
        LEVEL1_SESSION_FORMAT, path, component="level1_session", state_logger=logger,
        decode=Level1Session.from_dict, encode=Level1Session.to_dict,
    )


def _remove_active_file(path: Path) -> bool:
    """active ファイルを消す。G1 の封筒でない / 版が新しいファイルは消さない。

    消した (または元から無い) なら ``True``。
    """
    f = _active_file(path)
    f.load()
    if f.readonly:
        logger.warning("Leaving active session file %s in place (%s)", path, f.last_status)
        return False
    if path.exists():
        try:
            path.unlink()
        except OSError as e:
            logger.warning("Failed to remove active session file %s: %s", path, e)
            return False
    return True


def save_active_session(path: Path, session: Level1Session) -> None:
    """SUSPENDED な Level1Session を版付き封筒でアトミック保存する

    ディスク上のファイルが G1 の封筒でない / 版が新しいなら上書きしない (WARNING)。
    書き込みの失敗は従来どおり送出する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    f = _active_file(path)
    f.RAISE_ON_SAVE_ERROR = True
    f.load()
    f.payload = session
    if f.save():
        logger.debug("Active session saved: %s", path)


def load_active_session(path: Path) -> Level1Session | None:
    """active session ファイルがあればロードする。無い / 読めないなら None"""
    f = _active_file(path)
    return f.payload if f.load() else None


def archive_session(active_path: Path, history_dir: Path, session: Level1Session) -> Path:
    """完了した session を history へ移動する。

    active ファイルを削除し、`history_dir/{session_id}.json` に書き出す。
    履歴は :data:`HISTORY_KEEP` 件を超えた分を古い順に削除する。
    返り値はアーカイブ後のパス。
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    target = history_dir / f"{session.session_id}.json"
    archived = VersionedPayloadFile(
        LEVEL1_HISTORY_FORMAT, target, component="level1_session", state_logger=logger,
    )
    archived.RAISE_ON_SAVE_ERROR = True
    archived.payload = session.to_dict()
    archived.save()
    _remove_active_file(active_path)
    _prune_history(history_dir, keep=HISTORY_KEEP)
    logger.info("Session archived: %s", target)
    return target


def _prune_history(history_dir: Path, *, keep: int) -> int:
    """history_dir の session JSON を更新日時順に ``keep`` 件だけ残す。"""
    files = sorted(history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    removed = 0
    for old in files[:-keep] if keep > 0 else files:
        try:
            old.unlink()
            removed += 1
        except OSError as e:
            logger.warning("Failed to prune session history %s: %s", old, e)
    if removed:
        logger.info("Pruned %d old Level 1 session archives from %s", removed, history_dir)
    return removed


def discard_active_session(path: Path) -> None:
    """active session ファイルを破棄する（destructive cancel 後の復旧不能ケース等）

    G1 の封筒でない / 版が新しいファイルは破棄しない (このコードの持ち物ではない)。
    """
    existed = path.exists()
    if _remove_active_file(path) and existed:
        logger.info("Active session discarded: %s", path)
