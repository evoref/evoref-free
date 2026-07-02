"""

``develop=evolve`` レベルで出力される ``decision.jsonl`` / ``outcome.jsonl``
を継続的に tail-follow し、``trace_id`` で causal join した
:class:`JoinedPair` を非同期イテレータとして PolicyAdjuster (EvorefLearn
pillar) に供給する Loop pillar コンポーネント。

責務範囲 (CLAUDE.md §8 — pillar 境界):

* ``backend/free/loop/`` 配下に閉じる (Loop pillar)
* SemMem には書き込まない (それは PolicyAdjuster の責務)
* ``backend/free/learning/`` を import しない (Loop → Learn 禁止)

実装方針:

* tail follow は ``watchdog`` を使わず polling (1〜3 秒間隔) + 末尾 offset
  永続化で軽量実装。loop iteration が 30 秒〜数分単位なので秒オーダー
  の遅延は許容
* schema_version 不一致時は WARN + skip (loop の継続性優先)
* ローテーション検出: ファイルサイズが offset 未満 → ``.1`` リネーム
  発生と判断し、``.1`` ファイルの末尾を読み切ってから新ファイルへ
* decision / outcome は別ファイルなので **trace_id 単位の in-memory
  bounded LRU buffer (1024 件)** で JOIN
* 一定時間 (デフォルト 5 分) outcome が来ない decision は orphan として
  PolicyAdjuster に送る (フォールバック後にクラッシュした等の解析用)
* offset 永続化先は ``local/state/log_ingestor.json`` (再起動後の続き
  から読む)

JoinedPair の構造:

* ``decision``: ``decision.jsonl`` の 1 行 (dict)
* ``outcome``: ``outcome.jsonl`` の 1 行 (dict) または orphan 検出時 None

複数の decision エントリが同一 trace_id を共有する典型ケース (1 chat
リクエスト中に 3 つの decision_point を踏むなど) では、1 つの outcome
に対して N 個の JoinedPair が emit される。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.io import atomic_write_text
from backend.log_config import get_logger

logger = get_logger("loop.log_ingestor")


# ──────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────


SUPPORTED_SCHEMA_VERSION = 1
"""LogIngestor がサポートする ``schema_version``

異なる値のエントリは WARN + skip で読み飛ばし、loop の継続性を優先する
(fail-fast せず) 設計。将来 schema 移行時はここを bump して旧 version
の処理を分岐する。
"""

DEFAULT_POLL_INTERVAL_SEC = 2.0
"""ファイルポーリング間隔 (秒)。"""

DEFAULT_LRU_MAX = 1024
"""trace_id 単位の JOIN バッファ最大件数。超過時は最古 trace_id を orphan
として emit。"""

DEFAULT_ORPHAN_TIMEOUT_SEC = 300.0
"""decision を受信してから outcome を待つ最大時間 (秒、5 分)。経過した
trace_id は orphan として emit する。"""

DEFAULT_IO_BACKOFF_SEC = 60.0
"""I/O 失敗 (disk full / permission error 等) 時のバックオフ待機 (秒)。
永続的にループしないよう、ポーリング失敗後は本値だけ待ってからリトライ。
"""


# ──────────────────────────────────────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JoinedPair:
    """``decision`` と対応する ``outcome`` の causal join 結果。

    Attributes:
        trace_id: causal join のキー (``contextvars`` 由来 12 文字 hex)。
        decision: ``decision.jsonl`` の 1 エントリ (dict)。
        outcome: ``outcome.jsonl`` の 1 エントリ。orphan の場合 ``None``。
        is_orphan: ``outcome`` が ``None`` のとき ``True``。フォールバック
            後にクラッシュした等の解析用シグナル。
    """

    trace_id: str
    decision: dict[str, Any]
    outcome: dict[str, Any] | None

    @property
    def is_orphan(self) -> bool:
        return self.outcome is None


@dataclass
class _BufferedDecisions:
    """trace_id 単位の decision バッファ (LRU エントリ内部表現)。

    Attributes:
        decisions: 受信した decision エントリのリスト (受信順)。
        first_ts: 初回 decision を受信した monotonic 時刻 (orphan 判定用)。
    """

    decisions: list[dict[str, Any]] = field(default_factory=list)
    first_ts: float = 0.0


# ──────────────────────────────────────────────────────────────────────────
# 内部ヘルパ: offset 永続化
# ──────────────────────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """``path`` に JSON を原子的に書き込む (:func:`backend.io.atomic_write_text` 経由)。

    途中クラッシュで壊れた JSON を残さないため。Windows の書き込み競合時
    リトライと並行プロセス間の tmp 名衝突回避は ``backend.io`` が担う。
    """
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )


@dataclass
class _FileOffset:
    """1 ファイルの追跡情報。

    ``inode`` (Linux の st_ino / Windows の file index) で「同じ物理ファイル
    かどうか」を判定し、サイズ比較だけに頼らず堅牢にローテーション
    (rename) を検出する。Windows でも ``Path.stat().st_ino`` は GetFile
    InformationByHandle の nFileIndex に対応しており、rename を跨いで
    安定する。

    Attributes:
        inode: ``stat().st_ino`` の値 (0 はサポート対象外システムを表す
            sentinel として扱い、該当時はサイズ比較のみのフォールバック
            にする)。
        offset: 既読 byte 数 (再起動後はここから再開する)。
    """

    inode: int
    offset: int


def _load_offsets(state_path: Path) -> dict[str, _FileOffset]:
    """``state_path`` から offset map を読み込む。存在しない / 壊れて
    いる場合は空 dict を返す (起動時自然デフォルト)。"""
    if not state_path.exists():
        return {}
    try:
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "LogIngestor offset state unreadable, starting from scratch: %s",
            exc,
        )
        return {}
    raw = data.get("offsets", {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, _FileOffset] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        # 新形式: {"inode": int, "offset": int}
        if isinstance(value, dict):
            inode = value.get("inode")
            offset = value.get("offset")
            if isinstance(inode, (int, float)) and isinstance(offset, (int, float)):
                result[key] = _FileOffset(inode=int(inode), offset=int(offset))
        # 旧形式 (互換破棄、空オフセット扱い)
        # ※ 後方互換不要 (CLAUDE.md) のため bare int は読まない。
    return result


# ──────────────────────────────────────────────────────────────────────────
# LogIngestor
# ──────────────────────────────────────────────────────────────────────────


class LogIngestor:
    """``decision.jsonl`` / ``outcome.jsonl`` を tail-follow し JOIN する。

    使用例 (PolicyAdjuster からの典型結線)::

        ingestor = LogIngestor(debug_log_dir=..., state_path=...)
        await ingestor.start()
        try:
            async for pair in ingestor.stream_pairs():
                await adjuster.consume(pair)
        finally:
            await ingestor.stop()

    本クラスは Loop pillar 内に閉じ、PolicyAdjuster (Learn pillar) からは
    ``stream_pairs()`` の AsyncIterator 経由でのみアクセスする。
    """

    def __init__(
        self,
        debug_log_dir: Path,
        state_path: Path,
        *,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        lru_max: int = DEFAULT_LRU_MAX,
        orphan_timeout_sec: float = DEFAULT_ORPHAN_TIMEOUT_SEC,
        io_backoff_sec: float = DEFAULT_IO_BACKOFF_SEC,
        queue_max: int = 4096,
    ) -> None:
        if lru_max <= 0:
            raise ValueError(f"lru_max must be positive, got {lru_max}")
        if poll_interval_sec <= 0:
            raise ValueError(
                f"poll_interval_sec must be positive, got {poll_interval_sec}",
            )
        self.debug_log_dir = debug_log_dir
        self.state_path = state_path
        self.poll_interval_sec = poll_interval_sec
        self.lru_max = lru_max
        self.orphan_timeout_sec = orphan_timeout_sec
        self.io_backoff_sec = io_backoff_sec

        # filename -> _FileOffset (inode + byte offset)。新規ファイル名は
        # polling 時に追加。inode 比較で rename ベースの rotation を堅牢に
        # 検出する。
        self._offsets: dict[str, _FileOffset] = {}
        # trace_id -> _BufferedDecisions (LRU 順序維持)
        self._lru: OrderedDict[str, _BufferedDecisions] = OrderedDict()
        # JOIN 結果の AsyncIterator 用 queue
        self._queue: asyncio.Queue[JoinedPair] = asyncio.Queue(maxsize=queue_max)
        # tail-follow タスク制御
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # 観測カウンタ (テスト / デバッグ用、本番では未使用)
        self._stats: dict[str, int] = {
            "decisions_seen": 0,
            "outcomes_seen": 0,
            "pairs_emitted": 0,
            "orphans_emitted": 0,
            "schema_mismatches": 0,
            "malformed_lines": 0,
            "rotations_detected": 0,
        }

    # ------------------------------------------------------------------
    # ライフサイクル
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """offset を読み込み、ポーリングループを bg task として起動する。"""
        if self._task is not None:
            raise RuntimeError("LogIngestor already started")
        self._offsets = _load_offsets(self.state_path)
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "LogIngestor started: dir=%s state=%s offsets=%d",
            self.debug_log_dir, self.state_path, len(self._offsets),
        )

    async def stop(self) -> None:
        """ポーリングループを停止し、最終 orphan flush と offset 永続化を行う。

        shutdown 整合性のため、本メソッドは:

        1. ``_stop_event`` をセットしてループを抜けさせる
        2. 残バッファの全 decision を orphan として queue に push
        3. offset を永続化
        4. ``stream_pairs()`` 側で残 queue を drain できるよう sentinel を push
        """
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=self.poll_interval_sec * 3 + 5.0)
        except asyncio.TimeoutError:
            logger.warning("LogIngestor stop timed out, cancelling task")
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        # 最終 orphan flush
        await self._flush_remaining_orphans()
        # 最終 offset 永続化 (ループ内でも書いているが、shutdown 時の最新値を保証)
        try:
            self._save_offsets()
        except OSError as exc:
            logger.warning("Final offset save failed: %s", exc)

    # ------------------------------------------------------------------
    # 公開イテレータ
    # ------------------------------------------------------------------

    async def stream_pairs(self) -> AsyncIterator[JoinedPair]:
        """JOIN 済 :class:`JoinedPair` を非同期に yield する。

        ``stop()`` が呼ばれて queue が空になると終端する。
        """
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                return
            try:
                pair = await asyncio.wait_for(
                    self._queue.get(), timeout=self.poll_interval_sec,
                )
            except asyncio.TimeoutError:
                # stop が立っているか確認しつつポーリング継続
                continue
            yield pair

    # ------------------------------------------------------------------
    # 統計取得 (テスト / 観測用)
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """内部カウンタのスナップショットを返す (テスト / 観測用)。"""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # 内部実装: ポーリングループ
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._poll_once()
            except Exception as exc:
                # disk full / permission error 等。ログだけ出して continue。
                logger.warning(
                    "LogIngestor poll failed (will backoff %ss): %s",
                    self.io_backoff_sec, exc,
                )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.io_backoff_sec,
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.poll_interval_sec,
                )
            except asyncio.TimeoutError:
                pass

    async def _poll_once(self) -> None:
        """1 回分のポーリング: 新規行を読み、JOIN し、orphan を flush する。"""
        # ディレクトリ存在チェック (起動直後に未生成のケース)
        if not self.debug_log_dir.exists():
            return

        # decision_*.jsonl と outcome_*.jsonl をソートして処理
        # (古い日付 → 新しい日付の順、同名 .1/.2 ローテ世代も含む)
        decision_files = sorted(self.debug_log_dir.glob("decision_*.jsonl"))
        outcome_files = sorted(self.debug_log_dir.glob("outcome_*.jsonl"))

        for path in decision_files:
            await self._read_new_lines(path, "decision")
        for path in outcome_files:
            await self._read_new_lines(path, "outcome")

        # orphan 判定 + flush
        await self._flush_orphans()

        # offset 永続化 (atomic write)
        self._save_offsets()

    async def _read_new_lines(self, path: Path, category: str) -> None:
        """``path`` の前回 offset 以降を読み、各行を JOIN バッファに投入する。

        ローテーション検出 (inode 変化 / size < offset) 時は ``.1`` 世代の
        末尾を吸い取ってから本ファイルを offset=0 で読む。
        """
        try:
            st = path.stat()
        except OSError as exc:
            logger.warning("stat failed for %s: %s", path.name, exc)
            return
        size = st.st_size
        # st_ino は Linux の inode、Windows では nFileIndex (rename 跨ぎで
        # 安定)。0 を返すシステム (古い WSL 等) は inode 比較を諦めサイズ
        # 比較だけにフォールバックする。
        inode = int(st.st_ino) if st.st_ino else 0
        prev = self._offsets.get(path.name)

        if prev is None:
            # 初回読み込み (offset=0 から末尾まで)
            await self._read_from(path, 0, category, inode=inode)
            return

        rotation_detected = False
        if inode and prev.inode and inode != prev.inode:
            # inode 変化 = rotation (rename → 新規ファイル作成)
            rotation_detected = True
        elif size < prev.offset:
            # サイズ縮小 = rotation または truncate
            rotation_detected = True

        if rotation_detected:
            self._stats["rotations_detected"] += 1
            rotated = path.with_suffix(path.suffix + ".1")
            if rotated.exists():
                # 旧ファイル末尾を吸い取る (旧 inode と一致確認できるなら)
                try:
                    rotated_inode = int(rotated.stat().st_ino) or 0
                except OSError:
                    rotated_inode = 0
                if not prev.inode or not rotated_inode or rotated_inode == prev.inode:
                    await self._read_rotated_tail(rotated, prev.offset, category)
            # 本ファイルは offset=0 から再走査
            await self._read_from(path, 0, category, inode=inode)
            return

        if size == prev.offset:
            return  # 新規行なし
        await self._read_from(path, prev.offset, category, inode=inode)

    async def _read_from(
        self, path: Path, offset: int, category: str, *, inode: int,
    ) -> None:
        """``path`` の ``offset`` から末尾までを読み、行ごとに処理する。

        ``offset`` は ``self._offsets[path.name]`` に inode と一緒に書き戻す。
        """
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError as exc:
            logger.warning("read failed for %s: %s", path.name, exc)
            return

        new_offset = offset + len(data)
        self._offsets[path.name] = _FileOffset(inode=inode, offset=new_offset)

        await self._dispatch_lines(data, category, path.name)

    async def _read_rotated_tail(
        self, rotated: Path, offset: int, category: str,
    ) -> None:
        """``.1`` 世代ファイルの未読領域を 1 回だけ吸い取る (offset 永続化なし)。"""
        try:
            with open(rotated, "rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError as exc:
            logger.warning("read failed for rotated %s: %s", rotated.name, exc)
            return
        await self._dispatch_lines(data, category, rotated.name)

    async def _dispatch_lines(
        self, data: bytes, category: str, source_name: str,
    ) -> None:
        """raw bytes を行ごとにパースして ``_handle_entry`` に流す共通ヘルパ。"""
        text = data.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                self._stats["malformed_lines"] += 1
                logger.warning(
                    "malformed JSONL line in %s: %s", source_name, exc,
                )
                continue
            if not isinstance(entry, dict):
                self._stats["malformed_lines"] += 1
                continue
            await self._handle_entry(category, entry, source_name)

    async def _handle_entry(
        self, category: str, entry: dict[str, Any], source_file: str,
    ) -> None:
        """1 エントリを schema 検証 + JOIN バッファ投入する。"""
        schema_version = entry.get("schema_version")
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            self._stats["schema_mismatches"] += 1
            logger.warning(
                "schema_version mismatch in %s: got=%r expected=%d (skipped)",
                source_file, schema_version, SUPPORTED_SCHEMA_VERSION,
            )
            return

        trace_id = entry.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            # trace_id が無いエントリは causal join 不可能なので skip
            # (起動初期や非同期境界越えで欠落するエッジケース)
            return

        if category == "decision":
            self._stats["decisions_seen"] += 1
            await self._buffer_decision(trace_id, entry)
        elif category == "outcome":
            self._stats["outcomes_seen"] += 1
            await self._consume_outcome(trace_id, entry)

    async def _buffer_decision(
        self, trace_id: str, entry: dict[str, Any],
    ) -> None:
        """decision を LRU バッファに格納。LRU 超過なら最古を orphan として emit。"""
        buf = self._lru.get(trace_id)
        if buf is None:
            buf = _BufferedDecisions(decisions=[], first_ts=time.monotonic())
            self._lru[trace_id] = buf
        buf.decisions.append(entry)
        self._lru.move_to_end(trace_id)

        # LRU 超過時の eviction (最古を orphan で emit)
        while len(self._lru) > self.lru_max:
            evicted_tid, evicted = self._lru.popitem(last=False)
            for d in evicted.decisions:
                await self._emit_orphan(evicted_tid, d)

    async def _consume_outcome(
        self, trace_id: str, outcome: dict[str, Any],
    ) -> None:
        """outcome 受信時、該当 trace_id の全 decision と JOIN して emit する。

        decision がまだ来ていない trace_id (outcome 先着) は単純に skip。
        ``decision.jsonl`` と ``outcome.jsonl`` は別ファイルだが書込時刻は
        ほぼ同期するため、再順序化はバッファサイズが十分なら問題にならない。
        """
        buf = self._lru.pop(trace_id, None)
        if buf is None:
            return
        for d in buf.decisions:
            await self._emit_pair(trace_id, d, outcome)

    async def _emit_pair(
        self, trace_id: str, decision: dict[str, Any], outcome: dict[str, Any],
    ) -> None:
        pair = JoinedPair(trace_id=trace_id, decision=decision, outcome=outcome)
        await self._queue.put(pair)
        self._stats["pairs_emitted"] += 1

    async def _emit_orphan(
        self, trace_id: str, decision: dict[str, Any],
    ) -> None:
        pair = JoinedPair(trace_id=trace_id, decision=decision, outcome=None)
        await self._queue.put(pair)
        self._stats["orphans_emitted"] += 1

    async def _flush_orphans(self) -> None:
        """orphan_timeout を超えた trace_id を orphan として emit する。"""
        now = time.monotonic()
        cutoff = now - self.orphan_timeout_sec
        # 古い順に走査 (OrderedDict は挿入順保持)
        expired: list[str] = []
        for trace_id, buf in self._lru.items():
            if buf.first_ts < cutoff:
                expired.append(trace_id)
            else:
                # OrderedDict は move_to_end で更新されるため
                # first_ts も挿入順と相関する。ここで break すると
                # 後続のより新しいエントリは確実に未経過。
                break
        for trace_id in expired:
            buf = self._lru.pop(trace_id)
            for d in buf.decisions:
                await self._emit_orphan(trace_id, d)

    async def _flush_remaining_orphans(self) -> None:
        """shutdown 時に残バッファ全件を orphan で emit する。"""
        while self._lru:
            trace_id, buf = self._lru.popitem(last=False)
            for d in buf.decisions:
                await self._emit_orphan(trace_id, d)

    # ------------------------------------------------------------------
    # offset 永続化
    # ------------------------------------------------------------------

    def _save_offsets(self) -> None:
        """現在の offset map を ``state_path`` に atomic 書き込みする。"""
        payload = {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "offsets": {
                name: {"inode": fo.inode, "offset": fo.offset}
                for name, fo in self._offsets.items()
            },
        }
        _atomic_write_json(self.state_path, payload)


__all__ = [
    "DEFAULT_IO_BACKOFF_SEC",
    "DEFAULT_LRU_MAX",
    "DEFAULT_ORPHAN_TIMEOUT_SEC",
    "DEFAULT_POLL_INTERVAL_SEC",
    "JoinedPair",
    "LogIngestor",
    "SUPPORTED_SCHEMA_VERSION",
]
