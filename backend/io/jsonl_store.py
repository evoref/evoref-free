"""append-only JSONL ストア + atomic compaction

1 ファイル = 1 種類のレコード集合を JSONL (1 行 1 JSON) で永続化する。書き込み
の主要 API は :meth:`append` で、行を末尾に追記するだけ。削除は :meth:`tombstone`
で「削除マーカー行」を append する形を取る (物理削除は :meth:`maybe_compact`
の compaction で行う)。

トレードオフ:

- ファクト追加 1 件あたり O(1) (1 行 append) — 旧 dict 全件 dump の O(N) を回避
- ファイルが線形に伸びるため、定期的な :meth:`maybe_compact` で物理サイズと
  読込時間を縮小する (閾値超過時のみ ``AtomicWriter`` で再書き出し)

行の版 (c_05 §0.5.1):

- ``row_version`` を渡したストアは全ての行 (tombstone 行を含む) に ``_v`` を持つ。
  追記する行は ``serialize`` が現行の ``_v`` を刻む (刻んでいなければ ``ValueError``)。
- 読むとき ``_v`` の無い / 整数でない行は壊れた行として飛ばす。``_v`` がこの版より
  新しい行は読まずに数え (:meth:`newer_rows`)、そのファイルを **readonly** にする —
  追記・tombstone は :class:`DataReadonlyError`、compaction は書き直さない
  (新しい版の行を落として畳まない、c_05 §0.4.5)。

スレッド安全性:

- プロセス内の並行 ``append`` / ``tombstone`` / ``maybe_compact`` /
  ``load_all`` は ``threading.Lock`` で直列化する。multi-process write は
  対象外 (advisory lock を取らない、SemMem は 1 プロセス前提)。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from backend.io.atomic import AtomicWriter
from backend.io.readonly import DataReadonlyError, guard_write
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("io.jsonl_store")

__all__ = [
    "JSONLAppendStore",
    "ROW_VERSION_FIELD",
    "TOMBSTONE_KEY_FIELD",
    "TOMBSTONE_MARKER",
    "TOMBSTONE_REASON_FIELD",
    "TOMBSTONE_TS_FIELD",
]

T = TypeVar("T")

#: tombstone 行を識別するための予約フィールド。
#: tombstone 行は ``{"_tombstone": true, "_key": "<id>"}`` の形で書かれる。
#: serialize 関数の出力にこれらの予約フィールドを含めないこと (検知時 ValueError)。
TOMBSTONE_MARKER = "_tombstone"
TOMBSTONE_KEY_FIELD = "_key"
#: tombstone を書いた時刻 (ISO 8601 μs, ``Z``) と理由。以前は印とキーだけで、
#: compaction 後は削除の事実ごと消え「いつ / なぜ消えたか」を後から辿れなかった
#: (2026-09-05 監査)。
TOMBSTONE_TS_FIELD = "_deleted_at"
TOMBSTONE_REASON_FIELD = "_reason"
#: 行の版 (c_05 §0.5.1。``row_version`` を渡したストアの全行が持つ)。
ROW_VERSION_FIELD = "_v"

_ROW_OK, _ROW_BAD, _ROW_NEWER = 0, 1, 2


class JSONLAppendStore(Generic[T]):
    """append-only JSONL ストア。

    Args:
        path: JSONL ファイルパス。
        serialize: item を 1 行の JSON 文字列に変換する関数 (改行を含めないこと)。
        deserialize: 1 行の JSON 文字列から item を復元する関数。
        key_of: item から compaction 用の一意キーを抽出する関数。重複キーは
            **後勝ち** (後の行が前の行を上書き)。
        compact_threshold_lines: この行数を **超えた時のみ** :meth:`maybe_compact`
            が compaction 実行候補にする。デフォルト 1000。
        compact_threshold_ratio: ``live_count * ratio < total_lines`` の場合に
            compaction を実行する閾値。デフォルト 2.0 (= live が全行の 50% 未満)。
        debug_logger: 注入された ``DebugLogger`` があれば compaction 時に
            ``log_memory_op("jsonl_compact", ...)`` で観測情報を流す。
        row_version: 行の版 (``_v``、形式の版と同じ値)。渡すと全行が ``_v`` を持つ
            前提で読み書きする (モジュールの docstring)。

    Usage:
        >>> store = JSONLAppendStore[dict](
        ...     path,
        ...     serialize=lambda d: json.dumps(d, ensure_ascii=False),
        ...     deserialize=json.loads,
        ...     key_of=lambda d: d["id"],
        ... )
        >>> store.append({"id": "a", "v": 1})
        >>> store.append({"id": "b", "v": 2})
        >>> store.tombstone("a")
        >>> store.load_all()
        {'b': {'id': 'b', 'v': 2}}
    """

    def __init__(
        self,
        path: Path | str,
        serialize: Callable[[T], str],
        deserialize: Callable[[str], T],
        key_of: Callable[[T], str],
        *,
        compact_threshold_lines: int = 1000,
        compact_threshold_ratio: float = 2.0,
        debug_logger: "DebugLogger | None" = None,
        row_version: int | None = None,
    ) -> None:
        if compact_threshold_lines < 1:
            raise ValueError("compact_threshold_lines must be >= 1")
        if compact_threshold_ratio <= 1.0:
            raise ValueError("compact_threshold_ratio must be > 1.0")
        self._path = Path(path)
        self._serialize = serialize
        self._deserialize = deserialize
        self._key_of = key_of
        self._compact_threshold_lines = compact_threshold_lines
        self._compact_threshold_ratio = compact_threshold_ratio
        self._debug_logger = debug_logger
        self._row_version = row_version
        #: 版が新しい行の数 (最後の走査 / 読み込みの時点)。1 以上ならこのファイルは readonly。
        self._newer_rows = 0
        self._lock = threading.Lock()
        # 行数 / 生存キー集合をファイルから走査して初期化する。
        # 起動コストはファイル行数に比例 (O(N))。本ストアの典型用途 (数百-
        # 数千行) では問題にならない。
        self._total_lines = 0
        self._live_keys: set[str] = set()
        #: 末尾の途中切れを検査済みか (追記の前に 1 回だけ、c_05 §0.5.8)。
        self._tail_checked = False
        self._scan_existing()

    # ── 内部: 状態走査 ─────────────────────────────────────────────────

    def _scan_existing(self) -> None:
        """既存ファイルから ``total_lines`` / ``live_keys`` を再構築する。"""
        if not self._path.exists():
            return
        live: set[str] = set()
        total = 0
        newer = 0
        # 途中で切れた行は多バイト文字の途中で終わりうる。strict だと
        # UnicodeDecodeError で全体が読めなくなるので置換し、その行は壊れた行として飛ばす。
        with self._path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "skipping malformed JSONL line in %s: %s", self._path, e,
                    )
                    continue
                row = self._row_state(obj)
                if row == _ROW_NEWER:
                    newer += 1
                if row != _ROW_OK:
                    continue
                if isinstance(obj, dict) and obj.get(TOMBSTONE_MARKER):
                    key = obj.get(TOMBSTONE_KEY_FIELD)
                    if isinstance(key, str):
                        live.discard(key)
                    continue
                try:
                    item = self._deserialize(line)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        "skipping unparsable item in %s: %s", self._path, e,
                    )
                    continue
                live.add(self._key_of(item))
        self._total_lines = total
        self._live_keys = live
        self._set_newer(newer)

    def _row_state(self, obj: object) -> int:
        """行の版を判定する (``row_version`` を持たないストアは常に読む)。"""
        if self._row_version is None:
            return _ROW_OK
        version = obj.get(ROW_VERSION_FIELD) if isinstance(obj, dict) else None
        if type(version) is not int or version < 1:
            logger.warning("skipping a JSONL line without an integer _v in %s", self._path)
            return _ROW_BAD
        return _ROW_NEWER if version > self._row_version else _ROW_OK

    def _set_newer(self, count: int) -> None:
        if count and count != self._newer_rows:
            logger.warning(
                "%s has %d row(s) newer than _v %s; the file is read-only and is not folded",
                self._path, count, self._row_version,
            )
        self._newer_rows = count

    def _guard_newer(self) -> None:
        """版が新しい行を持つファイルへは書かない (c_05 §0.4.5)。"""
        if self._newer_rows:
            raise DataReadonlyError(self._path, f"{self._newer_rows} row(s) of a newer _v")

    def _terminate_torn_tail(self) -> None:
        """末尾が改行で終わっていなければ改行を 1 つ足す (ロックの下で最初の追記の前に 1 回)。

        途中で切れた行 (kill・電源断) を **切り詰めず** 改行で終端する — 断片は壊れた行
        として読み手が飛ばし、物理行の位置は保たれる。切り詰めると、行番号で位置を
        持つ読み手 (事象ログの ``folded_through``) がずれて以後の事象を畳まなくなる
        (c_05 §0.5.8)。readonly の間は ``guard_write`` が先に止めるので修復もしない。
        """
        if self._tail_checked:
            return
        self._tail_checked = True
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return
        if size == 0:
            return
        with self._path.open("rb") as f:
            f.seek(size - 1)
            last = f.read(1)
        if last == b"\n":
            return
        with self._path.open("ab") as f:
            f.write(b"\n")
        logger.warning("Terminated a torn last line in %s (%d bytes)", self._path, size)

    # ── 書き込み API ──────────────────────────────────────────────────

    def append(self, item: T) -> None:
        """1 件追記する。同じキーの既存 item は **後勝ち** で上書きされる
        (compaction まで物理的に古い行も残る)。
        """
        line = self._serialize(item)
        if "\n" in line or "\r" in line:
            raise ValueError("serialize() must not produce newline characters")
        # tombstone 用予約フィールドが混入していたら拒否
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"serialize() did not produce valid JSON: {e}") from e
        if self._row_version is not None and (
            not isinstance(obj, dict) or obj.get(ROW_VERSION_FIELD) != self._row_version
        ):
            raise ValueError(f"serialize() must stamp {ROW_VERSION_FIELD}={self._row_version}")
        if isinstance(obj, dict):
            for reserved in (
                TOMBSTONE_MARKER,
                TOMBSTONE_KEY_FIELD,
                TOMBSTONE_TS_FIELD,
                TOMBSTONE_REASON_FIELD,
            ):
                if reserved in obj:
                    raise ValueError(
                        f"serialize() output contains reserved tombstone field "
                        f"{reserved!r}",
                    )
        key = self._key_of(item)
        guard_write(self._path)
        with self._lock:
            self._guard_newer()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._terminate_torn_tail()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._total_lines += 1
            self._live_keys.add(key)

    def tombstone(self, key: str, *, reason: str | None = None) -> None:
        """指定キーの削除マーカーを append する。

        既に tombstone 済 / 未存在のキーに対しても安全 (no-op 相当の追記)。
        物理削除は :meth:`maybe_compact` 時に行われる。

        ``reason`` は削除理由の短いラベル (GC / supersede / user 等)。時刻と
        併せて記録し、監査に答えられるようにする。
        """
        from backend.utils import utc_now

        record: dict[str, object] = {}
        if self._row_version is not None:
            record[ROW_VERSION_FIELD] = self._row_version
        record[TOMBSTONE_MARKER] = True
        record[TOMBSTONE_KEY_FIELD] = key
        record[TOMBSTONE_TS_FIELD] = utc_now()
        if reason:
            record[TOMBSTONE_REASON_FIELD] = reason
        marker = json.dumps(record, ensure_ascii=False)
        guard_write(self._path)
        with self._lock:
            self._guard_newer()
            self._terminate_torn_tail()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(marker + "\n")
            self._total_lines += 1
            self._live_keys.discard(key)

    # ── compaction ────────────────────────────────────────────────────

    def maybe_compact(self) -> bool:
        """閾値を超えていれば compaction (物理再書き出し) を実行する。

        Returns:
            実際に compaction を実行した場合 ``True``、閾値未満で no-op の場合
            ``False``。
        """
        with self._lock:
            return self._maybe_compact_locked()

    def compact(self) -> None:
        """強制 compaction (閾値チェックを行わずに再書き出し。版が新しい行があれば書かない)。"""
        with self._lock:
            self._rewrite_locked()

    def _maybe_compact_locked(self) -> bool:
        live = len(self._live_keys)
        total = self._total_lines
        if total <= self._compact_threshold_lines:
            return False
        # live * ratio < total → dead row の割合が ratio に応じて大きい
        if live * self._compact_threshold_ratio >= total:
            return False
        return self._rewrite_locked()

    def _rewrite_locked(self) -> bool:
        items = self._load_locked()
        if self._newer_rows:
            return False  # 新しい版の行を落として書き直さない (c_05 §0.4.5)
        before_lines = self._total_lines
        before_dead = before_lines - len(items)
        with AtomicWriter(self._path, debug_logger=self._debug_logger) as f:
            for key in sorted(items.keys()):
                f.write(self._serialize(items[key]) + "\n")
        self._total_lines = len(items)
        self._live_keys = set(items.keys())
        if self._debug_logger is not None:
            try:
                self._debug_logger.log_memory_op(
                    "jsonl_compact",
                    {
                        "path": str(self._path),
                        "lines_before": before_lines,
                        "lines_after": self._total_lines,
                        "dead_removed": before_dead,
                    },
                )
            except Exception as log_err:
                logger.warning("DebugLogger.log_memory_op failed: %s", log_err)
        return True

    # ── 読み込み API ──────────────────────────────────────────────────

    def load_all(self) -> dict[str, T]:
        """ファイル全体を読み込み ``key -> item`` 辞書を返す。

        重複キーは後勝ち、tombstone 行は該当キーを除外する。malformed な行は
        WARNING ログを出してスキップする (例外は伝播しない)。
        """
        with self._lock:
            return self._load_locked()

    def _load_locked(self) -> dict[str, T]:
        if not self._path.exists():
            self._set_newer(0)
            return {}
        result: dict[str, T] = {}
        newer = 0
        with self._path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "skipping malformed JSONL line in %s: %s", self._path, e,
                    )
                    continue
                row = self._row_state(obj)
                if row == _ROW_NEWER:
                    newer += 1
                if row != _ROW_OK:
                    continue
                if isinstance(obj, dict) and obj.get(TOMBSTONE_MARKER):
                    key = obj.get(TOMBSTONE_KEY_FIELD)
                    if isinstance(key, str):
                        result.pop(key, None)
                    continue
                try:
                    item = self._deserialize(line)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        "skipping unparsable item in %s: %s", self._path, e,
                    )
                    continue
                result[self._key_of(item)] = item
        self._set_newer(newer)
        return result

    # ── 統計 / 観測 ──────────────────────────────────────────────────

    def live_count(self) -> int:
        """生存キー数 (tombstone 済を除く)。"""
        with self._lock:
            return len(self._live_keys)

    def total_lines(self) -> int:
        """物理行数 (tombstone 行を含む)。"""
        with self._lock:
            return self._total_lines

    def newer_rows(self) -> int:
        """版 (``_v``) がこのストアより新しい行の数。1 以上ならこのファイルは readonly。"""
        with self._lock:
            return self._newer_rows

    @property
    def path(self) -> Path:
        return self._path
