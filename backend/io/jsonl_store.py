"""append-only JSONL ストア + atomic compaction

1 ファイル = 1 種類のレコード集合を JSONL (1 行 1 JSON) で永続化する。書き込み
の主要 API は :meth:`append` で、行を末尾に追記するだけ。削除は :meth:`tombstone`
で「削除マーカー行」を append する形を取る (物理削除は :meth:`maybe_compact`
の compaction で行う)。

トレードオフ:

- ファクト追加 1 件あたり O(1) (1 行 append) — 旧 dict 全件 dump の O(N) を回避
- ファイルが線形に伸びるため、定期的な :meth:`maybe_compact` で物理サイズと
  読込時間を縮小する (閾値超過時のみ ``AtomicWriter`` で再書き出し)

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
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("io.jsonl_store")

__all__ = [
    "JSONLAppendStore",
    "TOMBSTONE_KEY_FIELD",
    "TOMBSTONE_MARKER",
]

T = TypeVar("T")

#: tombstone 行を識別するための予約フィールド。
#: tombstone 行は ``{"_tombstone": true, "_key": "<id>"}`` の形で書かれる。
#: serialize 関数の出力にこれらの予約フィールドを含めないこと (検知時 ValueError)。
TOMBSTONE_MARKER = "_tombstone"
TOMBSTONE_KEY_FIELD = "_key"


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
        self._lock = threading.Lock()
        # 行数 / 生存キー集合をファイルから走査して初期化する。
        # 起動コストはファイル行数に比例 (O(N))。本ストアの典型用途 (数百-
        # 数千行) では問題にならない。
        self._total_lines = 0
        self._live_keys: set[str] = set()
        self._scan_existing()

    # ── 内部: 状態走査 ─────────────────────────────────────────────────

    def _scan_existing(self) -> None:
        """既存ファイルから ``total_lines`` / ``live_keys`` を再構築する。"""
        if not self._path.exists():
            return
        live: set[str] = set()
        total = 0
        with self._path.open("r", encoding="utf-8") as f:
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
        if isinstance(obj, dict) and obj.get(TOMBSTONE_MARKER):
            raise ValueError(
                f"serialize() output contains reserved tombstone marker "
                f"{TOMBSTONE_MARKER!r}",
            )
        key = self._key_of(item)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._total_lines += 1
            self._live_keys.add(key)

    def tombstone(self, key: str) -> None:
        """指定キーの削除マーカーを append する。

        既に tombstone 済 / 未存在のキーに対しても安全 (no-op 相当の追記)。
        物理削除は :meth:`maybe_compact` 時に行われる。
        """
        marker = json.dumps(
            {TOMBSTONE_MARKER: True, TOMBSTONE_KEY_FIELD: key},
            ensure_ascii=False,
        )
        with self._lock:
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
        """強制 compaction (閾値チェックを行わずに必ず再書き出し)。"""
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
        self._rewrite_locked()
        return True

    def _rewrite_locked(self) -> None:
        items = self._load_locked()
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
            return {}
        result: dict[str, T] = {}
        with self._path.open("r", encoding="utf-8") as f:
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

    @property
    def path(self) -> Path:
        return self._path
