"""形式ごとの読み取りの状態 (``/api/status`` の ``data_health.formats``、c_07 §7.2)。

読み手がファイルを開いたときの分類 (:func:`backend.io.versioned.read_versioned`) と
Evidence ストアの readonly をプロセス内に溜めるだけの小さな台帳。data_health は
ここを読むだけで、ポーリングのたびに照合しない (全件の照合は ``evoref doctor``)。

- 載せるのは current 以外 (``newer`` / ``foreign`` / ``unmigratable`` / ``corrupt`` /
  ``readonly``) だけ。同じファイルを後で読めたら消す。
- 作り直せる形式 (derived / volatile) は読めなければ捨てて作り直すので載せない。
- ``backend/io`` の中だけで完結する (pillar を import しない)。
"""

from __future__ import annotations

import threading
from pathlib import Path

from backend.io.format_registry import FORMATS

#: 読み取りで current でなかった分類。
NON_CURRENT_STATES: frozenset[str] = frozenset({"newer", "foreign", "unmigratable", "corrupt"})
#: 1 形式に複数のファイルがあるとき代表に選ぶ順 (先ほど重い)。
_STATE_ORDER: tuple[str, ...] = ("newer", "unmigratable", "foreign", "readonly", "corrupt")
#: 1 形式の理由に並べるファイルの数。
_MAX_REASONS = 3

_lock = threading.Lock()
#: キー (ファイル / ストアのパス) → (format_id, state, reason)。
_entries: dict[str, tuple[str, str, str]] = {}


def report(format_id: str, key: str, state: str, reason: str) -> None:
    """``key`` (ファイル / ストアのパス) の状態を載せる (同じキーは上書き)。"""
    with _lock:
        _entries[key] = (format_id, state, reason)


def clear(key: str) -> None:
    """``key`` を読めた / 状態を解いた。"""
    if key not in _entries:  # 健全な読み取りでロックを取らない
        return
    with _lock:
        _entries.pop(key, None)


def observe_read(format_id: str, path: Path | str, status: str, detail: str) -> None:
    """版付きファイルの読み取りの分類を受け取る (:func:`read_versioned` が呼ぶ)。"""
    key = str(path)
    if status in ("current", "migrated"):
        clear(key)
        return
    if status not in NON_CURRENT_STATES:
        return
    spec = FORMATS.find(format_id)
    if spec is not None and spec.klass in ("derived", "volatile"):
        return
    name = Path(path).name
    report(format_id, key, status, f"{name}: {detail}" if detail else name)


def snapshot() -> dict[str, dict[str, str]]:
    """``{format_id: {"state", "reason"}}`` (current 以外の形式だけ。健全なら空)。"""
    with _lock:
        entries = list(_entries.values())
    grouped: dict[str, list[tuple[str, str]]] = {}
    for format_id, state, reason in entries:
        grouped.setdefault(format_id, []).append((state, reason))
    out: dict[str, dict[str, str]] = {}
    for format_id, items in sorted(grouped.items()):
        items.sort(key=lambda item: (_STATE_ORDER.index(item[0]) if item[0] in _STATE_ORDER else len(_STATE_ORDER), item[1]))
        reasons = [reason for _, reason in items[:_MAX_REASONS]]
        if len(items) > _MAX_REASONS:
            reasons.append(f"+{len(items) - _MAX_REASONS} more")
        out[format_id] = {"state": items[0][0], "reason": "; ".join(reasons)}
    return out


def reset() -> None:
    """空にする (テストと、同じプロセスでデータ根を切り替えるときだけ)。"""
    with _lock:
        _entries.clear()


__all__ = ["NON_CURRENT_STATES", "clear", "observe_read", "report", "reset", "snapshot"]
