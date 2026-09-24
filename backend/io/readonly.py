"""データ根の readonly (c_05 §0.4.2)。

readonly は **起動側で止める** のが主で (sleep-time の書き込み段・Level 1 / 2・
curator をスケジュールしない)、ここは最後の砦: ``store/`` への書き込み (原子的
書き込み・JSONL 追記・退避の改名) を :class:`DataReadonlyError` で拒否する。

握り潰し対策: 拒否した事実はリクエストごとの記録 (:func:`begin_request_scope`) に
残る。呼出側が ``except Exception`` で握り潰しても、API のミドルウェアがその記録を
見て応答を 423 / ``E0423`` に置き換える (全ルートに再送出の節を足さない)。
``logs/`` ``outputs/`` ``tmp/`` など ``store/`` の外は書ける。
"""

from __future__ import annotations

import contextvars
import os
import threading
from pathlib import Path


class DataReadonlyError(RuntimeError):
    """readonly 中に ``store/`` へ書こうとした。"""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"data root is read-only ({reason}); refused to write {path}")


_lock = threading.Lock()
_store_root: Path | None = None
_reason: str | None = None
_violations: contextvars.ContextVar[list[Path] | None] = contextvars.ContextVar(
    "evoref_readonly_violations", default=None,
)


def enable_readonly(store_root: Path, reason: str) -> None:
    """``store_root`` 配下を readonly にする (プロセスで 1 回、起動ゲートの後)。"""
    global _store_root, _reason
    with _lock:
        _store_root = Path(os.path.abspath(store_root))
        _reason = reason


def disable_readonly() -> None:
    """解除する (テストと、同じプロセスでデータ根を切り替えるときだけ)。"""
    global _store_root, _reason
    with _lock:
        _store_root = None
        _reason = None


def readonly_reason() -> str | None:
    return _reason


def is_readonly() -> bool:
    return _reason is not None


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.normcase(os.path.abspath(path))).relative_to(os.path.normcase(str(root)))
    except ValueError:
        return False
    return True


def guard_write(path: Path | str, *, outside_store: bool = False) -> None:
    """``path`` が readonly の ``store/`` 配下なら記録して :class:`DataReadonlyError`。

    ``outside_store=True`` は ``store/`` の外でも止めたい書き込み (インストール根の
    ``config.yaml`` — readonly の間は設定 API も止める) に使う。
    """
    root, reason = _store_root, _reason
    if reason is None or root is None:
        return
    target = Path(path)
    if not outside_store and not _inside(target, root):
        return
    scope = _violations.get()
    if scope is not None:
        scope.append(target)
    raise DataReadonlyError(target, reason)


def begin_request_scope() -> list[Path]:
    """リクエストの記録を始める。返したリストに拒否したパスが積まれる。"""
    scope: list[Path] = []
    _violations.set(scope)
    return scope


__all__ = [
    "DataReadonlyError",
    "begin_request_scope",
    "disable_readonly",
    "enable_readonly",
    "guard_write",
    "is_readonly",
    "readonly_reason",
]
