"""lazy-init バックオフキャッシュ

`/api/status` は、サーバが落ちている状態でも
毎回 lazy-init を試みていたため、httpx の接続失敗で 1 リクエスト 5〜7 秒の
遅延が発生していた。

本モジュールは直近の lazy-init 失敗時刻をモジュールスコープで記録し、
バックオフ期間内であれば実行をスキップする小さなガードを提供する。

- `kind`: "local" 等の識別子。呼び出し側が任意に決める
- 失敗した場合のみバックオフが効き、成功時はキャッシュをクリアする
- バックオフ期間が経過した次回の呼び出しでは再度 lazy-init を試行する
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

LAZY_BACKOFF_SEC: float = 30.0

_last_fail: dict[str, float] = {}


def should_skip(kind: str, *, now: float | None = None) -> bool:
    """直近の失敗から `LAZY_BACKOFF_SEC` 秒以内なら True を返す。"""
    if now is None:
        now = time.monotonic()
    last = _last_fail.get(kind, 0.0)
    return last > 0.0 and (now - last) < LAZY_BACKOFF_SEC


def mark_failure(kind: str) -> None:
    """lazy-init が失敗した直後に呼び出してバックオフを起動する。"""
    _last_fail[kind] = time.monotonic()


def mark_success(kind: str) -> None:
    """lazy-init が成功したらバックオフ状態をクリアする。"""
    _last_fail.pop(kind, None)


def reset(kind: str | None = None) -> None:
    """バックオフ状態をリセットする。テスト用。

    `kind` を省略すると全てのキーをクリアする。
    """
    if kind is None:
        _last_fail.clear()
    else:
        _last_fail.pop(kind, None)


async def guarded_lazy_connect(
    kind: str,
    connector: Callable[[], Awaitable[bool]],
) -> bool:
    """`connector` をバックオフガード付きで実行する。

    Args:
        kind: バックオフキー (例: "local")
        connector: 実際の lazy-init コルーチン。成功時 True を返すこと

    Returns:
        接続成功時 True / バックオフ中もしくは失敗時 False
    """
    if should_skip(kind):
        return False
    try:
        success = await connector()
    except Exception:
        mark_failure(kind)
        raise
    if success:
        mark_success(kind)
    else:
        mark_failure(kind)
    return success
