"""CLI coding モードの hook 抽象

Free 配布の CLI が coding モード固有の前処理 (例: ワークモード右ペイン用の
セッションコンテキスト初期化、Pro 専用ツール一覧の登録など) を Pro 拡張から
受けるための抽象境界。`develop_hook` と同じ自己登録パターンを採用する。

`backend.free.cli.coding_hook.get_coding_hook()` を初回呼び出しした際に、
Pro / Develop 配布が同梱されていれば `backend.pro.cli.coding_mode` を lazy
import し、Pro 側の自己登録を発火させる (Develop は Pro の上位互換のため
同じ Pro coding hook を共有する)。Pro / Develop が無い場合や
`EVOREF_EDITION=free` で強制 Free 化されている場合は no-op の
`_NoOpCodingHook` が返り、CLI 本体の挙動を変えない。

レイヤー責務:

- `backend/free/cli/coding_hook` — 抽象 + Free no-op 実装 (本モジュール)
- `backend/pro/cli/coding_mode` — Pro 拡張 (`ProCodingHook`)
- `backend/free/cli/{main,chat_loop}` — `get_coding_hook()` のみ参照
"""

from __future__ import annotations

from typing import Protocol

from backend.free.cli.cli_mode import is_cli_pro_edition
from backend.log_config import get_logger

logger = get_logger("cli.coding_hook")


class CodingHook(Protocol):
    """coding モード固有処理のフック契約。"""

    def is_available(self) -> bool:
        """coding モードが利用可能か (Pro なら True、Free フォールバックは False)。"""

    def on_mode_resolved(self, mode: str) -> None:
        """`main` がデフォルトモード解決を終えた直後に呼ばれる。

        coding モードで起動する場合の Pro 専用副作用 (例: 専用ツール登録) を実装する。
        Free フォールバックでは no-op。
        """


class _NoOpCodingHook:
    """Pro 未登録時のフォールバック。Free CLI では coding モードでも追加処理なし。"""

    def is_available(self) -> bool:
        return False

    def on_mode_resolved(self, mode: str) -> None:
        return None


# ──────────────────────────────────────────────────────────────────────────
# レジストリ
# ──────────────────────────────────────────────────────────────────────────

_hook: CodingHook | None = None
_bootstrapped: bool = False


def register_coding_hook(hook: CodingHook) -> None:
    """Pro 実装を登録する。Pro CLI モジュール (`backend.pro.cli.coding_mode`)
    が import 時に呼び出すことで Free CLI から参照可能になる。

    後勝ち (上書き可) — テストで stub に差し替えられるようにする。
    """
    global _hook
    _hook = hook


def reset_coding_hook() -> None:
    """テスト用: 登録状態をクリアして bootstrap をやり直せる状態に戻す。"""
    global _hook, _bootstrapped
    _hook = None
    _bootstrapped = False


def get_coding_hook() -> CodingHook:
    """現在登録されている CodingHook を返す。

    初回呼び出し時に、CLI が Pro エディションで動いている場合のみ
    `backend.pro.cli.coding_mode` の lazy import を試みる。Pro 配布が
    同梱されていない / import 失敗 / Free 環境のいずれかであれば
    `_NoOpCodingHook` を返す。
    """
    global _hook, _bootstrapped
    if _hook is not None:
        return _hook
    if not _bootstrapped:
        _bootstrapped = True
        if is_cli_pro_edition():
            try:
                # Pro 側 coding_mode の import が register_coding_hook を呼ぶ。
                # ImportError は Pro 未同梱 (Free 配布) の正常系。
                import backend.pro.cli.coding_mode  # noqa: F401
            except ImportError:
                logger.debug(
                    "backend.pro.cli.coding_mode not available; "
                    "using no-op coding hook",
                )
    if _hook is None:
        _hook = _NoOpCodingHook()
    return _hook
