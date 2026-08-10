"""CLI create モードの hook 抽象

Free 配布の CLI が create モード固有の前処理 (例: ワークモード右ペイン用の
セッションコンテキスト初期化、Pro 専用ツール一覧の登録など) を Pro 拡張から
受けるための抽象境界。`develop_hook` と同じ自己登録パターンを採用する。

`backend.free.cli.create_hook.get_create_hook()` を初回呼び出しした際に、
Pro / Develop 配布が同梱されていれば `backend.pro.cli.create_mode` を lazy
import し、Pro 側の自己登録を発火させる (Develop は Pro の上位互換のため
同じ Pro create hook を共有する)。Pro / Develop が無い場合や
`EVOREF_EDITION=free` で強制 Free 化されている場合は no-op の
`_NoOpCreateHook` が返り、CLI 本体の挙動を変えない。

レイヤー責務:

- `backend/free/cli/create_hook` — 抽象 + Free no-op 実装 (本モジュール)
- `backend/pro/cli/create_mode` — Pro 拡張 (`ProCreateHook`)
- `backend/free/cli/{main,chat_loop}` — `get_create_hook()` のみ参照
"""

from __future__ import annotations

from typing import Protocol

from backend.free.cli.cli_mode import is_cli_pro_edition
from backend.log_config import get_logger

logger = get_logger("cli.create_hook")


class CreateHook(Protocol):
    """create モード固有処理のフック契約。"""

    def is_available(self) -> bool:
        """create モードが利用可能か (Pro なら True、Free フォールバックは False)。"""

    def on_mode_resolved(self, mode: str) -> None:
        """`main` がデフォルトモード解決を終えた直後に呼ばれる。

        create モードで起動する場合の Pro 専用副作用 (例: 専用ツール登録) を実装する。
        Free フォールバックでは no-op。
        """


class _NoOpCreateHook:
    """Pro 未登録時のフォールバック。Free CLI では create モードでも追加処理なし。"""

    def is_available(self) -> bool:
        return False

    def on_mode_resolved(self, mode: str) -> None:  # noqa: ARG002
        return None


# ──────────────────────────────────────────────────────────────────────────
# レジストリ
# ──────────────────────────────────────────────────────────────────────────

_hook: CreateHook | None = None
_bootstrapped: bool = False


def register_create_hook(hook: CreateHook) -> None:
    """Pro 実装を登録する。Pro CLI モジュール (`backend.pro.cli.create_mode`)
    が import 時に呼び出すことで Free CLI から参照可能になる。

    後勝ち (上書き可) — テストで stub に差し替えられるようにする。
    """
    global _hook
    _hook = hook


def reset_create_hook() -> None:
    """テスト用: 登録状態をクリアして bootstrap をやり直せる状態に戻す。"""
    global _hook, _bootstrapped
    _hook = None
    _bootstrapped = False


def get_create_hook() -> CreateHook:
    """現在登録されている CreateHook を返す。

    初回呼び出し時に、CLI が Pro エディションで動いている場合のみ
    `backend.pro.cli.create_mode` の lazy import を試みる。Pro 配布が
    同梱されていない / import 失敗 / Free 環境のいずれかであれば
    `_NoOpCreateHook` を返す。
    """
    global _hook, _bootstrapped
    if _hook is not None:
        return _hook
    if not _bootstrapped:
        _bootstrapped = True
        if is_cli_pro_edition():
            try:
                # Pro 側 create_mode の import が register_create_hook を呼ぶ。
                # ImportError は Pro 未同梱 (Free 配布) の正常系。
                import backend.pro.cli.create_mode  # noqa: F401
            except ImportError:
                logger.debug(
                    "backend.pro.cli.create_mode not available; "
                    "using no-op create hook",
                )
    if _hook is None:
        _hook = _NoOpCreateHook()
    return _hook
