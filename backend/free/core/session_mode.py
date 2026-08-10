"""会話セッションの mode ("chat"/"create") 判定ヘルパー。

CLI 層 (``backend/free/cli/cli_mode.py``) には ``default_cli_mode()`` /
``coerce_cli_mode()`` という一元化関数と、それを強制する静的検査
(``test_pillar_boundary.py::test_cli_mode_no_hardcoded_create``) が既に
あるが、API/エージェント層には同種の一元化が無く、`mode == "create"` 相当の
判定が19ファイル・40箇所に直書きされていた。

``backend/free/core/`` は4 pillar (gen/mem/loop/learn) いずれからも境界検証の
対象外 (``test_pillar_boundary.py::_is_free_pillar_module`` が
``PILLAR_MODULE_PREFIXES`` に一致しないモジュールを無条件許可する) の、
既に確立された共有基盤の置き場所であるため、ここに置く。

無効値受領時のフォールバック挙動は呼び出し箇所によって異なる
(既定 ``"chat"`` に倒すもの、``None`` に倒すもの、呼び出し元の別の状態値に
倒すもの) ため、``normalize_session_mode()`` の機械的な適用はせず、
``is_valid_session_mode()`` で判定した上で個別のフォールバック値を維持する
呼び出し箇所もある。
"""

from __future__ import annotations

from typing import Literal

SessionMode = Literal["chat", "create"]

VALID_SESSION_MODES: frozenset[str] = frozenset({"chat", "create"})

DEFAULT_SESSION_MODE: SessionMode = "chat"

#: 旧名 → 現行名。``"coding"`` は ``"create"`` へ改名された (旧称の永続データ /
#: 旧クライアントからの受信を読めるようにするための入口互換)。移行済みデータでは
#: 出現しないが、移行漏れと外部クライアントの両方を安全側へ倒す。
LEGACY_SESSION_MODES: dict[str, SessionMode] = {"coding": "create"}


def canonicalize_session_mode(mode: str | None) -> SessionMode | None:
    """既知の mode を現行名へ正規化する。未知なら ``None``。

    現行名はそのまま、旧名 (:data:`LEGACY_SESSION_MODES`) は現行名へマップする。
    永続データの読み込みや API 受信など **入口** で使う。
    """
    if mode in VALID_SESSION_MODES:
        return mode  # type: ignore[return-value]
    return LEGACY_SESSION_MODES.get(mode or "")


def is_create_mode(mode: str | None) -> bool:
    """mode が厳密に ``"create"`` か。"""
    return mode == "create"


def is_chat_mode(mode: str | None) -> bool:
    """mode が厳密に ``"chat"`` か。"""
    return mode == "chat"


def is_valid_session_mode(mode: str | None) -> bool:
    """mode が既知の値 (``"chat"``/``"create"``) か。"""
    return mode in VALID_SESSION_MODES


def normalize_session_mode(
    mode: str | None, default: SessionMode = DEFAULT_SESSION_MODE,
) -> SessionMode:
    """未知/None の mode を ``default`` へフォールバックさせる。

    旧名 (:data:`LEGACY_SESSION_MODES`) は現行名へマップしてから返すため、
    改名前に書かれた永続データをそのまま読める。
    """
    return canonicalize_session_mode(mode) or default
