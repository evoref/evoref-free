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

def canonicalize_session_mode(mode: str | None) -> SessionMode | None:
    """既知の mode ならそのまま返す。未知 (旧名 ``"coding"`` を含む) なら ``None``。

    API 受信など **入口** の検証で使う。G1 は旧名の読み替えを持たない
    (G0 の永続データは読まない、c_05 §0.3)。
    """
    if mode in VALID_SESSION_MODES:
        return mode  # type: ignore[return-value]
    return None


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
    """未知/None の mode を ``default`` へフォールバックさせる。"""
    return canonicalize_session_mode(mode) or default
