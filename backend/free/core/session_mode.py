"""会話セッションの mode ("chat"/"coding") 判定ヘルパー。

CLI 層 (``backend/free/cli/cli_mode.py``) には ``default_cli_mode()`` /
``coerce_cli_mode()`` という一元化関数と、それを強制する静的検査
(``test_pillar_boundary.py::test_cli_mode_no_hardcoded_coding``) が既に
あるが、API/エージェント層には同種の一元化が無く、`mode == "coding"` 相当の
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

SessionMode = Literal["chat", "coding"]

VALID_SESSION_MODES: frozenset[str] = frozenset({"chat", "coding"})

DEFAULT_SESSION_MODE: SessionMode = "chat"


def is_coding_mode(mode: str | None) -> bool:
    """mode が厳密に ``"coding"`` か。"""
    return mode == "coding"


def is_chat_mode(mode: str | None) -> bool:
    """mode が厳密に ``"chat"`` か。"""
    return mode == "chat"


def is_valid_session_mode(mode: str | None) -> bool:
    """mode が既知の値 (``"chat"``/``"coding"``) か。"""
    return mode in VALID_SESSION_MODES


def normalize_session_mode(
    mode: str | None, default: SessionMode = DEFAULT_SESSION_MODE,
) -> SessionMode:
    """未知/None の mode を ``default`` へフォールバックさせる。"""
    return mode if mode in VALID_SESSION_MODES else default
