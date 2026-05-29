"""Policy フォールバック解決の共通ヘルパ (横断基盤)。

``PolicyInterpreter.get(domain, key[, mode])`` を呼び、未設定 / 未初期化 /
degraded mode (``policy is None``) のいずれでも安全に ``default`` へ縮退する
パターンを集約する。4 pillar 横断で 10+ 箇所に重複していた

    if policy is None:
        return default
    try:
        return policy.get(domain, key[, mode])
    except (KeyError, TypeError, ValueError):
        return default

を 1 関数に統一する。``backend/`` 直下に置くことで全 pillar から単方向に
import でき、新たな pillar 間依存を作らない (``PolicyInterpreter`` は
``TYPE_CHECKING`` 経由のため実行時 import は発生しない)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from backend.free.core.policy_interpreter import PolicyInterpreter

T = TypeVar("T")


def get_policy_value(
    policy: PolicyInterpreter | None,
    domain: str,
    key: str,
    default: T,
    *,
    mode: str | None = None,
    coerce_type: type | None = None,
) -> T:
    """policy からパラメータを取得し、失敗時は ``default`` を返す。

    Args:
        policy: ``PolicyInterpreter`` (degraded mode では ``None``)。
        domain: ポリシードメイン ("search" / "memory" / "learning" / ...)。
        key: パラメータ名。
        default: 取得失敗時のフォールバック (この型が戻り値型を決める)。
        mode: "chat" / "coding" など。``None`` のとき ``policy.get`` の既定
            (= "chat") を使う (mode 引数を持たない learning 等のドメイン向け)。
        coerce_type: 取得値に適用する型変換 (例: ``float``)。``None`` なら無変換。

    Returns:
        ``policy.get`` の結果 (``coerce_type`` 適用後)、または例外時 ``default``。
        ``policy is None`` のときは即 ``default``。
    """
    if policy is None:
        return default
    try:
        value = policy.get(domain, key) if mode is None else policy.get(domain, key, mode)
    except (KeyError, TypeError, ValueError):
        return default
    if coerce_type is not None:
        try:
            return coerce_type(value)
        except (TypeError, ValueError):
            return default
    return value
