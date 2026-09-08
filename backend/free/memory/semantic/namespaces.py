"""semantic ストアの namespace 規則 (c_16 §4.2)。

``subject`` の先頭要素が namespace で、**競合の勝ち方・減衰の既定・注入先**が
namespace ごとに違う。この差を 1 箇所へ集めるのがこのモジュール。

===========  ===================================  ==========  ==========
namespace    競合の勝者                            減衰既定     注入
===========  ===================================  ==========  ==========
``mem.*``    ``origin=user`` かつ ``as_of`` 新     なし        ``[関連する記憶]``
``know.*``   ``as_of`` 新 × ``confidence`` 高      domain 既定  ``[参考情報]``
``idx.*``    上書き (同 ``claim_key`` は旧を畳む)   なし        しない
``loop.*``   pillar 所有 (FACT_OWNERSHIP)          なし        しない
``learn.*``  pillar 所有 (FACT_OWNERSHIP)          なし        しない
===========  ===================================  ==========  ==========

**``mem.*`` と ``know.*`` は競合させない** (c_16 §4.2)。規則が違う 2 つを
1 レコードで競わせると、「ユーザーが言った」と「世の中でそうらしい」の
どちらが勝つかが順序に依存する。namespace が違えば競合対象外にする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from backend.log_config import get_logger

logger = get_logger("memory.semantic.namespaces")

#: 競合の解き方 (c_16 §4.2)。
CompetitionRule = Literal[
    "user_stated_wins",     # mem.*   : origin=user かつ as_of が新しい方
    "newest_confident_wins",  # know.* : as_of が新しく confidence が高い方
    "overwrite",            # idx.*   : 同じ claim_key は常に新しい方
    "pillar_owned",         # loop.* / learn.* : FACT_OWNERSHIP のまま
]

#: 既知の namespace。``subject`` の先頭セグメントがこれ以外なら
#: :data:`DEFAULT_NAMESPACE` (``mem``) の規則を当てる — 自然文 subject や
#: 抽出器のフォールバックが黙って競合対象外にならないようにするため。
KNOWN_NAMESPACES: tuple[str, ...] = ("mem", "know", "idx", "loop", "learn")

DEFAULT_NAMESPACE = "mem"

#: ``[関連する記憶]`` / ``[参考情報]`` のラベル本体は
#: :mod:`backend.free.memory.pipeline.injector` が i18n 経由で組む。ここは
#: 「どちらの枠に出すか」の識別子だけを持つ (UI 文字列を二重管理しない)。
InjectionFrame = Literal["memory", "reference"]


@dataclass(frozen=True, slots=True)
class NamespacePolicy:
    """1 namespace の規則 (c_16 §4.2 の表 1 行)。"""

    name: str
    competition: CompetitionRule
    #: ``None`` = 減衰なし。``know.*`` は domain 別に
    #: :func:`know_half_life_days` が上書きする。
    half_life_days: float | None
    #: ``[関連する記憶]`` / ``[参考情報]`` へ出してよいか。
    injectable: bool
    #: 注入先の枠 (``injectable=False`` なら ``None``)。
    frame: InjectionFrame | None
    #: 保持方針の件数上限 (``None`` = 無制限)。``idx.*`` のみ持つ。
    max_records_key: str | None = None

    @property
    def label(self) -> str:
        """観測ログ用の短い名前 (``mem`` / ``know`` …)。"""
        return self.name


NAMESPACE_POLICIES: dict[str, NamespacePolicy] = {
    "mem": NamespacePolicy(
        name="mem",
        competition="user_stated_wins",
        half_life_days=None,
        injectable=True,
        frame="memory",
    ),
    "know": NamespacePolicy(
        name="know",
        competition="newest_confident_wins",
        half_life_days=None,  # domain 別 (know_half_life_days) で決まる
        injectable=True,
        frame="reference",
    ),
    "idx": NamespacePolicy(
        name="idx",
        competition="overwrite",
        half_life_days=None,
        injectable=False,
        frame=None,
        max_records_key="idx_max_records",
    ),
    "loop": NamespacePolicy(
        name="loop",
        competition="pillar_owned",
        half_life_days=None,
        injectable=False,
        frame=None,
    ),
    "learn": NamespacePolicy(
        name="learn",
        competition="pillar_owned",
        half_life_days=None,
        injectable=False,
        frame=None,
    ),
}

#: ``know.<domain>`` の半減期既定 (c_16 §4.2 / §9)。``None`` = 減衰なし。
#: ``config.yaml`` の ``memory.evidence.know_half_life_days`` で上書きする。
DEFAULT_KNOW_HALF_LIFE_DAYS: dict[str, float | None] = {
    "news": 7.0,
    "economy": 1.0,
    "local": 30.0,
    "howto": None,
}


def namespace_of(subject: str) -> str:
    """``subject`` の namespace (先頭セグメント) を返す。

    未知の先頭セグメント (自然文 subject / 抽出器のフォールバック) は
    :data:`DEFAULT_NAMESPACE` へ倒す。空文字を返して「どの規則も当たらない」
    状態を作らない — 競合解決から静かに漏れるのがいちばん困る壊れ方で、
    実際 2026-08-30 の監査では規則の当たらないスロットの旧値が残り続けた。
    """
    if not subject:
        return DEFAULT_NAMESPACE
    head = subject.split(".", 1)[0]
    return head if head in KNOWN_NAMESPACES else DEFAULT_NAMESPACE


def policy_for(subject: str) -> NamespacePolicy:
    """``subject`` に当たる :class:`NamespacePolicy` を返す。"""
    return NAMESPACE_POLICIES[namespace_of(subject)]


def know_domain(subject: str) -> str:
    """``know.<domain>.<topic>`` の ``<domain>``。``know.*`` 以外は空文字。"""
    if namespace_of(subject) != "know":
        return ""
    parts = subject.split(".")
    return parts[1] if len(parts) >= 2 else ""


def know_half_life_days(
    subject: str, overrides: dict[str, Any] | None = None,
) -> float | None:
    """``know.<domain>`` の半減期 (日)。``know.*`` 以外と未知 domain は ``None``。

    Args:
        subject: ファクトの subject。
        overrides: ``memory.evidence.know_half_life_days`` (c_16 §9)。
            ``{"news": 7, "howto": null}`` の形。
    """
    domain = know_domain(subject)
    if not domain:
        return None
    table: dict[str, Any] = dict(DEFAULT_KNOW_HALF_LIFE_DAYS)
    if overrides:
        table.update(overrides)
    value = table.get(domain)
    if value is None:
        return None
    try:
        half_life = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid know_half_life_days for domain %r: %r (ignored)",
            domain, value,
        )
        return None
    return half_life if half_life > 0 else None


def competes(subject_a: str, subject_b: str) -> bool:
    """2 つの subject が同じ競合規則の下にあるか (c_16 §4.2)。

    namespace が違えば競合させない。``mem.*`` と ``know.*`` を 1 レコードで
    競わせないための門で、``idx.*`` / ``loop.*`` / ``learn.*`` にも同じ扱いを
    する (規則が違うもの同士は勝者を決められない)。
    """
    return namespace_of(subject_a) == namespace_of(subject_b)


def is_injectable(subject: str) -> bool:
    """``subject`` が ``[関連する記憶]`` / ``[参考情報]`` に出てよいか。"""
    return policy_for(subject).injectable


__all__ = [
    "DEFAULT_KNOW_HALF_LIFE_DAYS",
    "DEFAULT_NAMESPACE",
    "KNOWN_NAMESPACES",
    "NAMESPACE_POLICIES",
    "CompetitionRule",
    "InjectionFrame",
    "NamespacePolicy",
    "competes",
    "is_injectable",
    "know_domain",
    "know_half_life_days",
    "namespace_of",
    "policy_for",
]
