"""属性スロットの同定 — 「同じスロットか」を決める唯一の場所。

## なぜ 1 箇所に集めるか

「同じスロット」の定義が 4 系統に分裂していた (2026-09-01 監査):

===================== ============================================ ==========================================
層                    実装                                          スロットの定義
===================== ============================================ ==========================================
書込 (Step 8)          ``extraction._supersede_stale_slot_values``   ``(subject, predicate)`` 完全一致 + ``single_valued``
競合検出               ``_detect_groups`` / ``collect_pending_groups`` ``(subject, predicate)`` + 埋め込み類似度
注入 (読出)            ``injector._collapse_by_attribute``           subject の **末尾セグメント** (名前空間をまたぐ)
影の回収               ``store._supersede_generic_shadows``          ``mem.<kind>.user`` のみ
===================== ============================================ ==========================================

「汎用スロット」の定義も 2 つあった — ``injector._GENERIC_SUBJECT_TAIL`` は
8 語、``subject_key.is_generic_subject`` は ``user`` 1 語。``mem.world.assertion``
は注入側では汎用だが、ストア側の影の回収には掛からない。

実害は「畳み込みが永久に片付かない」こと。``_collapse_by_attribute`` は
``mem.personal.beverage`` と ``mem.preference.beverage`` を毎ターン 1 値へ
寄せるが、その結果を **ストアへ書き戻さない**。競合検出は
``(subject, predicate)`` 厳密一致なのでこの 2 つを対にできず supersede も
起きない。同じ計算を全ターン、永久にやり直していた。

## 2 つの概念を区別する

似ているが別物なので、名前で分ける:

- :func:`attribute_key` — 「この subject はどの実世界の属性を指すか」。
  名前空間をまたいだ世代の畳み込みキー。属性名を持たない subject は ``None``。
- :func:`is_generic_slot` — 「この subject は属性の解決に失敗した受け皿か」。
  そのファクトは固有の主張を持たない **発話の影** で、影の元が supersede
  されたら一緒に畳む (``SemanticFactStore._supersede_generic_shadows``)。

前者の除外集合 (:data:`NON_ATTRIBUTE_TAILS`) は後者 (:data:`GENERIC_ATTRIBUTE`)
の上位集合。``assertion`` は「何についての言明か」を答えた命名であって
フォールバックではないので、影としては扱わない。
"""

from __future__ import annotations

from typing import Final

from backend.free.memory.semantic.subject_key import (
    GENERIC_ATTRIBUTE,
    is_generic_subject as _is_generic_subject,
)

__all__ = [
    "GENERIC_ATTRIBUTE",
    "NON_ATTRIBUTE_TAILS",
    "attribute_key",
    "is_generic_slot",
]


NON_ATTRIBUTE_TAILS: Final[frozenset[str]] = frozenset({
    GENERIC_ATTRIBUTE,  # "user" — 属性を解決できなかった受け皿
    "assertion",        # mem.world.assertion (命名前の言明)
    "fact",
    "info",
    "note",
    "misc",
    "other",
    "value",
})
"""スロット同定に使えない末尾セグメント。

属性名が入っていない subject を属性単位の畳み込みから外す。同じキーに
無関係な事実が集まってしまうため。
"""


def attribute_key(subject: str) -> str | None:
    """subject から「実世界の属性名」を取り出す (純粋関数)。

    ``mem.personal.beverage`` / ``mem.preference.beverage`` はどちらも
    **同じ属性** を指すのに ``(subject, predicate)`` が違うためスロット
    畳み込みが効かない。末尾セグメントを属性キーとして扱い、名前空間をまたいで
    1 値へ寄せるために使う。

    実測 (2026-08-22 ライブ監査 2 回目、実ストア): 「好きな飲み物」が
    ``mem.personal.beverage`` / ``mem.preference.beverage`` /
    ``mem.personal.user`` / ``mem.preference.user`` の 4 スロットへ散り、
    緑茶 / ほうじ茶 / 紅茶 / コーヒーの歴代 4 世代がすべて live のまま
    注入されていた。

    会話要約 (``mem.decision.history.session.<id>``) は末尾がセッション ID
    なので互いに衝突せず、そのまま残る。

    Returns:
        属性キー。取り出せない (階層が浅い / 汎用語) 場合は ``None``。
    """
    parts = (subject or "").split(".")
    if len(parts) < 3:
        return None
    tail = parts[-1]
    if not tail or tail in NON_ATTRIBUTE_TAILS:
        return None
    return tail


def is_generic_slot(subject: str) -> bool:
    """``mem.<kind>.user`` — 属性としての身元を持たない汎用スロットか。

    汎用スロットは「分類できなかった発話」の置き場で、固有の属性を主張しない。
    そのため訂正の宛先にもならず、``asked_attrs`` の免除にも掛からない。実質
    その発話の **影** であり、影の元 (同じノートから起きた固有スロットの
    ファクト) が supersede されたら一緒に畳む必要がある。

    :data:`NON_ATTRIBUTE_TAILS` より **狭い**。``assertion`` 等は「属性名では
    ない」だけで、フォールバックの受け皿ではないため影として畳まない。

    実体は ``subject_key.is_generic_subject`` (下位モジュール側)。import の
    向きを一方向に保つため、ここでは委譲する。
    """
    return _is_generic_subject(subject)
