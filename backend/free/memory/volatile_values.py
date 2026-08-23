"""揮発する計測値の散文報告を判定する葉モジュール (依存なし)。

``extractors/mdp_trace.py`` の :func:`strip_volatile_measurements` と対になる。
あちらは **ツール出力そのもの** から時間で変わる値を落とし、こちらは
**それを言い直したアシスタント発話** を判定する。

``stores/short_term.py`` から使うため独立させてある (``extractors`` パッケージは
``stores.short_term`` を import するので循環する)。指標の集合は
``mdp_trace._VOLATILE_FIELD_SUBS`` と同じ 4 種類 (CPU 使用率 / GPU 使用率 /
空きメモリ / 空きディスク) に閉じておくこと — 片方だけ増やさない。
"""

from __future__ import annotations

import re


#: 揮発する計測値を **散文で言い直した** 形。
#:
#: :data:`_VOLATILE_FIELD_SUBS` と **同じ 4 種類の指標** (CPU 使用率 / GPU 使用率 /
#: 空きメモリ / 空きディスク) だけを対象にする。指標の集合は閉じているので語彙を
#: 増やし続ける類の判定にはならない。
_VOLATILE_PROSE_RE = re.compile(
    r"(?:空き|使用可能な|利用可能な|残り)\s*(?:メモリ|RAM|ディスク|容量|空間)"
    r"|(?:メモリ|RAM|ディスク|容量)\s*の?\s*空き"
    r"|(?:CPU|GPU)\s*(?:の)?\s*使用率"
    r"|(?:free|available)\s+(?:memory|ram|disk|space)"
    r"|(?:cpu|gpu)\s+usage",
    re.IGNORECASE,
)
#: 数値 + 単位 (計測値であることの裏取り)。
_VOLATILE_VALUE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|％|GB|MB|TB|ギガ|メガ)",
    re.IGNORECASE,
)
#: 計測値の言い直しを除いた残りが、この文字数未満なら「計測の報告だけ」とみなす。
_VOLATILE_PROSE_RESIDUE_MAX = 12


def is_volatile_measurement_report(text: str) -> bool:
    """発話が「揮発する計測値の報告」だけで出来ているか (純粋関数)。

    ツール出力そのもの (``is_tool_output``) は STM 以降へ渡さない扱いだが、
    **それを言い直したアシスタント発話** は素通りしていた。除外が 1 ホップで
    迂回されるため、結局同じ数値がエピソード記憶に焼き付く。

    実インシデント (2026-08-23 ライブ監査セット 1): 「空きメモリは25.3GBです。」
    「CPU使用率は8.1%です。」が STM ノートとして保存された (user 48 / assistant 48
    の 1:1 で、アシスタント発話が無条件にノート化されていた)。過去の監査
    (2026-08-19) では同種の値が後日 RAG 1 位で「現在の値」として再注入され、
    モデルが古い空き容量を今回の計測として答えている。

    「何を測ったか」ではなく「そのときいくつだったか」だけを落とす方針は
    :func:`strip_volatile_measurements` と同じ。計測の報告に実質的な本文が
    付いている場合 (残りが :data:`_VOLATILE_PROSE_RESIDUE_MAX` 文字以上) は
    保存する — 落とすのは報告だけで出来た発話に限る。
    """
    if not text:
        return False
    if not _VOLATILE_PROSE_RE.search(text) or not _VOLATILE_VALUE_RE.search(text):
        return False
    residue = _VOLATILE_PROSE_RE.sub("", text)
    residue = _VOLATILE_VALUE_RE.sub("", residue)
    residue = re.sub(r"[\s　はがをのでするですました。、,.:：]+", "", residue)
    return len(residue) < _VOLATILE_PROSE_RESIDUE_MAX
