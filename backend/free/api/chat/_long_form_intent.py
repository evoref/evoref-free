"""長文生成 (file_output_mode) の出力モード意図検出。

ユーザー指示から「機能ごと個別ファイル」(SPLIT) / 「単一ファイル詳細化」
(EXPAND) / 「現状の継続」(CONTINUE) / 「長文生成なし」(OFF) を分類する。

[docs/f_09_long_form_generation.md] 参照。本モジュールは
:mod:`backend.free.api.chat.chat_streaming` から呼ばれ、判定結果は
:meth:`backend.free.generation.orchestrator.LongFormOrchestrator.generate`
に ``long_form_mode`` kwarg として伝播する。
"""

from __future__ import annotations

import re

# enum 自体は EvorefGen pillar 側 (backend.free.generation.models) に SSOT。
# api 層からは検出関数とともに簡便に import できるよう再エクスポートする。
from backend.free.generation.models import LongFormMode

__all__ = ["LongFormMode", "detect_long_form_mode"]


# 「機能ごと個別ファイル」を示すヒント。SPLIT モードへの分岐シグナル。
# 日本語: 「機能ごと/毎/別/単位」+ 助詞、および「個別/別々/別」+「ファイル/出力」。
# 英語: per-feature / split files / one file each / individual files など。
_SPLIT_HINT_RE = re.compile(
    r"(?:"
    r"機能(?:ごと|毎|別|単位)(?:に|で|の)"
    r"|(?:個別|別々|別)(?:の)?(?:ファイル|出力)"
    r"|分割(?:し|して|出力)"
    r"|(?:複数|それぞれ)(?:の)?ファイル"
    r"|分けて(?:出力|保存|書)"
    r"|per[- ]?feature"
    r"|split(?:[^.!?\n]{0,30})files?"
    r"|one\s+file\s+(?:per|each)"
    r"|(?:individual|separate)\s+files?"
    r"|separate\s+(?:spec|specification|document)s?(?:\s+\w+)?\s+(?:for\s+each|per)\s+\w+"
    r")",
    re.IGNORECASE,
)

# 「詳細化 / 仕様書化 / 展開」を示すヒント。EXPAND モードへの分岐シグナル。
_EXPAND_HINT_RE = re.compile(
    r"(?:"
    r"詳細(?:化|に|な|の|を|設計)"
    r"|詳述"
    r"|詳しく"
    r"|仕様書"
    r"|詳細仕様"
    r"|展開(?:し|して)"
    r"|拡張(?:し|して)"
    r"|expand"
    r"|elaborate"
    r"|detailed?\s+(?:spec|design|specification)"
    r")",
    re.IGNORECASE,
)


def detect_long_form_mode(
    query: str,
    *,
    has_existing_content: bool,
    file_output_mode: bool,
) -> LongFormMode:
    """ユーザー指示から長文生成の出力モードを判定する。

    判定ロジック:

    1. ``file_output_mode=False`` (ファイル出力意図なし)
       → ``has_existing_content`` に応じて ``CONTINUE`` / ``OFF`` を返す
       (既存挙動互換)。
    2. ``file_output_mode=True``:

       - ``query`` が SPLIT ヒントを含む → :attr:`LongFormMode.SPLIT`
         (新規生成・既存参照のどちらでも発火)
       - ``query`` が EXPAND ヒントを含む → :attr:`LongFormMode.EXPAND`
         (新規生成・既存参照のどちらでも発火)
       - どちらのヒントもなく既存ファイルあり → :attr:`LongFormMode.CONTINUE`
         (従来動作)
       - どちらのヒントもなく既存ファイルなし → :attr:`LongFormMode.CONTINUE`
         (シングルパス書込み相当、既存挙動互換)。

    Args:
        query: ユーザー指示文
        has_existing_content: ``existing_content`` (既存ファイル content) が
            読み込まれているか。SPLIT/EXPAND の発火条件には影響しないが、
            EXPAND の context 取り込み有無を呼出側で分岐する際に使う。
        file_output_mode: 書込みヒント + ファイルパスが検出されているか

    Returns:
        判定されたモード。
    """
    if not file_output_mode:
        return LongFormMode.CONTINUE if has_existing_content else LongFormMode.OFF

    if _SPLIT_HINT_RE.search(query):
        return LongFormMode.SPLIT
    if _EXPAND_HINT_RE.search(query):
        return LongFormMode.EXPAND
    return LongFormMode.CONTINUE
