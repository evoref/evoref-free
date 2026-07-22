"""locale (GUI左下の言語設定) に応じた日本語/英語パターン選択の共有ヘルパー。

会話パイプラインのルールベース判定 (router.py / reactive.py /
content_detector.py / tool_call_judge.py / feedback.py / self_rag_judge.py)
はそれぞれ日本語専用または日英非対称な正規表現・語彙リストを持っていた。
これらを ``get_locale() == "en"`` で完全に切り替えるための共通ヘルパーを提供する。

``backend/free/core/`` は4 pillar (gen/mem/loop/learn) いずれの境界検証対象にも
含まれないため (test_pillar_boundary.py の PILLAR_MODULE_PREFIXES に不一致)、
``session_mode.py`` と同じ理由でここに置く。
"""

from __future__ import annotations

from typing import TypeVar

from backend.i18n_helper import get_locale

_T = TypeVar("_T")


def is_en_locale() -> bool:
    """現在の locale が 'en' か (呼び出し時点で評価する。キャッシュしない)。"""
    return get_locale() == "en"


def select_locale_variant(ja_variant: _T, en_variant: _T) -> _T:
    """locale=='en' なら en_variant、それ以外 (既定 'ja'・未知locale含む) は
    ja_variant を返す。"""
    return en_variant if is_en_locale() else ja_variant
