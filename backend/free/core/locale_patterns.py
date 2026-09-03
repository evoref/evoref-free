"""locale (GUI左下の言語設定) に応じた日本語/英語パターン選択の共有ヘルパー。

会話パイプラインのルールベース判定 (router.py / reactive.py /
content_detector.py / tool_call_judge.py / feedback.py / self_rag_judge.py)
はそれぞれ日本語専用または日英非対称な正規表現・語彙リストを持っていた。
これらを ``get_locale() == "en"`` で完全に切り替えるための共通ヘルパーを提供する。

``backend/free/core/`` は4 pillar (gen/mem/loop/learn) いずれの境界検証対象にも
含まれないため (test_pillar_boundary.py の PILLAR_MODULE_PREFIXES に不一致)、
``session_mode.py`` と同じ理由でここに置く。

**どのヘルパーを使うか** — GUI locale は「UI の言語」であって「いま打たれた
発話の言語」ではない。既定 'ja' のまま英語で打つ (逆も) 使い方が普通に起きる
ため、**入力の照合** を locale で排他選択してはいけない:

===============================  =========================================
用途                             使うもの
===============================  =========================================
入力にパターンが当たるかの判定   :func:`matches_either` (JA/EN 両方を評価)
言語固有のアルゴリズムの選択     :func:`select_by_script` (入力の字種で選ぶ)
ユーザーへ**出す**文字列の選択   :func:`select_locale_variant` (GUI locale)
===============================  =========================================

``select_locale_variant`` を入力の照合に使うと、その言語側の判定が丸ごと
効かなくなる (2026-07-22 に router で実測。英語の文書作成依頼が locale='ja'
のとき一度も long_form に分類されなかった)。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

from backend.i18n_helper import get_locale

if TYPE_CHECKING:
    from collections.abc import Iterable

_T = TypeVar("_T")


def is_en_locale() -> bool:
    """現在の locale が 'en' か (呼び出し時点で評価する。キャッシュしない)。"""
    return get_locale() == "en"


def select_locale_variant(ja_variant: _T, en_variant: _T) -> _T:
    """locale=='en' なら en_variant、それ以外 (既定 'ja'・未知locale含む) は
    ja_variant を返す。

    **ユーザーへ出す文字列 (定型応答 / プロンプト行 / メニュー) 専用。**
    入力の照合には使わないこと (モジュール docstring の表を参照)。
    """
    return en_variant if is_en_locale() else ja_variant


def matches_either(
    text: str, ja_pattern: re.Pattern[str], en_pattern: re.Pattern[str],
) -> bool:
    """JA / EN 両方のパターンを **locale に関わらず** 評価する (union)。

    片方だけを見る実装は、GUI locale と実際の入力言語が食い違った瞬間に
    その言語の判定を丸ごと落とす。パターンの誤検出コストは片側評価の
    取りこぼしより小さいので、入力の照合は既定で union に倒す。

    言語をまたいだ誤検出が実害になるのは、片方のパターンが **境界無しの
    短い ASCII 語** を持つときだけ。その場合はパターン側を
    ``intent_vocab.ascii_boundary`` で囲って直す (union をやめる理由にはしない)。
    """
    return bool(ja_pattern.search(text) or en_pattern.search(text))


def matches_any_either(
    text: str,
    ja_patterns: Iterable[re.Pattern[str]],
    en_patterns: Iterable[re.Pattern[str]],
) -> bool:
    """:func:`matches_either` のパターンリスト版。"""
    return any(p.search(text) for p in (*ja_patterns, *en_patterns))


#: 和文文字 (ひらがな / カタカナ / 漢字)。``text_quality._JA_CHAR_RE`` と同じ範囲。
#: あちらは「応答が日本語か」を 20 文字の下限付きで測る品質判定用なので、
#: 短いクエリの字種判定には使えない (別定義にしている理由)。
_JA_SCRIPT_RE = re.compile(r"[ぁ-んァ-ヶ一-龥]")


def has_japanese_script(text: str) -> bool:
    """テキストに和文文字が 1 つでも含まれるか (純粋関数)。

    GUI locale ではなく **入力そのもの** の言語を見るための判定。1 文字でも
    真とするのは、日本語のクエリに英単語が混ざる形 (「Python の GIL とは？」)
    が普通で、逆に英文へ和文が混ざる形は稀だから。
    """
    return bool(_JA_SCRIPT_RE.search(text or ""))


def select_by_script(text: str, ja_variant: _T, en_variant: _T) -> _T:
    """入力の字種で variant を選ぶ (和文文字を含めば ja_variant)。

    アルゴリズム自体が言語固有で union できない場合に使う — 文字クラスで
    内容語を切り出す日本語の縮約と、空白トークン + ストップワードで切り出す
    英語の縮約のように、「両方走らせる」が意味を成さない対。
    """
    return ja_variant if has_japanese_script(text) else en_variant
