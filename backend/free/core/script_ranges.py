"""文字種の Unicode レンジ — 正準定義 (SSOT)

`backend/free/**` の正規表現は日本語の文字クラスを **11 種のレンジに分裂**
させていた (2026-09-21 実測、32 箇所 / 15 ファイル)。差分は意図ではなく事故で、
同じ「日本語か」「漢字の連なりか」という判定が、どのモジュールを通るかで
違う答えを返していた:

- 漢字が ``U+4E00-9FA5`` (旧 URO 終端) と ``U+4E00-9FFF`` (ブロック終端) の
  2 通り。U+9FA6 以降の漢字が片方では漢字でない
- ひらがなが ``ぁ-ん`` (U+3041-3093) と ``ぁ-ゖ`` (U+3041-3096) の 2 通り。
  **``ゔ`` が前者から漏れる**
- カタカナが ``ァ-ヶ`` (U+30A1-30F6) と ``ァ-ヴ`` (U+30A1-30F4) の 2 通り。
  **``ヶ`` が後者から漏れる**。「3ヶ月」「一ヶ所」は日常語で、内容語抽出が
  そこで切れていた

そこで **レンジに名前を付け、用途ごとの違いは「どの名前を選ぶか」として
明示的に残す**。用途が違えば選ぶ名前が違ってよい (品質ゲートの「和文か」と
内容語抽出の「漢字の連なりか」は別の判定)。禁じるのは **名前を持たない生の
レンジ** で、それが偶然の差分の温床だった。

このモジュールは **import を一切持たない**。`core/text_quality.py` のように
意図的に無依存なモジュールからも引けるようにするため (循環を作らない)。
値は文字クラス ``[...]`` の **中身** (角括弧なし) なので、そのまま連結できる::

    from backend.free.core.script_ranges import HIRAGANA, KATAKANA, KANJI

    _JA_CHAR_RE = re.compile(f"[{HIRAGANA}{KATAKANA}{KANJI}]")

不変則 #14 / docs/c_17 §1 の「同一性」— 同じ判定が複数の実装を持たないこと。
生のレンジの再出現は `backend/free/core/tests/test_script_ranges.py` が拒否する。
"""

from __future__ import annotations

from typing import Final

#: ひらがな (U+3041 ぁ 〜 U+3096 ゖ)。``ゔ`` ``ゕ`` ``ゖ`` を含む。
HIRAGANA: Final = "ぁ-ゖ"

#: カタカナ (U+30A1 ァ 〜 U+30F6 ヶ)。``ヵ`` ``ヶ`` を含む。
KATAKANA: Final = "ァ-ヶ"

#: 長音記号 (U+30FC)。カタカナ語の一部なので内容語側では落とさない。
PROLONGED: Final = "ー"

#: カタカナ + 長音。カタカナ語 1 語を拾う用途はこちらを使う。
KATAKANA_WORD: Final = KATAKANA + PROLONGED

#: 漢字 / CJK 統合漢字 (U+4E00 一 〜 U+9FFF 鿿)。ブロック終端まで採る。
KANJI: Final = "一-鿿"

#: CJK 統合漢字 拡張 A (U+3400 㐀 〜 U+4DBF 䶿)。常用外まで拾う用途のみ。
KANJI_EXT_A: Final = "㐀-䶿"

#: CJK 互換漢字 (U+F900 豈 〜 U+FAFF 﫿)。
KANJI_COMPAT: Final = "豈-﫿"

#: 漢字の繰返し / 略号 (``々`` ``〆``)。レンジではないが内容語で漢字と同列に扱う。
KANJI_MARKS: Final = "々〆"

#: 半角カタカナ (U+FF66 ｦ 〜 U+FF9F ﾟ)。
HALFWIDTH_KATAKANA: Final = "ｦ-ﾟ"

#: ひらがなブロック全体 (U+3040 〜 U+309F)。濁点・繰返し記号まで含む掃き出し用。
HIRAGANA_BLOCK: Final = "぀-ゟ"

#: ひらがな + カタカナの 2 ブロック (U+3040 〜 U+30FF)。
#: 「和文文字が 1 つでもあるか」のような **粗い掃き出し** 用。
KANA_BLOCKS: Final = "぀-ヿ"

#: カタカナブロック全体 (U+30A0 ゠ 〜 U+30FF ヿ)。長音・中黒を含む。
KATAKANA_BLOCK: Final = "゠-ヿ"

#: 和文文字 (ひらがな + カタカナ + 漢字)。「日本語で書かれているか」の既定。
JAPANESE: Final = HIRAGANA + KATAKANA + KANJI

__all__ = [
    "HALFWIDTH_KATAKANA",
    "HIRAGANA",
    "HIRAGANA_BLOCK",
    "JAPANESE",
    "KANA_BLOCKS",
    "KANJI",
    "KANJI_COMPAT",
    "KANJI_EXT_A",
    "KANJI_MARKS",
    "KATAKANA",
    "KATAKANA_BLOCK",
    "KATAKANA_WORD",
    "PROLONGED",
]
