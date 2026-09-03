"""コンテンツ種別判定

設計書 f_08_long_form_generation.md §3.1 準拠。
パターンマッチによるコード/テキストの自動判定。
"""

from __future__ import annotations

import re

from backend.free.core.locale_patterns import matches_either
from backend.free.core.session_mode import is_create_mode
from backend.free.document_nouns import (
    DOCUMENT_NOUNS_NEEDS_SUFFIX,
    DOCUMENT_NOUNS_NEEDS_SUFFIX_EN,
    DOCUMENT_NOUNS_STANDALONE,
    DOCUMENT_NOUNS_STANDALONE_EN,
)
from backend.free.generation.models import ContentType

CODE_PATTERNS: list[str] = [
    r"(クラス|関数|メソッド|モジュール|ファイル).*(実装|作成|生成|書)",
    r"(implement|create|generate|write).*(class|function|module|file)",
    r"(コード|プログラム|スクリプト).*(書|作成|生成)",
    r"\.(py|ts|js|rs|go|java)\b",
    # DB スキーマ / DDL: テーブル定義・CREATE TABLE 等は構造化コード扱い。
    # 「データベース設計書」のような文書要求は TEXT 側「設計書」と tie になり、
    # chat モードの tie-break で TEXT に倒れるため過剰検出を避ける。
    r"(テーブル定義|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE|スキーマ定義|データベース\s*設計|\bDDL\b)",
    r"(?:create|define|design)\s+(?:a\s+)?(?:table|schema|database)\b",
]

# 旧 2 番目 ``(write|draft|compose).*(article|report|document|essay)`` と旧 5 番目
# ``(specification|design doc|requirements doc)`` と旧 6 番目の拡張子パターンは
# 削除した。JA 側だけが評価される時代に英語を最低限拾うための出張エントリで、
# union 化 (下記 TEXT_PATTERNS_ALL) 後は EN 側と二重になる。**ここは
# ``text_score`` として数える対象** なので、二重は単なる冗長ではなく
# スコアの水増しになり、`code_score > text_score` の判定を TEXT 側へ倒す
# (「C:\\tmp\\notes.md にクラス設計のサンプルコードを書いて」が CODE → TEXT に
# 化ける)。EN 側は語彙が広く ``\b`` と ``(?i)`` も付いているので上位互換。
TEXT_PATTERNS: list[str] = [
    # 動作動詞 (書/作成/生成) との共起が必要な文書名詞。router.py の
    # LONG_FORM_PATTERNS と backend/free/document_nouns.py で語彙を共有する。
    rf"({'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX)}).*(書|作成|生成)",
    r"(\d{3,})\s*[字文]",
    # 仕様書・設計書・要件定義書・手順書・議事録・送付状 等: 名詞単体でも TEXT
    # として扱う。「プログラムを作成するための仕様書を出力」のように、動作動詞が
    # 「出力」「保存」など多様で、かつコード関連語と共起しても文書要件のため。
    # 2026-07-15 に「受付フロー手順書」等が CODE 判定され .md へ Python コードが
    # 書き込まれた退行への対策で社内文書語彙 (手順書/議事録/送付状 等) を追加した
    # 経緯を含む。語彙は backend/free/document_nouns.py で共有する。
    rf"({'|'.join(DOCUMENT_NOUNS_STANDALONE)})",
]

# TEXT_PATTERNS の英語版。detect_content_type() は re.search() を flags 無しで
# 呼ぶため、大文字小文字を無視するには各パターン先頭に (?i) インライン修飾子を
# 明示する必要がある (文頭大文字化が頻出する英語の実用性のため)。
TEXT_PATTERNS_EN: list[str] = [
    rf"(?i)\b(?:write|draft|create|compose|prepare|put\s+together)\b"
    rf".*\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN)})\b"
    rf"|\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN)})\b.*\b(?:write|draft|create|compose|prepare)\b",
    r"(?i)\b(\d{3,})[\s-]?words?\b",
    rf"(?i)\b(?:{'|'.join(DOCUMENT_NOUNS_STANDALONE_EN)})\b",
    # 出力先が文書/データ拡張子ならテキスト成果物 (CODE_PATTERNS の
    # コード拡張子パターンと対称)。言語に依らないので EN 側に 1 本だけ置く。
    r"(?i)\.(md|txt|csv|docx|pptx|xlsx)\b",
]

#: JA / EN 双方を **locale に関わらず** 評価する union。GUI locale は UI の言語で
#: あって指示の言語ではないため、片側だけ見ると locale='en' のまま打った日本語の
#: 文書依頼 (「受付フロー手順書を作って」) が TEXT パターンに 1 つも当たらず
#: CODE と判定される — 2026-07-15 に .md へ Python コードが書き込まれた
#: インシデントそのものが英語ロケールで再発する形。
#: 各エントリは概念が重ならないよう上のコメントの通り整理済みで、
#: ``text_score`` の水増しは起きない。
TEXT_PATTERNS_ALL: list[str] = [*TEXT_PATTERNS, *TEXT_PATTERNS_EN]

# コード否定の文脈 (「プログラムではなくて文書が欲しい」等の訂正表現)。
# CODE_PATTERNS より優先して TEXT に確定させる。negation-blind だった旧実装は
# 「プログラムではなくて、文書としての手順書」を CODE 判定し、訂正ターンで
# Python ジェネレータを生成した (2026-07-15)。
_NEGATED_CODE_RE = re.compile(
    r"(?:コード|プログラム|スクリプト)\s*(?:じゃ|では|で\s*は)\s*な(?:く|い)",
)

# _NEGATED_CODE_RE の英語版 (TEXT_PATTERNS_EN 追加時に見落とされていた:
# 2026-07-22 監査で判明。英語ユーザーが同種の訂正 ("Not code, just the document
# itself") を行っても最優先ガードが働かず、2026-07-15 インシデントが英語
# ロケールで再発しうる状態だった)。
_NEGATED_CODE_RE_EN = re.compile(
    r"\b(?:not|isn'?t)\s+(?:a\s+)?(?:code|program|script)\b",
    re.IGNORECASE,
)


def detect_content_type(instruction: str, mode: str) -> ContentType:
    """コンテンツ種別を判定

    パターンは GUI locale で切り替えず JA / EN 双方を評価する。locale は UI の
    言語であって指示の言語ではないので、片側だけ見るとその言語の文書依頼が
    まるごと CODE 側へ倒れる (2026-07-15 の「手順書に Python コードが
    書き込まれる」インシデントの再現経路)。

    Args:
        instruction: ユーザー指示テキスト
        mode: 動作モード ("create" / "chat")
    """
    # 「コード/プログラムではなく」の明示否定は最優先で TEXT に確定
    if matches_either(instruction, _NEGATED_CODE_RE, _NEGATED_CODE_RE_EN):
        return ContentType.TEXT

    if is_create_mode(mode):
        if any(re.search(p, instruction) for p in TEXT_PATTERNS_ALL):
            return ContentType.TEXT
        return ContentType.CODE

    # CODE_PATTERNS は implement/create/generate/write 等の英語動詞を含むが
    # (?i) インライン修飾子が無い。TEXT_PATTERNS_EN 側は各パターン先頭に (?i)
    # を明示しているため (文頭大文字化が頻出する英語の実用性のため)、この
    # IGNORECASE 無しのままだと英語ロケールで文頭大文字化した指示
    # ("Implement a class Foo" 等) が CODE_PATTERNS 側だけ取りこぼし、
    # code_score/text_score が非対称に取りこぼす (2026-07-22 監査で判明)。
    code_score = sum(1 for p in CODE_PATTERNS if re.search(p, instruction, re.IGNORECASE))
    text_score = sum(1 for p in TEXT_PATTERNS_ALL if re.search(p, instruction))
    return ContentType.CODE if code_score > text_score else ContentType.TEXT
