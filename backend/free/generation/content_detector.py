"""コンテンツ種別判定

設計書 f_08_long_form_generation.md §11 準拠。
パターンマッチによるコード/テキストの自動判定。
"""

from __future__ import annotations

import re

from backend.free.core.locale_patterns import select_locale_variant
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

TEXT_PATTERNS: list[str] = [
    # 動作動詞 (書/作成/生成) との共起が必要な文書名詞。router.py の
    # LONG_FORM_PATTERNS と backend/free/document_nouns.py で語彙を共有する。
    rf"({'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX)}).*(書|作成|生成)",
    r"(write|draft|compose).*(article|report|document|essay)",
    r"(\d{3,})\s*[字文]",
    # 仕様書・設計書・要件定義書・手順書・議事録・送付状 等: 名詞単体でも TEXT
    # として扱う。「プログラムを作成するための仕様書を出力」のように、動作動詞が
    # 「出力」「保存」など多様で、かつコード関連語と共起しても文書要件のため。
    # 2026-07-15 に「受付フロー手順書」等が CODE 判定され .md へ Python コードが
    # 書き込まれた退行への対策で社内文書語彙 (手順書/議事録/送付状 等) を追加した
    # 経緯を含む。語彙は backend/free/document_nouns.py で共有する。
    rf"({'|'.join(DOCUMENT_NOUNS_STANDALONE)})",
    r"(specification|design[\s-]?doc(?:ument)?|requirements?[\s-]?doc(?:ument)?)",
    # 出力先が文書/データ拡張子ならテキスト成果物 (CODE_PATTERNS の
    # コード拡張子パターンと対称)。
    r"\.(md|txt|csv|docx|pptx|xlsx)\b",
]

# TEXT_PATTERNS の英語版。GUI 左下の言語設定が 'en' の場合のみ使う
# (TEXT_PATTERNS とは locale で完全に排他利用される。select_locale_variant 経由)。
# detect_content_type() は re.search() を flags 無しで呼ぶため、大文字小文字を
# 無視するには各パターン先頭に (?i) インライン修飾子を明示する必要がある
# (文頭大文字化が頻出する英語の実用性のため)。
TEXT_PATTERNS_EN: list[str] = [
    rf"(?i)\b(?:write|draft|create|compose|prepare|put\s+together)\b"
    rf".*\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN)})\b"
    rf"|\b(?:{'|'.join(DOCUMENT_NOUNS_NEEDS_SUFFIX_EN)})\b.*\b(?:write|draft|create|compose|prepare)\b",
    r"(?i)\b(\d{3,})[\s-]?words?\b",
    rf"(?i)\b(?:{'|'.join(DOCUMENT_NOUNS_STANDALONE_EN)})\b",
    r"(?i)\.(md|txt|csv|docx|pptx|xlsx)\b",
]

# コード否定の文脈 (「プログラムではなくて文書が欲しい」等の訂正表現)。
# CODE_PATTERNS より優先して TEXT に確定させる。negation-blind だった旧実装は
# 「プログラムではなくて、文書としての手順書」を CODE 判定し、訂正ターンで
# Python ジェネレータを生成した (2026-07-15)。
_NEGATED_CODE_RE = re.compile(
    r"(?:コード|プログラム|スクリプト)\s*(?:じゃ|では|で\s*は)\s*な(?:く|い)",
)

# _NEGATED_CODE_RE の英語版。GUI 左下の言語設定が 'en' の場合のみ使う
# (TEXT_PATTERNS_EN 追加時に見落とされていた: 2026-07-22 監査で判明。
# 英語ユーザーが同種の訂正 ("Not code, just the document itself") を
# 行っても最優先ガードが働かず、2026-07-15 インシデントが英語ロケールで
# 再発しうる状態だった)。
_NEGATED_CODE_RE_EN = re.compile(
    r"\b(?:not|isn'?t)\s+(?:a\s+)?(?:code|program|script)\b",
    re.IGNORECASE,
)


def detect_content_type(instruction: str, mode: str) -> ContentType:
    """コンテンツ種別を判定

    Args:
        instruction: ユーザー指示テキスト
        mode: 動作モード ("create" / "chat")
    """
    # 「コード/プログラムではなく」の明示否定は最優先で TEXT に確定
    negated_code_re = select_locale_variant(_NEGATED_CODE_RE, _NEGATED_CODE_RE_EN)
    if negated_code_re.search(instruction):
        return ContentType.TEXT

    text_patterns = select_locale_variant(TEXT_PATTERNS, TEXT_PATTERNS_EN)

    if is_create_mode(mode):
        if any(re.search(p, instruction) for p in text_patterns):
            return ContentType.TEXT
        return ContentType.CODE

    # CODE_PATTERNS は implement/create/generate/write 等の英語動詞を含むが
    # (?i) インライン修飾子が無い。TEXT_PATTERNS_EN 側は各パターン先頭に (?i)
    # を明示しているため (文頭大文字化が頻出する英語の実用性のため)、この
    # IGNORECASE 無しのままだと英語ロケールで文頭大文字化した指示
    # ("Implement a class Foo" 等) が CODE_PATTERNS 側だけ取りこぼし、
    # code_score/text_score が非対称に取りこぼす (2026-07-22 監査で判明)。
    code_score = sum(1 for p in CODE_PATTERNS if re.search(p, instruction, re.IGNORECASE))
    text_score = sum(1 for p in text_patterns if re.search(p, instruction))
    return ContentType.CODE if code_score > text_score else ContentType.TEXT
