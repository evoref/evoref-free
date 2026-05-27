"""コンテンツ種別判定

設計書 f_09_long_form_generation.md §11 準拠。
パターンマッチによるコード/テキストの自動判定。
"""

from __future__ import annotations

import re

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
    r"(記事|レポート|文書|論文|ブログ|説明文|マニュアル).*(書|作成|生成)",
    r"(write|draft|compose).*(article|report|document|essay)",
    r"(\d{3,})\s*[字文]",
    # 仕様書・設計書・要件定義書: 名詞単体でも TEXT として扱う。
    # 「プログラムを作成するための仕様書を出力」のように、動作動詞が
    # 「出力」「保存」など多様で、かつコード関連語と共起しても文書要件のため。
    r"(仕様書|要件定義書?|設計書|設計仕様書?|基本設計書|詳細設計書|要求仕様書|デザインドキュメント)",
    r"(specification|design[\s-]?doc(?:ument)?|requirements?[\s-]?doc(?:ument)?)",
]


def detect_content_type(instruction: str, mode: str) -> ContentType:
    """コンテンツ種別を判定

    Args:
        instruction: ユーザー指示テキスト
        mode: 動作モード ("coding" / "chat")
    """
    if mode == "coding":
        if any(re.search(p, instruction) for p in TEXT_PATTERNS):
            return ContentType.TEXT
        return ContentType.CODE

    code_score = sum(1 for p in CODE_PATTERNS if re.search(p, instruction))
    text_score = sum(1 for p in TEXT_PATTERNS if re.search(p, instruction))
    return ContentType.CODE if code_score > text_score else ContentType.TEXT
