"""出力ファイルの形式 (拡張子) 推論とディレクトリ解決。

long_form (api 層 ``chat_streaming``) と meta_cognitive (agent 層) の双方が
「ユーザー指示文から出力拡張子を推論し、ディレクトリ指定を ``output_<UTC><ext>`` に
解決する」ために共有する。api → agent の依存方向を保つため agent 層に置く
(meta_cognitive が api を import すると循環するため)。
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.utils import utc_compact_stamp

# 明示的な拡張子指定 (「report.xlsx」「.docx で」等)。最優先で尊重する。
_EXPLICIT_EXT_RE = re.compile(
    r"\.(xlsx|xls|docx|doc|pptx|ppt|csv|md|txt)(?:\b|ファイル|形式|で|に|を)",
    re.IGNORECASE,
)
# 表計算 (Excel) 出力意図。「一覧表」「図表」「年表」等の "表" 誤検出を避け、
# 明確なシグナルのみ採用する。
_XLSX_HINT_RE = re.compile(
    r"(?:excel|エクセル|スプレッドシート|表計算|xlsx)",
    re.IGNORECASE,
)
# 文書 (Word) 出力意図。「キーワード」等の誤検出を避け ASCII 境界を要求する。
_DOCX_HINT_RE = re.compile(
    r"(?:(?<![A-Za-z])word(?![A-Za-z])|ワード文書|ワードファイル|docx)",
    re.IGNORECASE,
)
# プレゼン (PowerPoint) 出力意図。素の「プレゼン」「スライド」は内容語 (「プレゼンを
# 要約」「スライドの作り方」) に多く誤検出するため採用しない。成果物を明確に指す語
# (パワポ / パワーポイント / プレゼンテーション / スライド資料・形式 / pptx) のみ拾う。
_PPTX_HINT_RE = re.compile(
    r"(?:(?<![A-Za-z])powerpoint(?![A-Za-z])"
    r"|パワーポイント|パワポ|プレゼンテーション"
    r"|スライド資料|スライド形式|pptx)",
    re.IGNORECASE,
)
# CSV 出力意図。
_CSV_HINT_RE = re.compile(r"(?:csv|カンマ区切り)", re.IGNORECASE)
# Markdown 出力意図。``.md`` 指定に加え、ドット無しの「md 形式」「md ファイル」も拾う。
_MD_HINT_RE = re.compile(
    r"(?:"
    r"\.md(?:\b|ファイル|形式|で|に|を)"
    r"|md[\s ]?(?:形式|ファイル|で出力|で保存|で書)"
    r"|markdown"
    r"|マークダウン"
    r")",
    re.IGNORECASE,
)

# スプレッドシート/表形式の出力先拡張子。これらは GFM マークダウン表として
# 生成させ、write_file → ContentConverter.from_markdown → XlsxWriter で実セル化する。
TABLE_OUTPUT_EXTS: frozenset[str] = frozenset({".xlsx", ".xls", ".csv", ".ods"})

# リッチ文書 (Word / PowerPoint) の出力先拡張子。スプレッドシートではないが、取得
# 済みの実テーブルを GFM として直接書き込めば export Writer が実テーブルに描画する。
# `TABLE_OUTPUT_EXTS` (xlsx 専用契約) とは別集合に保ち、`is_table_output` の意味を
# 変えない (例: `report.docx` は従来どおり is_table_output=False)。
RICH_TABLE_OUTPUT_EXTS: frozenset[str] = frozenset({".docx", ".pptx"})

# 取得済み実テーブルを決定論的に書き込むべき出力先 (表計算 + リッチ文書) の和集合。
FETCHED_TABLE_EXTS: frozenset[str] = TABLE_OUTPUT_EXTS | RICH_TABLE_OUTPUT_EXTS


def infer_output_extension(query: str, default: str = ".txt") -> str:
    """ユーザー指示文から出力ファイルの拡張子を推論する。

    明示拡張子 → xlsx → docx → pptx → csv → md の順に判定し、いずれも該当しなければ
    ``default`` を返す。pptx を docx の後に置くのは、「Excel の表をパワポにも」等の
    曖昧クエリで先行フォーマット (xlsx/docx) を優先し既存挙動を保つため。

    Args:
        query: ユーザー指示文
        default: 推論できなかった場合に返す既定拡張子 (先頭ドット必須)

    Returns:
        ``.xlsx`` / ``.docx`` / ``.csv`` / ``.md`` / ``.txt`` 等の拡張子文字列。
    """
    m = _EXPLICIT_EXT_RE.search(query)
    if m:
        return "." + m.group(1).lower()
    if _XLSX_HINT_RE.search(query):
        return ".xlsx"
    if _DOCX_HINT_RE.search(query):
        return ".docx"
    if _PPTX_HINT_RE.search(query):
        return ".pptx"
    if _CSV_HINT_RE.search(query):
        return ".csv"
    if _MD_HINT_RE.search(query):
        return ".md"
    return default


def resolve_dir_output_path(file_path: str, query: str) -> str:
    """``file_path`` が既存ディレクトリなら ``dir/output_<UTC><ext>`` に解決する。

    ファイル指定・空文字・解決不能パスは原文のまま返す。``<ext>`` は ``query`` から
    ``infer_output_extension`` で推論する。``write_file`` がディレクトリ指定を
    エラーにする問題を、書込み前にファイル名へ解決して回避する。
    """
    if not file_path:
        return file_path
    try:
        p = Path(file_path)
        if p.is_dir():
            ext = infer_output_extension(query)
            return str(p / f"output_{utc_compact_stamp()}{ext}")
    except OSError:
        pass
    return file_path


def is_table_output(file_path: str) -> bool:
    """``file_path`` の拡張子がスプレッドシート/表形式か判定する。"""
    return Path(file_path).suffix.lower() in TABLE_OUTPUT_EXTS


def is_rich_table_output(file_path: str) -> bool:
    """``file_path`` の拡張子がリッチ文書 (Word / PowerPoint) か判定する。"""
    return Path(file_path).suffix.lower() in RICH_TABLE_OUTPUT_EXTS


def wants_fetched_table(file_path: str) -> bool:
    """取得済み実テーブルを決定論的に書き込むべき出力先か (表計算 or リッチ文書)。"""
    return Path(file_path).suffix.lower() in FETCHED_TABLE_EXTS
