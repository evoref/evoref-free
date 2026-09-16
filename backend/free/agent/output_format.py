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


#: 発話が **裸の名前で** 挙げているファイル。直前が区切り文字のものは除く
#: (``E:\\out\\report.md`` の ``report.md`` を裸名と取ると、保存先が同じ
#: ディレクトリのときに読み元を上書きしてしまう)。
_BARE_FILENAME_RE = re.compile(
    r"(?<![\\/\w])(?P<name>[^\s\\/、。「」『』（）()\[\]]+\.[A-Za-z0-9]{1,8})(?![\\/])",
)


def named_output_basename(query: str) -> str | None:
    """発話が裸の名前で挙げている出力ファイル名 (曖昧なら ``None``)。

    「点検記録.txt を作って」「Append two lines to 点検記録.txt」のように
    **名前だけ** が書かれている場合に、その綴りを返す。候補が 2 つ以上あれば
    どれが書込み先か決められないので ``None``。
    """
    names = {m.group("name") for m in _BARE_FILENAME_RE.finditer(query or "")}
    return next(iter(names)) if len(names) == 1 else None


#: 裸の名前が **書込み先として** 挙がっている形。日本語は格助詞 + 書込み動詞
#: (「点検記録.txt を作って」「点検記録.txt に二行追記する」)、英語は後置の
#: ``to`` / ``into`` (「Append two lines to 点検記録.txt」)。読み元
#: (「report.md を読んで」) を拾わないために動詞まで見る。
_BARE_WRITE_TARGET_JA_RE = re.compile(
    r"(?<![\\/\w])(?P<name>[^\s\\/、。「」『』（）()\[\]]+\.[A-Za-z0-9]{1,8})(?![\\/])"
    r"\s*(?:[をにへ]|に対して)\s*[^\s、。]{0,10}?"
    # 活用形まで見る。終止形だけだと「点検記録.txt を作って」が外れる
    # (語形が 1 つ外れると判定ごと落ちる、の典型)。``作業`` のような別語に
    # 当たらないよう、語幹の直後の 1 文字まで固定する。
    r"(?:書(?:き|い|く|込|出)|追記|保存|出力|作(?:成|っ|り|る)|生成|セーブ|上書き)",
)
_BARE_WRITE_TARGET_EN_RE = re.compile(
    r"\b(?:to|into)\s+"
    r"(?<![\\/])(?P<name>[^\s\\/、。「」『』（）()\[\]]+\.[A-Za-z0-9]{1,8})(?![\\/])",
    re.IGNORECASE,
)


def named_write_target_basename(text: str) -> str | None:
    """発話/タスクが **書込み先として** 名指ししている裸のファイル名。

    :func:`named_output_basename` との違いは動詞を見ること。読み元として挙がって
    いるだけの名前 (「report.md を読んで要約して」) を書込み先と取り違えると、
    事故を防ぐつもりの補正が **読み元の上書き** を起こす。候補が 2 つ以上なら
    どれとも決められないので ``None``。
    """
    body = text or ""
    names = {
        m.group("name")
        for pattern in (_BARE_WRITE_TARGET_JA_RE, _BARE_WRITE_TARGET_EN_RE)
        for m in pattern.finditer(body)
    }
    return next(iter(names)) if len(names) == 1 else None


def redirect_unnamed_overwrite(file_path: str, *contexts: str) -> str:
    """誰も名指ししていない **既存ファイル** への上書きを、名指しの対象へ戻す。

    実インシデント (2026-09-16 ライブ監査 F-13): 「同じファイルに二行追記して
    ください」(対象 ``点検記録.txt``) で、直前のターンの一覧に出ていただけの
    ``drive.log`` が書込み先に選ばれ、**4,761 バイトで上書き**された。書込みは
    成功するのでどこにも失敗が出ず、ユーザーには「追記しました」としか見えない。

    条件は 3 つすべて。1 つでも欠けたら触らない (推測で書込み先を動かさない):

    1. 書込み先が **既に存在するファイル** — 新規作成は壊すものが無い
    2. その名前が task / query の **どこにも現れない** — 名指しされていれば
       ユーザーの指示どおり
    3. task / query が **書込み先として** ちょうど 1 つの名前を挙げている
       (:func:`named_write_target_basename`)

    振り替え先は「書込み先として挙がっている名前」を、モデルが選んだ
    ディレクトリの下に置いたもの。
    """
    if not file_path:
        return file_path
    try:
        p = Path(file_path)
        if not p.is_file():
            return file_path
    except OSError:
        return file_path
    haystack = " ".join(c or "" for c in contexts)
    if p.name and p.name in haystack:
        return file_path
    named = named_write_target_basename(haystack)
    if not named or named == p.name:
        return file_path
    return str(p.parent / named)


def resolve_dir_output_path(file_path: str, query: str) -> str:
    """``file_path`` が既存ディレクトリならその下のファイル名へ解決する。

    ファイル指定・空文字・解決不能パスは原文のまま返す。``write_file`` が
    ディレクトリ指定をエラーにする問題を、書込み前にファイル名へ解決して回避
    する。

    名前は **発話が挙げていればその綴り**、無ければ ``output_<UTC><ext>``
    (``<ext>`` は :func:`infer_output_extension` の推論)。発話の名前を優先
    するのは、生成名にするとユーザーが指したファイルが別名で作られ、以後の
    追記・読み出しが行方不明になるため — 実インシデント (2026-09-16 ライブ
    監査 F-06): 「同じファイルに二行追記してください」(対象 ``点検記録.txt``)
    が ``E:\\tmp\\live_audit_20260916\\output_20260916T010728Z.txt`` へ書かれ、
    次のターンの読み出しは ``File not found: …\\点検記録.txt`` になった。
    """
    if not file_path:
        return file_path
    try:
        p = Path(file_path)
        if p.is_dir():
            named = named_output_basename(query)
            if named:
                return str(p / named)
            ext = infer_output_extension(query)
            return str(p / f"output_{utc_compact_stamp()}{ext}")
    except OSError:
        pass
    return file_path


#: ``write_file`` の戻り値 (``Written 158 bytes to E:\tmp\a.txt``) から書込み先を
#: 拾うパターン。書いた事実を読み直す側 (最終応答の本文提示 / SSE の書込み先
#: 表示) が共有する SSOT — 各所で書き写すと片方だけ形式追随に失敗する。
WRITTEN_PATH_RE = re.compile(r"Written\s+\d+\s+bytes?\s+to\s+(.+?)\s*$", re.MULTILINE)


def anchor_relative_output_path(file_path: str) -> str:
    """錨の無い相対パスを既定の出力先 (``local_paths.outputs_dir``) へ寄せる。

    規則 (書込み経路すべてで共通):

    - 絶対パス (``E:\\tmp\\a.md`` / ``/home/u/a.md``) → そのまま
    - ``./`` ``../`` ``~`` 始まり → **明示的な相対指定** なのでそのまま
      (``..`` は ``write_file`` の traversal ガードが別途拒否する)
    - それ以外の相対パス (``compose.yaml`` / ``deploy/compose.yaml``) →
      ``<outputs_dir>/<相対パス>``

    最後の 1 行が本題。裸の名前をそのまま ``write_file`` へ渡すとプロセスの
    CWD (= リポジトリ直下) に着地し、ユーザーが指してもいない場所へゴミが
    残る (実インシデント 2026-09-08 監査 F-05: リポジトリ直下に
    ``compose.yaml`` が作られた)。純粋関数ではない (config を読む) が、
    config 未ロードでも既定値へ解決する。
    """
    if not file_path:
        return file_path
    normalized = file_path.replace("\\", "/")
    if normalized.startswith(("./", "../", "~")):
        return file_path
    p = Path(file_path)
    # ドライブ文字付き (``E:\tmp``) は POSIX 上で is_absolute() が False に
    # なるため、明示的に見る (Windows で書かれたパスがテストで寄せられない)。
    if p.is_absolute() or (len(file_path) > 1 and file_path[1] == ":"):
        return file_path

    from backend.config import resolve_outputs_dir

    return str(resolve_outputs_dir() / p)


def is_table_output(file_path: str) -> bool:
    """``file_path`` の拡張子がスプレッドシート/表形式か判定する。"""
    return Path(file_path).suffix.lower() in TABLE_OUTPUT_EXTS


def is_rich_table_output(file_path: str) -> bool:
    """``file_path`` の拡張子がリッチ文書 (Word / PowerPoint) か判定する。"""
    return Path(file_path).suffix.lower() in RICH_TABLE_OUTPUT_EXTS


#: 画像 / 図形を実体化できる出力形式 (docs/f_11_file_export.md §3.1)。
#: ``RICH_TABLE_OUTPUT_EXTS`` は「取得済みテーブルを決定論的に書く」対象の
#: 集合なので ODF を含まない。画像・図形の案内はそれとは別の軸で決める。
MEDIA_CAPABLE_OUTPUT_EXTS: frozenset[str] = frozenset(
    {".docx", ".pptx", ".odt", ".odp"},
)


def is_media_capable_output(file_path: str) -> bool:
    """``file_path`` が画像 (と形式によっては図形) を埋め込める形式か。"""
    return Path(file_path).suffix.lower() in MEDIA_CAPABLE_OUTPUT_EXTS


def wants_fetched_table(file_path: str) -> bool:
    """取得済み実テーブルを決定論的に書き込むべき出力先か (表計算 or リッチ文書)。"""
    return Path(file_path).suffix.lower() in FETCHED_TABLE_EXTS
