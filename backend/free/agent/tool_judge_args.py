"""クエリ → ツール引数の抽出 (純粋関数)

ファイルパス / 検索語 / 行範囲 / 算術式など、**依頼文から決定論的に決まる引数**
の抽出器を集める。モデルの転記に委ねると層ごとに結果が割れるため、抽出は必ず
ここを通す。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# クエリ先頭の URL を抽出する。非 ASCII (CJK 等) を除外し、「URL + 日本語」
# 入力で末尾テキストを URL に取り込まないようにする
# (例: https://news.yahoo.co.jp/で取得して... → https://news.yahoo.co.jp/ のみ)。
_URL_IN_QUERY_RE = re.compile(r"(https?://[^\s\]）」』\u0080-\U0010ffff]+)")
def _normalize_path_text(text: str) -> str:
    """パス照合用にセパレータと大小文字を正規化する (純粋関数)。"""
    return text.replace("\\", "/").casefold()
def _coerce_positive_int(value: object) -> int | None:
    """aux の型崩れ JSON 由来の値を正の int へ正規化する (int / 数値文字列 /
    整数値 float を受理)。bool や非数値、0 以下は ``None`` を返す。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            n = int(s)
            return n if n > 0 else None
    return None
#: 「プロジェクトのルート」を指す表現。列挙対象をカレントディレクトリに解決する。
_PROJECT_ROOT_REFERENCE_RE = re.compile(
    r"(?:プロジェクト|リポジトリ|ルート|トップ(?:レベル)?|一番上)"
    r"|(?<![A-Za-z])(?:project|repo(?:sitory)?|root|top[-\s]?level)(?![A-Za-z])",
    re.IGNORECASE,
)

#: ``<名前> ディレクトリ`` / ``<名前> フォルダ`` の ``<名前>`` を取る。パス片として
#: ありうる文字だけを許し、和文は取らない (「このディレクトリ」の「この」等を
#: 対象名と誤認しないため)。
_NAMED_DIRECTORY_RE = re.compile(
    r"([A-Za-z0-9._/\\-]+)\s*(?:ディレクトリ|フォルダ)"
    r"|(?:director(?:y|ies)|folders?)\s+([A-Za-z0-9._/\\-]+)",
    re.IGNORECASE,
)


def resolve_listing_directory(query: str, root: Path) -> str | None:
    """列挙対象のディレクトリを解決する。**実在するものだけ**返す (純粋関数)。

    存在しないパスを返さないのは、捏造パスを実行しても失敗するだけで価値が無い
    ためで、``_READ_PATH_TOOLS`` の方針と同じ。解決できなければ ``None`` を返し、
    呼び出し側は後段の層 (aux 判定) へ委ねる — 当てずっぽうの引数でツールを
    撃つより、シグナルだけ立てて判断を渡すほうが安全。
    """
    for match in _NAMED_DIRECTORY_RE.finditer(query):
        name = match.group(1) or match.group(2)
        if not name:
            continue
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = root / name
        if candidate.is_dir():
            return name
    if _PROJECT_ROOT_REFERENCE_RE.search(query):
        return "."
    return None
def _extract_search_pattern(query: str) -> str:
    """クエリから検索パターンを抽出する

    「検索」「search」等のキーワード自体を除外し、
    実際の検索対象となる語句を返す。

    例:
        "関数名 hello を検索して" → "hello"
        "search for parse_config" → "parse_config"
        "grep pattern" → "pattern"
    """
    # バッククォート内のパターン
    m = re.search(r'`([^`]+)`', query)
    if m:
        return m.group(1)

    # 引用符内のパターン
    m = re.search(r'[「"\'](.*?)[」"\']', query)
    if m:
        return m.group(1)

    # 検索/search/grep/find 等を除去した残りからキーワードを抽出
    cleaned = re.sub(
        r"(?:を|で|して|する|しろ|で検索|を検索|検索して|検索する"
        r"|search\s+(?:for|in)|grep|find|検索)",
        " ", query,
    )
    # 英数字・アンダースコアで構成されるトークンを探す
    tokens = re.findall(r"[A-Za-z_]\w{2,}", cleaned)
    if tokens:
        return tokens[0]

    return ""


# ディレクトリパス抽出用: ドライブレター配下のパスセグメントを解析する。
# 各セグメントは「\」直後が非空白文字で始まる前提とする
# (``[A-Za-z0-9_.]`` から開始し、内部は空白を含んでよい)。
# 「...\aa\ with the content」のように、ディレクトリ指定の直後に自然文
# (英語の説明文) が「\」+ 空白で続くケースを誤ってパスセグメントとして
# 飲み込まないための境界条件 (#incident: 日本語ファイル名クエリで
# planner が生成した英語タスク記述の一部がパスに混入した)。
# 実在の Windows パスでバックスラッシュ直後が空白になることはない
# ("Program Files" のようにセグメント内部に空白を含むのは許容する)。
#: ドライブレター付きパスの区切り。Windows はスラッシュ区切りも等価に受け付け、
#: ユーザーもツール出力もそちらを書く。バックスラッシュ限定にしていたため
#: ``E:/tmp/a.txt`` が 1 つも抽出できず、ルール層が read_file を選べないまま
#: aux 層へ落ちていた (実インシデント 2026-08-04 ライブ監査: 同じ依頼が
#: read_file / search_history / ツール未発火に割れる原因)。
_DIR_PATH_RE = re.compile(
    r"([A-Za-z]:(?:[\\/][A-Za-z0-9_.][A-Za-z0-9_. -]*)*)",
)

# クォート文字の対応表（開き, 閉じ）。ファイル名の語幹 (拡張子直前) が
# 日本語等の非ASCIIの場合に、明示的にクォートされたファイル名を抽出する
# 際に使う。
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'), ("'", "'"), ("「", "」"), ("『", "』"),
)


def _extract_quoted_filename(query: str) -> str | None:
    """クォートで明示的に囲まれたファイル名を抽出する（非ASCII語幹対応）。

    ``[A-Za-z0-9_-]+\\.ext`` 前提の ASCII 限定パターンでは、「テスト.docx」の
    ように拡張子直前が日本語等の非ASCIIだと一切マッチしない。クォートで
    明示されていれば語幹の文字種を問わず抽出する（クォート無しの非ASCII
    語幹は文中の地の文と区別できず誤検出リスクが高いため対象外）。
    """
    for open_q, close_q in _QUOTE_PAIRS:
        m = re.search(
            re.escape(open_q)
            + r"([^\n" + re.escape(open_q) + re.escape(close_q) + r"]{1,200}"
            r"\.[A-Za-z0-9]{1,10})"
            + re.escape(close_q),
            query,
        )
        if m:
            return m.group(1).strip()
    return None
#: 「最初の 3 行」「先頭 10 行」「first 5 lines」等、ファイル先頭からの行数指定。
#: 全角数字も拾う (日本語入力では「３行」になりやすい)。
_HEAD_LINES_RE = re.compile(
    r"(?:最初|先頭|冒頭|頭|first|head|top)\D{0,6}?([0-9０-９]{1,4})\s*(?:行|lines?)",
)

#: 「このファイルは存在しますか」= 有無だけを問う質問。
_FILE_EXISTENCE_RE = re.compile(
    r"(?:存在し|ありますか|あるか|残ってい|消えてい|できてい"
    r"|\bexists?\b|\bis there\b|\bstill there\b)",
    re.IGNORECASE,
)
#: 本文そのものを求める語。存在確認と併記されていれば内容要求が優先される
#: (「まだ存在しますか？先頭3行だけ見せてください」)。
_FILE_CONTENT_REQUEST_RE = re.compile(
    r"(?:見せ|見たい|中身|内容|読[んみむ]|表示|出力|全文|何文字|文字数|何行|行数"
    r"|\bshow\b|\bcontent\b|\bread\b|\bdisplay\b|\bprint\b|\bdump\b)",
    re.IGNORECASE,
)


def asks_file_existence_only(query: str) -> bool:
    """ファイルの有無だけを問い、本文は求めていないか。

    有無だけを聞かれているのに ``read_file`` を範囲指定なしで撃つと全文が
    ツール結果として返り、モデルはそれを回答に丸ごと復唱する。

    2026-08-16 ライブ監査ターン 14「E:\\...\\README.md というファイルは存在
    しますか？」: 3,331 文字の全文が返り、モデルは全文の復唱を始めて
    **ちょうど 1,024 トークン (llama.max_tokens の既定値) で表の途中で切断**
    された。yes/no の質問に **197 秒** かけ、しかも回答は未完だった。

    ``read_file`` は先頭にメタ行 ``[file: ... | lines: N | chars: M]`` を付ける
    ので、1 行だけ読めば「存在する / 何行・何文字か」は決定論的に答えられる。
    """
    return bool(
        _FILE_EXISTENCE_RE.search(query)
        and not _FILE_CONTENT_REQUEST_RE.search(query),
    )


def _extract_head_line_count(query: str) -> int | None:
    """「最初の N 行」の N を返す (指定が無ければ ``None``)。

    本文全体を渡すとモデルが行数指定を守らずほぼ全文を出力するため
    (実測 2026-08-05: NOTICE.md の「最初の 3 行」で約 1,264 文字を出力)、
    read_file 側で切り出せるようにツール引数へ渡す。
    """
    m = _HEAD_LINES_RE.search(query)
    if not m:
        return None
    try:
        count = int(m.group(1).translate(_ZENKAKU_DIGITS))
    except ValueError:
        return None
    return count if count > 0 else None
#: 全角数字 → ASCII。
_ZENKAKU_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
def _extract_file_path(query: str) -> str:
    """クエリからファイルパスを抽出する

    日本語の自然言語テキストからファイルパスを抽出する。
    「e:\\直下にa.txtのファイル名で...」→ 「e:\\a.txt」のように、
    ドライブレターとファイル名を組み合わせて解釈する。
    抽出後、連続バックスラッシュ (\\\\) をシングル (\\) に正規化する。
    """
    # URL はファイル名抽出の対象から除外する。URL ドメイン (例: soccer.yahoo.co.jp)
    # が「co.jp」のようなファイル名として誤抽出されるのを防ぐ。
    query = _URL_IN_QUERY_RE.sub(" ", query)

    # 1a. 非 ASCII を含みうるフルパス: E:\tmp\日本語テスト.txt / E:/tmp/日本語.txt
    #     ASCII 限定にすると日本語ファイル名が拡張子の手前で切れ、切り詰めた
    #     パスがたまたま実在ディレクトリだと read_file ではなく list_directory が
    #     選ばれ、実在するファイルを「見つからない」と答える (実測 2026-08-05)。
    #     区切りは \ と / の双方を受ける。バックスラッシュ限定だと ``E:/tmp/a.txt``
    #     が 1 つも抽出できず、同じ依頼が read_file / search_history / ツール
    #     未発火に割れていた (実測 2026-08-04)。
    #     地の文を飲み込まないための境界条件は 2 つ:
    #       - 空白 (半角/全角) とクォートを含まない (「E:\tmp に置いた report.txt」)
    #       - ドライブ直下ではなく 1 階層以上下 (「e:\直下にa.txtのファイル名で」)。
    #         ドライブ直下 + 非 ASCII は地の文と構造的に区別できないため、
    #         従来どおり Pattern 2 (ドライブ + ファイル名) に委ねる。
    m = re.search(
        r"[A-Za-z]:[\\/][^\s　\"'「」『』\\/]+[\\/][^\s　\"'「」『』]*\.[A-Za-z0-9]{1,10}",
        query,
    )
    if m:
        return _normalize_path_separators(m.group(0))

    # 1b. 空白を含む ASCII パス: C:\Program Files\app.exe
    #     空白を許容する代償として本体は ASCII 限定にし、地の文 (日本語) で
    #     停止させる。区切りは 1a と同様に \ と / の双方を受ける。
    m = re.search(r"[A-Za-z]:[\\/][A-Za-z0-9_.\\/ -]+\.[A-Za-z0-9]{1,10}", query)
    if m:
        return _normalize_path_separators(m.group(0).rstrip(" "))

    # 2. ドライブレター + 自然言語でのファイル名指定
    #    例: 「e:\直下にa.txtのファイル名で」→ e:\a.txt
    #    ディレクトリとファイル名が日本語/全角スペースで分断されていても、
    #    ディレクトリ部 (Pattern 3 と同じ捕捉) を取り出してファイル名と結合し、
    #    サブ階層を保持する。深い階層が無い (ドライブ直下指定) 場合のみ
    #    従来どおりドライブ直下へフォールバックする。
    #    \w は日本語にもマッチするため ASCII 限定で検索
    drive_match = re.search(r"([A-Za-z]):[\\/]", query)
    file_match = re.search(r"([A-Za-z0-9_-]+\.[A-Za-z0-9]{1,10})(?=[^A-Za-z0-9_.]|$)", query)
    # ファイル名の語幹が非ASCII (日本語等) だと file_match はマッチしない
    # ("テスト.docx" 等)。その場合はクォートで明示されたファイル名を拾う。
    filename = file_match.group(1) if file_match else _extract_quoted_filename(query)
    if drive_match and filename:
        dir_match = _DIR_PATH_RE.search(query)
        if dir_match:
            # セグメント内部は空白を許容するため ("Program Files" 等)、末尾に
            # 地の文へ続く空白が巻き込まれることがある (例: "aa に保存して" の
            # "aa " )。rstrip() で末尾空白 (全角含む) を落としてから区切りも除去。
            # さらに英語の地の文が空白のみで続くケース ("aa in Excel format") は
            # 実在チェックで切り落とす。
            directory = _trim_nonexistent_path_tail(
                _normalize_path_separators(
                    dir_match.group(1).rstrip(),
                ).rstrip("\\/"),
            )
            return f"{directory}\\{filename}"
        return f"{drive_match.group(1)}:\\{filename}"

    # 3. ディレクトリパスのみ（ファイル名なし）: E:\xxx\ や E:\xxx 等
    #    配下のファイルを参照する文脈では、ディレクトリパスを返す。
    #    全角スペース (U+3000) 等の Unicode 空白や文末で終端しても、
    #    セグメント単位で解析する _DIR_PATH_RE が自然に正しい境界で止まる。
    if drive_match:
        dir_match = _DIR_PATH_RE.search(query)
        if dir_match:
            return _trim_nonexistent_path_tail(
                _normalize_path_separators(dir_match.group(1).rstrip()),
            )

    # 4. Unix パス: /home/user/file.txt
    m = re.search(r"(?:^|[\s　])((?:/[\w._-]+){2,})", query)
    if m:
        return m.group(1)

    # 5. bare ファイル名 (拡張子付き): dice_roller.py / README.md / app.svelte
    #    ドライブレターも Unix パスもない場合のフォールバック。CWD 相対として
    #    のタスクを出したとき write_file の auto-recovery / fast-path が働くようにする。
    #    誤検出防止のため拡張子は英字始まりに限定 (「3.12」「v1.2」等を弾く)。
    m = re.search(
        r"(?:^|[\s　`'\"(\[])"
        r"([A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\.[A-Za-z][A-Za-z0-9]{0,9})"
        r"(?=$|[\s　`'\")\].,;:!?])",
        query,
    )
    if m:
        return m.group(1)

    return ""
# --- 算術式抽出 (calculate ツールの決定論的ルーティング) ---------------------
# 全角の数字・演算子を ASCII へ寄せる。カタカナ長音符 (ー) や罫線 (―) は
# 日本語語中に頻出するため意図的に含めない (マイナスへ誤変換すると
# 「コーヒー」等が式断片に見えてしまう)。
_ARITH_NORMALIZE = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "＋": "+", "－": "-", "−": "-",
    "×": "*", "✕": "*", "＊": "*",
    "÷": "/", "／": "/", "％": "%", "＾": "^",
    "（": "(", "）": ")", "．": ".",
})
# 算術式になりうる文字だけからなる連続領域
_ARITH_RUN_RE = re.compile(r"[0-9.+\-*/%^()\s]+")
# 日付・バージョン番号の誤検出除け (2026-07-27 は BinOp として parse できてしまう)
_ARITH_DATE_LIKE_RE = re.compile(
    r"^(?:\d{4}\s*-\s*\d{1,2}\s*-\s*\d{1,2}"
    r"|\d{1,2}\s*/\s*\d{1,2}(?:\s*/\s*\d{2,4})?)$",
)
# 「式の値を求めている」ことの手掛かり。式だけが裸で書かれた場合は不要。
_ARITH_REQUEST_CUE_RE = re.compile(
    r"(?:いくつ|いくら|答え|計算|求め|何になる|=|＝"
    r"|(?<![A-Za-z])calculate(?![A-Za-z])|(?<![A-Za-z])compute(?![A-Za-z])"
    r"|what\s+is|how\s+much|(?<![A-Za-z])equals?(?![A-Za-z]))",
    re.IGNORECASE,
)
# 式の直後に助詞と疑問符しか残らない形 (「1+1は？」「12*34」) も計算依頼とみなす
_ARITH_BARE_TAIL_RE = re.compile(r"^[\s　]*(?:とは|って|は|の)?[\s　]*[?？。!！]*$")
_ARITH_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)


def _is_numeric_expression(expression: str) -> bool:
    """``expression`` が数値リテラルと算術演算子だけで構成されるか (純粋関数)。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    has_operator = False
    for node in ast.walk(tree):
        if not isinstance(node, _ARITH_SAFE_NODES):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False
        if isinstance(node, ast.BinOp):
            has_operator = True
    return has_operator


def _extract_arithmetic_expression(query: str) -> str:
    """クエリに書かれた算術式を Python 構文へ正規化して返す (純粋関数)。

    「1234 × 5678 はいくつですか？」のような明示的な計算依頼で ``calculate``
    を決定論的に発火させるための抽出器。ルール層は従来「計算」の字句しか
    見ておらず、式そのものを書かれるとツール無しで base の暗算に落ちて
    誤答していた (実インシデント 2026-07-27 ライブ検証: 1234 × 5678 に
    7060672 と回答。正解は 7006652)。

    誤検出を避けるため、以下をすべて満たす場合のみ式を返す:

    * 数値リテラルと算術演算子のみで構成され、二項演算を 1 つ以上含む
    * 日付 (2026-07-27) / 日付表記 (7/27) ではない
    * 値を尋ねる手掛かり語があるか、式の前に文が無く後ろも助詞・疑問符だけ
      (「12*34」「1+1は？」のような裸の式)

    Returns:
        正規化済みの式。抽出できなければ空文字列。
    """
    normalized = query.translate(_ARITH_NORMALIZE)
    for match in _ARITH_RUN_RE.finditer(normalized):
        candidate = match.group(0).strip()
        if not candidate or _ARITH_DATE_LIKE_RE.match(candidate):
            continue
        # ^ は Python では XOR。書かれた意図は冪乗なので ** へ寄せる。
        candidate = candidate.replace("^", "**")
        if not _is_numeric_expression(candidate):
            continue
        head = normalized[: match.start()]
        tail = normalized[match.end():]
        bare = (
            not any(c.isalnum() for c in head)
            and _ARITH_BARE_TAIL_RE.match(tail) is not None
        )
        if bare or _ARITH_REQUEST_CUE_RE.search(normalized):
            return candidate
    return ""
def _normalize_path_separators(path: str) -> str:
    """連続バックスラッシュをシングルに正規化する

    LLM や JSON パース経由でパスが二重エスケープされるケースに対応。
    例: E:\\\\xxx\\\\tetris.py → E:\\xxx\\tetris.py
    """
    # 連続する2つ以上の \ を1つに置換
    return re.sub(r"\\{2,}", r"\\", path)


def _trim_nonexistent_path_tail(path: str) -> str:
    """実在チェックに基づき、パス末尾へ混入した自然文トークンを切り落とす。

    ``_DIR_PATH_RE`` はセグメント内部の空白を許容する ("Program Files") ため、
    LLM 生成のタスク記述がパス直後に空白 + 英語の修飾語を続けると
    (実インシデント: ``...to C:\\...\\Desktop\\aa in Excel format``) 地の文が
    末尾セグメントへ飲み込まれ、実在しない拡張子なしパスへの平文書込みに
    化ける (リッチ文書経路・検証ゲートをすべてバイパス)。

    捕捉パスが実在しない場合のみ、空白区切りトークンを右から 1 つずつ外し
    ながら「実在する最長の空白境界プレフィックス」を探して返す。地の文の
    混入は必ず空白境界で起きるため、バックスラッシュ境界では分割しない。
    どのプレフィックスも実在しなければ原文のまま返す (新規パスの指定を
    壊さない)。
    """
    try:
        if not path or Path(path).exists():
            return path
    except (OSError, ValueError):
        return path
    candidate = path
    while " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0].rstrip()
        if not candidate or candidate.endswith(":"):
            break
        try:
            if Path(candidate).exists():
                return candidate
        except (OSError, ValueError):
            break
    return path
