"""クエリ → ツール引数の抽出 (純粋関数)

ファイルパス / 検索語 / 行範囲 / 算術式など、**依頼文から決定論的に決まる引数**
の抽出器を集める。モデルの転記に委ねると層ごとに結果が割れるため、抽出は必ず
ここを通す。
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path
from typing import NamedTuple

from backend.log_config import get_logger

logger = get_logger("agent.tool_judge_args")

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
#: 括り記号の対応表 (開き, 閉じ)。開きと閉じは **対で** 見る — 開きの ``"`` を
#: 閉じの ``」`` で受けるような混在は許さない (文中の引用符が別の括りの内側まで
#: 飲み込むため)。
_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'), ("“", "”"), ("「", "」"), ("『", "』"),
)
#: ``'`` の括り。英文の所有格・短縮形 (``don't`` / ``it's``) と区別できないので
#: 既定では採らず、検索語・ファイル名の抽出 (ユーザーが明示的に括る文脈) だけが
#: ``single_quotes=True`` で受ける。
_SINGLE_QUOTE_PAIR: tuple[str, str] = ("'", "'")


@functools.lru_cache(maxsize=None)
def _quoted_span_regex(pairs: tuple[tuple[str, str], ...], max_len: int) -> re.Pattern:
    """括り対ごとの alternation を組む (中身は同じ括り記号と改行を含まない)。"""
    alts = []
    for open_q, close_q in pairs:
        inner = f"[^\\n{re.escape(open_q)}{re.escape(close_q)}]{{1,{max_len}}}"
        alts.append(f"{re.escape(open_q)}({inner}){re.escape(close_q)}")
    return re.compile("|".join(alts))


def quoted_spans(
    query: str, *, max_len: int = 400, single_quotes: bool = False,
) -> list[str]:
    """クエリ中で括られた文字列を出現順に (重複なく、前後空白を除いて) 返す。

    判定系で括りを取り出す **唯一の実装**。以前は 4 箇所 (判定本体の
    ``_first_quoted_span`` / 履歴の ``quoted_search_terms`` / 検索語抽出 /
    クォート付きファイル名) が微妙に違う正規表現を各自で持っていた。
    ``max_len`` は中身の長さ上限 (呼出側の用途で変える: 検索語 40 / 訳文 400)。
    """
    pairs = _QUOTE_PAIRS + ((_SINGLE_QUOTE_PAIR,) if single_quotes else ())
    out: list[str] = []
    for m in _quoted_span_regex(pairs, max_len).finditer(query or ""):
        span = next((g for g in m.groups() if g), "").strip()
        if span and span not in out:
            out.append(span)
    return out


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
    quoted = quoted_spans(query, single_quotes=True)
    if quoted:
        return quoted[0]

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
#: セグメント部の繰り返しは ``*`` ではなく ``+``。**区切りの無い裸のドライブ
#: レター (``E:``) にはマッチさせない**。``*`` だとセグメントが非 ASCII で
#: 始まるクエリ (「E:\日本語.txt を読んで」) で 0 回マッチが成立し、
#: ディレクトリとして ``"E:"`` を返していた — 実在しないパスであり、しかも
#: ファイル要求がディレクトリ要求に化ける。ドライブ直下そのものを指す
#: ``E:\`` は第 2 選択肢で拾う。
#: セグメント内の空白は、直後が次のドライブレター (``X:``) なら取らない — 地の文を挟んだ
#: 次のパス (``Copy Q:\in\data to Q:\out``) のドライブレターを飲み込み、2 つ目を失っていた。
#: 最後のセグメントの空白と文末の ``.`` はここでは取りすぎてよく、境界は
#: :func:`iter_drive_dir_paths` が決める。区切りの重複 (二重エスケープの ``E:\\\\xxx``) は
#: 1 つの区切りとして受け、正規化は呼出側 (``_normalize_path_separators``) が行う。
#: ドライブレターの左は英字でないこと — ``in Japanese:\\docs`` の ``e:\\docs`` を拾わない。
_DIR_PATH_RE = re.compile(
    r"(?<![A-Za-z])"
    r"([A-Za-z]:(?:[\\/]+[A-Za-z0-9_.](?:[A-Za-z0-9_.-]| (?![A-Za-z]:))*)+|[A-Za-z]:[\\/])",
)

#: 丸ごと括られたドライブレター付きパス (``"…"`` / ``“…”`` / ``「…」`` / ``『…』`` / ``'…'``)。
#: 中身は ``_DIR_PATH_RE`` と同じ ASCII に空白を足した文字に限る — 括りの中が和文
#: (「E:\\tmp\\xに保存して」) なら発話の引用なので、括りを見ずに ``_DIR_PATH_RE`` の境界へ落とす。
#: 開きの直後がドライブレターである形だけを見るので、英文の短縮形の ``'`` (``don't``) は括りにならない。
_QUOTED_DRIVE_PATH_RE = re.compile("|".join(
    f"{re.escape(open_q)}\\s*([A-Za-z]:(?:[\\\\/][A-Za-z0-9_. -]*)+){re.escape(close_q)}"
    for open_q, close_q in (*_QUOTE_PAIRS, _SINGLE_QUOTE_PAIR)
))

#: 空白入りの名前の実在を確かめる語数の上限 (最後のセグメント)。長い英文が続いても
#: ファイルシステムへの問い合わせを語数ぶん撃たない。
_EXISTING_NAME_MAX_WORDS = 8


class DrivePath(NamedTuple):
    """依頼文のドライブレター付きパス 1 件 (:func:`iter_drive_dir_paths`)。"""

    start: int
    #: 括りなら閉じ記号の後、括り無しなら ``_DIR_PATH_RE`` の一致の終わり
    #: (直後の文字で「非 ASCII のセグメントの手前で切れた」を見分ける呼出側がある)。
    end: int
    path: str
    #: 最後のセグメントを英文の手前の空白で切った (``new app`` → ``new``)。切らない読み方も
    #: ありうるので、配信先の根は親まで広げて両方を覆う (``explicit_query_dirs``)。
    ambiguous: bool


def _dir_path_boundary(raw: str, following: str) -> tuple[str, bool]:
    """引用符の無いパス (``_DIR_PATH_RE`` の一致) の末尾の境界を構造で決める (f_03 §4.x)。

    ``following`` は一致の直後の文字列。戻り値は ``(パス, 曖昧か)``。最後のセグメントに空白が
    あるときだけ判断が要り、上から順に:

    1. 直後 (空白を飛ばした次の文字) が非 ASCII (和文) → 空白入りのセグメントを丸ごと受ける
       (``E:\\tmp\\my app フォルダに`` / ``E:\\tmp\\My Report.docx に保存して``)。
    2. 空白入りの名前が **実在する** → 長い方から受ける (``C:\\Program Files`` 型、
       最大 ``_EXISTING_NAME_MAX_WORDS`` 語)。
    3. 最初の語の直後に空白 1 つで拡張子付きの語が続く → その 2 語でファイル名
       (``Save it as Q:\\tmp\\My Report.docx``)。間に語があれば (``todo and name it main.py``) 続きにしない。
    4. それ以外 (英文や文末が続く) → 最初の空白で切り、曖昧と印を付ける
       (``in the E:\\tmp\\todo folder.`` → ``E:\\tmp\\todo``)。``new dir`` と ``todo folder`` は字面で
       区別できないので、語形は列挙しない。空白入りの名前を確実に渡すには括る。

    空白の後に区切りが続く (``C:\\Program Files\\MyApp``) なら、その空白はセグメントの内部。
    末尾の ``.`` は文末の句読点で、Windows は名前の末尾の ``.`` を持てないので構造的にパスではない
    (内部の ``.`` — ``v1.2`` / ``report.txt`` — は残す)。

    以前は空白を取りすぎてから「実在する最長の接頭辞」へ切り詰めていたため、未作成の
    フォルダでは地の文ごと残った (``...\\Desktop\\aa in Excel format`` への平文書込み、
    2026-09-28 R16 の ``in the E:\\tmp\\todo folder.`` → ``E:\\tmp\\todo folder.``)。
    """
    path = _normalize_path_separators(raw.rstrip())
    cut = max(path.rfind("\\"), path.rfind("/")) + 1
    head, words = path[:cut], path[cut:].split(" ")
    if len(words) == 1:
        return head + words[0].rstrip("."), False
    next_char = following.lstrip()[:1]
    if next_char and not next_char.isascii():
        return head + " ".join(words).rstrip(". "), False
    for n in range(min(len(words), _EXISTING_NAME_MAX_WORDS), 1, -1):
        candidate = head + " ".join(words[:n]).rstrip(". ")
        try:
            if Path(candidate).exists():
                return candidate, False
        except (OSError, ValueError):
            break
    first, second = words[0], words[1].rstrip(".")
    if (
        not first.endswith(".")
        and not _QUOTED_FILENAME_EXT_RE.search(first)
        and _QUOTED_FILENAME_EXT_RE.search(second)
    ):
        return f"{head}{first} {second}", False
    return head + first.rstrip("."), True


def iter_drive_dir_paths(query: str) -> list[DrivePath]:
    """依頼文のドライブレター付きパスを出現順に返す (:class:`DrivePath`)。

    ドライブパスの末尾の境界 (文末の句読点 / 空白の後の地の文 / 括られた空白入りの名前) を
    決める実装 (f_03 §4.x)。明示フォルダの抽出 (``explicit_query_dirs``) とファイルパスの抽出
    (``_extract_file_path_literal`` の Pattern 1b / 2 / 3) が共有する。括られたパスはそのまま
    (空白入りの名前も) 受け、括りの内側の一致は二重に数えない。区切りの正規化はするが、
    末尾の区切りは落とさない (ドライブ直下 ``E:\\`` を残すため)。

    別の境界がまだ残っている (統合は残件): ``intent_vocab.EXPLICIT_WINDOWS_PATH_RE`` + 句読点の
    rstrip (``explicit_query_dirs`` の実在パス / ``_explicit_path_named``)、Pattern 1a (非 ASCII を含む
    フルパス)、``_extract_last_file_path`` の区切り。
    """
    text = query or ""
    found: list[DrivePath] = []
    quoted: list[tuple[int, int]] = []
    for m in _QUOTED_DRIVE_PATH_RE.finditer(text):
        span = next((g for g in m.groups() if g), "").strip()
        found.append(DrivePath(m.start(), m.end(), _normalize_path_separators(span), False))
        quoted.append((m.start(), m.end()))
    for m in _DIR_PATH_RE.finditer(text):
        if any(start <= m.start() < end for start, end in quoted):
            continue
        path, ambiguous = _dir_path_boundary(m.group(1), text[m.end():m.end() + 16])
        found.append(DrivePath(m.start(), m.end(), path, ambiguous))
    return sorted(found)

#: 括られた文字列 / 空白を含みうるドライブパス (Pattern 1b) がファイル名と言えるための拡張子 (末尾)。
_QUOTED_FILENAME_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,10}$")


def _extract_quoted_filename(query: str) -> str | None:
    """クォートで明示的に囲まれたファイル名を抽出する（非ASCII語幹対応）。

    ``[A-Za-z0-9_-]+\\.ext`` 前提の ASCII 限定パターンでは、「テスト.docx」の
    ように拡張子直前が日本語等の非ASCIIだと一切マッチしない。クォートで
    明示されていれば語幹の文字種を問わず抽出する（クォート無しの非ASCII
    語幹は文中の地の文と区別できず誤検出リスクが高いため対象外）。
    """
    for span in quoted_spans(query, max_len=200, single_quotes=True):
        if _QUOTED_FILENAME_EXT_RE.search(span):
            return span
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
    """クエリからファイルパスを抽出する (暗黙参照は台帳へフォールバック)"""
    found = _extract_file_path_literal(query)
    if found:
        return found
    # 文字列としてのパスが無い依頼 (「保存したファイルを読んで」「その中身を
    # 見せて」) は、いくら正規表現を足しても解けない。解けるのは「直前に何を
    # 書いたか」という観測事実の側で、それは ToolsRegistry が実行時に知って
    # いる (file_ledger のモジュール docstring を参照)。
    #
    # 実インシデント (2026-08-27 ライブ監査、再現 2/2):
    #   T13-7 「保存したファイルを読んで、構文エラーがないか確認してください。」
    #         → read_file が 1 回も撃たれず 238 秒後に「確認できません」
    #   T15-7 「その中身を見せてください。」→「表示できません」
    # どちらも実ファイルは存在し、明示パスを与えた T05 では正常に動いていた。
    return _resolve_implicit_file_path(query)


def _resolve_implicit_file_path(query: str) -> str:
    """暗黙参照を直近に触れたファイルへ解決する (無ければ空文字)。

    ゲートは 2 条件の AND:

    1. この発話が指示/過去参照 + ファイル対象語 (``references_recent_file``)
    2. **観測事実** — この会話で実際にファイルを読み書きしている

    2 を必須にするのが要点。「その中身」だけで台帳を引くと、ファイルに一度も
    触れていないターンでも直前の何かを掴んでしまう。
    """
    from backend.free.agent.file_ledger import (
        last_file_path,
        references_recent_file,
    )
    from backend.free.agent.tool_ledger import current_session_id

    if not references_recent_file(query):
        return ""
    session_id = current_session_id()
    if not session_id:
        return ""
    resolved = last_file_path(session_id)
    if resolved:
        logger.info(
            "Implicit file reference resolved to the most recent file: %s",
            resolved,
        )
    return resolved


def _extract_file_path_literal(query: str) -> str:
    """クエリの **文字列から** ファイルパスを抽出する

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
    #     末尾の境界はディレクトリと同じ iter_drive_dir_paths で決める — 以前は
    #     空白を無条件に受け、``in the E:\tmp\todo folder. Save a.py`` を
    #     ``E:\tmp\todo folder. Save a.py`` と読んだ (2026-09-28 R16)。空白入りの
    #     ファイル名は括られたとき・和文が続くとき・実在するとき・最初の語に空白 1 つで
    #     隣り合うとき (``My Report.docx``) だけ。
    dir_paths = iter_drive_dir_paths(query)
    for found in dir_paths:
        if _QUOTED_FILENAME_EXT_RE.search(found.path):
            return found.path

    # 2. ドライブレター + 自然言語でのファイル名指定
    #    例: 「e:\直下にa.txtのファイル名で」→ e:\a.txt
    #    ディレクトリとファイル名が日本語/全角スペースで分断されていても、
    #    ディレクトリ部 (Pattern 3 と同じ捕捉) を取り出してファイル名と結合し、
    #    サブ階層を保持する。深い階層が無い (ドライブ直下指定) 場合のみ
    #    従来どおりドライブ直下へフォールバックする。
    #    \w は日本語にもマッチするため ASCII 限定で検索
    drive_match = re.search(r"(?<![A-Za-z])([A-Za-z]):[\\/]", query)
    file_match = re.search(r"([A-Za-z0-9_-]+\.[A-Za-z0-9]{1,10})(?=[^A-Za-z0-9_.]|$)", query)
    # ファイル名の語幹が非ASCII (日本語等) だと file_match はマッチしない
    # ("テスト.docx" 等)。その場合はクォートで明示されたファイル名を拾う。
    filename = file_match.group(1) if file_match else _extract_quoted_filename(query)
    if filename is None and drive_match:
        # ドライブ直下の非 ASCII ファイル名 (``E:\日本語.txt``)。Pattern 1a は
        # 「ドライブ直下 + 非 ASCII」を地の文と区別できないとして除外している
        # (「e:\直下にa.txtのファイル名で」が丸ごと 1 つのパスに見えてしまう) が、
        # **クエリ全体に ASCII のファイル名もクォート付きファイル名も無い**
        # ときだけは曖昧さが無い (上の 2 つが先に効くため、除外例は必ずそちらで
        # 拾われる)。区切り直後から拡張子までを取る。
        m = re.search(
            r"[A-Za-z]:[\\/]([^\s　\"'「」『』\\/]{1,128}\.[A-Za-z][A-Za-z0-9]{0,9})",
            query,
        )
        if m:
            filename = m.group(1)
    if drive_match and filename:
        if dir_paths:
            # 末尾の地の文 ("aa in Excel format" / "todo folder.") は
            # iter_drive_dir_paths が境界で落とす。
            directory = dir_paths[0].path.rstrip("\\/")
            return f"{directory}\\{filename}"
        return f"{drive_match.group(1)}:\\{filename}"

    # 3. ディレクトリパスのみ（ファイル名なし）: E:\xxx\ や E:\xxx 等
    #    配下のファイルを参照する文脈では、ディレクトリパスを返す。
    #    全角スペース (U+3000) 等の Unicode 空白や文末で終端しても、
    #    セグメント単位で解析する _DIR_PATH_RE が自然に正しい境界で止まる。
    if dir_paths:
        return dir_paths[0].path

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

    # 6. クォートで明示された非 ASCII ファイル名 (「メモ.txt」)。
    #    Pattern 5 は語幹を ASCII に限っているので日本語のファイル名を拾えず、
    #    ドライブレターが無いクエリでは Pattern 2 の quoted fallback にも
    #    到達しなかった。クォートされている以上、地の文との区別は付いている。
    quoted = _extract_quoted_filename(query)
    if quoted:
        return quoted

    return ""


#: 書き込み先を後置で導く語 (英語)。``to`` / ``into`` の **後ろ** が宛先。
#: タスクグラフの記述は英語で生成されるため、実際に踏むのはこちら
#: (実測: ``Read the content of audit2_b.txt / Append the content to audit2_a.txt``)。
_WRITE_DEST_MARKERS_EN: tuple[str, ...] = (" into ", " onto ", " to ")

#: 書き込み先を前置で導く語 (日本語)。これらの **手前** が宛先。
#: 「の末尾に」は「A の中身を B の末尾に追記」の形を拾うために要る。
_WRITE_DEST_MARKER_JA_RE = re.compile(
    r"(?:の)?(?:末尾|先頭|後ろ)?\s*[にへ]\s*(?:追記|書[きい]|保存|出力|コピー|貼)",
)

#: 日本語の助詞・区切り。パス候補の切り出しに使う。
#: **コロンは入れない** — ``E:\\tmp\\out.md`` のドライブレターを切ってしまい、
#: 残骸が抽出器に掛からなくなる (実測でこの 1 件だけ落ちた)。
_PATH_SPLIT_RE = re.compile(r"[\s　、。,;「」『』\"'（）()]+|[をのがはへに]")


def _extract_last_file_path(text: str) -> str:
    """``text`` の中で **最後に現れる** ファイルパスを返す。

    ``_extract_file_path`` は最初の 1 件で打ち切るため、末尾側の宛先を取れない。
    パス候補は空白・助詞を含まないので、区切って右から順に既存の抽出器へ
    掛ければよい (パターン集合を二重管理しない)。
    """
    chunks = [c for c in _PATH_SPLIT_RE.split(text or "") if c]
    for chunk in reversed(chunks):
        found = _extract_file_path(chunk)
        if found:
            return found
    return ""


def extract_write_target_path(text: str) -> str:
    """**書き込み先** のファイルパスを返す (見つからなければ空文字列)。

    ``_extract_file_path`` は最初に現れたパスを返す。1 ファイルしか出てこない
    依頼ではそれが正しいが、**2 ファイルが登場するタスクでは先頭は常に
    source (読む側)** なので、書き込み先が source に化ける。

    実インシデント (2026-08-26 ライブ監査 T7-7):
    「audit2_b.txt の中身を audit2_a.txt の末尾に追記してください。」に対し
    ``write_file({'file_path': 'E:\\tmp\\audit2_b.txt', 'content': 'ブラボー'})``
    が実行され、**source を自分の中身で上書き**した。a.txt は未変更のまま、
    失敗の報告も無い (表示は「audit2_b.txt に書き込みました」で正直だが、
    ユーザーの指示は実行されていない)。日英どちらの語順でも同じに再現する。

    選び方は 3 段:

    1. 英語の後置マーカー (``to`` / ``into``) の **後ろ** から採る
    2. 日本語の前置マーカー (``に追記`` / ``の末尾に書き`` 等) の **手前** で
       最後に現れるパスを採る
    3. どちらも無ければ従来どおり ``_extract_file_path`` (先頭一致)

    3 を残すのは、マーカーが無い普通の依頼 (「E:\\tmp\\a.txt に書いて」) で
    挙動を変えないため。
    """
    body = text or ""
    if not body:
        return ""
    for marker in _WRITE_DEST_MARKERS_EN:
        idx = body.lower().rfind(marker)
        if idx == -1:
            continue
        found = _extract_file_path(body[idx + len(marker):])
        if found:
            return found
    m = _WRITE_DEST_MARKER_JA_RE.search(body)
    if m:
        found = _extract_last_file_path(body[:m.start()])
        if found:
            return found
    return _extract_file_path(body)


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
# 桁区切り入りの数字 (``1,280`` / ``1,234,567.5``)。
_ARITH_GROUPED_DIGITS_RE = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])")
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
    # 桁区切り (``1,280``) は先に落とす。``_ARITH_RUN_RE`` はカンマを式の文字と
    # 見ないので、残したままだと「1,280 × 37 × 1.08」の run が「280 × 37 × 1.08」
    # から始まり、正しく計算された嘘 (11188.8) が回答の根拠に載る
    # (2026-09-05 ライブ監査 T15)。
    normalized = _ARITH_GROUPED_DIGITS_RE.sub(
        lambda m: m.group(0).replace(",", ""), query.translate(_ARITH_NORMALIZE),
    )
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
