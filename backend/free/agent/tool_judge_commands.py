"""実行可能クエリ → シェルコマンドの合成と readonly 判定

ルール表 (``_EXECUTABLE_QUERY_COMMANDS``) によるコマンド生成と、生成/引き当てた
コマンドを撃ってよいかの判定 (readonly 検証 / リコール適合) をまとめる。
"""

from __future__ import annotations

import datetime
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.free.agent.safety_patterns import reject_readonly_violation
from backend.free.agent.tool_judge_grounding import _numeric_literals
from backend.free.core.intent_vocab import DATETIME_QUERY_RE, is_plain_statement
from backend.log_config import get_logger

logger = get_logger("agent.tool_call_judge")

#: readonly の allow-list (python のみ) から漏れるが、**状態を変えないことが
#: 明らかな**検査コマンドの実行ファイル名。
#:
#: 用途は「拒否されたコマンドが *変更の試み* だったのか *測定の試み* だったのか」
#: の振り分けのみで、**実行可否は一切変わらない** (どちらも allow-list 違反として
#: 拒否される)。変わるのは base へ足す注記が ``_UNPERFORMED_ACTION_GUIDANCE``
#: (何も実行していない) か ``_UNMEASURED_FACT_GUIDANCE`` (測っていない) かだけ。
#:
#: 実インシデント (2026-08-15 ライブ監査 ターン12): 「本当に削除されましたか？
#: 確認して。」にネイティブ層が ``test -f <path>`` を選び、allow-list 違反で
#: 拒否 → 一律 ``_action_blocked`` が立ち「状態を変える操作を実行していない」の
#: 注記が入った結果、base が「ファイルの存在確認を行うツールが利用できない」と
#: 誤った説明で締めた (実際は read_file / list_directory が使える)。
#:
#: mutation を read と誤分類すると完了の捏造 (2026-08-08 の ``echo >> file``)
#: に戻るため、**曖昧なものは載せない**。判定不能なら従来どおり action 扱い。
_READONLY_INSPECT_COMMANDS: frozenset[str] = frozenset({
    "test", "ls", "dir", "cat", "type", "stat",
    "head", "tail", "wc", "grep", "findstr", "where", "which",
})


def _command_is_readonly_inspection(command: str) -> bool:
    """``command`` が「状態を変えない検査」と確実に言えるか。

    リダイレクト (``>`` / ``>>``) や連鎖 (``&&`` / ``;`` / ``|``) を含む場合は、
    先頭が検査コマンドでも後続で状態を変えうるので False を返す
    (``test -f x && rm x`` のような形を read と誤分類しない)。
    """
    if not command or any(t in command for t in (">", ">>", "&&", "||", ";", "|")):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    return Path(tokens[0]).name.lower().removesuffix(".exe") in (
        _READONLY_INSPECT_COMMANDS
    )


def _readonly_command_rejected(exec_tool: str, command: str) -> bool:
    """readonly ツールに載せる ``command`` が readonly 検証に違反するか。

    ``exec_tool`` が ``run_command_readonly`` のときだけ
    ``reject_readonly_violation`` を適用する (create の run_command は対象外)。
    judge 段でこれを弾くと、synth が返した非 readonly コマンド (PowerShell
    スニペット等) が実行段の "Error: readonly violation" ではなく no_tool に
    倒れ、LLM 知識回答へクリーンに落ちる。実行段のラッパ検証は最終防衛として
    別途残る (二重ガード)。
    """
    if exec_tool != "run_command_readonly":
        return False
    reject = reject_readonly_violation(command)
    if reject is not None:
        logger.info(
            "Readonly executable command rejected at judge stage (%s): %s",
            reject, command[:80],
        )
        return True
    return False
# ユーザークエリからドライブレターを抽出するパターン
# 「Eドライブ」「C:」「D drive」等のパターンにマッチし、
# 単一の英字（ドライブレター）をキャプチャする。
# ASCII 境界を使用して "PCの" 等の複数文字並びに誤マッチしないよう、
# 直前が英字でないことを保証する。
_DRIVE_LETTER_RE = re.compile(
    r"(?:^|[^A-Za-z])([A-Za-z])(?::|\s*ドライブ|\s*drive(?![A-Za-z]))",
    re.IGNORECASE,
)
def _build_spec_command(query: str) -> str:
    """システムスペックコマンドを生成する

    クエリにドライブレター指定（「Eドライブ」「C:」等）が含まれる場合は、
    そのドライブの容量を取得する。指定がなければシステムドライブ
    (Windows は %SystemDrive%、Unix は '/')。
    Windows / Unix の両方で動作するよう、パスはフォワードスラッシュで構築する
    （shutil.disk_usage は Windows でも 'E:/' を受理する）。

    フォールバックはかつてカレントディレクトリ ('.') だったが、これは backend
    プロセスの起動位置という**ユーザーから見えない値**に測定対象が依存する。
    実測 (2026-07-27 ライブ監査): 「C ドライブの空き容量は?」→ C: の 138 GB を
    回答した直後、「さっき調べた空き容量はディスク全体の何%?」でドライブ名が
    落ちて '.' にフォールバックし、cwd のある E: (553 GB free) を測って
    「さっき調べた空き容量 553 GB」と自己矛盾した回答を返した。
    「この PC の空き容量」はシステムドライブを指すのが自然で、かつ起動位置に
    依存せず決定論的になる。
    """
    m = _DRIVE_LETTER_RE.search(query)
    if m:
        letter = m.group(1).upper()
        py_path = f"'{letter}:/'"
    else:
        # サブプロセス側で評価する (実行ホストのシステムドライブを見る)。
        # os は既に import 済みで、os.environ / .get とも readonly guard の
        # 禁止属性ではない。
        py_path = "(os.environ.get('SystemDrive','C:')+'/' if os.name=='nt' else '/')"
    return (
        "python -c \""
        "import platform,os,shutil;"
        " print('OS:',platform.platform());"
        " print('CPU:',platform.processor() or platform.machine());"
        " print('Cores:',os.cpu_count());"
        f" t,u,f=shutil.disk_usage({py_path});"
        " print('Disk:',t//(1024**3),'GB total,',f//(1024**3),'GB free')"
        "\""
    )


# 現在時刻 / 日付クエリ。executable 判定の中で最も曖昧さが小さく、aux が
# 否定票を返しても regex 結果を維持してよい唯一の高特異度パターン
# (``_upgrade_command_via_aux`` の降格例外)。
# 定義は core.intent_vocab が SSOT (agent.router が同一定義を持っていたが、
# ``(?!間)`` ガードの有無など細部が食い違っていた)。
_DATETIME_QUERY_RE = DATETIME_QUERY_RE

#: 「N 日後 / N 年前」等の相対日付。単位の直後に 前/後 を要求するので
#: 「1 月 3 日」のような絶対日付には掛からない。
_RELATIVE_OFFSET_RE = re.compile(
    r"(\d{1,4})\s*(週間|週|[かヶケヵ箇]月|月|日|年)\s*(前|後|先)",
)

#: コマンドが日付演算をしている印。``_build_datetime_command`` が相対日付用に
#: 生成するコマンドは必ずどちらかを含む。リコールで引き当てた過去のコマンドが
#: 相対日付クエリに答えられるかの判定に使う (``recalled_command_fits_query``)。
_DATE_ARITHMETIC_RE = re.compile(r"timedelta|datetime\.datetime\(|datetime\.date\(")

#: 完全に特定された絶対日付 (年・月・日がすべて書かれている)。年を必須にする
#: ことで「1 月 3 日は何曜日ですか」のような年抜き表現は従来どおり除外する。
_ABSOLUTE_DATE_RE = re.compile(
    r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
    r"|(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
)

#: 年が書かれていない日付 (「9 月 14 日」「9/14」)。会話で日付を口にするときの
#: **もっとも自然な形**だが、``_ABSOLUTE_DATE_RE`` が 4 桁年を必須にするため
#: 日付として一切認識されていなかった。年は実行時に ``n.year`` で埋める
#: (ビルド時に埋めると ``mem.world.executable_command`` として学習された
#: コマンドが翌年に誤答する)。
#:
#: 実インシデント (2026-08-25 ライブ監査 T2-3 / T2-9):
#:
#: - 「9 月 14 日まであと何日ですか？」→ ``_day_count_command`` が日付を 1 つも
#:   拾えず now-only コマンドへ落ち、「20 日」はモデルの暗算だった。
#: - 「9 月 14 日の 3 週間前は何月何日ですか？」→ 相対オフセットが常に *今日* を
#:   基点にするため ``target: 2026-08-04`` を出力 (正しくは 8/24)。回答本文は
#:   8 月 24 日で、**自分のツール出力と食い違う**表示になった。
#:
#: ``\d{1,2}/\d{1,2}`` は日付以外 (分数・比率) とも衝突するため、月日いずれも
#: 実在する範囲のときだけ採用する (``_iter_query_dates`` で検証)。
_PARTIAL_DATE_RE = re.compile(
    r"(\d{1,2})\s*月\s*(\d{1,2})\s*日"
    r"|(?<![\d/-])(\d{1,2})/(\d{1,2})(?![\d/-])",
)

#: 曜日を尋ねていることを示す語。
_WEEKDAY_ASK_RE = re.compile(
    r"曜日|(?<![A-Za-z])day\s+of\s+the\s+week(?![A-Za-z])"
    r"|(?<![A-Za-z])weekday(?![A-Za-z])",
    re.IGNORECASE,
)

#: 相対日付の単位 → コマンド生成の種別。
_OFFSET_UNITS = {
    "日": "days", "週": "weeks", "週間": "weeks",
    "月": "months", "か月": "months", "ヶ月": "months",
    "ケ月": "months", "ヵ月": "months", "箇月": "months",
    "年": "years",
}

#: 現在日時のみを返す既定コマンド。
#:
#: ``astimezone()`` を付けて **UTC オフセット付き**で出力する。プロンプトには
#: 別途 ``[現在日時 (UTC基準)]`` が注入されており、コマンド出力が naive
#: ローカル時刻だと 2 つの時計が無印で並ぶ。JST では 00:00-09:00 の間、
#: ローカル日付と UTC 日付が 1 日ずれるため、モデルはどちらを「今日」と
#: 呼ぶべきか判断できない (2026-08-05 ライブ監査で構造として確認)。
#:
#: 曜日 (``%A``) も出す。出さないと「今日は何曜日」以外のターンで曜日に触れた
#: とき、モデルが日付から曜日を暗算して外す (実測 2026-08-22 ライブ監査の
#: 修正検証: 「今の時刻を教えてください。」→「2026年8月22日（金）の午前1時37分」。
#: 8/22 は土曜)。``datetime.datetime.now()`` は ``_DATE_ARITHMETIC_RE``
#: (``datetime.datetime(``) に一致しないので、リコール時の日付演算ガードは
#: この追加後も従来どおり now-only コマンドを弾く。
_DATETIME_NOW_COMMAND = (
    'python -c "import datetime; n=datetime.datetime.now().astimezone();'
    " print(n); print('weekday:', n.strftime('%A'))\""
)

#: 相対日付コマンドの共通前置き (now と目標日を両方出す)。
_REL_PREFIX = 'python -c "import datetime; n=datetime.datetime.now().astimezone();'
_REL_SUFFIX = (
    " print('now:',n);"
    " print('target:',t.strftime('%Y-%m-%d (%A)'))\""
)


def _absolute_weekday_command(query: str) -> str:
    """「YYYY年M月D日は何曜日？」用に、その日の曜日を計算するコマンドを返す。

    該当しない (曜日を尋ねていない / 年月日が揃っていない / 実在しない日付) 場合は
    空文字列を返し、呼び出し側が従来のコマンドへ倒す。

    相対日付は既に Python で計算させているのに、**絶対日付だけモデルの暗算に
    委ねられていた**。ツールは現在時刻しか返さないため、曜日は完全に base の
    記憶頼みになる (実インシデント 2026-08-19 ライブ監査 ターン19:
    「西暦2000年1月1日は何曜日でしたか？」で ``datetime.now()`` だけが実行され、
    回答の「土曜日」はツールで裏取りされていなかった)。

    年なしの「9 月 14 日は何曜日？」も対象にする。年は実行時の ``n.year``
    (:data:`_PARTIAL_DATE_RE`)。
    """
    if not _WEEKDAY_ASK_RE.search(query or ""):
        return ""
    dates = _iter_query_dates(query)
    if not dates:
        return ""
    return (
        'python -c "import datetime;'
        " n=datetime.datetime.now().astimezone();"
        f" t={dates[0].datetime_expr()};"
        " print('target:',t.strftime('%Y-%m-%d (%A)'))\""
    )


#: 「あと何日」「何日間」「残り日数」等、**2 点間の日数** を尋ねる語。
_DAY_COUNT_ASK_RE = re.compile(
    # ``何日です`` は「何月何日ですか」(= 日付を訊く問い) も飲み込む。年なし日付を
    # 読むようになって初めて実害が出た: 「9 月 14 日の 3 週間前は何月何日ですか？」
    # が日数カウント側に取られ、``days: 20`` (今日から 9/14 までの日数) を返した。
    # 直前が ``何月`` のときだけ除外し、「あと何日ですか」は従来どおり拾う。
    r"何日間|あと何日|残り\s*(?:の)?\s*日数|日数は|何日ある|(?<!何月)何日です"
    r"|まで(?:は)?\s*何日"
    r"|(?<![A-Za-z])how\s+many\s+days(?![A-Za-z])"
    r"|(?<![A-Za-z])days\s+(?:left|remaining|until)(?![A-Za-z])",
    re.IGNORECASE,
)

#: 「今週 / 来週 / 再来週 / 先週 / 先々週」の何曜日、という指定。
#:
#: 相対オフセット (``_RELATIVE_OFFSET_RE``) は数字を要求するので掛からず、
#: 絶対日付でもないため **now-only コマンドへ落ちて曜日→日付の変換が丸ごと
#: モデルの暗算に残っていた**。当たるかどうかは運になる。
#:
#: 実インシデント (2026-08-26 ライブ監査 T9-3): 当日 8/26 (水) に
#: 「来週の金曜日に歯科の予約があります。」と伝えたところ「来週の金曜日は
#: 2026年8月28日」と応答した。8/28 は **今週の**金曜で、来週の金曜は 9/4。
#: 前日の監査では同型の「来週の月曜日」に正答しており、暗算の当否が
#: 揺れていることが確認できる。
#:
#: 週の起点は月曜 (ISO / 日本の慣行)。``今週`` はその週、``来週`` は +7 日、
#: ``再来週`` は +14 日、``先週`` は -7 日、``先々週`` は -14 日。
_WEEK_OFFSETS: dict[str, int] = {
    "今週": 0, "こんしゅう": 0,
    "来週": 7, "らいしゅう": 7,
    "再来週": 14, "さらいしゅう": 14,
    "先週": -7, "せんしゅう": -7,
    "先々週": -14, "せんせんしゅう": -14,
}

#: 曜日名 → ``datetime.weekday()`` の値 (月曜 = 0)。
_WEEKDAY_INDEX: dict[str, int] = {
    "月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6,
}

_WEEK_OF_WEEKDAY_RE = re.compile(
    r"(先々週|再来週|今週|来週|先週|こんしゅう|らいしゅう|さらいしゅう"
    r"|せんせんしゅう|せんしゅう)"
    r"\s*の?\s*([月火水木金土日])曜",
)


def _week_of_weekday_command(query: str) -> str:
    """「来週の金曜日」型の指定を Python に解かせるコマンドを返す。

    該当しなければ空文字列 (呼び出し側が従来のコマンドへ倒す)。
    週の起点は月曜で計算する (:data:`_WEEK_OFFSETS`)。
    """
    m = _WEEK_OF_WEEKDAY_RE.search(query or "")
    if m is None:
        return ""
    offset = _WEEK_OFFSETS.get(m.group(1))
    weekday = _WEEKDAY_INDEX.get(m.group(2))
    if offset is None or weekday is None:
        return ""
    return (
        'python -c "import datetime;'
        " n=datetime.datetime.now().astimezone(); t=n.date();"
        # その週の月曜へ寄せてから、週オフセットと曜日を足す。
        f" w=t-datetime.timedelta(days=t.weekday())+datetime.timedelta(days={offset}+{weekday});"
        " print('now:',n);"
        " print('target:',w.strftime('%Y-%m-%d (%A)'))\""
    )


#: 「今年の残り」「年末まで」— 年末までの日数。
_YEAR_REMAINDER_RE = re.compile(
    r"今年.{0,6}(?:残り|あと)|年末まで|(?:残り|あと).{0,4}今年"
    r"|(?<![A-Za-z])rest\s+of\s+(?:the\s+)?year(?![A-Za-z])",
    re.IGNORECASE,
)


def _iter_absolute_dates(query: str) -> list[tuple[int, int, int]]:
    """クエリ中の実在する絶対日付 (年月日が揃ったもの) を出現順に返す。"""
    found: list[tuple[int, int, int]] = []
    for m in _ABSOLUTE_DATE_RE.finditer(query or ""):
        parts = m.groups()
        triple = parts[:3] if parts[0] is not None else parts[3:]
        try:
            year, month, day = (int(v) for v in triple)
            datetime.date(year, month, day)
        except (TypeError, ValueError):
            continue
        found.append((year, month, day))
    return found


#: ``_iter_query_dates`` が返す 1 件。``year`` が ``None`` なら年が書かれて
#: いない日付で、コマンド側では ``n.year`` (実行時の年) で埋める。
#: ``start`` はクエリ内の出現位置 (相対オフセットの基点判定に使う)。
@dataclass(frozen=True, slots=True)
class _QueryDate:
    year: int | None
    month: int
    day: int
    start: int
    end: int

    def date_expr(self) -> str:
        """``datetime.date(...)`` 式 (年なしは実行時の ``n.year`` で埋める)。"""
        year = str(self.year) if self.year is not None else "n.year"
        return f"datetime.date({year},{self.month},{self.day})"

    def datetime_expr(self) -> str:
        """``datetime.datetime(...)`` 式 (年なしは実行時の ``n.year``)。"""
        year = str(self.year) if self.year is not None else "n.year"
        return f"datetime.datetime({year},{self.month},{self.day})"


def _iter_query_dates(query: str) -> list[_QueryDate]:
    """クエリ中の日付を出現順に返す (年なしの「9 月 14 日」も含む)。

    年が揃った日付を先に採り、その範囲に重なる年なしマッチは捨てる
    (「2026年9月14日」の中の「9月14日」を二重に数えないため)。年なしの月日は
    その年の実在日として妥当なもの (うるう年を跨ぐ 2/29 も含めて 1〜12 月 /
    1〜31 日) だけ採用する。
    """
    q = query or ""
    found: list[_QueryDate] = []
    spans: list[tuple[int, int]] = []
    for m in _ABSOLUTE_DATE_RE.finditer(q):
        parts = m.groups()
        triple = parts[:3] if parts[0] is not None else parts[3:]
        try:
            year, month, day = (int(v) for v in triple)
            datetime.date(year, month, day)
        except (TypeError, ValueError):
            continue
        found.append(_QueryDate(year, month, day, m.start(), m.end()))
        spans.append((m.start(), m.end()))
    for m in _PARTIAL_DATE_RE.finditer(q):
        if any(s <= m.start() < e for s, e in spans):
            continue
        parts = m.groups()
        pair = parts[:2] if parts[0] is not None else parts[2:]
        try:
            month, day = (int(v) for v in pair)
        except (TypeError, ValueError):
            continue
        # 年が分からないのでうるう年を仮定して実在判定する (2/29 を落とさない)。
        try:
            datetime.date(2024, month, day)
        except ValueError:
            continue
        found.append(_QueryDate(None, month, day, m.start(), m.end()))
    found.sort(key=lambda d: d.start)
    return found


def _day_count_command(query: str) -> str:
    """日数を数えるクエリ用に、差分まで Python に計算させるコマンドを返す。

    該当しなければ空文字列 (呼び出し側が従来のコマンドへ倒す)。

    相対日付 (「100 日後」) と絶対日付の曜日は既に Python 側で計算させて
    いるのに、**2 点間の日数だけモデルの暗算に残っていた**。ツールは現在時刻
    しか返さないので、月ごとの日数の足し上げと引き算がそのまま出力に乗る。
    実インシデント 2026-08-22 ライブ監査:

    - 「締め切りまであと何日ありますか？」(締切 2026-10-15 / 当日 2026-08-21)
      → ``run_command_readonly`` は現在日時だけを返し、回答は「25 日」。正解 55。
    - 「今年の残り日数は何日ですか？」→ 1〜7 月の合計を 182 日 (正 212)、
      さらに ``365 - 203`` を 62 と誤り、回答は「62 日」。正解 132。

    同じ日に「2026年8月21日から2026年10月15日までは何日間ありますか？」
    (両端が本文にある) は 55 日と正答している。数えられないのではなく、
    **一方の端がツール出力や記憶から来ると崩れる**。

    対応するのはクエリだけで両端が決まる 3 形:

    1. 日付が 2 つ → その差
    2. 日付が 1 つ → 今日との差
    3. 「今年の残り / 年末まで」 → 今日から 12/31 までの差

    日付は年なし (「9 月 14 日」) も対象で、年は実行時に ``n.year`` で埋める
    (:data:`_PARTIAL_DATE_RE`)。年を要求していた頃は「9 月 14 日まであと何日
    ですか？」が now-only コマンドへ落ち、日数はモデルの暗算に残っていた。
    """
    if not _DAY_COUNT_ASK_RE.search(query or ""):
        return ""
    dates = _iter_query_dates(query)
    if len(dates) >= 2:
        a, b = dates[0], dates[1]
        return (
            'python -c "import datetime;'
            " n=datetime.datetime.now().astimezone();"
            f" a={a.date_expr()}; b={b.date_expr()};"
            " print('from:',a); print('to:',b);"
            " print('days:',abs((b-a).days))\""
        )
    if len(dates) == 1:
        return (
            'python -c "import datetime;'
            " n=datetime.datetime.now().astimezone(); t=n.date();"
            f" a={dates[0].date_expr()};"
            " print('now:',n); print('target:',a);"
            " print('days:',(a-t).days)\""
        )
    if _YEAR_REMAINDER_RE.search(query or ""):
        return (
            'python -c "import datetime;'
            " n=datetime.datetime.now().astimezone(); t=n.date();"
            " e=datetime.date(t.year,12,31);"
            " print('now:',n); print('year_end:',e);"
            " print('days:',(e-t).days)\""
        )
    return ""


#: 「(過去に述べられた事実) は何日でしたか？」型の想起。``何月`` / ``何日`` /
#: ``何時`` は ``DATETIME_QUERY_RE`` に載っているため、**ユーザーが以前伝えた
#: 日付を訊き直しただけ**のターンでも現在日時コマンドが撃たれていた。
#: 実インシデント (2026-08-22 ライブ監査 2 回目 ターン 18/26):
#: 「私の誕生日は何年何月何日でしたか？」「誕生日は変わっていませんよね？
#: 何日でしたか？」の 2 ターンで ``datetime.now()`` が実行された。求められて
#: いるのは記憶の想起であって現在時刻ではなく、注入された「今日の日付」は
#: 誤答の材料にしかならない。
#:
#: 抑止は **now-only コマンドに落ちる分岐だけ** に掛ける。絶対日付
#: (「1987年3月14日は何曜日でしたか？」) や相対日付 (「3年前の今日は何曜日
#: でしたか？」) は過去形でも計算が要るため、従来どおり撃つ。
_PAST_RECALL_TAIL_RE = re.compile(
    r"でした(?:か|っけ)|だった(?:か|っけ)|だっけ"
    r"|(?:と)?(?:言|いい|伝え|教え)(?:い?ました|った)"
    r"|覚えて(?:い|ま|る)",
)

#: 「現在」を指す語。1 つでもあれば now-only コマンドを抑止しない。
#: ``いま`` / ``きょう`` は 2 文字の部分文字列で、無関係な語に埋もれる
#: (変わって**いま**せん / **興味** → きょうみ)。実際に「誕生日は変わって
#: いませんよね？何日でしたか？」の「て**いま**せん」が現在アンカーとして
#: 誤ヒットし、抑止が効かなかった。後続文字で除外する。
_PRESENT_ANCHOR_RE = re.compile(
    r"今日|本日|現在|ただいま|只今|今[のはがもへ、。 ]|今$"
    r"|いま(?![すせしそまん])|きょう(?![みりょ])"
    r"|(?<![A-Za-z])(?:now|today|current|currently)(?![A-Za-z])",
    re.IGNORECASE,
)


def _relative_anchor(query: str, offset_start: int) -> "_QueryDate | None":
    """相対オフセットの基点になる日付を返す (無ければ ``None`` = 今日基点)。

    採るのは **オフセット表現より前に完全に現れている** 日付だけ。重なりを
    許すと「9 月 14 日前」のような表現で ``14日前`` を相対オフセット、
    ``9月14日`` を基点として二重に読んでしまう。
    """
    candidates = [d for d in _iter_query_dates(query) if d.end <= offset_start]
    return candidates[-1] if candidates else None


def _is_past_fact_recall(query: str) -> bool:
    """現在日時ではなく「以前述べられた日付」を訊いているか (純粋関数)。"""
    q = query or ""
    if _PRESENT_ANCHOR_RE.search(q):
        return False
    return bool(_PAST_RECALL_TAIL_RE.search(q))


def _build_datetime_command(query: str) -> str:
    """日付 / 時刻クエリ用のコマンドを組み立てる。

    相対表現 (「3 年前の今日」「今日から 100 日後」) が含まれる場合は **目標日と
    その曜日まで Python に計算させる**。現在時刻だけを渡してモデルに暗算させると
    外す (実インシデント 2026-08-07 ライブ監査: 「3 年前の今日は何曜日でしたか？」
    に「火曜日」と回答。2023-08-07 は月曜日)。同じ日に「今日から 100 日後」は
    正答しており、暗算が当たるかどうかは運になっていた。

    年月日が揃った絶対日付の曜日を尋ねる場合も同じ理由で Python に計算させる
    (``_absolute_weekday_command``)。2 点間の日数も同様 (``_day_count_command``)。

    相対表現の **基点** は既定では今日だが、クエリ内でオフセット表現より前に
    日付が書かれていればその日付を基点にする。基点を常に今日にしていたため
    「9 月 14 日の 3 週間前は何月何日ですか？」が *今日から* 3 週間前を計算し、
    ``target: 2026-08-04`` (正しくは 8/24) を出力していた
    (2026-08-25 ライブ監査 T2-9)。

    どれでもなければ従来どおり現在日時のみを返す。
    """
    # 日数カウントは **両端が決まるときだけ** コマンドを返す。返せたならそれが
    # 最も具体的なので優先する。
    day_count = _day_count_command(query)
    if day_count:
        return day_count
    # 「来週の金曜日」型は数字を伴わないので相対オフセットに掛からず、絶対日付
    # でもないため now-only へ落ちて曜日→日付の変換が暗算に残っていた。
    # 日数カウントが空を返した後に見る — 「今週の金曜日は何日ですか？」は
    # ``何日です`` で日数側の語彙に掛かるが両端が決まらず空になるので、ここで
    # 拾える (now-only より常に情報が多い)。
    week_of = _week_of_weekday_command(query)
    if week_of:
        return week_of
    m = _RELATIVE_OFFSET_RE.search(query or "")
    if m is None:
        absolute = _absolute_weekday_command(query)
        if absolute:
            return absolute
        # now-only へ落ちる分岐だけ、過去事実の想起を抑止する
        # (_is_past_fact_recall 参照)。
        if _is_past_fact_recall(query):
            return ""
        return _DATETIME_NOW_COMMAND
    kind = _OFFSET_UNITS.get(m.group(2))
    if kind is None:
        return _DATETIME_NOW_COMMAND
    n = int(m.group(1))
    signed = -n if m.group(3) == "前" else n

    anchor = _relative_anchor(query, m.start())
    if anchor is None:
        base = " b=n;"
        base_echo = ""
    else:
        base = f" b={anchor.datetime_expr()};"
        base_echo = " print('base:',b.strftime('%Y-%m-%d'));"

    if kind in ("days", "weeks"):
        body = base + f" t=b+datetime.timedelta({kind}={signed});"
    else:
        # 月/年は timedelta で表せない。月末クランプ (1/31 の 1 か月後 = 2/28 等)
        # を含めて構築する。``calendar`` は readonly guard の許可モジュール外、
        # ``datetime.replace`` は禁止属性なのでコンストラクタで組み立てる。
        total = " tm=(b.year*12+b.month-1)+" + str(
            signed * 12 if kind == "years" else signed,
        ) + ";"
        body = (
            base
            + total
            + " y=tm//12; mo=tm%12+1;"
            " lp=(y%4==0 and (y%100!=0 or y%400==0));"
            " dim=[31,29 if lp else 28,31,30,31,30,31,31,30,31,30,31][mo-1];"
            " t=datetime.datetime(y,mo,min(b.day,dim));"
        )
    return _REL_PREFIX + body + base_echo + _REL_SUFFIX

# Python 実行で正確に答えられるシステム情報クエリのコマンドマッピング
# パターンにマッチしたクエリに対して、具体的な Python コマンドを生成する。
# コマンドは Windows cmd.exe / Unix sh の両方で動作するよう、
# 外側を "..." で囲み内側で '...' を使用する。
# 第二要素が Callable の場合はクエリ文字列を渡して動的に生成する
_EXECUTABLE_QUERY_COMMANDS: list[tuple[re.Pattern, "str | Callable[[str], str]"]] = [
    # 現在時刻 / 日付 (「何月|何日|何曜日」は明確な疑問語のみ追加、
    # 「今日|明日|昨日」単独は誤検出するため見送り)
    # ``astimezone()`` を付けて **UTC オフセット付き**で出力する。プロンプトには
    # 別途 ``[現在日時 (UTC基準)]`` が注入されており、コマンド出力が naive
    # ローカル時刻だと 2 つの時計が無印で並ぶ。JST では 00:00-09:00 の間、
    # ローカル日付と UTC 日付が 1 日ずれるため、モデルはどちらを「今日」と
    # 呼ぶべきか判断できない (2026-08-05 ライブ監査で構造として確認。当日は
    # 22:43 JST = 13:43 UTC で偶然一致しており表面化しなかった)。
    # オフセットを添えれば両者の関係が出力から読み取れる。
    (_DATETIME_QUERY_RE, _build_datetime_command),
    # システムスペック（OS / CPU / コア数 / ディスク）
    # ドライブレター指定があれば指定ドライブの容量を返す
    # CPU 等の英字略語は ASCII 境界必須 ("program" の 'ram' 誤マッチ対策)
    # spec(s)? で複数形 ("PC specs") も許容する。
    # メモリ / memory / RAM は 2026-07-27 に外した。GPU/VRAM (下記) と同じ理由で、
    # コマンドが搭載メモリ量を一切出力しないのにパターンだけ一致して発火し、
    # サブプロセスと 1 ターンを消費した末に「ツール結果にメモリ容量の数値は
    # 記載されていません」としか返せなかった (実測: 「この PC のメモリは何 GB
    # 積んでいますか？」)。Windows で搭載 RAM を取る手段 (ctypes / wmic /
    # Get-CimInstance) は _READONLY_SAFE_MODULES / 危険コマンド判定が全て拒否
    # するため、正しい情報を返すコマンドへ差し替える経路は存在しない。
    # ``capacity`` は 2026-08-09 に外した。他の語と違い **データ項目名として
    # 普通に現れる** ため、機械スペックの要求とは限らない (2 回目のライブ監査:
    # 「同じ表を JSON 配列にしてください。キーは category, fee, capacity で…」
    # という純粋な整形依頼で OS/CPU/コア数の取得コマンドが撃たれた。
    # `capacity` を別名に変えると発火しない = この語が唯一の引き金だった)。
    # ``容量`` 単独も 2026-08-10 に外した (同じ理由。「DBの容量」「総容量」
    # 「容量効率」で spec コマンドが撃たれた)。機器を名指しする質問は
    # ストレージ / ディスク / ドライブ / disk / drive 側で拾えるので
    # 取りこぼしは実質無い。「空き容量」「残り容量」「使用容量」は残す。
    # ``disk`` / ``storage`` は他の ASCII トークンと同じく境界必須へ揃える
    # (このファイルの規約。境界無しだと部分一致で誤爆する)。
    # ``CPU`` は 2026-08-18 に「処理の種別を表す複合語」を除外した。RAM /
    # GPU / capacity / 容量 と違いトークンごと外すことはできない (「CPU の型番を
    # 教えて」「What's my CPU model?」は spec コマンドが正しく答えられる唯一の
    # 引き金) が、``CPU バウンド`` / ``CPU-bound`` / ``CPU 集約`` は **この
    # マシンの部品ではなくワークロードの分類名** で、機械スペックの要求では
    # 決してない。実インシデント (2026-08-18 ライブ監査 ターン4):
    # 「Python の GIL があることで、CPU バウンド処理と I/O バウンド処理で
    # スレッドの効果がどう違うのか、簡潔に説明してください。」という純粋な
    # 知識質問で OS/CPU/コア数/ディスクの取得コマンドが撃たれ、無関係な実測値が
    # 「唯一の事実根拠」枠で base に渡された。層1 の知識質問ゲートは
    # ``_query_has_tool_signal`` が True になるため到達できず、この層より後段に
    # あるため構造的に救済不能。
    (re.compile(
        r"(?:スペック|(?<![A-Za-z])CPU(?![A-Za-z])(?!\s*[-‐-—]?\s*(?:bound|バウンド|集約))"
        r"|ディスク|(?:空き|残り|使用)容量|ストレージ|ドライブ"
        r"|(?<![A-Za-z])disk(?![A-Za-z])|(?<![A-Za-z])storage(?![A-Za-z])"
        r"|(?<![A-Za-z])specs?(?![A-Za-z])"
        r"|(?<![A-Za-z])drive(?![A-Za-z]))",
        re.IGNORECASE,
    ), _build_spec_command),
    # GPU / VRAM のエントリは 2026-07-25 に削除した。
    # コマンドが platform.platform() / platform.machine() しか実行しておらず
    # GPU 型番も VRAM 容量も一切返さないのに、実行が成功扱いになっていた
    # (実測: 「さっき伝えた GPU は？」→ "Platform: Windows-11 / Machine: AMD64" →
    #  「ツール結果に GPU 型番は含まれていません」と誤答)。
    # safety_patterns._READONLY_SAFE_MODULES が wmic / Get-CimInstance /
    # nvidia-smi / 外部ライブラリをすべて拒否するため、正しい情報を返すコマンドへ
    # 差し替える経路は存在しない。エントリを消すと _infer_tool が引数なしを返し
    # _suppress_commandless_run_command が no_tool へ落とすので、GPU/VRAM は
    # 会話履歴と LLM 知識に委ねる (そちらの方が誤答が少ない)。
    # IP アドレス / ホスト名
    (re.compile(
        r"(?:IP\s*アドレス|ホスト名"
        r"|(?<![A-Za-z])hostname(?![A-Za-z])"
        r"|(?<![A-Za-z])ip\s*address)",
        re.IGNORECASE,
    ), "python -c \""
       "import socket;"
       " h=socket.gethostname();"
       " print('Hostname:',h);"
       " print('IP:',socket.gethostbyname(h))"
       "\""),
    # OS
    (re.compile(
        r"(?:(?<![A-Za-z])OS(?![A-Za-z])|オペレーティングシステム"
        r"|(?<![A-Za-z])Windows(?![A-Za-z])"
        r"|(?<![A-Za-z])Linux(?![A-Za-z])"
        r"|(?<![A-Za-z])Mac(?![A-Za-z]))",
        re.IGNORECASE,
    ), "python -c \""
       "import platform,sys;"
       " print(platform.platform());"
       " print(sys.platform,platform.release())"
       "\""),
    # Python バージョン
    (re.compile(
        r"(?:Python|python)\s*(?:バージョン|version)",
        re.IGNORECASE,
    ), "python --version"),
    # 環境変数
    (re.compile(
        r"(?:環境変数|(?<![A-Za-z])env(?![A-Za-z])"
        r"|(?<![A-Za-z])PATH(?![A-Za-z]))",
        re.IGNORECASE,
    ), "python -c \""
       "import os;"
       " [print(k,'=',v[:80]) for k,v in sorted(os.environ.items())[:30]]"
       "\""),
]
def recalled_command_fits_query(
    command: str, origin_query: str, query: str,
) -> bool:
    """引き当てたコマンドを別クエリへ再生してよいかを判定する (純粋関数)。

    executable_command リコールの根拠は embedding 類似度と過去成功率だけで、
    コマンドに焼き込まれた「そのクエリ固有の値」を見ていない。日付や日数の
    ような値が本文へ埋まったコマンドを類似クエリへ再生すると、質問と無関係な
    数字を「ツールで確かめた事実」として提示してしまう
    (実インシデント 2026-07-29 ライブ監査: 「私の誕生日は3月14日です。今日から
    誕生日まであと何日ですか。」から学習した ``datetime.date(y,3,14)`` 入りの
    コマンドが、類似度 0.52 で「2026年3月15日から11月8日までは何日間ですか」へ
    再生され、無関係な ``228`` が返った)。

    コマンドと **合成元クエリ** の両方に現れる数値をクエリ由来のパラメータと
    みなし、それが今回のクエリに無ければ再生を拒否する。合成元クエリに数値が
    無いコマンド (``1024**3`` を含むディスク容量取得等) は構造上の定数しか
    持たないため、そのまま再利用できる。

    Args:
        command: 引き当てたコマンド文字列。
        origin_query: そのコマンドを合成した元のクエリ (fact.object)。
            空なら判定不能として True を返す (従来挙動を維持)。
        query: 今回のクエリ。
    """
    # 相対日付を尋ねているのに、引き当てたコマンドが日付演算を含まない場合は
    # 拒否する。数値パラメータを持たないコマンド (現在時刻の print だけ) は
    # 下の literal 判定を無条件に通ってしまい、「今日から100日後」に対して
    # 現在時刻だけが返る。差分はモデルの暗算に倒れ、当たるかどうかが運になる
    # (実インシデント 2026-08-08 ライブ監査: 修正済みの _build_datetime_command
    # ではなく、修正前に学習した現在時刻コマンドが sim=0.69 で再生された)。
    if _RELATIVE_OFFSET_RE.search(query or "") and not _DATE_ARITHMETIC_RE.search(
        command,
    ):
        return False
    # 日数を数えるクエリも同じ。``_day_count_command`` が答えを出せる形なのに
    # 日付演算を含まないコマンドを引き当てると、ビルダの出力が捨てられて差分が
    # 暗算に戻る。実測 (2026-08-22 ライブ監査、修正の実機検証):
    # 「今年の残り日数は何日ですか？」に対し sim=0.4563 (下限 0.45) で
    # 「今から100日後」由来の **現在時刻 print だけ** のコマンドが再生され、
    # 回答は 134 日 (正 131)。相対日付ガードは ``_RELATIVE_OFFSET_RE`` を
    # 見るので、オフセット表現の無いこの形には掛からなかった。
    if _day_count_command(query) and not _DATE_ARITHMETIC_RE.search(command):
        return False
    # 過去に述べられた日付の想起 (「私の誕生日は何年何月何日でしたか？」) には
    # 現在日時コマンドを撃たない。``_build_datetime_command`` 側は抑止済みだが、
    # **リコール層はビルダを通らない** ため素通りしていた。実測
    # (2026-08-22 ライブ監査 2 回目の修正検証、セット2 ターン42):
    # ビルダ側の抑止を入れた直後の再測定で ``executable_command_recall_matched``
    # として同じ now-only コマンドが再生された。日付演算を含むコマンド
    # (絶対日付・相対日付) は従来どおり通す。
    if _is_past_fact_recall(query) and not _DATE_ARITHMETIC_RE.search(command):
        return False
    # クエリに **絶対日付が書かれている** なら、引き当てたコマンドはその年月日を
    # 含んでいなければならない。「日付演算を含むか」だけでは足りない —
    # 別の日付が焼き込まれたコマンドも ``_DATE_ARITHMETIC_RE`` に当たるため。
    # 実インシデント (2026-08-22 ライブ監査 2 回目 セット2 ターン48):
    # 「今日から2027年1月1日まで何日ありますか？」に対し「今年の残り日数」用の
    # ``datetime.date(t.year,12,31)`` 入りコマンドが再生され、``days: 131``
    # (今年の残り) が返った。origin_query 側に数値が無いため既存の
    # literal 判定も素通りしていた。ビルダ (``_day_count_command``) は
    # クエリの日付でコマンドを組むので、リコールで代用する理由が無い。
    date_m = _ABSOLUTE_DATE_RE.search(query or "")
    if date_m:
        groups = [g for g in date_m.groups() if g]
        if not all(str(int(g)) in command for g in groups):
            return False
    if not origin_query:
        return True
    parameters = _numeric_literals(command) & _numeric_literals(origin_query)
    if not parameters:
        return True
    return parameters <= _numeric_literals(query)
def _infer_executable_command(query: str) -> str:
    """executable query パターンから具体的な Python コマンドを生成する

    _EXECUTABLE_QUERY_COMMANDS の各パターンを順に照合し、
    最初にマッチしたコマンドを返す。
    マッチしない場合（数値処理・データ処理等）は空文字列を返す。

    ルール表は語彙一致なので、ユーザーの自己申告 (「ターミナルは Windows
    Terminal を使っています。」) にも当たる。問い・依頼のマーカーが無い平叙文は
    実行要求ではないため、照合前に落とす (2026-08-19 ライブ監査 ターン3 で
    ``platform.platform()`` が撃たれた)。

    Returns:
        生成されたシェルコマンド。該当なしの場合は空文字列。
    """
    if is_plain_statement(query):
        logger.debug("Plain statement, no executable command: %s", query[:50])
        return ""
    for pattern, command in _EXECUTABLE_QUERY_COMMANDS:
        if pattern.search(query):
            if callable(command):
                built = command(query)
                # ビルダが「このクエリには撃たない」と判断した場合 (空文字) は
                # 後続のパターンへ委ねる。ここで即 return すると、先頭に居る
                # 日時パターンが後続表 (スペック / ディスク等) を飲み込む。
                if not built:
                    continue
                return built
            return command
    return ""
