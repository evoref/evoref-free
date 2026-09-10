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

from backend.free.core.date_math_cue import (
    DATE_MATH_CUE_RE,
    query_has_date_math_cue,
)
from backend.free.agent.safety_patterns import reject_readonly_violation
from backend.free.agent.tool_judge_grounding import _numeric_literals
from backend.free.core.intent_vocab import (
    CPU_HARDWARE_TERM,
    DATETIME_QUERY_RE,
    ENV_VAR_TERMS,
    NETWORK_IDENTITY_TERMS,
    OS_QUERY_TERMS,
    PAST_RECALL_TAIL_RE,
    PYTHON_VERSION_TERMS,
    STORAGE_SPEC_TERMS,
    is_plain_statement,
    is_practice_advice_query,
)
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
#: (ビルド時に埋めると ``idx.command`` として学習された
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


#: 年なしの月日を **過去側** の巡りへ寄せる問い (「8 月 1 日から何日経ちましたか」)。
#: 該当しなければ未来側 (「あと何日」「まで何日」) に寄せる。日数を尋ねる問いの
#: 語彙 (:data:`_DAY_COUNT_ASK_RE`) は圧倒的に未来向きなので、既定を未来にして
#: 過去向きの語が出たときだけ反転させる。
_DAY_COUNT_BACKWARD_RE = re.compile(
    r"経ち|経過|過ぎ|以来|から今日|から現在"
    r"|(?<![A-Za-z])since(?![A-Za-z])"
    r"|(?<![A-Za-z])ago(?![A-Za-z])",
    re.IGNORECASE,
)

#: 「今年の残り」「年末まで」— 年末までの日数。
_YEAR_REMAINDER_RE = re.compile(
    r"今年.{0,6}(?:残り|あと)|年末まで|(?:残り|あと).{0,4}今年"
    r"|(?<![A-Za-z])rest\s+of\s+(?:the\s+)?year(?![A-Za-z])",
    re.IGNORECASE,
)


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

    def occurrence_stmt(self, var: str, *, forward: bool) -> str:
        """年なしの月日を **今日を基準にした最寄りの巡り** へ寄せる文を返す。

        年が書かれている日付はそのまま (寄せない)。年が無い日付を実行時の年で
        固定すると、既に過ぎた月日では日数が負になる。

        実インシデント (2026-08-28 ライブ監査 T03-7):
        「私の誕生日は5月3日です。今日から次の誕生日まで何日ですか。」
        (当日 2026-08-28) で ``a=datetime.date(n.year,5,3)`` が組まれ、
        ツール出力は ``days: -117``。モデルはこれを使わず「260日です」と答えた
        (正解 248 日)。**負の日数はどちらの向きの問いにも答えになっていない**
        ので、そもそも渡さない。

        ``forward`` が真なら今日以降の最初の巡り、偽なら今日以前の最後の巡りを
        採る。2/29 は 4 年に一度しか無いので、実在する年まで進める / 戻す。
        """
        if self.year is not None:
            return f" {var}={self.date_expr()};"
        if (self.month, self.day) == (2, 29):
            years = (
                "range(n.year,n.year+9)" if forward else "range(n.year,n.year-9,-1)"
            )
            cmp_ = ">=" if forward else "<="
            pick = "min" if forward else "max"
            return (
                f" {var}={pick}(d0 for d0 in (datetime.date(y,2,29)"
                f" for y in {years}"
                " if y%4==0 and (y%100!=0 or y%400==0))"
                f" if d0{cmp_}t);"
            )
        shift = "+1" if forward else "-1"
        cmp_ = ">=" if forward else "<="
        return (
            f" {var}={self.date_expr()};"
            f" {var}={var} if {var}{cmp_}t"
            f" else datetime.date(n.year{shift},{self.month},{self.day});"
        )


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
        forward = not _DAY_COUNT_BACKWARD_RE.search(query or "")
        occurrence = dates[0].occurrence_stmt("a", forward=forward)
        # 差は問いの向きで引く。「まであと何日」は a-t、「から何日経ったか」は
        # t-a。年なしの日付は上で最寄りの巡りへ寄せてあるので、どちらも非負になる。
        diff = "(a-t).days" if forward else "(t-a).days"
        return (
            'python -c "import datetime;'
            " n=datetime.datetime.now().astimezone(); t=n.date();"
            f"{occurrence}"
            " print('now:',n); print('target:',a);"
            f" print('days:',{diff})\""
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
#: 語彙は core.intent_vocab が SSOT (``tool_judge_history`` の既出対象の
#: 尋ね直し判定と同じ文末形)。後方互換で旧名を残す。
_PAST_RECALL_TAIL_RE = PAST_RECALL_TAIL_RE

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


# ===========================================================================
# 日付演算の意図 (date_intent) — パラメータだけ取り、コマンドはコードが組む
# ===========================================================================
#
# 上のカスケード (``_day_count`` → ``_week_of_weekday`` → ``_RELATIVE_OFFSET_RE``
# → ``_absolute_weekday``) は語形ごとの正規表現で、**新しい言い回しが来るたびに
# now-only コマンドへ落ちて日付演算がモデルの暗算に戻る**。このファイルのコメント
# 自身が 2026-08-19 / 08-22 / 08-25 / 08-26 の同型インシデントを記録しており、
# 語形を 1 つずつ足す方針では次の語形で必ずまた漏れる。
#
# 実インシデント (2026-09-08 ライブ監査 T19、5/5 誤答):
# 「2026 年 9 月 8 日（火）から数えて、土日を除いた 30 営業日目は何月何日ですか。」
# → 単位 ``営業日`` も接尾 ``目`` もカスケードに無く now-only コマンドだけが
#    実行され、回答「10月14日」は暗算 (正しくは 10/19)。
#
# 対処は語形の追加ではなく **層を 1 つ足すこと**: 日付演算の手掛かりがあるのに
# カスケードが now-only へ落ちたときだけ、文法制約 JSON (``date_intent``、
# CLAUDE.md §6 #1 の許容範囲) で **パラメータだけ** を取り、コマンド文字列は
# 本モジュールが決定論的に組む。**LLM にコードは書かせない** (シェルへ渡るのは
# 常にここで組んだテンプレート)。

#: 日付演算をしている手掛かり。SSOT は :mod:`backend.free.core.date_math_cue`
#: (few-shot の採用拒否とも共有する)。**カスケードが now-only へ落ちたときだけ**
#: 見るので、「今日は何日ですか」のような現在日時の問いは巻き込まない。
_DATE_MATH_CUE_RE = DATE_MATH_CUE_RE

#: 起点より **過去側** を尋ねている印。``DateIntent.direction`` を LLM が
#: 返すようになった後も、値域を守らない場面 (forward を誤って返す) の
#: **コード側の安全弁**として残す (``_effective_direction`` 参照、
#: 2026-09-09 監査 G-06: 「逆算して」に forward のコマンドが組まれ、
#: モデルが暗算で答えを捨てた)。
_DATE_BACKWARD_RE = re.compile(
    r"日前|週間前|[かヶケヵ箇]月前|年前|逆算|遡"
    r"|(?<![A-Za-z])(?:before|earlier|ago)(?![A-Za-z])",
    re.IGNORECASE,
)

#: ``date_intent`` が受け付ける演算の種別 (``none`` は日付演算でない)。
_DATE_INTENT_KINDS = frozenset({
    "business_days_from", "days_from", "days_between", "weekday_of",
})

#: ``n`` の上限。スキーマ側は 100000 まで許すが、コマンドは候補日を列挙する
#: 内包表記なので、実行時間が読める範囲へコード側で絞る。
_MAX_DATE_INTENT_N = 2000

#: 祝日リストの上限 (スキーマと同じ)。
_MAX_DATE_INTENT_HOLIDAYS = 32

#: ``date_intent`` のシステムプロンプト。**判断ではなくパラメータの抽出だけ**を
#: 命じる (層 5.95 の式合成と同じ立て付け)。コードを書かせないことが要点。
DATE_INTENT_SYSTEM = (
    "あなたは日付計算のパラメータ抽出器です。ユーザーの最後の質問に答えるために"
    "必要な日付演算のパラメータだけを JSON で返してください。"
    "回答本文・説明・プログラムは書かないこと。\n"
    "kind の選び方:\n"
    "- business_days_from: 起点から N 営業日 (土日・祝日を除く) 後の日付\n"
    "- days_from: 起点から N 日後 (暦日)\n"
    "- days_between: 2 つの日付の間の日数\n"
    "- weekday_of: ある日付の曜日\n"
    "- none: 日付の計算が不要\n"
    "規則:\n"
    "- start / end は YYYY-MM-DD 形式、または今日なら today と書くこと。\n"
    "- 質問と直前の会話に書かれていない日付・日数を発明しないこと。追い質問"
    "(「その日から」「〜だとすると」) の起点日や日数は直前の会話から取ること。\n"
    "- skip_weekends は土日を数えないときだけ true。\n"
    "- holidays には質問文に明示された休日だけを YYYY-MM-DD で入れること。休日の"
    "日付を自分で発明しないこと。\n"
    "- count_start_day は起点の日を 1 日目と数えるときだけ true。\n"
    "- direction は起点より後 (通常) なら forward、起点より前・逆算・遡るなら"
    " backward。\n"
    "- excluded_weekdays には、質問が特定の曜日を作業日・営業日から除外すると"
    "言っているとき (例: 「毎週水曜日は定例会議で作業できない」) だけ、その曜日を"
    "0=月曜〜6=日曜の数字で入れること。言及が無ければ空配列。\n"
    "- 休日や除外条件だけを付け足す追い質問 (「その週の火曜日が休みだとしたら」"
    "「その日が祝日なら」) は none ではなく、直前の演算と同じ kind・起点・日数・"
    "向き・起点の数え方を返し、追加の休日を holidays に (直前の回答の週から日付を"
    "求めて YYYY-MM-DD で) 入れること。\n"
    "- 使わない項目は start/end に today、n に 0、holidays / excluded_weekdays に"
    "空配列、direction に forward を入れること。"
)
DATE_INTENT_SYSTEM_EN = (
    "You are a date-arithmetic parameter extractor. Return only the parameters "
    "needed to answer the user's last question as JSON. Do not write the reply, "
    "an explanation, or any code.\n"
    "Choosing kind:\n"
    "- business_days_from: the date N business days (weekends/holidays skipped) "
    "from the start date\n"
    "- days_from: the date N calendar days from the start date\n"
    "- days_between: the number of days between two dates\n"
    "- weekday_of: the day of the week of one date\n"
    "- none: no date arithmetic is needed\n"
    "Rules:\n"
    "- start / end must be YYYY-MM-DD, or the literal today.\n"
    "- Never invent a date or count that is neither in the question nor in the "
    "preceding conversation; a follow-up question takes its start date and "
    "count from the previous turns.\n"
    "- Set skip_weekends only when Saturdays and Sundays must not be counted.\n"
    "- Put in holidays only the dates the question states, as YYYY-MM-DD. Never "
    "invent a holiday date.\n"
    "- Set count_start_day only when the start date itself counts as day 1.\n"
    "- direction is forward when counting after the start date (the usual case), "
    "backward when the question counts back from it (e.g. \"working backward\", "
    "\"before\", \"ago\").\n"
    "- excluded_weekdays: only when the question says a specific weekday is not a "
    "working/business day (e.g. \"Wednesdays are a standing meeting, no work "
    "then\"), list it as 0=Monday .. 6=Sunday. Empty if not mentioned.\n"
    "- A follow-up that only adds a holiday or an exclusion (\"what if that "
    "Tuesday is a holiday\") is not none: return the same kind, start, n, "
    "direction and counting as the previous computation, and add the extra "
    "holiday to holidays as YYYY-MM-DD (derive it from the week of the previous "
    "answer).\n"
    "- For unused fields use today for start/end, 0 for n, an empty list for "
    "holidays and excluded_weekdays, and forward for direction."
)


@dataclass(frozen=True, slots=True)
class DateIntentParams:
    """検証済みの日付演算パラメータ。

    ``start`` / ``end`` が ``None`` は「今日」(コマンド実行時に解決する)。
    ビルド時に今日の日付を焼き込むと、``idx.command`` として学習された
    コマンドが翌日以降に誤答する (:class:`_QueryDate` と同じ理由)。
    """

    kind: str
    start: datetime.date | None
    end: datetime.date | None
    n: int
    skip_weekends: bool
    holidays: tuple[datetime.date, ...]
    count_start_day: bool
    #: ``"forward"`` (起点より後) か ``"backward"`` (起点より前 / 逆算)。
    #: ``_effective_direction`` が ``_DATE_BACKWARD_RE`` で上書きしうる
    #: (2026-09-09 監査 G-06)。
    direction: str = "forward"
    #: 作業日・営業日から除外する曜日 (0=月曜〜6=日曜)。``skip_weekends`` とは
    #: 独立 — 両者は :func:`_business_day_candidates` で合算する。
    excluded_weekdays: tuple[int, ...] = ()


def command_lacks_date_arithmetic(command: str) -> bool:
    """コマンドが日付演算を **していない** か (純粋関数)。

    ``_DATE_ARITHMETIC_RE`` に当たらないコマンド (現在日時の print だけ等) は、
    日付演算クエリに対して「答えを含まない出力」しか返さない。
    """
    return not _DATE_ARITHMETIC_RE.search(command or "")


def _parse_intent_date(value: object) -> "datetime.date | None | str":
    """``start`` / ``end`` の 1 項目を解釈する。

    Returns:
        ``datetime.date`` (確定日) / ``None`` (今日) / ``"invalid"`` (棄却)。
    """
    if not isinstance(value, str):
        return "invalid"
    text = value.strip().lower()
    if text in ("", "today", "now", "現在", "今日"):
        return None
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return "invalid"


def parse_date_intent(payload: object) -> DateIntentParams | None:
    """``date_intent`` の応答を検証して :class:`DateIntentParams` にする。

    **コード側の検証がこの層の安全弁**。文法制約 JSON は形は守らせるが値域は
    守らないため (``json_schema`` は enum と型までしか強制しない)、日付が ISO
    として読めること・``n`` が現実的な範囲にあること・祝日がすべて読めること・
    ``excluded_weekdays`` が 0-6 の整数であることをここで確かめる。1 つでも
    欠ければ ``None`` を返し、呼出側は now-only のままにして「ツールで検証
    していない」印 (``unexplained_date_math``) を立てる。``direction`` /
    ``excluded_weekdays`` は省略可 (未指定時は forward / 空)。
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind not in _DATE_INTENT_KINDS:
        return None
    start = _parse_intent_date(payload.get("start"))
    end = _parse_intent_date(payload.get("end"))
    if start == "invalid" or end == "invalid":
        return None
    raw_n = payload.get("n")
    n = raw_n if isinstance(raw_n, int) and not isinstance(raw_n, bool) else 0
    if kind in ("business_days_from", "days_from") and not 1 <= n <= _MAX_DATE_INTENT_N:
        return None
    exclusions = parse_date_intent_exclusions(payload)
    if exclusions is None:
        return None
    holidays, excluded_weekdays = exclusions
    skip_weekends = bool(payload.get("skip_weekends"))
    if kind == "business_days_from":
        # 種別自体が「営業日で数える」なので、モデルが false を返しても従う
        # 理由が無い (false なら days_from と区別が付かない)。
        skip_weekends = True
    raw_direction = payload.get("direction")
    direction = raw_direction if raw_direction in ("forward", "backward") else "forward"
    return DateIntentParams(
        kind=kind,
        start=start if isinstance(start, datetime.date) else None,
        end=end if isinstance(end, datetime.date) else None,
        n=n,
        skip_weekends=skip_weekends,
        holidays=holidays,
        count_start_day=bool(payload.get("count_start_day")),
        direction=direction,
        excluded_weekdays=excluded_weekdays,
    )


def parse_date_intent_exclusions(
    payload: object,
) -> tuple[tuple[datetime.date, ...], tuple[int, ...]] | None:
    """``date_intent`` 応答の除外条件 (``holidays`` / ``excluded_weekdays``) だけを検証する。

    ``kind`` に関わらず読める — 条件だけを足す追い質問で抽出器が ``kind: none`` を
    返しても、除外条件が入っていれば直前の演算に継げる (B-03 の続き)。
    形式違反は ``None``。
    """
    if not isinstance(payload, dict):
        return None
    raw_holidays = payload.get("holidays")
    holidays: list[datetime.date] = []
    if raw_holidays is not None:
        if not isinstance(raw_holidays, list):
            return None
        if len(raw_holidays) > _MAX_DATE_INTENT_HOLIDAYS:
            return None
        for item in raw_holidays:
            parsed = _parse_intent_date(item)
            if not isinstance(parsed, datetime.date):
                return None
            holidays.append(parsed)
    raw_excluded = payload.get("excluded_weekdays")
    excluded_weekdays: list[int] = []
    if raw_excluded is not None:
        if not isinstance(raw_excluded, list):
            return None
        if len(raw_excluded) > 7:
            return None
        for item in raw_excluded:
            if (
                not isinstance(item, int)
                or isinstance(item, bool)
                or not 0 <= item <= 6
            ):
                return None
            excluded_weekdays.append(item)
    return tuple(holidays), tuple(sorted(set(excluded_weekdays)))


def _business_day_candidates(
    start: datetime.date,
    span: int,
    *,
    skip_weekends: bool,
    holidays: tuple[datetime.date, ...],
    excluded_weekdays: tuple[int, ...] = (),
    direction: str = "forward",
) -> list[datetime.date]:
    """``start`` から ``span`` 日分のうち、数える対象になる日を並べる。

    ``direction == "backward"`` なら ``start`` より前へ辿る (2026-09-09 監査
    G-06: 「逆算」で前向きコマンドが組まれ、モデルが暗算で捨てた)。
    ``skip_weekends`` (土日) と ``excluded_weekdays`` (任意の曜日、例:
    毎週水曜が定例会議で作業不可) は独立指定で、除外対象は合算する。
    """
    step = -1 if direction == "backward" else 1
    days = [start + datetime.timedelta(days=step * i) for i in range(span)]
    excluded = set(excluded_weekdays)
    if skip_weekends:
        excluded |= {5, 6}
    return [d for d in days if d.weekday() not in excluded and d not in holidays]


def _candidate_span(n: int, holidays: tuple[datetime.date, ...]) -> int:
    """``n`` 営業日を確実に含む候補日数 (土日で 5/7 に減るので 2n + 余裕)。"""
    return n * 2 + len(holidays) * 3 + 14


def business_day_target(
    start: datetime.date,
    n: int,
    *,
    skip_weekends: bool = True,
    holidays: tuple[datetime.date, ...] = (),
    count_start_day: bool = True,
    excluded_weekdays: tuple[int, ...] = (),
    direction: str = "forward",
) -> datetime.date:
    """``start`` から数えて ``n`` 営業日目の日付 (純粋関数)。

    ``count_start_day`` が真なら ``start`` を 1 日目と数える (``start`` が
    休みなら次の営業日が 1 日目)。偽なら ``start`` の翌営業日が 1 日目。
    ``direction == "backward"`` なら ``start`` より前へ ``n`` 営業日辿る
    (「その日から逆算して」、2026-09-09 監査 G-06)。生成コマンドのテンプレート
    と **同じ数え方** を実装しており、``test_date_intent_command.py`` が
    両者の一致を検証する。

    候補列の先頭は ``start`` 自身が営業日のときだけ ``start`` になる。
    ``start`` が休み (土日 / 祝日 / 除外曜日) のときは先頭が既に「翌営業日 =
    1 日目」なので、``count_start_day`` に関わらず index は ``n - 1``。
    ``n`` のままだと 1 日ずれる — 実インシデント (2026-09-09 ライブ監査 (d)
    D-04): 「10/30 (金) 納品の 7 営業日前、毎週金曜は出荷不可」で起点の金曜が
    候補から落ち、``b[7]`` が 8 営業日前の 10/19 (正: 10/20) になった。
    """
    candidates = _business_day_candidates(
        start, _candidate_span(n, holidays),
        skip_weekends=skip_weekends, holidays=holidays,
        excluded_weekdays=excluded_weekdays, direction=direction,
    )
    start_is_business_day = bool(candidates) and candidates[0] == start
    index = n - 1 if (count_start_day or not start_is_business_day) else n
    return candidates[index]


def business_days_between(
    start: datetime.date,
    end: datetime.date,
    *,
    skip_weekends: bool = True,
    holidays: tuple[datetime.date, ...] = (),
    count_start_day: bool = True,
    excluded_weekdays: tuple[int, ...] = (),
) -> int:
    """``start``〜``end`` の間の営業日数 (純粋関数、両端を含む数え方)。

    ``excluded_weekdays`` (「毎週水曜日が定休日」) は ``business_day_target`` と
    同じく土日と合算する (2026-09-10 ライブ監査 (f) F-03: 期間の数え上げだけ
    曜日除外を持たず、追い質問が暗算に落ちていた)。
    """
    lo, hi = (start, end) if start <= end else (end, start)
    candidates = _business_day_candidates(
        lo, (hi - lo).days + 1, skip_weekends=skip_weekends, holidays=holidays,
        excluded_weekdays=excluded_weekdays,
    )
    return len(candidates) - (0 if count_start_day else 1)


def _date_literal(value: datetime.date | None) -> str:
    """コマンドへ埋める日付式 (``None`` は実行時の今日)。"""
    if value is None:
        return "n.date()"
    return f"datetime.date({value.year},{value.month},{value.day})"


def _holiday_set_literal(holidays: tuple[datetime.date, ...]) -> str:
    """祝日集合のリテラル (空集合は ``set()``)。"""
    if not holidays:
        return "set()"
    return "{" + ",".join(_date_literal(d) for d in holidays) + "}"


def _weekday_set_literal(weekdays: "set[int]") -> str:
    """除外曜日集合のリテラル (空集合は ``set()``)。"""
    if not weekdays:
        return "set()"
    return "{" + ",".join(str(w) for w in sorted(weekdays)) + "}"


def _effective_direction(direction: str, query: str) -> str:
    """規則側の逆算手掛かりを、LLM の forward 宣言より優先する。

    文法制約 JSON は値域までは守らない (CLAUDE.md の既知の落とし穴) ため、
    ``direction`` が forward でもクエリに逆算の手掛かり (``_DATE_BACKWARD_RE``)
    があれば backward を採る。LLM が既に backward を返しているときはそのまま
    (手掛かり語を持たない逆算表現も拾える、2026-09-09 監査 G-06)。
    """
    if direction == "backward":
        return "backward"
    if _DATE_BACKWARD_RE.search(query or ""):
        return "backward"
    return "forward"


#: 生成コマンドの共通前置き。
_DATE_INTENT_PREFIX = 'python -c "import datetime; n=datetime.datetime.now().astimezone();'


def build_date_intent_command(params: DateIntentParams, query: str = "") -> str:
    """検証済みパラメータから **決定論的に** コマンドを組む。

    出力には結果の日付だけでなく **数え方の前提** (向き / 起点を 1 日目と
    数えたか / 土日・除外曜日を除いたか / 除いた祝日の件数) も print する。
    起点日の数え方は自然言語では曖昧で、前提を書かない回答は検算できない
    (2026-09-08 F-06)。

    Args:
        params: :func:`parse_date_intent` が返した検証済みパラメータ。
        query: 元のクエリ。``business_days_from`` / ``days_from`` の向き
            (前 / 後) の LLM 宣言に対する **コード側の上書き判定**
            (:func:`_effective_direction`) にだけ使う。

    Returns:
        ``python -c "..."`` 形式のコマンド。組めない場合は空文字列。
    """
    start = _date_literal(params.start)
    if params.kind == "business_days_from":
        span = _candidate_span(params.n, params.holidays)
        # index は実行時に決める: 起点が休みなら候補の先頭が既に 1 日目
        # (:func:`business_day_target` と同じ数え方、D-04)。
        index = (
            f"{params.n}-1" if params.count_start_day
            else f"{params.n}-(0 if b[0]==s else 1)"
        )
        direction = _effective_direction(params.direction, query)
        step = -1 if direction == "backward" else 1
        excluded = set(params.excluded_weekdays)
        if params.skip_weekends:
            excluded |= {5, 6}
        return (
            _DATE_INTENT_PREFIX
            + f" s={start};"
            f" hs={_holiday_set_literal(params.holidays)};"
            f" ws={_weekday_set_literal(excluded)};"
            f" c=[s+datetime.timedelta(days={step}*i) for i in range({span})];"
            " b=[d for d in c if d.weekday() not in ws and d not in hs];"
            f" t=b[{index}];"
            " print('now:',n); print('start:',s.strftime('%Y-%m-%d (%A)'));"
            f" print('direction:',{direction!r});"
            f" print('skip_weekends:',{params.skip_weekends});"
            f" print('excluded_weekdays:',{sorted(excluded)!r});"
            f" print('holidays_excluded:',{len(params.holidays)});"
            f" print('start_counted_as_day1:',{params.count_start_day});"
            f" print('business_day_number:',{params.n});"
            " print('target:',t.strftime('%Y-%m-%d (%A)'))\""
        )
    if params.kind == "days_from":
        direction = _effective_direction(params.direction, query)
        signed = -params.n if direction == "backward" else params.n
        return (
            _DATE_INTENT_PREFIX
            + f" s={start};"
            f" t=s+datetime.timedelta(days={signed});"
            " print('now:',n); print('start:',s.strftime('%Y-%m-%d (%A)'));"
            f" print('direction:',{direction!r});"
            f" print('offset_days:',{signed});"
            " print('target:',t.strftime('%Y-%m-%d (%A)'))\""
        )
    if params.kind == "days_between":
        if params.skip_weekends or params.holidays or params.excluded_weekdays:
            adjust = 0 if params.count_start_day else 1
            # 曜日除外は起点からの数え (business_days_from) と同じ合算 (F-03)。
            excluded = set(params.excluded_weekdays)
            if params.skip_weekends:
                excluded |= {5, 6}
            return (
                _DATE_INTENT_PREFIX
                + f" a={start}; z={_date_literal(params.end)};"
                f" hs={_holiday_set_literal(params.holidays)};"
                f" ws={_weekday_set_literal(excluded)};"
                " lo=min(a,z); hi=max(a,z);"
                " c=[lo+datetime.timedelta(days=i) for i in range((hi-lo).days+1)];"
                " b=[d for d in c if d.weekday() not in ws and d not in hs];"
                " print('now:',n); print('from:',lo.strftime('%Y-%m-%d (%A)'));"
                " print('to:',hi.strftime('%Y-%m-%d (%A)'));"
                f" print('skip_weekends:',{params.skip_weekends});"
                f" print('excluded_weekdays:',{sorted(excluded)!r});"
                f" print('holidays_excluded:',{len(params.holidays)});"
                f" print('start_counted_as_day1:',{params.count_start_day});"
                f" print('business_days:',len(b)-{adjust})\""
            )
        return (
            _DATE_INTENT_PREFIX
            + f" a={start}; z={_date_literal(params.end)};"
            " lo=min(a,z); hi=max(a,z);"
            " print('now:',n); print('from:',lo.strftime('%Y-%m-%d (%A)'));"
            " print('to:',hi.strftime('%Y-%m-%d (%A)'));"
            " print('days:',(hi-lo).days)\""
        )
    if params.kind == "weekday_of":
        return (
            _DATE_INTENT_PREFIX
            + f" s={start};"
            " print('now:',n);"
            " print('target:',s.strftime('%Y-%m-%d (%A)'))\""
        )
    return ""


def inherit_date_intent(
    previous: DateIntentParams, current: DateIntentParams,
) -> DateIntentParams:
    """条件だけを変える追い質問で、直前の演算パラメータを **継ぐ** (純粋関数)。

    「その週の火曜日が休みだとしたら、着手日はどうなりますか」は自分では
    起点・日数・向き・起点の数え方を言わない。抽出器はそれらを直前の会話から
    読み直すが、同じ会話を 2 度読んでも同じ値になる保証は無い — 実機では
    直前の「6 営業日前に着手」で ``count_start_day=False`` (10/12) だったものが
    追い質問で ``True`` に振れ、休日 10/13 を除いても 10/12 のまま「変わり
    ません」と答えた (正 10/9、2026-09-09 ライブ監査 B-03)。

    追い質問が持ち込めるのは **除外条件だけ** (祝日 / 除外曜日) なので、
    それ以外は直前の値を採り、除外条件は和を取る。継ぐかどうか (今回の発話が
    手掛かり語を持たない追い質問か) は呼出側が決める。
    """
    return DateIntentParams(
        kind=previous.kind,
        start=previous.start,
        end=previous.end,
        n=previous.n,
        skip_weekends=previous.skip_weekends or current.skip_weekends,
        holidays=tuple(sorted(set(previous.holidays) | set(current.holidays))),
        count_start_day=previous.count_start_day,
        direction=previous.direction,
        excluded_weekdays=tuple(sorted(
            set(previous.excluded_weekdays) | set(current.excluded_weekdays),
        )),
    )


#: 追い質問が直前の結果に対して相対に置く休日。「その週の火曜日が休み」は
#: 直前の目標日と同じ週の火曜日、「その日が祝日」は目標日そのもの。曜日名は
#: 閉じた集合なので語形の網ではなく **構造の解決** として持つ (抽出器は
#: 「直前の回答の週から日付を求めよ」と指示しても 3 回中 3 回 ``holidays`` を
#: 空で返した — 2026-09-09 ライブ監査 B-03 / 検証 V02/4・V03/2)。
_FOLLOW_UP_WEEK_HOLIDAY_RE = re.compile(
    r"(?:その|同じ|当該)週の([月火水木金土日])曜"
)
_FOLLOW_UP_SAME_DAY_HOLIDAY_RE = re.compile(
    r"(?:その日|当日|着手日|納品日|完了日|目標日)(?:自体|そのもの)?が(?:休み|休日|祝日|休業)"
)
_WEEKDAY_INDEX = {c: i for i, c in enumerate("月火水木金土日")}


def resolve_date_intent_target(
    params: DateIntentParams, today: datetime.date,
) -> datetime.date | None:
    """検証済みパラメータの **目標日** をコード側で求める (純粋関数)。

    生成コマンドと同じ数え方 (:func:`business_day_target`)。``days_between`` /
    ``weekday_of`` は目標日を持たないので ``None``。
    """
    start = params.start or today
    if params.kind == "business_days_from":
        return business_day_target(
            start, params.n, skip_weekends=params.skip_weekends,
            holidays=params.holidays, count_start_day=params.count_start_day,
            excluded_weekdays=params.excluded_weekdays, direction=params.direction,
        )
    if params.kind == "days_from":
        step = -1 if params.direction == "backward" else 1
        return start + datetime.timedelta(days=step * params.n)
    return None


#: 「毎週水曜日は作業できない」— 曜日の除外。曜日名は閉じた集合。
_FOLLOW_UP_WEEKDAY_EXCLUSION_RE = re.compile(
    r"毎週([月火水木金土日])曜"
)


def follow_up_excluded_weekdays_from_query(query: str) -> tuple[int, ...]:
    """追い質問が除外する曜日 (0=月曜〜6=日曜) を解決する (純粋関数)。

    「毎週水曜日は編集会議で作業できないとすると」→ ``(2,)``。抽出器は同じ
    追い質問で ``kind: none`` を返す回がある (2026-09-09 ライブ監査 C-02 の
    続き) ので、除外曜日もコード側で確定させる。
    """
    if not query:
        return ()
    return tuple(sorted({
        _WEEKDAY_INDEX[m.group(1)]
        for m in _FOLLOW_UP_WEEKDAY_EXCLUSION_RE.finditer(query)
    }))


def follow_up_holidays_from_query(
    query: str, previous_target: datetime.date | None,
) -> tuple[datetime.date, ...]:
    """追い質問が直前の目標日に相対して置く休日を解決する (純粋関数)。

    「その週の火曜日が休みだとしたら」→ 目標日と同じ週 (月曜始まり) の火曜日。
    「その日が祝日なら」→ 目標日。解決できなければ空。
    """
    if previous_target is None or not query:
        return ()
    found: list[datetime.date] = []
    for m in _FOLLOW_UP_WEEK_HOLIDAY_RE.finditer(query):
        monday = previous_target - datetime.timedelta(days=previous_target.weekday())
        found.append(monday + datetime.timedelta(days=_WEEKDAY_INDEX[m.group(1)]))
    if _FOLLOW_UP_SAME_DAY_HOLIDAY_RE.search(query):
        found.append(previous_target)
    return tuple(sorted(set(found)))


#: 起点を直前の結果に置く照応 (「その日から 5 営業日後」「そこから 3 日後」)。
_ANAPHORIC_START_RE = re.compile(
    r"(?:その日|同日|そこ|その日付|当日|上の日|上記の日)(?:を起点に|から|以降|の)"
)
#: 直前のアシスタント発話に現れる日付 (和暦表記 / ISO)。最後に現れたものを
#: 「直前の結果」と読む (回答は結論の日付で終わることが多い)。
_ANSWER_DATE_RE = re.compile(
    r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
    r"|(\d{4})-(\d{2})-(\d{2})"
)


#: 「<月>の営業日数」— 期間が月で閉じている数え上げ。月は閉じた集合なので
#: 抽出器に任せず (実測: 「11 月の営業日数は…祝日 11/3・11/23 を除いて」で
#: ``kind: none``、モデルの暗算は土日を 8 日と数えて 20 (正 19)。2026-09-10
#: ライブ監査 (f) F-01) コード側で ``days_between`` を組む。
_MONTH_BUSINESS_DAYS_RE = re.compile(
    r"(?P<month>今月|来月|再来月|先月"
    r"|(?P<year>\d{4})\s*年\s*(?P<ynum>\d{1,2})\s*月"
    r"|(?P<num>\d{1,2})\s*月)"
    r"(?:の|中の|における)?\s*(?:営業日|稼働日|平日)(?:数|は何日|はいくつ|の日数)"
)
#: 発話に明示された祝日 / 休業日 (「祝日の 10 月 12 日」「11 月 3 日、11 月 23 日」)。
_EXPLICIT_MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_RELATIVE_MONTH_OFFSET = {"今月": 0, "来月": 1, "再来月": 2, "先月": -1}


def _resolve_month(
    token: str, today: datetime.date, *, year: int | None, num: int | None,
) -> tuple[int, int] | None:
    """月の語 (今月 / 来月 / N 月 / YYYY 年 N 月) を (年, 月) に解決する。

    裸の「N 月」は今日から見て **直近の同名月** (今月以降)。
    """
    offset = _RELATIVE_MONTH_OFFSET.get(token)
    if offset is not None:
        m = today.month - 1 + offset
        return today.year + m // 12, m % 12 + 1
    if num is None or not 1 <= num <= 12:
        return None
    if year is not None:
        return year, num
    if num >= today.month:
        return today.year, num
    return today.year + 1, num


def month_business_days_from_query(
    query: str, today: datetime.date,
) -> DateIntentParams | None:
    """「<月>の営業日数」を ``days_between`` のパラメータに解決する (純粋関数)。

    土日は除き、発話に列挙された「M 月 D 日」のうち **その月のもの** を祝日として
    除く。月の初日と末日が両端で、両端を含めて数える。
    """
    m = _MONTH_BUSINESS_DAYS_RE.search(query or "")
    if m is None:
        return None
    token = m.group("month")
    year = int(m.group("year")) if m.group("year") else None
    raw_num = m.group("ynum") or m.group("num")
    num = int(raw_num) if raw_num else None
    resolved = _resolve_month(token, today, year=year, num=num)
    if resolved is None:
        return None
    y, mo = resolved
    first = datetime.date(y, mo, 1)
    last = datetime.date(y + mo // 12, mo % 12 + 1, 1) - datetime.timedelta(days=1)
    holidays: list[datetime.date] = []
    for hm, hd in _EXPLICIT_MONTH_DAY_RE.findall(query or ""):
        if int(hm) != mo:
            continue
        try:
            holidays.append(datetime.date(y, mo, int(hd)))
        except ValueError:
            continue
    return DateIntentParams(
        kind="days_between", start=first, end=last, n=0, skip_weekends=True,
        holidays=tuple(sorted(set(holidays))), count_start_day=True,
        direction="forward", excluded_weekdays=(),
    )


#: 「<月>の第 N X 曜日」— 月と序数と曜日で閉じた日付。抽出器には kind が無く
#: (``weekday_of`` は「ある日付の曜日」)、now-only に落ちてモデルの暗算 +
#: 「ツールで検証していない」の開示になっていた (2026-09-10 (f) 検証 V05/2)。
_NTH_WEEKDAY_OF_MONTH_RE = re.compile(
    r"(?P<month>今月|来月|再来月|先月"
    r"|(?P<year>\d{4})\s*年\s*(?P<ynum>\d{1,2})\s*月"
    r"|(?P<num>\d{1,2})\s*月)"
    r"(?:の)?\s*第\s*(?P<nth>[1-5１-５一二三四五])\s*(?P<wd>[月火水木金土日])曜"
)
_KANJI_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}


def nth_weekday_of_month_from_query(
    query: str, today: datetime.date,
) -> DateIntentParams | None:
    """「来月の第 2 月曜日」を ``weekday_of`` (日付確定) に解決する (純粋関数)。"""
    m = _NTH_WEEKDAY_OF_MONTH_RE.search(query or "")
    if m is None:
        return None
    year = int(m.group("year")) if m.group("year") else None
    raw_num = m.group("ynum") or m.group("num")
    resolved = _resolve_month(
        m.group("month"), today, year=year, num=int(raw_num) if raw_num else None,
    )
    if resolved is None:
        return None
    y, mo = resolved
    nth_raw = m.group("nth")
    nth = _KANJI_DIGITS.get(nth_raw) or int(nth_raw.translate(
        str.maketrans("１２３４５", "12345"),
    ))
    weekday = _WEEKDAY_INDEX[m.group("wd")]
    first = datetime.date(y, mo, 1)
    offset = (weekday - first.weekday()) % 7
    target = first + datetime.timedelta(days=offset + 7 * (nth - 1))
    if target.month != mo:
        return None
    return DateIntentParams(
        kind="weekday_of", start=target, end=None, n=0, skip_weekends=False,
        holidays=(), count_start_day=False, direction="forward",
        excluded_weekdays=(),
    )


def query_has_anaphoric_start(query: str) -> bool:
    """追い質問が起点を直前の結果 (「その日」) に置いているか (純粋関数)。"""
    return bool(_ANAPHORIC_START_RE.search(query or ""))


def last_date_in_text(text: str) -> datetime.date | None:
    """本文の **最後** に現れる日付 (純粋関数)。無ければ ``None``。"""
    found = None
    for m in _ANSWER_DATE_RE.finditer(text or ""):
        parts = [g for g in m.groups() if g is not None]
        try:
            found = datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return found


def previous_answer_date(conversation: list[dict] | None, *, before: str = "") -> datetime.date | None:
    """直前のアシスタント発話が述べた日付 (純粋関数)。

    「今日から 3 週間後は？」→「2026年9月30日（水曜日）です」の次の
    「その日から 5 営業日後は？」は、起点が **直前の回答の日付** で、今日では
    ない。抽出器は会話を添えても ``start: today`` を返し、9/9 起点で 9/16 と
    答えた (2026-09-09 ライブ監査 (e) E-02、正 10/7)。照応の解決は抽出器に
    任せず、直前のアシスタント発話の最後の日付をコード側で採る。
    ``before`` に今回の発話を渡すと、末尾に積まれた今回の user ターンより前の
    assistant 発話を探す。
    """
    turns = list(conversation or [])
    seen_self = not before or not any(
        str(m.get("role") or "") == "user"
        and str(m.get("content") or "").strip() == before.strip()
        for m in turns
    )
    for msg in reversed(turns):
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "")
        if not seen_self:
            if role == "user" and content.strip() == before.strip():
                seen_self = True
            continue
        if role == "assistant":
            found = last_date_in_text(content)
            if found is not None:
                return found
            return None
    return None


def date_intent_command_from_params(params: DateIntentParams, query: str = "") -> str:
    """検証済みパラメータからコマンドを組む (readonly 検査込み、純粋関数)。"""
    command = build_date_intent_command(params, query)
    if not command:
        return ""
    if _readonly_command_rejected("run_command_readonly", command):
        return ""
    return command


def date_intent_command_from_payload(payload: object, query: str = "") -> str:
    """``date_intent`` の生応答からコマンドを組む (検証込み、純粋関数)。

    ``kind == "none"`` / 検証失敗 / readonly 違反はすべて空文字列を返し、
    呼出側は now-only のまま「未検証」の印を立てる。
    """
    params = parse_date_intent(payload)
    if params is None:
        return ""
    return date_intent_command_from_params(params, query)


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
    # 「唯一の事実根拠」枠で base に渡された。
    # 語彙は core.intent_vocab が SSOT。ここは **メモリ / GPU 系
    # (``MEMORY_SPEC_TERMS``) を意図的に載せない** 唯一の消費側 (上記の理由)。
    (re.compile(
        f"(?:{STORAGE_SPEC_TERMS}|{CPU_HARDWARE_TERM})",
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
        f"(?:{NETWORK_IDENTITY_TERMS})",
        re.IGNORECASE,
    ), "python -c \""
       "import socket;"
       " h=socket.gethostname();"
       " print('Hostname:',h);"
       " print('IP:',socket.gethostbyname(h))"
       "\""),
    # OS
    (re.compile(
        f"(?:{OS_QUERY_TERMS})",
        re.IGNORECASE,
    ), "python -c \""
       "import platform,sys;"
       " print(platform.platform());"
       " print(sys.platform,platform.release())"
       "\""),
    # Python バージョン
    (re.compile(
        f"(?:{PYTHON_VERSION_TERMS})",
        re.IGNORECASE,
    ), "python --version"),
    # 環境変数 — 名前だけを列挙する。値を出すと ``*_TOKEN`` 等がプロンプトと
    # debug JSONL に平文で載る (2026-09-05 ライブ監査 F-16)。値が要るなら
    # ユーザーが変数名を指定して個別に聞く形に倒す。
    (re.compile(
        f"(?:{ENV_VAR_TERMS})",
        re.IGNORECASE,
    ), "python -c \""
       "import os;"
       " print(', '.join(sorted(os.environ)))"
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
    # 日付演算の手掛かり (「30 営業日目」「三日後」) があるクエリも同じ。
    # リコール層はビルダを通らないので、過去ターンの now-only コマンドを
    # 引き当てると ``date_intent`` 層 (日付演算の唯一の受け皿) の前に確定して
    # しまい、暗算に戻る。日付演算を含むコマンドは従来どおり通す。
    if _DATE_MATH_CUE_RE.search(query or "") and not _DATE_ARITHMETIC_RE.search(
        command,
    ):
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
    if is_practice_advice_query(query):
        logger.debug("Practice/advice question, no executable command: %s", query[:50])
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
