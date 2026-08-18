"""実行可能クエリ → シェルコマンドの合成と readonly 判定

ルール表 (``_EXECUTABLE_QUERY_COMMANDS``) によるコマンド生成と、生成/引き当てた
コマンドを撃ってよいかの判定 (readonly 検証 / リコール適合) をまとめる。
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from pathlib import Path

from backend.free.agent.safety_patterns import reject_readonly_violation
from backend.free.agent.tool_judge_grounding import _numeric_literals
from backend.free.core.intent_vocab import DATETIME_QUERY_RE
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
_DATETIME_NOW_COMMAND = (
    'python -c "import datetime; print(datetime.datetime.now().astimezone())"'
)

#: 相対日付コマンドの共通前置き (now と目標日を両方出す)。
_REL_PREFIX = 'python -c "import datetime; n=datetime.datetime.now().astimezone();'
_REL_SUFFIX = (
    " print('now:',n);"
    " print('target:',t.strftime('%Y-%m-%d (%A)'))\""
)


def _build_datetime_command(query: str) -> str:
    """日付 / 時刻クエリ用のコマンドを組み立てる。

    相対表現 (「3 年前の今日」「今日から 100 日後」) が含まれる場合は **目標日と
    その曜日まで Python に計算させる**。現在時刻だけを渡してモデルに暗算させると
    外す (実インシデント 2026-08-07 ライブ監査: 「3 年前の今日は何曜日でしたか？」
    に「火曜日」と回答。2023-08-07 は月曜日)。同じ日に「今日から 100 日後」は
    正答しており、暗算が当たるかどうかは運になっていた。

    相対表現が無ければ従来どおり現在日時のみを返す。
    """
    m = _RELATIVE_OFFSET_RE.search(query or "")
    if m is None:
        return _DATETIME_NOW_COMMAND
    kind = _OFFSET_UNITS.get(m.group(2))
    if kind is None:
        return _DATETIME_NOW_COMMAND
    n = int(m.group(1))
    signed = -n if m.group(3) == "前" else n

    if kind in ("days", "weeks"):
        body = f" t=n+datetime.timedelta({kind}={signed});"
    else:
        # 月/年は timedelta で表せない。月末クランプ (1/31 の 1 か月後 = 2/28 等)
        # を含めて構築する。``calendar`` は readonly guard の許可モジュール外、
        # ``datetime.replace`` は禁止属性なのでコンストラクタで組み立てる。
        total = " tm=(n.year*12+n.month-1)+" + str(
            signed * 12 if kind == "years" else signed,
        ) + ";"
        body = (
            total
            + " y=tm//12; mo=tm%12+1;"
            " lp=(y%4==0 and (y%100!=0 or y%400==0));"
            " dim=[31,29 if lp else 28,31,30,31,30,31,31,30,31,30,31][mo-1];"
            " t=datetime.datetime(y,mo,min(n.day,dim));"
        )
    return _REL_PREFIX + body + _REL_SUFFIX

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

    Returns:
        生成されたシェルコマンド。該当なしの場合は空文字列。
    """
    for pattern, command in _EXECUTABLE_QUERY_COMMANDS:
        if pattern.search(query):
            if callable(command):
                return command(query)
            return command
    return ""
