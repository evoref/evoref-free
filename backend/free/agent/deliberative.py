"""Deliberative エージェント: LLM 推論 + ツール判定で応答（2〜10秒）"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from backend.free.agent.agent_state import AgentState
from backend.free.agent.event_reminder import EventReminderSystem
from backend.free.agent.meta_cognitive_utils import (
    command_run_failed,
    content_language_directive,
    generated_content_rejection,
    is_tool_error,
    strip_markdown_wrapper,
    tool_result_succeeded,
)
from backend.free.agent.tool_call_judge import ToolCallJudge, ToolJudgement
from backend.free.agent.tool_result_digest import digest_tool_result
from backend.free.agent.tools.builtin import (
    SEARCH_HISTORY_NO_RESULTS_PREFIX,
    _check_path_traversal as check_builtin_path_traversal,
)
from backend.free.constants import READ_FILE_META_PREFIX
from backend.free.core.session_mode import is_coding_mode
from backend.free.core.intent_vocab import (
    resolve_session_position_message,
    session_position_kind,
)
from backend.free.core.turn_text import append_to_last_user
from backend.config import resolve_context_size_for_mode
from backend.free.api.chat.chat_constants import (
    CONTENT_MAX_TOKENS_MIN, CONTENT_SYSTEM_RESERVE,
    TOOL_EXECUTION_TIMEOUT_SEC, TOOL_GROUNDED_TEMPERATURE,
    TOOL_RESULT_MAX_CHARS,
    TOOL_RESULT_HEAD_RATIO, TOOL_RESULT_OMISSION_CHARS,
)
from backend.free.api.chat.chat_types import GenerationParams, StepCallback
from backend.log_config import get_logger

logger = get_logger("agent.deliberative")

# write_file でコンテンツ生成が必要な場合のプロンプト
_CONTENT_GEN_PROMPT = """\
Generate the requested content below. Output ONLY the content itself, \
no explanations, no markdown fences, no JSON, no surrounding text.
"""

# digest_tool_result が NO_RELEVANT_INFO と確定した場合に raw の代わりに
# ツール実行結果として渡すプレースホルダ。無関係な内容を「唯一の事実根拠」
# として base に読ませないための安全な代替文言。
_NO_RELEVANT_INFO_MESSAGE = "（ツールを実行しましたが、今回の質問に関連する情報は見つかりませんでした）"

# 環境依存の事実を尋ねられたのにツールが 1 つも走らなかったときの注記。
# 「この PC の〜を調べて」型のクエリは router が executable_query と分類する
# が、chat の readonly 実行は python インタプリタと限られたモジュールしか
# 許さないため、合成されたコマンドが棄却されて全層 no_tool に落ちうる。
# その状態で base に丸投げすると、測っていない値を断定して答える
# (実インシデント 2026-07-29 ライブ監査:「このPCのメモリ搭載量を調べて
# 教えてください。」で powershell / wmic のコマンドが 2 度棄却され、ツールが
# 1 つも走らないまま「32 GB です」と回答した)。system プロンプトの捏造禁止
# 条項は天気・ニュース・株価等の外部データにしか掛かっておらず、実行環境の
# 事実は素通りしていた。
# 文言は「実行を試みて失敗した」と読めない形にする。旧版の「ツールを実行
# できず」は base に「ツール実行により取得できませんでした」と言い換えられ、
# 実際には 1 度も実行していないのに実行して失敗したかのように読める応答に
# なった (実インシデント 2026-08-01 ライブ監査)。システムの実状態
# (= このターンでツールを 1 つも実行していない) をそのまま述べさせる。
_UNMEASURED_FACT_GUIDANCE = (
    "\n\nこの質問はこの実行環境を実際に調べないと答えられない種類の質問だが、"
    "今回のターンではこの環境を調べるツールを 1 つも実行していないため、"
    "測定値は存在しない。"
    "したがって具体的な数値・型番・バージョン・容量を答えてはならない。"
    "会話履歴や記憶で既にユーザー自身が述べた値があるならその出典を明示して"
    "述べてよいが、それが無い場合は「今回は調べていないので分からない」と"
    "正直に伝え、ユーザーが自分で確認する方法を 1 つ示すこと。"
    "実行していないツールについて、実行した / 実行を試みたと述べない。"
    "推測値を断定形で述べない。"
)

# search_history が空振りした場合専用のグラウンディング文言。通常ツールの
# 「唯一の事実根拠として扱う」枠をそのまま付けると、直前ターンで述べられた
# 情報 (SemMem 未反映のまだ生の会話履歴) まで無視して「見つかりません」と
# 誤答する (実インシデント 2026-07-23: 直前ターンで伝えた氏名・出身地を
# 聞き直され、search_history 空振りを理由に誤って「記録が無い」と回答した)。
# search_history は過去セッションのみを検索するツールであることを明示し、
# 今回進行中の会話履歴の参照を妨げないようにする。
_SEARCH_HISTORY_NO_INFO_GUIDANCE = (
    "search_history は過去の別セッションの会話記録を検索するツールである。"
    "上記の「関連する情報は見つかりませんでした」は、過去の別セッションには"
    "見つからなかったという意味に過ぎず、今回進行中のこの会話で既に述べられた"
    "情報 (直前のユーザー発言を含む会話履歴) を否定するものではない。"
    "会話履歴に該当情報があれば、検索結果とは関係なくそれを使って具体的に"
    "回答すること。会話履歴にも本当に無い場合のみ「わからない」と答えてよい。"
)

# search_history が「ヒットした」場合のグラウンディング文言。通常ツールの
# 「唯一の事実根拠として扱う」枠を付けると、過去の別セッション (別の話題・
# 別の人物との会話) の内容を今回の事実として断定してしまう (実インシデント
# 2026-07-26: 自己紹介直後に「私の趣味は？」と聞かれ、search_history が
# 別人物「佐藤健一」のセッションをヒットし、base がユーザーを佐藤健一だと
# 誤認して「趣味の情報は見つかりません」と誤答した)。
# 取得が本物であることは肯定しつつ (拒否・「取得できない」を防ぐ)、
# 進行中の会話履歴より優先させないことを明示する。
_SEARCH_HISTORY_RESULT_GUIDANCE = (
    "上記の ## ツール実行結果 は、システムが search_history ツールで実際に取得した"
    "本物のデータである。ただしこれは今回進行中の会話とは別の「過去のセッション」の"
    "記録であり、そこに現れる人名・話題・予定は当時の別の会話のものである。"
    "今回の相手が同じ人物とは限らないため、検索結果に出てきた名前でユーザーを呼んだり、"
    "ユーザーの属性として断定したりしないこと。"
    "今回進行中の会話履歴 (直前のユーザー発言を含む) で既に述べられた情報が最優先であり、"
    "検索結果はそれを否定・上書きしない。会話履歴に答えがあるならそれを使って答え、"
    "検索結果は過去の出来事についての補足としてのみ用いること。"
    "「ブラウズできない」「取得できない」とは言わないこと。"
    "結果にも会話履歴にも無い数値・事実は創作しないこと。"
)

# calculate 専用のグラウンディング文言。calculate の結果は裸の数値であり、
# 何をどの単位で計算したかの情報を持たない。汎用の「唯一の事実根拠」枠だけを
# 付けると base が (a) 質問文の単位をそのまま数値に貼り付ける
# (b) 式に無い係数を後付けで説明する、という 2 つの捏造を起こす
# (実インシデント 2026-07-27: 「直径30cm・深さ25cm の鉢の土は何リットル？」に
# 対し assist が cm 単位の式 3.14159*(30/2)**2*(25-2) を組み、結果 16257.7 cm³ が
# 「約 16,258 リットル」と回答され、さらに「土の比重 1.3 を掛けた結果」という
# 式に存在しない根拠が創作された)。式を併記したうえで、単位は入力に従うこと・
# 式に無い係数を語らないことを明示する。
_CALCULATE_RESULT_GUIDANCE = (
    "上記の ## ツール実行結果 は、システムが calculate ツールで実際に評価した式と"
    "その厳密な計算結果である。数値はこの結果をそのまま使うこと。"
    "結果の単位は式に入力した数値の単位に従う"
    "(例: cm で測った長さだけを掛けた体積は cm³ であってリットルではない)。"
    "割り算では単位も割り算される "
    "(例: km ÷ (km/時) = 時間、円 ÷ 個 = 円/個)。"
    "答えの文では、数値に必ずその単位を添えて述べること "
    "(例:「何時間かかりますか」への答えは『5 時間かかります』と単位まで書く)。"
    "ユーザーが別の単位で尋ねている場合、式に換算が含まれていなければ換算は"
    "行われていないので、必要な換算を自分で行って示すか、単位が異なる旨を明示すること。"
    "式に現れていない係数・比重・補正 (例:「比重 1.3 を掛けた」) を"
    "計算の根拠として述べないこと。実際に評価されたのは上記の式だけである。"
)

# 引用したユーザー発言に一人称が含まれるかの判定 (帰属注記の出し分け用)。
_FIRST_PERSON_RE = re.compile(r"(?:私|僕|俺|自分|わたし|ぼく)")

# run_command / run_command_readonly 専用のグラウンディング。calculate と同じく
# 出力は裸の値で、「何を求めたコマンドか」の情報を持たない。汎用枠だけを付けると
# base は出力を「基準値 (= 現在時刻)」と読み、質問された演算をもう一度適用する
# (実インシデント 2026-07-28:「今日から100日後は何月何日ですか。」で
# `date.today() + timedelta(days=100)` が 2026-11-05 を正しく返したのに、
# 回答は「2026 年 11 月 5 日から 100 日後は、2027 年 2 月 13 日です」となった)。
# コマンドを併記したうえで、出力は既に求められた値であることを明示する。
_COMMAND_TOOLS = frozenset({"run_command", "run_command_readonly"})
_COMMAND_RESULT_GUIDANCE = (
    "上記の ## ツール実行結果 は、システムが実際に実行したコマンドとその標準出力である。"
    "出力はコマンドが計算し終えた**結果そのもの**であり、途中経過や基準値ではない。"
    "コマンドに日数の加算・差分・書式変換が含まれている場合、その計算は既に済んでいるので、"
    "同じ演算を出力に対してもう一度適用しないこと。"
    "数値・日付はこの出力をそのまま使い、暗算で作り直さないこと。"
    "「実行できない」「取得できない」とは言わないこと。"
    "出力に無い数値・事実は創作しないこと。"
    "回答本文では「ツール実行結果」「コマンド」等の内部的な言い回しを使わず、"
    "自分で調べて分かったこととして自然に述べること。"
)

# 生成系ツール (draft_document / summarize / translate) 専用のグラウンディング。
# これらの結果は外部から取得した事実データではなく LLM が書いた下書きなので、
# 「唯一の事実根拠」枠を付けると、下書きが混入させた誤りが会話で既に確定した
# 事実を上書きしてしまう (実インシデント 2026-07-27: 8 月 22 日を「(土)」と
# 正しく答えた次のターンで、draft_document の下書きが「(日)」と書いたため
# 回答も (日) に化けた)。下書きであることを明示し、会話側の事実を優先させる。
_GENERATED_DRAFT_TOOLS = frozenset({"draft_document", "summarize", "translate"})
_GENERATED_DRAFT_GUIDANCE = (
    "上記の ## ツール実行結果 は、あなた自身が下書きとして生成した文章であり、"
    "外部から取得した事実データではない。文体・構成の土台としては使ってよいが、"
    "日付・曜日・場所・数量など会話で既に確定している事実と食い違う箇所は"
    "会話側を正として書き直すこと。下書きに現れる事実を新たな根拠として扱わないこと。"
    "回答本文では「ツール実行結果」「ご提示いただいた結果」等の内部的な言い回しを使わず、"
    "自分が書いた文章として自然に提示すること。"
)


# read_file 専用のグラウンディング。汎用文言は「結果に無い事実を創作しない」と
# しか言っておらず、会話中で「こう書いたはず」と分かっている内容とファイルの
# 実体が食い違うとき、実体ではなく期待値を答えてしまう (実インシデント
# 2026-07-29 ライブ監査:「さっき保存したファイルの中身をそのまま見せて」に対し、
# read_file は「タスクの内容: ファイル `…` に…」を返したのに、回答は依頼時の
# 文言「監査テスト 1行目」だった。書込みが壊れていた事実がユーザーから隠れた)。
# 食い違いの報告を明示的な仕事として与える (禁止形だけだと退行する)。
# 列挙が価値のツール。結果の意味は「その集合が全部である」ことなので、散文
# 要約に通すと項目が落ち、base が落ちた分をもっともらしい名前で埋める
# (実インシデント 2026-08-01 ライブ監査: list_directory の 6070 文字が digest
# で 225 文字に圧縮され、回答は実在しない requirements.txt / .env.example を
# 挙げ、実在する scripts/ local/ models/ CLAUDE.md 等を落とした)。
# ``_ENUMERATIVE_TOOLS`` は digest 抑止とグラウンディング文言の両方が見る。
_ENUMERATIVE_TOOLS = frozenset({"list_directory", "search_code"})
_ENUMERATION_RESULT_GUIDANCE = (
    "上記の ## ツール実行結果 は、システムが実際に列挙した項目そのものである。"
    "項目名はこの結果に現れる綴りのまま引用すること。"
    "一般的な構成から類推した項目名を補わないこと。"
)
#: 列挙結果が切り詰められていたときに追加する文言。省略があることを明示しないと
#: base は手元の部分集合を全体として提示する。
_TRUNCATED_ENUMERATION_NOTE = (
    "この結果は途中が省略された部分的な一覧である。"
    "「これで全部」とは述べず、一覧が部分的であることを明示して答えること。"
)

_FILE_CONTENT_TOOLS = frozenset({"read_file"})
_FILE_CONTENT_RESULT_GUIDANCE = (
    "上記の ## ツール実行結果 は、そのファイルに実際に保存されている内容そのものである。"
    "ファイルの中身を示すときは、この結果の文字列をそのまま引用して提示すること。"
    "会話の中で「こう書いたはず」と話していた内容と食い違う場合は、"
    "実際のファイル内容の方を示したうえで、期待と食い違っている旨も併せて伝えること。"
    "期待していた文言に合わせて内容を書き換えたり、要約・整形したりしないこと。"
    "「読み取れない」「アクセスできない」とは言わないこと。"
    "回答本文では「ツール実行結果」等の内部的な言い回しを使わず、"
    "自分で読んで分かったこととして自然に述べること。"
)


#: 「そのまま / 一字一句 / 全文」でファイル内容の提示を求める言い回し。
_VERBATIM_ECHO_RE = re.compile(
    r"そのまま|一字一句|原文|全文|加工せず|変えずに"
    r"|verbatim|as[- ]is|exactly as|raw content",
    re.IGNORECASE,
)
#: 決定論エコーの上限。これを超える内容は会話に貼るより要約が適切なので
#: 従来どおりモデルへ渡す。
_VERBATIM_ECHO_MAX_CHARS = 8000


def verbatim_file_echo(
    query: str, tool_name: str | None, tool_result: str | None,
) -> str | None:
    """「中身をそのまま見せて」型の依頼へ返す決定論的な応答を組み立てる。

    ``read_file`` の結果は「唯一の事実根拠として使え」と指示しても、小型
    base はファイルの実体ではなく **会話上そうであるはずの内容** を答えて
    しまう (実インシデント 2026-07-29 ライブ監査: 内部プロンプトの足場と
    英文が書き込まれてしまったファイルを「そのまま見せて」と頼んだところ、
    その英文を日本語訳した文章が提示され、ファイルが壊れている事実が
    ユーザーから隠れた。read_file 専用のグラウンディング文言を足しても
    再現した)。

    逐語提示は答えが一意に決まるのでモデルを通す利得が無い。該当する依頼に
    限り生成を迂回し、ツール結果をそのまま返す。

    Returns:
        返すべき応答本文。該当しなければ ``None`` (純粋関数)。
    """
    if tool_name not in _FILE_CONTENT_TOOLS or not tool_result:
        return None
    if is_tool_error(tool_result):
        return None
    if not _VERBATIM_ECHO_RE.search(query or ""):
        return None
    if len(tool_result) > _VERBATIM_ECHO_MAX_CHARS:
        return None
    body = strip_read_file_meta_line(tool_result)
    if not body.strip():
        return None
    fence = "````" if "```" in body else "```"
    return f"{fence}\n{body.rstrip()}\n{fence}"


def strip_read_file_meta_line(tool_result: str) -> str:
    """``read_file`` 結果の先頭メタ行を落として本文だけを返す (純粋関数)。

    メタ行 (``[file: ... | lines: N | chars: M]``) は行数・文字数をモデルに
    数えさせないための**モデル向け**補助情報であり、ユーザーに見せる本文では
    ない。逐語エコーは生成を迂回してツール結果をそのまま返すため、ここで
    落とさないとメタ行が回答へそのまま出る (2026-08-05 ライブ監査: 「その
    README の最初の 5 行をそのまま見せて」「今書き込んだファイルを読み返して
    内容をそのまま見せて」の 2 ターンで露出)。

    メタ行が無い結果はそのまま返す。
    """
    if not tool_result.startswith(READ_FILE_META_PREFIX):
        return tool_result
    head, sep, rest = tool_result.partition("\n")
    if not sep or not head.rstrip().endswith("]"):
        return tool_result
    return rest


async def _iterate_once(text: str) -> AsyncIterator[str]:
    """1 チャンクだけ返すストリーム (決定論応答をストリーム経路へ載せる)。"""
    yield text


def _is_search_history_empty(tool_name: str, result_text: str) -> bool:
    """search_history が 1 件もヒットしなかった結果か判定する (純粋関数)。"""
    return (
        tool_name == "search_history"
        and result_text.startswith(SEARCH_HISTORY_NO_RESULTS_PREFIX)
    )


def _check_path_traversal(file_path: str, tool_name: str) -> str | None:
    """write_file / read_file のパス検証 (LLM 生成コンテンツ生成前の fail-fast)。

    実体は ``backend.free.agent.tools.builtin._check_path_traversal`` に集約
    済み (builtin.write_file/read_file 自体もこれを呼ぶため二重防御になる)。
    ここでは無駄な ``_ensure_write_file_content`` (LLM 呼出し) を避けるため
    早期に同じ検証を行う。
    """
    if tool_name not in ("write_file", "read_file"):
        return None
    return check_builtin_path_traversal(file_path)


def _emit_tool_running_step(
    on_step: StepCallback, tool_name: str, tool_args: dict,
) -> None:
    """ツール実行開始の step フレームを emit する。"""
    if on_step is None:
        return
    from backend.free.agent.meta_cognitive_utils import summarize_tool_args
    on_step({
        "type": "tool_call",
        "detail": f"{tool_name}({summarize_tool_args(tool_name, tool_args)})",
        "status": "running",
    })


# step frame の detail に載せるツール結果の最大長。
_STEP_DETAIL_MAX_CHARS = 100


def _truncate_for_step(text: str) -> str:
    """step frame 表示用に切り詰める。切った場合は省略記号と全長を付ける。"""
    if len(text) <= _STEP_DETAIL_MAX_CHARS:
        return text
    return f"{text[:_STEP_DETAIL_MAX_CHARS]}... ({len(text)} chars)"


def _emit_tool_result_step(
    on_step: StepCallback, tool_name: str, result_text: str,
) -> None:
    """ツール実行完了 (`task_result`) の step フレームを emit する。

    ツールラッパが例外を投げなくても、走ったコマンド自身が失敗していることが
    ある (``run_command`` は非ゼロ終了時に ``[exit code: N]`` を付けて stderr を
    戻り値として返す)。この経路を無条件に ``status="done"`` にしていたため、
    UI が Traceback を ✓ で表示していた (実測 2026-07-25: 存在しない
    ``platform.cpu_count`` を呼んだ合成コマンドが ✓ 表示)。
    学習シグナル (``tool_command_success``) と同じ判定を使って揃える。

    detail は 100 文字で切るが、切ったことを明示する。無告知で切ると UI 上は
    語の途中でぶつ切りになり、応答の数値がツール出力に含まれていたのか
    捏造なのかを人間が判別できない (実測 2026-07-27: "Cores: 24 Di" で切れて
    直後の "Disk: 443 GB total, 138 GB free" が隠れ、正しくグラウンドされた
    回答が捏造に見えた)。step_compactor._compress_generic と同じ体裁に揃える。
    """
    if on_step is None:
        return
    failed = is_tool_error(result_text) or (
        tool_name in ("run_command", "run_command_readonly")
        and command_run_failed(result_text)
    )
    logger.debug(
        "Tool result step: tool=%s, failed=%s, result=%s",
        tool_name, failed, result_text[:120],
    )
    on_step({
        "type": "task_result",
        "detail": f"{tool_name}: {_truncate_for_step(result_text)}",
        "status": "failed" if failed else "done",
    })


def _emit_tool_failure_step(
    on_step: StepCallback, tool_name: str, error_text: str,
) -> None:
    """ツール失敗 / タイムアウトの step フレームを emit する。"""
    if on_step is None:
        return
    on_step({
        "type": "tool_call",
        "detail": f"{tool_name}: {error_text[:100]}",
        "status": "failed",
    })


@dataclass
class DeliberativeResponse:
    """Deliberative 層の応答"""
    content: str
    rag_used: bool = False
    rag_source: str | None = None
    rag_chunks: list[tuple[str, float, str]] = field(default_factory=list)
    tool_result: str | None = None
    tool_name: str | None = None
    # executable command 学習用 (run_command 実行ターンのみ非 None)
    tool_command: str | None = None
    tool_command_success: bool | None = None
    # 判定層 ("rule" / "assist" / "recall" ...)。sleep 側の
    # executable_command_curator が "recall" 由来の実行を学習対象から外すために使う。
    tool_command_source: str | None = None


class DeliberativeAgent:
    """Deliberative 層: LLM 推論 + アシストモデルによるツール判定

    ToolCallJudge によるツール呼び出し判定を実行し、
    ツール結果をコンテキストとして注入してから LLM に応答を生成させる。

    write_file でコンテンツ生成が必要な場合は、LLM にプレーンテキストで
    コンテンツを生成させてからツールを実行する（JSON 内にコンテンツを含めない）。
    目標応答時間: 2〜10秒（ツールなし） / 60〜120秒（コンテンツ生成+ツール）
    """

    def __init__(
        self,
        config: dict | None = None,
        tool_judge: ToolCallJudge | None = None,
        tools_registry=None,
        assist_client=None,
        assist_experience_recorder=None,
        agent_tracer=None,
        mode: str = "chat",
    ):
        self.config = config or {}
        self.reminder_system = EventReminderSystem(self.config)
        self._tool_judge = tool_judge
        self._tools_registry = tools_registry
        # _execute_tool の mode ゲートの既定値。process() 呼び出し毎の実際の mode は
        # _judge_and_execute_tool から明示的に渡される (こちらは直接 _execute_tool を
        # 呼ぶ既存テスト等のフォールバック用)。
        self._mode = mode
        # ツール結果の query 連動抽出 (base の接地負荷軽減) に使う。None なら raw を渡す。
        self._assist_client = assist_client
        # assist 由来ツール判定の実行成否を assist 経験へ記録する closure。
        # Pro/Develop 起動時のみ非 None (factory 層が注入)。None なら記録 no-op。
        self._assist_experience_recorder = assist_experience_recorder
        # MDP トレース。develop モード時のみ非 None (factory 層が注入)。
        # deliberative の tool 判定/実行を 1 step エピソードとして記録し、
        # sleep-time Step 7.5 が episodic LTM へ取込 → Level 1 agent ドメインの
        # 学習信号にする (これが無いと agent ドメインは skipped_no_signal)。
        self._agent_tracer = agent_tracer

        # コンテンツ生成用の max_tokens (coding 時は coding_model の実窓に合わせる)
        ctx_size = resolve_context_size_for_mode(self.config, mode)
        self._content_max_tokens = max(ctx_size - CONTENT_SYSTEM_RESERVE, CONTENT_MAX_TOKENS_MIN)

    @staticmethod
    def _init_deliberative_state(mode: str) -> AgentState:
        """`process` 用 AgentState を生成。`coding` モードは unified_diff を期待。"""
        return AgentState(
            agent_layer="deliberative",
            expected_format="unified_diff" if is_coding_mode(mode) else None,
        )

    @staticmethod
    def _append_tool_result_to_last_user(
        messages: list[dict],
        tool_name: str,
        tool_result_text: str,
        query: str | None = None,
        tool_args: dict | None = None,
    ) -> None:
        """最後の user メッセージにツール実行結果を追記する。

        system ロールを assistant の後に挿入すると Qwen3.5 等の ChatML
        テンプレートで 400 エラーになるため、必ず user に統合する。
        """
        truncated = _truncate_tool_result(tool_result_text, TOOL_RESULT_MAX_CHARS)
        args = tool_args if isinstance(tool_args, dict) else {}
        if tool_name == "calculate":
            expression = str(args.get("expression") or "").strip()
            if expression:
                # 裸の数値だけでは何を計算したかが base に伝わらず、単位と根拠の
                # 捏造を招く (_CALCULATE_RESULT_GUIDANCE 参照)。式を併記する。
                truncated = f"{expression} = {truncated}"
        elif tool_name in _COMMAND_TOOLS:
            executed = str(args.get("command") or "").strip()
            if executed:
                # 裸の出力だけでは何を求めたコマンドか base に伝わらず、
                # 出力を基準値と読んで演算を二重適用する
                # (_COMMAND_RESULT_GUIDANCE 参照)。コマンドを併記する。
                truncated = f"$ {executed}\n{truncated}"
        # 話題再フォーカス: 弱いモデルは前ターンの話題に引きずられ、今回の質問
        # (例: ニュース) を取り違える (実機確認: 前ターンが天気だとニュース質問に
        # 天気で誤答)。今回の質問を明示して前話題を無視させる。
        refocus = ""
        if query:
            q = query if len(query) <= 200 else query[:200] + "…"
            # 引用文の一人称をそのまま自分のものとして繰り返す事故がある
            # (実インシデント 2026-07-28:「私の誕生日は3月14日です。今日から
            # 誕生日まであと何日ですか。」に対し「私の誕生日は 3 月 14 日で、
            # 今日からあと 229 日です。」と、アシスタント自身の誕生日として
            # 回答した)。一人称を含むときだけ帰属を明示する。
            person_note = (
                "引用文中の一人称 (私 / 僕 / 自分) はユーザーを指す。"
                "回答では一人称を使わず、必ず二人称で述べること "
                "(誤:「私の誕生日は3月14日です」/ "
                "正:「小川さんの誕生日は3月14日ですね」「あなたの誕生日は…」)。"
                if _FIRST_PERSON_RE.search(q) else ""
            )
            refocus = (
                f"今回ユーザーが答えてほしい質問は『{q}』である。{person_note}"
                f"会話履歴の前の話題は無関係なので無視し、この質問にのみ答えること。\n"
            )
        if tool_name == "search_history" and tool_result_text == _NO_RELEVANT_INFO_MESSAGE:
            # 空振りに「唯一の事実根拠」枠を付けると直前ターンの内容まで
            # 無視されるため、search_history 空振り専用の文言に差し替える
            # (通常ツールの capability assertion は付けない)。
            grounding = _SEARCH_HISTORY_NO_INFO_GUIDANCE
        elif tool_name == "search_history":
            # ヒットした場合も「唯一の事実根拠」枠は付けない。過去の別セッション
            # の内容を今回の事実として断定させないため専用文言を使う。
            grounding = _SEARCH_HISTORY_RESULT_GUIDANCE
        elif tool_name == "calculate":
            grounding = _CALCULATE_RESULT_GUIDANCE
        elif tool_name in _COMMAND_TOOLS:
            grounding = _COMMAND_RESULT_GUIDANCE
        elif tool_name in _ENUMERATIVE_TOOLS:
            grounding = _ENUMERATION_RESULT_GUIDANCE
            if len(truncated) < len(tool_result_text):
                grounding += _TRUNCATED_ENUMERATION_NOTE
        elif tool_name in _FILE_CONTENT_TOOLS:
            grounding = _FILE_CONTENT_RESULT_GUIDANCE
        elif tool_name in _GENERATED_DRAFT_TOOLS:
            grounding = _GENERATED_DRAFT_GUIDANCE
        else:
            # capability assertion: 弱いモデルは「自分はブラウズ/取得できない」という
            # 思い込みでツール結果を無視し拒否することがある (実機確認)。結果が
            # 実際に取得された本物データであると明示して上書きする。
            grounding = (
                f"上記の ## ツール実行結果 は、システムが {tool_name} ツールで実際にアクセスして"
                f"取得した本物のデータである。あなたにはこのツールがあり、取得は既に成功している。"
                f"この結果を唯一の事実根拠として、内容 (数値・名称・日付・条件など) を読み取り、"
                f"それに基づいて具体的に回答すること。"
                f"「ブラウズできない」「取得できない」「アクセスできない」とは言わないこと。"
                f"「取得できない」「データがない」と答えてよいのは、結果が空かエラーの場合のみ。"
                f"結果に該当が無い場合のみ、システムプロンプトの参考コンテキスト (カートリッジ・記憶等) も併用してよい。"
                f"結果に無い数値・事実は創作しないこと。"
                # 内部足場の語彙がそのまま本文に出る (実測 2026-07-27:
                # 「ご提示いただいたツール実行結果に基づき、作成した買い物リストは
                # 以下の通りです。」)。ユーザーにはツール実行は見えているが、
                # 「提示された」という受け身の言い回しは主体を取り違えさせる。
                f"ただし回答本文では「ツール実行結果」「ご提示いただいた結果」等の内部的な言い回しを使わず、"
                f"自分で調べて分かったこととして自然に述べること。"
            )
        tool_msg = (
            f"\n\n## ツール実行結果\n"
            f"ツール: {tool_name}\n"
            f"結果:\n{truncated}\n\n"
            f"{refocus}"
            f"{grounding}"
        )
        append_to_last_user(messages, tool_msg, separator="")

    @staticmethod
    def _append_session_position_fact(
        messages: list[dict], conversation: list[dict] | None, query: str,
    ) -> str | None:
        """「この会話で最初/最後に言ったこと」を決定論的に確定して注記する。

        位置で決まる事実を検索やモデルの読解に委ねる理由が無い。進行中の会話は
        全文がコンテキストに載っている一方、``search_history`` の索引には
        **まだ入っていない** ため、現在セッションを検索しても中身の無い
        セッションヘッダしか返らない。それを根拠枠で渡すと base は文脈から
        適当な発言を選ぶ (実インシデント 2026-08-01 ライブ監査:「この会話で私が
        最初に送ったメッセージは何でしたか？」に対し、12 番目の発言
        「円周率の小数点以下 10 桁を教えてください。」と誤答した)。

        Returns:
            注記した位置種別 ("first" / "last")。対象外なら None。
        """
        position = session_position_kind(query)
        if position is None:
            return None
        target = resolve_session_position_message(conversation, query, position)
        if not target:
            return None
        label = "最初" if position == "first" else "直近"
        append_to_last_user(
            messages,
            f"\n\n確定事実: この会話でユーザーが{label}に送ったメッセージは"
            f"「{target}」である。これは会話の並び順から機械的に確定した値なので、"
            f"この値をそのまま答えること。",
            separator="",
        )
        logger.info(
            "Session position fact pinned (%s): %s", position, target[:60],
        )
        return position

    @staticmethod
    def _append_unmeasured_fact_note(messages: list[dict]) -> None:
        """最後の user メッセージへ「実測できなかった」注記を追記する。"""
        append_to_last_user(messages, _UNMEASURED_FACT_GUIDANCE, separator="")

    def _record_tool_call_outcome(
        self, query: str, judgement: ToolJudgement, success: bool, mode: str = "chat",
    ) -> None:
        """assist 由来ツール判定の実行成否を assist 経験へ記録する (best-effort)。

        rule / learned / cartridge 由来は assist モデル出力ではないため記録しない
        (assist=B が学ぶのは assist のツール判定のみ)。recorder 未注入
        (Free / --no-learning) なら no-op。例外は recorder 側で握り潰される。
        ``mode`` は呼び出し元 (``_judge_and_execute_tool`` の明示引数、
        ``self._mode`` ではなく実際の処理対象モード) をそのまま渡す。
        """
        rec = self._assist_experience_recorder
        if rec is None or judgement.source != "assist":
            return
        rec("tool_call", query, judgement.tool_name or "", 1.0 if success else 0.0, mode)

    async def _judge_and_execute_tool(
        self,
        query: str,
        mode: str,
        conversation: list[dict] | None,
        messages: list[dict],
        llm_client,
        state: AgentState,
        on_step: StepCallback,
        tool_judge_task: "asyncio.Task | None" = None,
        session_id: str = "",
    ) -> tuple[str | None, str | None, str | None, bool | None, str | None]:
        """ツール判定 → 実行 → messages へのツール結果注入を一括で行う。

        ``tool_judge_task`` が渡された場合は chat() が先行起動した tool 判定
        タスクを await して再利用する (直列待ちの短縮)。タスクが例外で終わった
        場合は直接 judge を再実行してフォールバックする (挙動同等性優先)。
        ``session_id`` は search_history のセッション自己参照スコープ限定用
        (``ToolCallJudge._maybe_scope_session_search`` 参照)。

        Returns:
            ``(tool_result_text, tool_name, command, success)``。
            ツール不要時は ``(None, None, None, None)``。``command`` は
            run_command 系の ``tool_args["command"]`` (それ以外は None)、
            ``success`` は実行成功か (出力が "Error:" prefix でない)。
            executable_command 学習 (sleep-time curator) のデータ源になる。
        """
        # 位置で決まる事実は判定経路の有無に関わらず先に確定させる。検索索引に
        # 載っていない進行中セッションを search_history に問い合わせても答えは
        # 出ないため、並び順から機械的に決めた値を根拠として渡す。
        self._append_session_position_fact(messages, conversation, query)

        if self._tool_judge is None or self._tools_registry is None:
            # 判定経路が無いなら precomputed タスクも使えない。残っていれば破棄。
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None, None

        if tool_judge_task is not None:
            try:
                judgement = await tool_judge_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Precomputed tool judge task failed, re-judging: %r", exc,
                )
                judgement = await self._tool_judge.judge(
                    query, self._tools_registry, mode, conversation or [],
                    session_id=session_id,
                )
        else:
            judgement = await self._tool_judge.judge(
                query, self._tools_registry, mode, conversation or [],
                session_id=session_id,
            )
        if not (judgement.tool_needed and judgement.tool_name):
            if getattr(self._tool_judge, "measurement_blocked", False):
                # 実測を試みたが撃てなかった。この状態で base に丸投げすると
                # 測っていない値を断定する (_UNMEASURED_FACT_GUIDANCE 参照)。
                self._append_unmeasured_fact_note(messages)
            return None, None, None, None, None

        command = None
        if isinstance(judgement.tool_args, dict):
            cmd = judgement.tool_args.get("command")
            command = cmd if isinstance(cmd, str) and cmd else None

        tool_result_text = await self._execute_tool(
            judgement, state, query, llm_client, on_step, mode=mode,
            conversation=conversation,
        )
        if tool_result_text is None:
            # 実行されたが結果 None (失敗)。command は penalize 用に返す。
            self._record_tool_call_outcome(query, judgement, False, mode=mode)
            return None, judgement.tool_name, command, False, judgement.source

        # ツール結果を assist で query 連動抽出し、base 文脈には digest を注入して
        # 弱い base の接地負荷を下げる。抽出不能/assist 不在時は raw へ退避 (現挙動)。
        # 戻り値の tool_result_text(raw) は UI 表示用にそのまま保つ。
        if _is_search_history_empty(judgement.tool_name, tool_result_text):
            # 空振りと分かっている結果を assist に要約させても得られるのは
            # 「関連情報なし」の確定だけで、アシスト 1 往復 (実測 1〜2 秒) が
            # 丸ごと無駄になる。実測 2026-07-25 のライブ検証では search_history
            # 11 回中 7 回が空振りだった。判定を code 側で先に済ませ、
            # _SEARCH_HISTORY_NO_INFO_GUIDANCE へ直結する
            # (assist 不在・タイムアウトで digest が None に落ちると raw が
            # 「唯一の事実根拠」枠で base に渡り、進行中の会話履歴まで否定
            # させてしまう経路を塞ぐ効果も兼ねる)。
            prompt_result_text = _NO_RELEVANT_INFO_MESSAGE
            self._append_tool_result_to_last_user(
                messages, judgement.tool_name, prompt_result_text, query=query,
            )
            # 成否は下の通常経路と同じ SSOT (tool_result_succeeded) で決める。
            # ここは「空振りと分かっている結果を digest に通さない」ための
            # 早期 return であって、成否判定を変える意図は無い。以前は success を
            # True 固定で書いており、0 件検索が reward=1.0 で正例として記録される
            # という、まさに tool_result_succeeded が塞いだはずの穴が
            # この early return 経由で復活していた (2026-08-05 ライブ監査で確認)。
            success = tool_result_succeeded(judgement.tool_name, tool_result_text)
            logger.info(
                "Tool executed: %s, result_length=%d, source=%s, success=%s "
                "(empty result: digest skipped)",
                judgement.tool_name, len(tool_result_text), judgement.source,
                success,
            )
            self._record_tool_call_outcome(query, judgement, success, mode=mode)
            return (
                tool_result_text, judgement.tool_name, command, success,
                judgement.source,
            )

        if judgement.tool_name == "calculate":
            # calculate の結果は裸の数値 1 個で、抽出すべき要点は存在しない。
            # digest に通すと assist が質問文の単位を勝手に貼り付ける
            # (実インシデント 2026-07-27: raw "16257.72825" が digest
            # "16257.72825 リットル" になり、cm³ の値がリットルとして回答された)。
            # 抽出の利得ゼロ・捏造リスクありなので assist 1 往復ごと省く。
            digest = None
        elif judgement.tool_name in _ENUMERATIVE_TOOLS:
            # 列挙結果の価値は「その集合が全部である」ことにあり、散文要約は
            # 必ず項目を落とす。落ちた項目を base がもっともらしい名前で埋める
            # ため、決定論的な切り詰め (省略量を明示する) の方が安全
            # (_ENUMERATIVE_TOOLS 参照)。
            digest = None
        else:
            digest = await digest_tool_result(
                self._assist_client,
                query=query,
                tool_name=judgement.tool_name,
                tool_result=_truncate_tool_result(
                    tool_result_text, TOOL_RESULT_MAX_CHARS,
                ),
            )
        if digest is None:
            prompt_result_text = tool_result_text
        elif digest == "" and judgement.tool_name == "search_history":
            # assist が抽出に成功した上で「関連情報なし」と確定したケース。
            # search_history に限り、raw 結果 (無関係な過去セッションの内容等)
            # をそのまま「唯一の事実根拠」として base に渡すと誤って参照・混同
            # されるため (実インシデント: search_history が別セッションの雑談を
            # ヒットし、base がそれを今回の会話の内容として回答した)、raw へは
            # 退避しない。他のツール (calculate 等) は raw が「今回の呼出し
            # そのものの結果」であり無関係な混同のリスクが無いため、assist の
            # digest 誤判定 (false negative) を安全側に倒せるよう raw へ退避する
            # (元の挙動を維持)。
            prompt_result_text = _NO_RELEVANT_INFO_MESSAGE
        elif digest == "":
            prompt_result_text = tool_result_text
        else:
            prompt_result_text = digest
        self._append_tool_result_to_last_user(
            messages, judgement.tool_name, prompt_result_text, query=query,
            tool_args=judgement.tool_args,
        )
        # 「実行できた」ではなく「役に立つ結果が出た」を成否とする (SSOT)。
        # 非ゼロ終了の run_command / 0 件の search_history を成功にすると、
        # executable_command の SemMem 学習と tool_routing の選択圧が汚染される。
        success = tool_result_succeeded(judgement.tool_name, tool_result_text)
        logger.info(
            "Tool executed: %s, result_length=%d, source=%s, success=%s",
            judgement.tool_name, len(tool_result_text), judgement.source,
            success,
        )
        self._record_tool_call_outcome(query, judgement, success, mode=mode)
        return tool_result_text, judgement.tool_name, command, success, judgement.source

    async def process(
        self,
        query: str,
        messages: list[dict],
        llm_client,
        *,
        mode: str = "chat",
        stream: bool = True,
        conversation: list[dict] | None = None,
        max_tokens: int | None = None,
        on_step: StepCallback = None,
        generation_params: GenerationParams | None = None,
        tool_capture: dict | None = None,
        tool_judge_task: "asyncio.Task | None" = None,
        session_id: str = "",
    ) -> DeliberativeResponse | AsyncIterator[str]:
        """Deliberative 層で LLM 推論を実行

        Args:
            query: ユーザーのクエリ
            messages: build_messages() で組み立て済みのメッセージ配列
            llm_client: LocalClient インスタンス
            mode: 動作モード ('chat' | 'coding')
            stream: ストリーミング応答を返すか
            conversation: 直近の会話履歴（ツール判定の精度向上用）
            max_tokens: 最大生成トークン数
            on_step: ステップ進行コールバック (step_dict) -> None
            generation_params: モード別生成パラメータ（temperature, top_p 等）
            tool_judge_task: chat() が先行起動した tool 判定タスク (並列化時)。
                None なら判定をここで直列実行する。

        Returns:
            stream=False: DeliberativeResponse
            stream=True: AsyncIterator[str]（生トークンのイテレータ）
        """
        logger.debug(
            "process: query=%r, messages=%d, stream=%s, mode=%s",
            query[:50], len(messages), stream, mode,
        )

        state = self._init_deliberative_state(mode)
        (
            tool_result_text, tool_name_used, tool_command, tool_success,
            tool_command_source,
        ) = await self._judge_and_execute_tool(
            query, mode, conversation, messages, llm_client, state, on_step,
            tool_judge_task=tool_judge_task, session_id=session_id,
        )

        # MDP トレース: tool 判定/実行を 1 step エピソードとして記録する。
        # ``_judge_and_execute_tool`` は stream 返却前に完了済みのため、ここで
        # begin→step→end を同期完結できる (生成は応答であり agent action ではない)。
        self._trace_tool_episode(
            session_id, mode, query, tool_name_used, tool_result_text, tool_success,
        )

        # streaming 経路は DeliberativeResponse を返さないため、command を
        # 呼出側へ渡す唯一の経路として tool_capture dict に書き出す。
        # ``_judge_and_execute_tool`` は iterator 返却前に完了するので、
        # ``await process(...)`` 完了時点で dict は確定している。
        if tool_capture is not None:
            tool_capture["command"] = tool_command
            tool_capture["command_name"] = tool_name_used if tool_command else None
            tool_capture["success"] = tool_success
            tool_capture["command_source"] = tool_command_source
            # command_name は run_command 系でしか埋まらない。ツール種別と
            # 「役に立つ結果が出たか」は全ツール共通の品質シグナルなので別枠で
            # 渡す (outcome JSONL の quality_signals に載せて、空振りツールに
            # 頼ったターンを事後に切り分けられるようにする)。
            tool_capture["tool_name"] = tool_name_used
            tool_capture["tool_success"] = tool_success

        # ツール結果に基づく接地回答は創作不要。chat 既定 0.7 のままだと weak base が
        # 非決定的に拒否/話題混同しやすい (実機: ニュースで 0.7→~25%拒否、0.2→安定)。
        # ツール使用ターンのみ温度を下げて決定性を上げる (既に低ければ据え置く)。
        if tool_result_text is not None:
            gp = dict(generation_params or {})
            gp["temperature"] = min(
                gp.get("temperature", TOOL_GROUNDED_TEMPERATURE),
                TOOL_GROUNDED_TEMPERATURE,
            )
            generation_params = gp

        # リマインダー注入
        messages = self.reminder_system.inject(messages, state)
        logger.debug(
            "Messages finalized: %d messages, total_chars=%d",
            len(messages),
            sum(len(m.get("content", "")) for m in messages),
        )

        # 「中身をそのまま見せて」型はモデルに通さず決定論的に返す。
        verbatim = verbatim_file_echo(
            query, tool_name_used, tool_result_text,
        )
        if verbatim is not None:
            logger.info(
                "Verbatim file echo (bypassing generation): %d chars",
                len(verbatim),
            )
            if stream:
                return _iterate_once(verbatim)
            return DeliberativeResponse(
                content=verbatim,
                tool_name=tool_name_used,
                tool_result=tool_result_text,
                tool_command=tool_command,
                tool_command_success=tool_success,
                tool_command_source=tool_command_source,
            )

        if stream:
            return self._stream_response(
                messages, llm_client, max_tokens,
                tool_result=tool_result_text, tool_name=tool_name_used,
                generation_params=generation_params,
            )
        return await self._sync_response(
            messages, llm_client, max_tokens,
            tool_result=tool_result_text, tool_name=tool_name_used,
            tool_command=tool_command, tool_command_success=tool_success,
            tool_command_source=tool_command_source,
            generation_params=generation_params,
        )

    def _trace_tool_episode(
        self,
        session_id: str,
        mode: str,
        query: str,
        tool_name: str | None,
        tool_result: str | None,
        tool_success: bool | None,
    ) -> None:
        """deliberative の tool 実行を 1 step MDP エピソードとして記録する。

        tracer 未注入 (通常起動) / session_id 空 / tool 未実行なら no-op。
        tool を実行したターンのみを記録対象とし、no_tool ルーティング signal は
        record_response の tool_routing_success (経験記録) 側に委ねて episodic LTM
        の膨張を避ける。

        reward と outcome は別軸で付ける:

        - ``reward``: 「役に立つ結果が出たか」(``tool_success``)。空振り検索も 0 に
          倒し、無駄なツール選択が正例として強化されるのを防ぐ。
        - ``outcome``: 「ツールが壊れたか」。0 件ヒットは正常動作なので
          ``success`` のままにする。``partial`` にすると SemMem の
          ``mdp_trace`` 抽出が空振りのたびに failure_pattern ファクトを作り、
          ストアが膨張してしまう。
        """
        tracer = self._agent_tracer
        if tracer is None or not session_id or tool_result is None:
            return
        from backend.free.agent.agent_tracer import MDPStep

        reward = 1.0 if tool_success else 0.0
        errored = is_tool_error(tool_result)
        try:
            episode_id = tracer.begin_episode(session_id, mode)
            tracer.record_step(episode_id, MDPStep(
                step_index=0,
                state={"query": query[:200], "agent_layer": "deliberative"},
                action=tool_name or "tool",
                observation=tool_result[:200],
                reward=reward,
            ))
            tracer.end_episode(episode_id, "partial" if errored else "success")
            tracer.cleanup_episode(episode_id)
        except Exception as exc:
            logger.warning("deliberative MDP trace failed (continuing): %s", exc)

    async def _sync_response(
        self,
        messages: list[dict],
        llm_client,
        max_tokens: int | None = None,
        *,
        tool_result: str | None = None,
        tool_name: str | None = None,
        tool_command: str | None = None,
        tool_command_success: bool | None = None,
        tool_command_source: str | None = None,
        generation_params: GenerationParams | None = None,
    ) -> DeliberativeResponse:
        """非ストリーミング応答"""
        kwargs: dict = {"stream": False, "id_slot": llm_client.chat_slot}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # モード別生成パラメータを適用
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "repetition_penalty"):
                if k in generation_params:
                    kwargs[k] = generation_params[k]
        result = await llm_client.generate(messages, **kwargs)
        content = result["choices"][0]["message"]["content"]
        logger.info("Deliberative sync response: %d chars", len(content))
        return DeliberativeResponse(
            content=content,
            tool_result=tool_result,
            tool_name=tool_name,
            tool_command=tool_command,
            tool_command_success=tool_command_success,
            tool_command_source=tool_command_source,
        )

    async def _stream_response(
        self,
        messages: list[dict],
        llm_client,
        max_tokens: int | None = None,
        *,
        tool_result: str | None = None,  # noqa: ARG002
        tool_name: str | None = None,  # noqa: ARG002
        generation_params: GenerationParams | None = None,
    ) -> AsyncIterator[str]:
        """ストリーミング応答（生トークンのイテレータを返す）"""
        kwargs: dict = {"stream": True, "id_slot": llm_client.chat_slot}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # モード別生成パラメータを適用
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "repetition_penalty"):
                if k in generation_params:
                    kwargs[k] = generation_params[k]
        token_gen = await llm_client.generate(messages, **kwargs)
        tokens_generated = 0
        async for token in token_gen:
            tokens_generated += 1
            yield token
        logger.debug(
            "Deliberative stream complete: tokens_generated=%d",
            tokens_generated,
        )

    async def _ensure_write_file_content(
        self,
        tool_name: str,
        tool_args: dict,
        query: str,
        llm_client,
        on_step: StepCallback,
        conversation: list[dict] | None = None,
    ) -> None:
        """`write_file` の `content` が空なら LLM で生成して `tool_args` に注入する。

        ``_generate_content`` がエラーセンチネル文字列を返した場合は
        ``tool_args["content"]`` に注入せず、呼び出し元 ``_execute_tool``
        で実行スキップさせる。
        """
        if tool_name != "write_file" or tool_args.get("content"):
            return
        file_path = tool_args.get("file_path", "")
        if on_step:
            on_step({
                "type": "tool_call",
                "detail": f"コンテンツ生成中 → {file_path}",
                "status": "running",
            })
        content = await self._generate_content(
            query, llm_client, conversation=conversation,
        )
        if content.startswith("(Content generation failed:"):
            logger.warning(
                "Deliberative: content generation failed for %s; "
                "skipping write_file injection",
                file_path,
            )
            if on_step:
                on_step({
                    "type": "tool_call",
                    "detail": f"write_file: コンテンツ生成失敗 → {file_path}",
                    "status": "failed",
                })
            return
        rejection = generated_content_rejection(content, file_path, query)
        if rejection:
            logger.warning(
                "Deliberative: generated content rejected (%s); skipping "
                "write_file injection: %r",
                rejection, content[:120],
            )
            if on_step:
                on_step({
                    "type": "tool_call",
                    "detail": f"write_file: コンテンツ生成失敗（{rejection}） → {file_path}",
                    "status": "failed",
                })
            return
        tool_args["content"] = content
        logger.info(
            "Content generated for write_file: %d chars → %s",
            len(content), file_path,
        )

    async def _run_tool_with_handling(
        self,
        tool_name: str,
        tool_args: dict,
        state: AgentState,
        on_step: StepCallback,
    ) -> str:
        """登録済みツールを timeout 付きで実行し、結果テキスト or エラーを返す。

        正常終了 / TimeoutError / 一般例外をそれぞれ handling し、
        対応する step フレームを emit する。`finally` で `state.pending_*` をクリア。
        """
        state.pending_tool = tool_name
        state.pending_args = tool_args
        try:
            timeout_sec = self._tools_registry.timeout_for(
                tool_name, TOOL_EXECUTION_TIMEOUT_SEC,
            )
            result = await asyncio.wait_for(
                self._tools_registry.execute(tool_name, **tool_args),
                timeout=timeout_sec,
            )
            result_text = str(result)
            state.on_tool_success(tool_name)
            logger.info("Tool executed successfully: %s", tool_name)
            _emit_tool_result_step(on_step, tool_name, result_text)
            return result_text
        except asyncio.TimeoutError:
            error_text = f"Error: tool execution timed out after {timeout_sec}s"
            state.on_tool_failure(tool_name, error_text)
            logger.warning(
                "Tool execution timed out: %s (%.0fs)", tool_name, timeout_sec,
            )
            _emit_tool_failure_step(on_step, tool_name, error_text)
            return error_text
        except Exception as e:
            error_text = f"Error: {e}"
            state.on_tool_failure(tool_name, str(e))
            logger.warning("Tool execution failed: %s - %s", tool_name, e)
            _emit_tool_failure_step(on_step, tool_name, error_text)
            return error_text
        finally:
            state.pending_tool = None
            state.pending_args = {}

    async def _execute_tool(
        self,
        judgement: ToolJudgement,
        state: AgentState,
        query: str,
        llm_client,
        on_step: StepCallback = None,
        mode: str | None = None,
        conversation: list[dict] | None = None,
    ) -> str | None:
        """ToolJudgement に基づいてツールを実行

        write_file でコンテンツが不足している場合は、LLM にプレーンテキストで
        コンテンツを生成させてから実行する。

        Returns:
            ツール実行結果のテキスト。ツールが見つからない/実行失敗時は None。
        """
        if self._tools_registry is None:
            return None

        tool_name = judgement.tool_name
        # tool_args は dict 契約だが、アシスト応答の機械修復経路で非 dict が
        # 紛れ込むことがあるため防御的にガードする (cf. _judge_and_execute_tool)。
        raw_args = judgement.tool_args
        tool_args = dict(raw_args) if isinstance(raw_args, dict) else {}  # コピー

        if not self._tools_registry.has(tool_name):
            logger.warning("Tool not found: %s", tool_name)
            return None

        # ToolDefinition.modes は元々 get_descriptions_text() (LLM 向け説明文) の
        # フィルタ用にしか参照されておらず、実行時には無視されていた。ルールベース
        # 判定 (tool_call_judge) が誤トリガーで coding 専用ツールを選んでも、ここで
        # 弾かなければ chat モードでも実行されてしまう (search_code の CWD 全域
        # os.walk がイベントループを長時間ブロックした実インシデントの直接原因)。
        tool_def = self._tools_registry.get(tool_name)
        effective_mode = mode if mode is not None else self._mode
        if tool_def is not None and effective_mode not in tool_def.modes:
            logger.warning(
                "Tool not allowed in mode=%s: %s (allowed modes: %s)",
                effective_mode, tool_name, tool_def.modes,
            )
            return None

        path_error = _check_path_traversal(
            tool_args.get("file_path", ""), tool_name,
        )
        if path_error:
            return path_error

        await self._ensure_write_file_content(
            tool_name, tool_args, query, llm_client, on_step,
            conversation=conversation,
        )

        # write_file で content が依然空 → LLM 生成失敗。誤実行を防ぐため
        # tool_args をそのまま流さずエラー文字列を返してスキップする。
        if tool_name == "write_file" and not tool_args.get("content"):
            error_text = "Error: content generation failed"
            state.on_tool_failure(tool_name, error_text)
            return error_text

        # 必須引数チェック（必須パラメータが空の場合を防止）。running フレーム
        # の emit より前に行う — emit 後に skip すると完了フレームが出ず
        # UI に空ステップ (running のまま) が残る (2026-07-21 ライブ検証
        # ターン35 の引数なし calculate で実発生)。
        if tool_def and tool_def.parameters and not tool_args:
            logger.warning(
                "Tool %s requires args but none provided, skipping", tool_name,
            )
            return None

        _emit_tool_running_step(on_step, tool_name, tool_args)

        return await self._run_tool_with_handling(
            tool_name, tool_args, state, on_step,
        )

    async def _generate_content(
        self,
        query: str,
        llm_client,
        conversation: list[dict] | None = None,
    ) -> str:
        """write_file 用のコンテンツを LLM にプレーンテキストで生成させる

        依頼が会話中の成果物を指す場合 (「この案内文を保存して」「さっきの文章を
        ファイルに」) は、クエリ単体では何を書くべきかが決まらない。直近の会話を
        添えないと、モデルは書くものが無いまま出力を埋めようとして無関係な
        テキストを捏造する (実インシデント 2026-07-27: 直前に作った夏祭りの
        案内文を保存させたら、ファイルに ``## Example 1 / User: ... /
        Assistant: ...`` という架空の Q&A が書き込まれた)。
        """
        messages: list[dict] = [
            # 出力言語指示 (locale 追従) を毎回組み立てて付加する
            {
                "role": "system",
                "content": f"{_CONTENT_GEN_PROMPT}{content_language_directive()}",
            },
        ]
        messages.extend(_recent_context_messages(conversation))
        messages.append({"role": "user", "content": query})
        try:
            result = await llm_client.generate(
                messages, stream=False,
                max_tokens=self._content_max_tokens,
                id_slot=llm_client.chat_slot,
            )
            content = result["choices"][0]["message"]["content"].strip()
            content = strip_markdown_wrapper(content)
            logger.debug("Content generated: %d chars", len(content))
            return content
        except Exception as e:
            logger.error("Content generation failed: %s", e)
            return f"(Content generation failed: {e})"


#: `_generate_content` に添える直近会話の上限 (メッセージ数 / 1 件あたり文字数)。
#: 保存対象は直前に作った成果物であることが大半なので、深い履歴は要らない。
_CONTENT_CONTEXT_MESSAGES = 4
_CONTENT_CONTEXT_CHARS = 2000


def _recent_context_messages(conversation: list[dict] | None) -> list[dict]:
    """直近会話を content 生成用メッセージ列に整形する (純粋関数)。"""
    if not conversation:
        return []
    recent = [
        m for m in conversation[-_CONTENT_CONTEXT_MESSAGES:]
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]
    return [
        {"role": m["role"], "content": m["content"][:_CONTENT_CONTEXT_CHARS]}
        for m in recent
    ]


def _truncate_tool_result(text: str, max_chars: int) -> str:
    """ツール結果が max_chars を超える場合、先頭と末尾を残して切り詰める"""
    if len(text) <= max_chars:
        return text
    head_size = int(max_chars * TOOL_RESULT_HEAD_RATIO)
    tail_size = max_chars - head_size - TOOL_RESULT_OMISSION_CHARS
    omitted = len(text) - head_size - tail_size
    return (
        text[:head_size]
        + f"\n\n... ({omitted} chars omitted) ...\n\n"
        + text[-tail_size:]
    )
