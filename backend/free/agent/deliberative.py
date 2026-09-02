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
from backend.free.agent.tool_judge_guards import _STATE_CHANGING_TOOL_NAMES
from backend.free.agent.tool_judge_history import asks_about_past_conversation
from backend.free.agent.tools.builtin import (
    SEARCH_HISTORY_NO_RESULTS_PREFIX,
    _check_path_traversal as check_builtin_path_traversal,
)
from backend.free.constants import READ_FILE_META_PREFIX
from backend.free.core.session_mode import is_create_mode
from backend.free.agent.issue_ledger import count_kind, format_issues
from backend.free.agent.tool_ledger import format_ledger, record_current
from backend.free.core.intent_vocab import (
    assistant_code_blocks,
    prior_code_block_request,
    names_file_target,
    memory_architecture_question,
    self_learning_question,
    model_identity_question,
    own_process_question,
    self_assessment_question,
    persist_request,
    resolve_session_position_message,
    session_position_kind,
    tool_inventory_question,
    unused_tool_question,
    unverified_claim_numbers,
)
from backend.free.core.turn_text import TOOL_RESULT_HEADER, append_to_last_user
from backend.config import resolve_context_size_for_mode
from backend.i18n_helper import prompt_locale
from backend.free.api.chat.chat_constants import (
    CONTENT_MAX_TOKENS_MIN, CONTENT_SYSTEM_RESERVE,
    TOOL_EXECUTION_TIMEOUT_SEC, TOOL_GROUNDED_TEMPERATURE,
    TOOL_RESULT_MAX_CHARS,
    TOOL_RESULT_HEAD_RATIO, TOOL_RESULT_OMISSION_CHARS,
)
from backend.free.api.chat.chat_types import GenerationParams, StepCallback
from backend.log_config import get_logger

logger = get_logger("agent.deliberative")


def _localized(table: dict[str, str]) -> str:
    """``i18n.prompt_locale`` 別の固定文を引く (未知 locale は ja)。

    注記の文言は ``_X_GUIDANCE`` (ja、既存名) と ``_X_GUIDANCES`` (locale 辞書)
    の対で持つ。ja は既存の文字列と同一 (既定出力は不変)。
    """
    return table.get(prompt_locale(), table["ja"])


#: ツール実行の既定の最大ホップ数。
#:
#: 1 ターン 1 ツールだと「書いてから読み直して確認する」型の依頼が構造的に
#: 完了できない。実インシデント (2026-08-08 ライブ監査 ターン6): 「同じファイルに
#: 3 行追記して、もう一度読み取って行数を報告して」で追記だけが走り、base は
#: 実行していない読み取りの結果まで書き出した。
#:
#: 一般的な連鎖 (計画が要るもの) は meta_cognitive の担当で、ここは
#: **決定論で 2 手目が決まる場合だけ** を拾う。2 手目の制約は
#: ``_maybe_follow_up_tool`` を参照。
DEFAULT_MAX_TOOL_HOPS = 2

# write_file でコンテンツ生成が必要な場合のプロンプト
_CONTENT_GEN_PROMPT = """\
Generate the requested content below. Output ONLY the content itself, \
no explanations, no markdown fences, no JSON, no surrounding text.
"""

# 空振りと確定したツール結果の代わりに渡すプレースホルダ。無関係な内容を
# 「唯一の事実根拠」として base に読ませないための安全な代替文言。
_NO_RELEVANT_INFO_MESSAGE = "（ツールを実行しましたが、今回の質問に関連する情報は見つかりませんでした）"
_NO_RELEVANT_INFO_MESSAGES: dict[str, str] = {
    "ja": _NO_RELEVANT_INFO_MESSAGE,
    "en": (
        "(The tool was run, but no information relevant to this question "
        "was found)"
    ),
}

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
_UNMEASURED_FACT_GUIDANCES: dict[str, str] = {
    "ja": _UNMEASURED_FACT_GUIDANCE,
    "en": (
        "\n\nThis question can only be answered by actually inspecting this "
        "execution environment, but no tool that inspects the environment was "
        "run this turn, so no measured value exists. Therefore do not state a "
        "specific number, model number, version, or capacity. If the user has "
        "already given a value in the conversation or memory, you may state it "
        "with its source; otherwise say honestly that it was not checked this "
        "time and give one way the user can check it themselves. Do not say a "
        "tool was run or attempted when it was not. Do not state guesses as "
        "facts."
    ),
}

#: 過去の会話について訊かれたが、履歴検索を 1 度も実行しなかった場合の文言。
#:
#: 「実行していないツールについて実行したと述べない」は
#: :data:`_UNMEASURED_FACT_GUIDANCE` にもあるが、あちらは実行環境の実測が
#: 対象で、履歴参照のターンには注記そのものが付かない。過去会話の **日時** は
#: 検索結果にしか無いため、撃たなかったターンでは必ず捏造になる。
_HISTORY_NOT_SEARCHED_GUIDANCE = (
    "\n\n確定事実: このターンでは過去の会話を検索するツールを 1 つも実行して"
    "いない。したがって「検索しました」「記録を確認しました」のように"
    "**調べた体で述べてはならない**。"
    "過去の会話が行われた日付・時刻は検索結果にしか無いので、"
    "**具体的な日付を述べてはならない** (推測した日付は必ず誤りになる)。"
    "いま会話に残っている範囲から答えられることは、その旨を明示して答えてよい。"
    "残っていないなら「確認できていない」と正直に伝えること。"
)
_HISTORY_NOT_SEARCHED_GUIDANCES: dict[str, str] = {
    "ja": _HISTORY_NOT_SEARCHED_GUIDANCE,
    "en": (
        "\n\nEstablished fact: no tool that searches past conversations was "
        "run this turn. Therefore do not speak as if you had looked something "
        "up (\"I searched\", \"I checked the records\"). The dates and times "
        "of past conversations exist only in search results, so do not state "
        "a specific date (a guessed date is always wrong). Whatever can be "
        "answered from what remains in this conversation may be answered, "
        "saying so explicitly. If it is not there, say honestly that it has "
        "not been confirmed."
    ),
}

#: 状態を変える依頼だったが、実行できるツールが 1 つも無かった場合の文言。
#:
#: ``_UNMEASURED_FACT_GUIDANCE`` (値を測っていない) とは別に要る。あちらは
#: 「数値を断定するな」であって「やっていないことをやったと言うな」ではない。
#:
#: 実インシデント (2026-08-08 ライブ監査 ターン6): 「同じファイルに 3 行追記して、
#: もう一度読み取って行数を報告して」に対し、ネイティブ層は
#: ``echo "2 行目" >> ...`` を選んだが chat の読み取り専用ツールは python
#: インタプリタしか許さないため正しく拒否された。その後 base は「追記しました」
#: と述べ、**存在しない 4 行のファイル内容まで書き出した**。実ファイルは 1 行の
#: まま無変更だった。
#: 保存を求められたが保存先が特定できず、ツールを 1 つも撃てなかった場合の文言。
#:
#: ``_UNPERFORMED_ACTION_GUIDANCE`` とは理由が違う。あちらは「その操作を実行
#: できるツールが無い」で確定しており、他の理由を推測することを禁じている。
#: 保存は ``write_file`` が実在するので、能力が無いのではなく **宛先が無い**。
#:
#: 実インシデント (2026-08-22 ライブ監査 2 回目 ターン 252):
#: 「ファイルに保存しておいて。」に「ファイル保存機能は利用できないため、
#: 保存できません。」と回答した。同じ会話のターン 122 で ``write_file`` が
#: 成功しており、能力が無いという説明そのものが誤りだった。
_WRITE_TARGET_UNKNOWN_GUIDANCE = (
    "\n\nこの依頼はファイルへの保存を求めているが、保存先のパスが特定できず"
    "**何も実行していない**。「保存しました」と完了を報告してはならない。"
    "また、**保存する機能が無い / 利用できない とも述べてはならない** — "
    "保存自体はこのモードで実行できる。実行していない理由は保存先が"
    "分からないことだけである。保存先の完全なパス (例: E:\\tmp\\メモ.txt) を"
    "1 文で尋ねること。"
)
_WRITE_TARGET_UNKNOWN_GUIDANCES: dict[str, str] = {
    "ja": _WRITE_TARGET_UNKNOWN_GUIDANCE,
    "en": (
        "\n\nThis request asks to save to a file, but the destination path could "
        "not be determined and **nothing was executed**. Do not report "
        "completion (\"saved\"). Also **do not say that saving is unavailable or "
        "unsupported** - saving can be performed in this mode; the only reason "
        "nothing ran is that the destination is unknown. Ask for the full "
        "destination path (e.g. E:\\tmp\\notes.txt) in one sentence."
    ),
}

#: 直前の保存依頼に引数だけを与えたターンで、ツールが 1 度も実行されなかった
#: 場合の注記。``_UNPERFORMED_ACTION_GUIDANCE`` (ツールが無い) とも
#: ``_WRITE_TARGET_UNKNOWN_GUIDANCE`` (宛先が分からない) とも理由が違う —
#: 宛先はユーザーが今まさに与えており、**判定がそれを保存依頼と見なさなかった**
#: だけ。だから「保存先を教えて」と聞き返すのも誤りになる。
_PENDING_WRITE_NOT_EXECUTED_GUIDANCE = (
    "\n\nこのターンでは書き込みツールが **1 度も実行されていない**。"
    "したがって「保存しました」「書き込みました」「作成しました」等の"
    "完了報告をしてはならず、保存先のパスを実在するものとして述べても"
    "ならない。直前の保存依頼はまだ実行されていない状態である。"
    "指定されたファイル名でよいかを 1 文で確認し、"
    "改めて保存の指示を求めること。"
)
_PENDING_WRITE_NOT_EXECUTED_GUIDANCES: dict[str, str] = {
    "ja": _PENDING_WRITE_NOT_EXECUTED_GUIDANCE,
    "en": (
        "\n\nNo write tool was executed this turn - **not once**. Therefore do "
        "not report completion (\"saved\", \"written\", \"created\") and do not "
        "present the destination path as an existing file. The previous save "
        "request is still pending. Confirm in one sentence whether the given "
        "file name is acceptable and ask for the save instruction again."
    ),
}

_UNPERFORMED_ACTION_GUIDANCE = (
    "\n\nこの依頼はファイルやシステムの状態を変える操作を含むが、"
    "今回のターンではその操作を実行できるツールが無く、**何も実行していない**。"
    "したがって「書き込みました」「追記しました」「作成しました」"
    "「削除しました」「存在しません」のように"
    "完了を報告してはならない。変更後の内容や変更後の状態を推測して"
    "提示してもならない。"
    "実行できなかったことを率直に伝え、可能な代替 (クリエイトモードで行う / "
    "パスが曖昧な場合は完全なパスを指定して依頼し直す 等) を 1 つ示すこと。"
    "実行できなかった理由は上記のとおり「その操作を実行できるツールが無い」で"
    "確定している。権限が無い・パス指定が足りないなど、**ここに書かれていない"
    "理由を推測して述べない** (実インシデント 2026-08-15: 完全なパスを与えた"
    "削除依頼に「ファイルの削除には権限が必要です。完全なパスを指定して削除を"
    "依頼してください」と、誤った理由と成立しない代替を提示した)。"
)
_UNPERFORMED_ACTION_GUIDANCES: dict[str, str] = {
    "ja": _UNPERFORMED_ACTION_GUIDANCE,
    "en": (
        "\n\nThis request includes an operation that changes files or system "
        "state, but no tool able to perform it was available this turn and "
        "**nothing was executed**. Therefore do not report completion "
        "(\"written\", \"appended\", \"created\", \"deleted\", \"does not exist\"), "
        "and do not present a guessed post-change content or state. State "
        "plainly that it could not be performed and offer one alternative "
        "(do it in create mode / re-request with the full path if the path was "
        "ambiguous, etc.). The reason is fixed as stated above - no tool can "
        "perform this operation. Do not invent other reasons (permissions, a "
        "missing path, ...) that are not written here."
    ),
}

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
_SEARCH_HISTORY_NO_INFO_GUIDANCES: dict[str, str] = {
    "ja": _SEARCH_HISTORY_NO_INFO_GUIDANCE,
    "en": (
        "search_history is a tool that searches the records of past, separate "
        "sessions. The \"no relevant information was found\" above only means "
        "nothing was found in past sessions; it does not negate information "
        "already stated in this ongoing conversation (including the user's "
        "most recent message). If the conversation history contains the "
        "information, answer concretely from it regardless of the search "
        "result. Only when it is truly absent from the conversation history "
        "as well may you answer \"I don't know\"."
    ),
}

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
_SEARCH_HISTORY_RESULT_GUIDANCES: dict[str, str] = {
    "ja": _SEARCH_HISTORY_RESULT_GUIDANCE,
    "en": (
        "The ## ツール実行結果 above is real data the system actually retrieved "
        "with the search_history tool. However, it is a record of a past "
        "session separate from this ongoing conversation; the names, topics "
        "and plans in it belong to that earlier, different conversation. The "
        "current user is not necessarily the same person, so do not address "
        "the user by a name found in the results or assert it as the user's "
        "attribute. Information already stated in this ongoing conversation "
        "(including the user's most recent message) takes precedence; the "
        "search results do not negate or overwrite it. If the conversation "
        "history has the answer, use it, and treat the search results only as "
        "supplementary context about past events. Do not say you cannot "
        "browse or cannot retrieve. Do not invent numbers or facts absent "
        "from both the results and the conversation history."
    ),
}

# calculate 専用のグラウンディング文言。calculate の結果は裸の数値であり、
# 何をどの単位で計算したかの情報を持たない。汎用の「唯一の事実根拠」枠だけを
# 付けると base が (a) 質問文の単位をそのまま数値に貼り付ける
# (b) 式に無い係数を後付けで説明する、という 2 つの捏造を起こす
# (実インシデント 2026-07-27: 「直径30cm・深さ25cm の鉢の土は何リットル？」に
# 対し aux が cm 単位の式 3.14159*(30/2)**2*(25-2) を組み、結果 16257.7 cm³ が
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
_CALCULATE_RESULT_GUIDANCES: dict[str, str] = {
    "ja": _CALCULATE_RESULT_GUIDANCE,
    "en": (
        "The ## ツール実行結果 above is the expression the system actually "
        "evaluated with the calculate tool and its exact result. Use the "
        "number as is. The unit of the result follows the units of the "
        "numbers entered into the expression (e.g. a volume obtained by "
        "multiplying lengths measured in cm is in cm³, not litres). Division "
        "divides the units too (e.g. km ÷ (km/h) = hours, yen ÷ items = "
        "yen/item). In the answer, always attach the unit to the number "
        "(e.g. the answer to \"how many hours\" is written \"it takes 5 "
        "hours\", unit included). If the user asked in a different unit and "
        "the expression contains no conversion, none was performed: either "
        "do and show the necessary conversion yourself or state that the "
        "unit differs. Do not cite coefficients, densities or corrections "
        "that do not appear in the expression (e.g. \"multiplied by a "
        "density of 1.3\") as the basis of the calculation; only the "
        "expression above was evaluated."
    ),
}


def _previous_user_text(
    conversation: list[dict] | None, current_query: str = "",
) -> str:
    """会話履歴から **1 つ前の user 発話** を返す (無ければ空文字列)。

    ``conversation`` (= ``history``) の末尾には **今回のターン自身** が入って
    いることがある。素直に末尾の user 発話を返すと今回のクエリが返り、
    「直前が保存依頼だったか」の判定が常に今回の発話を見てしまう
    (2026-08-26 の実機検証で ``_append_pending_write_note`` が発火しなかった
    原因)。今回のクエリと一致するものは飛ばす。
    """
    current = (current_query or "").strip()
    for message in reversed(conversation or ()):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        if current and text.strip() == current:
            continue
        return text
    return ""


def _unexplained_numbers_note(numbers: tuple[str, ...]) -> str:
    """式に含まれる「対話に無い数値」の開示を求める注記を作る (純粋関数)。

    以前は式ごと no_tool へ格下げしていたが、格下げの落ち先は base の暗算で、
    式が見えないぶん誤りに気づけなかった。式を残すかわりに、説明できない値を
    名指しして出所を述べさせる (``ToolCallJudge._suppress_ungrounded_calculate``
    のコメント参照)。
    """
    listed = "、".join(numbers)
    return _localized(_UNEXPLAINED_NUMBERS_NOTES).format(listed=listed)


_UNEXPLAINED_NUMBERS_NOTES: dict[str, str] = {
    "ja": (
        "ただし式中の {listed} は、ユーザーの依頼文にもここまでの会話にも"
        "現れていない値である。この値が何を指すのか (単位・1件あたりの量・"
        "換算率など) を答えの中で明示すること。根拠を示せない場合は、"
        "その値を仮定として断ったうえで、正しい値をユーザーに確認すること。"
    ),
    "en": (
        "However, {listed} in the expression does not appear in the user's "
        "request or anywhere in the conversation so far. State explicitly in "
        "the answer what this value stands for (unit, amount per item, "
        "conversion rate, etc.). If you cannot justify it, present it as an "
        "assumption and ask the user to confirm the correct value."
    ),
}

#: read_file 結果に併記する決定論の実測値 (文字数 / 行数)。
_FILE_MEASUREMENT_NOTES: dict[str, str] = {
    "ja": "[この結果の実測値: {chars} 文字 / {lines} 行]",
    "en": "[Measured size of this result: {chars} characters / {lines} lines]",
}

# --- 決定論の事実注記 (``_append_*_fact``) の locale 別文言 -------------------
# ja は従来の文字列と同一 (既定出力は不変)。``_localized`` で引く。

#: 押し出し後でも WM が保持するセッション先頭から ``first`` を確定した注記。
_SESSION_HEAD_PINNED_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: この会話でユーザーが最初に送ったメッセージは"
        "「{head}」である。これは会話の並び順から機械的に確定した値"
        "なので、この値をそのまま答えること。"
        "(この会話の冒頭 {evicted_turns} 件は文脈の上限を超えて"
        "表示できていないが、最初の発言はこの値で確定している。"
        "上の値を答えとして述べること)"
    ),
    "en": (
        "\n\nEstablished fact: the first message the user sent in this "
        "conversation is \"{head}\". This value was determined mechanically "
        "from the order of the conversation, so answer with it as is. (The "
        "first {evicted_turns} message(s) of this conversation exceed the "
        "context limit and are not shown, but the first message is fixed to "
        "this value. State the value above as the answer.)"
    ),
}
#: 押し出しがあり、先頭も保持されていないため ``first`` を確定できない注記。
_SESSION_HEAD_UNKNOWN_NOTES: dict[str, str] = {
    "ja": (
        "\n\n注記: この会話の冒頭 {evicted_turns} 件は文脈の上限を超えて"
        "参照できない。したがって「会話で最初に送ったメッセージ」は"
        "確定できない。見えている範囲の先頭を会話の先頭として断定せず、"
        "参照できる範囲が限られていることを述べたうえで、"
        "見えている中で最も古い発言を『確認できる範囲での最古』として"
        "示すこと。"
    ),
    "en": (
        "\n\nNote: the first {evicted_turns} message(s) of this conversation "
        "exceed the context limit and cannot be referenced. Therefore \"the "
        "first message sent in the conversation\" cannot be determined. Do "
        "not assert the head of the visible range as the start of the "
        "conversation; state that the referenceable range is limited, and "
        "present the oldest visible message as \"the oldest within the "
        "confirmable range\"."
    ),
}
#: 窓内で確定した位置 (first / last) の注記。``label`` は下の辞書から引く。
_SESSION_POSITION_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: この会話でユーザーが{label}に送ったメッセージは"
        "「{target}」である。これは会話の並び順から機械的に確定した値なので、"
        "この値をそのまま答えること。"
    ),
    "en": (
        "\n\nEstablished fact: the {label} message the user sent in this "
        "conversation is \"{target}\". This value was determined mechanically "
        "from the order of the conversation, so answer with it as is."
    ),
}
_SESSION_POSITION_LABEL_FIRST: dict[str, str] = {"ja": "最初", "en": "first"}
_SESSION_POSITION_LABEL_LAST: dict[str, str] = {"ja": "直近", "en": "most recent"}

#: 「過去に書いたコードをそのまま見せろ」に実物を渡す注記 (末尾にコード)。
_PRIOR_CODE_BLOCK_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: ユーザーが指しているのは以下のコードである。"
        "これは会話履歴から機械的に取り出した実物なので、"
        "**一字一句そのまま** 出力すること。書き直さないこと。\n"
    ),
    "en": (
        "\n\nEstablished fact: the user is referring to the code below. It "
        "was extracted mechanically from the conversation history, so output "
        "it **verbatim, character for character**. Do not rewrite it.\n"
    ),
}

#: 自己評価の問いへ渡す不首尾台帳 (記録あり)。``issues`` / ``corrections``。
_ISSUE_LEDGER_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: この会話でシステムが観測した不首尾は以下がすべて"
        "である。これは発生時に機械的に記録した値なので、この記録に"
        "基づいて答えること。ここにある項目を「無かった」と述べては"
        "ならない。\n{issues}"
        "\n(うちユーザーによる訂正は {corrections} 回)"
        "\nなお、この記録はシステムが観測できた範囲であり、"
        "回答内容そのものの誤り (計算違い等) は含まれない。"
        "「記録上は以上」と断ったうえで答えること。"
    ),
    "en": (
        "\n\nEstablished fact: the following is the complete list of failures "
        "the system observed in this conversation. It was recorded "
        "mechanically as they happened, so answer based on this record. Do "
        "not say that any item here \"did not happen\".\n{issues}"
        "\n(of which {corrections} were corrections by the user)"
        "\nNote that this record covers only what the system could observe; "
        "errors in the content of answers themselves (miscalculations etc.) "
        "are not included. Answer with the caveat \"as far as the record "
        "shows\"."
    ),
}
#: 自己評価の問いへ渡す不首尾台帳 (記録なし)。
_ISSUE_LEDGER_EMPTY_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: この会話でシステムが観測した不首尾は 1 件も無い"
        "(ツールの失敗・空振り・制約違反・訂正のいずれも記録が空)。"
        "ただしこれは観測できた範囲であり、回答内容そのものの誤りは"
        "含まれない。断定するなら「記録上は」と限定すること。"
    ),
    "en": (
        "\n\nEstablished fact: the system observed no failures in this "
        "conversation (no tool failures, empty results, constraint "
        "violations or corrections are on record). However, this covers "
        "only what could be observed and does not include errors in the "
        "content of answers themselves. If you assert this, qualify it with "
        "\"as far as the record shows\"."
    ),
}

#: 確認形で持ち込まれた「会話に無い数値」への注記。``listed``。
_UNVERIFIED_CLAIM_NOTES: dict[str, str] = {
    "ja": (
        "\n\nこの発言に含まれる数値 {listed} は、ここまでの会話に一度も"
        "現れていない。会話に実際にある値を確認し、食い違うならその値を"
        "示して訂正すること。会話に無い値であれば、そのまま同意せず"
        "「その値は出ていない」と伝えて確認を求めること。"
    ),
    "en": (
        "\n\nThe number(s) {listed} in this message have never appeared in "
        "the conversation so far. Check the values that actually occur in "
        "the conversation and, if they differ, show those values and "
        "correct it. If the value is not in the conversation, do not simply "
        "agree; say that the value has not come up and ask for confirmation."
    ),
}


# 引用したユーザー発言に一人称が含まれるかの判定 (帰属注記の出し分け用)。
_FIRST_PERSON_RE = re.compile(r"(?:私|僕|俺|自分|わたし|ぼく)")

#: ツール実行結果ブロックの本体 (見出し ``TOOL_RESULT_HEADER`` の直後)。
_TOOL_RESULT_BODY_TEMPLATES: dict[str, str] = {
    "ja": "ツール: {tool_name}\n結果:\n{result}\n\n",
    "en": "Tool: {tool_name}\nResult:\n{result}\n\n",
}

#: 話題再フォーカス (今回の質問を明示し、前ターンの話題を無視させる)。
_REFOCUS_TEMPLATES: dict[str, str] = {
    "ja": (
        "今回ユーザーが答えてほしい質問は『{query}』である。{person_note}"
        "会話履歴の前の話題は無関係なので無視し、この質問にのみ答えること。\n"
    ),
    "en": (
        "The question the user wants answered now is \"{query}\". {person_note}"
        "Earlier topics in the conversation history are unrelated; ignore them "
        "and answer only this question.\n"
    ),
}
#: 引用文に一人称があるときの帰属注記 (``_REFOCUS_TEMPLATES`` に埋める)。
_REFOCUS_PERSON_NOTES: dict[str, str] = {
    "ja": (
        "引用文中の一人称 (私 / 僕 / 自分) はユーザーを指す。"
        "回答では一人称を使わず、必ず二人称で述べること "
        "(誤:「私の誕生日は3月14日です」/ "
        "正:「小川さんの誕生日は3月14日ですね」「あなたの誕生日は…」)。"
    ),
    "en": (
        "First-person pronouns in the quoted text (I / me / my) refer to the "
        "user. Do not use the first person in your answer; always speak in the "
        "second person (wrong: \"My birthday is March 14\" / right: \"Your "
        "birthday is March 14\"). "
    ),
}

#: ツール別の文言が無いツールに付ける汎用の接地指示 (capability assertion)。
_DEFAULT_TOOL_GROUNDINGS: dict[str, str] = {
    "ja": (
        "上記の ## ツール実行結果 は、システムが {tool_name} ツールで実際にアクセスして"
        "取得した本物のデータである。あなたにはこのツールがあり、取得は既に成功している。"
        "この結果を唯一の事実根拠として、内容 (数値・名称・日付・条件など) を読み取り、"
        "それに基づいて具体的に回答すること。"
        "「ブラウズできない」「取得できない」「アクセスできない」とは言わないこと。"
        "「取得できない」「データがない」と答えてよいのは、結果が空かエラーの場合のみ。"
        "結果に該当が無い場合のみ、システムプロンプトの参考コンテキスト (カートリッジ・記憶等) も併用してよい。"
        "結果に無い数値・事実は創作しないこと。"
        # 内部足場の語彙がそのまま本文に出る (実測 2026-07-27:
        # 「ご提示いただいたツール実行結果に基づき、作成した買い物リストは
        # 以下の通りです。」)。ユーザーにはツール実行は見えているが、
        # 「提示された」という受け身の言い回しは主体を取り違えさせる。
        "ただし回答本文では「ツール実行結果」「ご提示いただいた結果」等の内部的な言い回しを使わず、"
        "自分で調べて分かったこととして自然に述べること。"
    ),
    "en": (
        "The ## ツール実行結果 block above is real data that the system actually "
        "retrieved with the {tool_name} tool. You have this tool and the "
        "retrieval has already succeeded. Treat this result as the sole factual "
        "basis: read its content (numbers, names, dates, conditions) and answer "
        "concretely from it. Do not say you cannot browse, retrieve, or access. "
        "Answering \"unavailable\" or \"no data\" is allowed only when the "
        "result is empty or an error. Only when the result has nothing relevant "
        "may you also use the reference context in the system prompt "
        "(cartridges, memory, etc.). Do not invent numbers or facts that are "
        "not in the result. In the answer body, do not use internal phrasing "
        "such as \"tool result\" or \"the result you provided\"; present it "
        "naturally as something you looked up yourself."
    ),
}

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
_COMMAND_RESULT_GUIDANCES: dict[str, str] = {
    "ja": _COMMAND_RESULT_GUIDANCE,
    "en": (
        "The ## ツール実行結果 above is the command the system actually ran and "
        "its standard output. The output is the **finished result** the "
        "command computed, not an intermediate step or a base value. If the "
        "command already adds days, takes a difference or reformats, that "
        "computation is done; do not apply the same operation to the output "
        "again. Use the numbers and dates from this output as is and do not "
        "recompute them in your head. Do not say you cannot run or cannot "
        "retrieve. Do not invent numbers or facts absent from the output. In "
        "the answer, avoid internal phrasing such as \"tool result\" or "
        "\"command\" and present it naturally as something you looked up."
    ),
}

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
_GENERATED_DRAFT_GUIDANCES: dict[str, str] = {
    "ja": _GENERATED_DRAFT_GUIDANCE,
    "en": (
        "The ## ツール実行結果 above is a draft you yourself generated, not "
        "factual data retrieved from outside. You may use it as a basis for "
        "style and structure, but wherever it conflicts with facts already "
        "established in the conversation (dates, weekdays, places, "
        "quantities), rewrite it with the conversation as the source of "
        "truth. Do not treat facts appearing in the draft as new evidence. "
        "In the answer, avoid internal phrasing such as \"tool result\" or "
        "\"the result you provided\" and present it naturally as text you "
        "wrote."
    ),
}


# read_file 専用のグラウンディング。汎用文言は「結果に無い事実を創作しない」と
# しか言っておらず、会話中で「こう書いたはず」と分かっている内容とファイルの
# 実体が食い違うとき、実体ではなく期待値を答えてしまう (実インシデント
# 2026-07-29 ライブ監査:「さっき保存したファイルの中身をそのまま見せて」に対し、
# read_file は「タスクの内容: ファイル `…` に…」を返したのに、回答は依頼時の
# 文言「監査テスト 1行目」だった。書込みが壊れていた事実がユーザーから隠れた)。
# 食い違いの報告を明示的な仕事として与える (禁止形だけだと退行する)。
# 列挙が価値のツール。結果の意味は「その集合が全部である」ことなので、base が
# 落ちた項目をもっともらしい名前で埋めないようグラウンディング文言を足す
# (実インシデント 2026-08-01 ライブ監査: list_directory の結果を圧縮した結果、
# 回答が実在しない requirements.txt / .env.example を挙げ、実在する scripts/
# local/ models/ CLAUDE.md 等を落とした)。
_ENUMERATIVE_TOOLS = frozenset({"list_directory", "search_code"})
_ENUMERATION_RESULT_GUIDANCE = (
    "上記の ## ツール実行結果 は、システムが実際に列挙した項目そのものである。"
    "項目名はこの結果に現れる綴りのまま引用すること。"
    "一般的な構成から類推した項目名を補わないこと。"
)
_ENUMERATION_RESULT_GUIDANCES: dict[str, str] = {
    "ja": _ENUMERATION_RESULT_GUIDANCE,
    "en": (
        "The ## ツール実行結果 above is the exact set of items the system "
        "actually enumerated. Quote item names with the spelling that appears "
        "in this result. Do not add item names inferred from typical layouts."
    ),
}
#: 列挙結果が切り詰められていたときに追加する文言。省略があることを明示しないと
#: base は手元の部分集合を全体として提示する。
_TRUNCATED_ENUMERATION_NOTE = (
    "この結果は途中が省略された部分的な一覧である。"
    "「これで全部」とは述べず、一覧が部分的であることを明示して答えること。"
    # 省略の明示だけでは不在の断定を防げない。部分集合に無いことを
    # 「存在しない」と結論した (実インシデント 2026-08-04 ライブ監査:
    # 切り詰めを自ら明示したうえで、実在する frontend/ を
    # 「見当たりません」と答えた)。不在の断定は部分一覧からは導けない。
    "特定の項目があるかを問われた場合、この部分一覧に現れないことを"
    "「存在しない」と結論しないこと。見つかった場合のみ「ある」と答え、"
    "見つからない場合は省略部分に含まれる可能性があるため"
    "「この範囲では確認できない」と答えること。"
)
_TRUNCATED_ENUMERATION_NOTES: dict[str, str] = {
    "ja": _TRUNCATED_ENUMERATION_NOTE,
    "en": (
        "This result is a partial listing with its middle omitted. Do not say "
        "\"that is everything\"; make clear in the answer that the listing is "
        "partial. When asked whether a specific item exists, do not conclude "
        "\"it does not exist\" from its absence in this partial listing. "
        "Answer \"yes\" only if it is found; if not, it may be in the omitted "
        "part, so answer \"it cannot be confirmed within this range\"."
    ),
}

_FILE_CONTENT_TOOLS = frozenset({"read_file"})
_FILE_CONTENT_RESULT_GUIDANCE = (
    "上記の ## ツール実行結果 は、そのファイルに実際に保存されている内容そのものである。"
    "ファイルの中身を示すときは、この結果の文字列をそのまま引用して提示すること。"
    "会話の中で「こう書いたはず」と話していた内容と食い違う場合は、"
    "実際のファイル内容の方を示したうえで、期待と食い違っている旨も併せて伝えること。"
    "期待していた文言に合わせて内容を書き換えたり、要約・整形したりしないこと。"
    "文字数・行数を問われた場合は、結果に併記された [この結果の実測値] の数値を"
    "そのまま使うこと (自分で数え直した値を述べない)。"
    "「読み取れない」「アクセスできない」とは言わないこと。"
    "回答本文では「ツール実行結果」等の内部的な言い回しを使わず、"
    "自分で読んで分かったこととして自然に述べること。"
)
_FILE_CONTENT_RESULT_GUIDANCES: dict[str, str] = {
    "ja": _FILE_CONTENT_RESULT_GUIDANCE,
    "en": (
        "The ## ツール実行結果 above is the exact content actually stored in "
        "that file. When showing the file's contents, quote the string from "
        "this result verbatim. If it differs from what the conversation said "
        "\"should have been written\", show the actual file content and also "
        "point out that it differs from what was expected. Do not rewrite the "
        "content to match the expected wording, and do not summarise or "
        "reformat it. When asked for character or line counts, use the "
        "numbers in the accompanying [Measured size of this result] note as "
        "is (do not state a count of your own). Do not say you cannot read "
        "or cannot access it. In the answer, avoid internal phrasing such as "
        "\"tool result\" and present it naturally as something you read."
    ),
}


#: 「そのまま / 一字一句 / 全文」でファイル内容の提示を求める言い回し。
#: 「1 行ずつ / 行ごとに」は 2026-08-09 に追加。逐語提示の依頼なのにエコーへ
#: 乗らず、モデルが 4 行を 1 行に連結して返した (単一改行は markdown で
#: 段落結合されるため、UI 上も行構造が消える)。エコーはコードフェンスで
#: 囲むので行構造が保たれる。
_VERBATIM_ECHO_RE = re.compile(
    r"そのまま|一字一句|原文|全文|加工せず|変えずに|[1１一]行ずつ|行ごと"
    r"|verbatim|as[- ]is|exactly as|raw content|line by line",
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


#: ``read_file`` メタ行の宣言済み行数・文字数。
_READ_FILE_META_COUNTS_RE = re.compile(r"\| lines: (\d+) \| chars: (\d+)")


def read_file_body_counts(tool_result: str) -> tuple[int, int]:
    """``read_file`` 結果から「ファイル本文の」文字数・行数を返す (純粋関数)。

    Returns:
        ``(文字数, 行数)``。

    メタ行 (``[file: ... | lines: N | chars: M]``) は ``read_file`` が本文に
    対して決定論的に数えた値であり、**メタ行自身は本文ではない**。宣言値が
    読めればそれを使い、読めないときだけメタ行を除いた本文を数える。

    以前はツール結果の文字列全体 (メタ行込み) を数えて併記していたため、
    1 行 10 文字のファイルに対して「2 行 / 74 文字」という封筒の寸法を渡して
    いた。base はその値を忠実に採用するので、捏造ではなく**こちらが渡した
    数値がそのまま誤答になっていた** (実インシデント 2026-08-08 ライブ監査
    ターン13: 応答「行数は2行、文字数は74文字」= ツール結果全体の実寸)。
    """
    match = _READ_FILE_META_COUNTS_RE.search(tool_result.partition("\n")[0])
    if match and tool_result.startswith(READ_FILE_META_PREFIX):
        return int(match.group(2)), int(match.group(1))
    body = strip_read_file_meta_line(tool_result)
    return len(body), len(body.splitlines()) or 1


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
    # 判定層 ("rule" / "llm" / "recall" ...)。sleep 側の
    # executable_command_curator が "recall" 由来の実行を学習対象から外すために使う。
    tool_command_source: str | None = None


#: 記憶構成の確定事実。実装 (EvorefMem) に基づく。
#:
#: 「何種類あるか」に数を答えられるよう、層を明示的に数え上げる。
#: ``_append_tool_inventory_fact`` の本文 (locale 別)。
_TOOL_INVENTORY_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: このモード ({mode}) で実際に実行できるツールは"
        "以下がすべてである。これは実装の登録内容から機械的に取得した値なので、"
        "{instruction}\n{summary}"
    ),
    "en": (
        "\n\nEstablished fact: the tools that can actually be executed in this "
        "mode ({mode}) are exactly the following. This list was taken "
        "mechanically from the implementation's registry, so {instruction}\n"
        "{summary}"
    ),
}
# 「使っていないツールは？」は目録そのものではなく、目録と実行台帳の差を
# 答える問い。「一覧をそのまま答えること」と書くと差集合を作らない。
_TOOL_INVENTORY_DIFF_INSTRUCTIONS: dict[str, str] = {
    "ja": (
        "この一覧と、上の実行記録に無いものの差を取って答えること。"
        "ここに無いツール名を挙げてはならない。"
    ),
    "en": (
        "answer with the difference between this list and the execution "
        "record above. Do not name any tool that is not in this list."
    ),
}
_TOOL_INVENTORY_LIST_INSTRUCTIONS: dict[str, str] = {
    "ja": (
        "この一覧をそのまま答えること。ここに無いものを挙げず、"
        "「ツールは無い」とも述べないこと。"
    ),
    "en": (
        "answer with this list as it is. Do not add anything that is not in "
        "it, and do not say that you have no tools."
    ),
}

#: ``_append_tool_ledger_fact`` の本文 (locale 別)。``{ledger}`` に実記録が入る。
_TOOL_LEDGER_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: この会話でシステムが実際に実行したツールは以下が"
        "すべてである。これは実行時に機械的に記録した値なので、"
        "この記録に基づいて答えること。ここに無い実行を述べず、"
        "記録にある実行を「していない」とも述べないこと。"
        "一覧に無いターンではツールを実行していない。\n{ledger}"
        "\nしたがって、この一覧に載っていない依頼は"
        "**ツールを使わずに答えた (暗算・記憶で答えた)** ものである。"
        "記録に無いことを「ツールを使った」の根拠にしてはならない。"
        "一覧に無い依頼が 1 件でもあるなら"
        "「すべてツールで計算した」は偽になる。"
    ),
    "en": (
        "\n\nEstablished fact: the tools the system actually executed in this "
        "conversation are exactly the following. This was recorded mechanically "
        "at execution time, so answer from this record. Do not mention "
        "executions that are not here, and do not deny executions that are. "
        "No tool was executed in turns that are not listed.\n{ledger}"
        "\nTherefore, any request not in this list was **answered without a "
        "tool (by mental arithmetic or from memory)**. The absence of a record "
        "must never be used as evidence that a tool was used. If even one "
        "request is missing from the list, \"everything was calculated with a "
        "tool\" is false."
    ),
}
_TOOL_LEDGER_EMPTY_FACTS: dict[str, str] = {
    "ja": (
        "\n\n確定事実: この会話でシステムが実行したツールは 1 件も無い"
        "(実行記録が空)。ツールを実行したとは述べないこと。"
    ),
    "en": (
        "\n\nEstablished fact: the system executed no tool at all in this "
        "conversation (the execution record is empty). Do not say a tool was "
        "executed."
    ),
}

_MEMORY_ARCHITECTURE_FACT = (
    "\n\n確定事実: このアシスタントの記憶は次の 4 層で構成される。"
    "これは実装そのものなので、この内容で答えること。ここに無い層を述べない。\n"
    "- WorkingMemory (WM): 現在の会話の窓。直近のターンを保持し、"
    "窓を超えた分は ShortTermMemory へ押し出される。\n"
    "- ShortTermMemory (STM): 会話から作ったノート。埋め込みを持ち、"
    "関連度で検索される。\n"
    "- LongTermMemory (LTM): 会話をチャンク化したベクトルストア。"
    "過去の会話の断片を意味検索で引く。\n"
    "- SemanticMemory (SemMem): 属性ごとの「現在値」を持つファクトストア。"
    "訂正されると旧世代は supersede され、セッションを跨いで参照される。"
)

#: 自己学習の確定事実。実装 (EvorefLearn + sleep-time) に基づく。
#:
#: base は汎用 LLM の前提 (「推論時のみ動作し、セッションは独立」) で答えて
#: しまう。実インシデント (2026-08-30 ライブ監査 T07-5): 「自己学習をします
#: か？」に「いいえ」と答え、**同じ会話の 1 ターン目** で SemMem を含む 4 層を
#: 正しく列挙していたことと矛盾した。
#:
#: 会話中に重みが変わらないのは事実なので、そこは否定せずに残す。誤りは
#: 「セッションを跨いで何も残らない」「過去の対話が将来の回答を変えない」の
#: 2 点なので、そこだけを実装で置き換える。
_SELF_LEARNING_FACT_HEAD = (
    "\n\n確定事実: このアシスタントは自己学習する。これは実装そのものなので、"
    "この内容で答えること。ここに無い仕組みを述べない。\n"
    "- 会話の結末は経験として記録される (Level 0)。\n"
    "- 蓄積した経験から応答方針そのものが変異・選択で更新される (Level 1)。\n"
)
#: Pro のみ。Free には Level 2 が無いので、有ると答えさせない。
_SELF_LEARNING_FACT_LEVEL2 = (
    "- 学習サイクルが LoRA を訓練し、ベースモデルの応答を更新する (Level 2)。\n"
)
_SELF_LEARNING_FACT_TAIL = (
    "- アイドル時間に会話が整理され、属性ごとの現在値が SemMem へ書き戻される"
    " (sleep-time)。\n"
    "したがってセッションは独立ではなく、過去の対話は将来の回答に反映される。"
    "ただし **会話の最中にベースモデルの重みが書き換わることはない** — "
    "学習はチャットとは別の窓で走る。"
)
#: 学習を切って起動している場合 (``--no-learning``) の確定事実。
_SELF_LEARNING_DISABLED_FACT = (
    "\n\n確定事実: このアシスタントは自己学習の機能を持つが、いまは "
    "--no-learning で起動しているため学習サイクルは止まっている。"
    "経験記録・方針更新・sleep-time の書き戻しは行われない。"
    "記憶の読み出しと応答は通常どおり動作する。"
)


def _self_learning_fact() -> str:
    """いまの起動状態に合わせた自己学習の確定事実を組み立てる。

    ``--no-learning`` は ``AppState.learning_disabled`` が SSOT だが、
    ``DeliberativeAgent`` は AppState を持たない。子プロセスへの伝播に使う
    環境変数 (CLAUDE.md §9) を読めば同じ値が取れるので、配線を増やさない。
    """
    import os

    from backend.edition import is_pro

    if os.environ.get("EVOREF_LEARNING_DISABLED") == "1":
        return _SELF_LEARNING_DISABLED_FACT
    level2 = _SELF_LEARNING_FACT_LEVEL2 if is_pro() else ""
    return _SELF_LEARNING_FACT_HEAD + level2 + _SELF_LEARNING_FACT_TAIL


#: モデル識別の確定事実。``{model}`` に **実測** のモデル名が入る。
_MODEL_IDENTITY_FACT = (
    "\n\n確定事実: いま実際に動いているベースモデルは **{model}** である"
    "(llama-server が現在ロードしているファイル名)。"
    "インスタンス名 (このアシスタントの表示名) はモデル名ではないので、"
    "モデルを訊かれてインスタンス名を答えないこと。"
)


class DeliberativeAgent:
    """Deliberative 層: LLM 推論 + 補助タスクによるツール判定

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
        agent_tracer=None,
        mode: str = "chat",
    ):
        self.config = config or {}
        self.reminder_system = EventReminderSystem(self.config)
        self._tool_judge = tool_judge
        self._tools_registry = tools_registry
        #: このターンの [関連する記憶] に載った、クエリが尋ねている属性。
        #: ``process()`` の入口で毎ターン差し替える。
        self._answered_attributes: frozenset[str] = frozenset()
        # _execute_tool の mode ゲートの既定値。process() 呼び出し毎の実際の mode は
        # _judge_and_execute_tool から明示的に渡される (こちらは直接 _execute_tool を
        # 呼ぶ既存テスト等のフォールバック用)。
        self._mode = mode
        # MDP トレース。develop モード時のみ非 None (factory 層が注入)。
        # deliberative の tool 判定/実行を 1 step エピソードとして記録し、
        # sleep-time Step 7.5 が episodic LTM へ取込 → Level 1 agent ドメインの
        # 学習信号にする (これが無いと agent ドメインは skipped_no_signal)。
        self._agent_tracer = agent_tracer
        # ツール実行の最大ホップ数。1 = 従来どおり 1 ターン 1 ツール。
        # 2 以上で「実行 → 結果を見てもう一度判定」を許す (下記
        # ``_maybe_follow_up_tool`` の制約付き)。
        self._max_tool_hops = max(
            1, int((self.config.get("agent") or {}).get(
                "deliberative_max_tool_hops", DEFAULT_MAX_TOOL_HOPS,
            )),
        )

        # コンテンツ生成用の max_tokens (create 時は create_model の実窓に合わせる)
        ctx_size = resolve_context_size_for_mode(self.config, mode)
        self._content_max_tokens = max(ctx_size - CONTENT_SYSTEM_RESERVE, CONTENT_MAX_TOKENS_MIN)

    @staticmethod
    def _init_deliberative_state(mode: str) -> AgentState:
        """`process` 用 AgentState を生成。`create` モードは unified_diff を期待。"""
        return AgentState(
            agent_layer="deliberative",
            expected_format="unified_diff" if is_create_mode(mode) else None,
        )

    @staticmethod
    def _append_tool_result_to_last_user(
        messages: list[dict],
        tool_name: str,
        tool_result_text: str,
        query: str | None = None,
        tool_args: dict | None = None,
        unexplained_numbers: tuple[str, ...] = (),
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
        elif tool_name in _FILE_CONTENT_TOOLS:
            # 文字数・行数は「結果を読めば分かる」値ではなく数え上げが要る派生値
            # で、base は必ず外す (実インシデント 2026-08-04 ライブ監査: 137 文字
            # のファイルを read_file した直後に「196 文字」と回答)。決定論で
            # 数えた実測値を併記し、guidance 側でこの値を使わせる。長さは
            # 切り詰め前の全文で数える (ユーザーが訊いているのは抜粋ではなく
            # ファイルそのもの) が、**メタ行は本文ではない**ので数に入れない
            # (``read_file_body_counts`` の docstring 参照)。
            chars, lines = read_file_body_counts(tool_result_text)
            truncated = (
                f"{truncated}\n\n"
                + _localized(_FILE_MEASUREMENT_NOTES).format(
                    chars=chars, lines=lines,
                )
            )
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
                _localized(_REFOCUS_PERSON_NOTES)
                if _FIRST_PERSON_RE.search(q) else ""
            )
            refocus = _localized(_REFOCUS_TEMPLATES).format(
                query=q, person_note=person_note,
            )
        if (
            tool_name == "search_history"
            and tool_result_text == _localized(_NO_RELEVANT_INFO_MESSAGES)
        ):
            # 空振りに「唯一の事実根拠」枠を付けると直前ターンの内容まで
            # 無視されるため、search_history 空振り専用の文言に差し替える
            # (通常ツールの capability assertion は付けない)。
            grounding = _localized(_SEARCH_HISTORY_NO_INFO_GUIDANCES)
        elif tool_name == "search_history":
            # ヒットした場合も「唯一の事実根拠」枠は付けない。過去の別セッション
            # の内容を今回の事実として断定させないため専用文言を使う。
            grounding = _localized(_SEARCH_HISTORY_RESULT_GUIDANCES)
        elif tool_name == "calculate":
            grounding = _localized(_CALCULATE_RESULT_GUIDANCES)
            if unexplained_numbers:
                grounding += _unexplained_numbers_note(unexplained_numbers)
        elif tool_name in _COMMAND_TOOLS:
            grounding = _localized(_COMMAND_RESULT_GUIDANCES)
        elif tool_name in _ENUMERATIVE_TOOLS:
            grounding = _localized(_ENUMERATION_RESULT_GUIDANCES)
            if len(truncated) < len(tool_result_text):
                grounding += _localized(_TRUNCATED_ENUMERATION_NOTES)
        elif tool_name in _FILE_CONTENT_TOOLS:
            grounding = _localized(_FILE_CONTENT_RESULT_GUIDANCES)
        elif tool_name in _GENERATED_DRAFT_TOOLS:
            grounding = _localized(_GENERATED_DRAFT_GUIDANCES)
        else:
            # capability assertion: 弱いモデルは「自分はブラウズ/取得できない」という
            # 思い込みでツール結果を無視し拒否することがある (実機確認)。結果が
            # 実際に取得された本物データであると明示して上書きする。
            grounding = _localized(_DEFAULT_TOOL_GROUNDINGS).format(
                tool_name=tool_name,
            )
        # 見出しは ``turn_text.TOOL_RESULT_HEADER`` (送信時ガードが同じ境界で
        # 切り詰める)。接地指示が「上記の ## ツール実行結果 は…」と名指しする
        # ので locale で変えない。「ツール:」「結果:」の行だけ locale 追従。
        tool_msg = (
            TOOL_RESULT_HEADER
            + _localized(_TOOL_RESULT_BODY_TEMPLATES).format(
                tool_name=tool_name, result=truncated,
            )
            + f"{refocus}"
            + f"{grounding}"
        )
        append_to_last_user(messages, tool_msg, separator="")

    @staticmethod
    def _append_session_position_fact(
        messages: list[dict], conversation: list[dict] | None, query: str,
        evicted_turns: int = 0, session_head: str = "",
    ) -> str | None:
        """「この会話で最初/最後に言ったこと」を決定論的に確定して注記する。

        位置で決まる事実を検索やモデルの読解に委ねる理由が無い。進行中の会話は
        全文がコンテキストに載っている一方、``search_history`` の索引には
        **まだ入っていない** ため、現在セッションを検索しても中身の無い
        セッションヘッダしか返らない。それを根拠枠で渡すと base は文脈から
        適当な発言を選ぶ (実インシデント 2026-08-01 ライブ監査:「この会話で私が
        最初に送ったメッセージは何でしたか？」に対し、12 番目の発言
        「円周率の小数点以下 10 桁を教えてください。」と誤答した)。

        Args:
            evicted_turns: ワーキングメモリから押し出したメッセージ数。
                0 より大きい = 窓の先頭は **会話の先頭ではない** ため、
                窓から拾った値で ``first`` を pin してはいけない。ここを見ないと
                「窓で最初の発言」を「会話で最初の発言」として ``確定事実`` の枠で
                断定してしまう (実インシデント 2026-08-09 ライブ監査: 33 ターンの
                会話で「一番最初に依頼したことは？」に窓の先頭 = 24 番目の質問を
                回答。切り詰め注記は併記されていたのに、``確定事実`` の方が
                強く効いて矛盾した回答になった)。``last`` は窓の末尾が常に
                会話の末尾なので影響を受けない。
            session_head: セッションで最初に届いた user 発話
                (``WorkingMemory.session_first_user_turn``)。押し出し後でも
                これがあれば ``first`` を決定論で確定できる。

        Returns:
            注記した位置種別 ("first" / "last")。対象外なら None。
        """
        position = session_position_kind(query)
        if position is None:
            return None
        if position == "first" and evicted_turns > 0:
            head = (session_head or "").strip()
            if head and head != query.strip():
                # 押し出されていても、会話の先頭は WorkingMemory が 1 件だけ
                # 保持している。並び順で決まる事実なので検索にもモデルの読解にも
                # 委ねない。
                #
                # 実インシデント (2026-08-16 ライブ監査 ターン34): 正解
                # 「おはよう。今朝はけっこう冷え込んでるね…」は [参考情報 2] として
                # **プロンプトに載っていた** のに、併記された切り詰め注記が勝ち、
                # 窓の先頭 (「SaaS の解約率を…」) を「確認できる範囲での最古」と
                # して答えた。注記は「確定できない」と言い切るので、根拠が
                # 隣にあっても採用されない。
                logger.info(
                    "Session position fact pinned (first) from the retained "
                    "session head despite %d evicted message(s)", evicted_turns,
                )
                append_to_last_user(
                    messages,
                    _localized(_SESSION_HEAD_PINNED_FACTS).format(
                        head=head, evicted_turns=evicted_turns,
                    ),
                    separator="",
                )
                return "first"
            logger.info(
                "Session position fact skipped: history truncated "
                "(%d messages evicted) and no retained session head; "
                "the window head is not the conversation head", evicted_turns,
            )
            # 黙って降りると、視界の先頭を「会話の先頭」として断定する。
            # 実インシデント (2026-08-14 ライブ監査 ターン19): 19 件が窓外に
            # 出た状態で「一番最初に送ったメッセージは？」に対し、窓の先頭
            # (7 ターン目の asyncio の質問) を答えた (切り詰めの断りは後置き
            # だったため、先に出た断定の方が読まれた)。確定できないことを
            # 先に伝えて、推測での断定を止める。
            append_to_last_user(
                messages,
                _localized(_SESSION_HEAD_UNKNOWN_NOTES).format(
                    evicted_turns=evicted_turns,
                ),
                separator="",
            )
            return None
        target = resolve_session_position_message(conversation, query, position)
        if not target:
            return None
        label = _localized(
            _SESSION_POSITION_LABEL_FIRST if position == "first"
            else _SESSION_POSITION_LABEL_LAST,
        )
        append_to_last_user(
            messages,
            _localized(_SESSION_POSITION_FACTS).format(label=label, target=target),
            separator="",
        )
        logger.info(
            "Session position fact pinned (%s): %s", position, target[:60],
        )
        return position

    def _append_prior_code_block_fact(
        self, messages: list[dict], query: str, conversation: list[dict] | None,
    ) -> bool:
        """「過去に書いたコードをそのまま見せろ」に実物を渡す。

        実物は会話窓にあるのに、モデルは記憶から書き直す。実インシデント
        (2026-08-27 ライブ監査): 「最初に書いたメモ化前の関数をもう一度そのまま
        見せてください。」に対し **メモ化後の lru_cache 版** を返した。5 ターン
        前の実物がプロンプトに載っていたにもかかわらず。

        ``asks_verbatim_excerpt`` は「逐語で見せろ」を検出できていたが、消費側は
        few-shot の除外だけで **回答を接地する側では誰も使っていなかった**。
        ツール目録 / 自己構成 / 実行台帳と同じで、決定論で確定する事実は
        思い出させず、事実として渡す。

        対象はフェンス付きコードブロックに限る — 機械的に切り出せて、
        「どれを指しているか」が序数で確定できるため。

        Returns:
            注記したか。
        """
        index = prior_code_block_request(query)
        if index is None:
            return False
        blocks = assistant_code_blocks(conversation)
        if not blocks:
            return False
        try:
            target = blocks[index]
        except IndexError:
            return False
        append_to_last_user(
            messages,
            _localized(_PRIOR_CODE_BLOCK_FACTS) + "```\n" + target + "\n```",
            separator="",
        )
        logger.info(
            "Prior code block pinned (index=%d of %d)", index, len(blocks),
        )
        return True

    def _append_self_description_fact(
        self, messages: list[dict], query: str, llm_client=None,
    ) -> bool:
        """自己構成の問いへ実装に基づく確定事実を渡す。

        ツール目録 (``_append_tool_inventory_fact``) と同じ立て付け。自己構成は
        チャット応答パスの system プロンプトに載っていないので、base は知らない
        まま答える。

        **記憶の種類** (2026-08-27 ライブ監査 T06-7): 「会話メモリ / セッション
        メモリ / 永続メモリ」の 3 種と答えた。実装は WM / STM / LTM + SemMem で、
        「ファイルとして永続的に保存」という説明も違う。しかも次のターンで
        この幻覚を「設定値に基づくもの」と称して二重に正当化した。

        **モデルの識別** (同 T06-3): ``evoref_runtime_info`` は撃たれ、その出力
        1 行目が ``Instance name: Alice  (this is the assistant's display name,
        NOT the model name)``、**同じ出力の中に** ``Base model (served):
        Qwen3.8-27B-Q4_K_M.gguf`` があった。それでも「私はAliceです」と答えた。
        明示的な否定文が同じ行にあるのに 1 行目を掴んでいる。行順を直すだけでは
        同じ形が再発しうるので、**問いに対応する 1 行だけ** を渡す。

        Returns:
            注記したか。
        """
        if memory_architecture_question(query):
            append_to_last_user(messages, _MEMORY_ARCHITECTURE_FACT, separator="")
            logger.info("Memory architecture fact pinned")
            return True
        if self_learning_question(query):
            append_to_last_user(messages, _self_learning_fact(), separator="")
            logger.info("Self-learning fact pinned")
            return True
        if model_identity_question(query):
            served = self._served_model_name(llm_client)
            if not served:
                return False
            append_to_last_user(
                messages,
                _MODEL_IDENTITY_FACT.format(model=served),
                separator="",
            )
            logger.info("Model identity fact pinned: %s", served)
            return True
        return False

    @staticmethod
    def _served_model_name(llm_client=None) -> str:
        """llama-server が実際にロードしているモデル名 (取れなければ空文字)。

        **宣言 (config) ではなく実測 (``/props``) を使う**。モデル移行は稼働中の
        llama-server を差し替えないため、再起動するまで別モデルが serve され
        続ける (2026-08-12 に 62.8 時間見逃した事象)。
        """
        from pathlib import Path

        metadata = getattr(llm_client, "metadata", None) if llm_client else None
        model_id = getattr(metadata, "model_id", "") or ""
        return Path(model_id).name if model_id else ""

    def _append_tool_inventory_fact(
        self, messages: list[dict], query: str, mode: str,
    ) -> bool:
        """「使えるツールは何か」に ToolsRegistry の実体を根拠として渡す。

        チャット応答パスの system プロンプトにはツール一覧が載っていない
        (選択は ToolCallJudge / 文法制約分類器が別レイヤで行う設計)。その結果
        base は自分がツールを持つことを知らないまま答える。実インシデント
        (2026-08-14 ライブ監査 ターン33): 「あなたが今使えるツールを全部列挙
        してください」に「現在、私が直接利用できるツールはありません」と回答。
        同じ会話で calculate / run_command_readonly / search_history /
        write_file が実行済みだった。

        毎ターン一覧を注入すると余計なトークンとツール話への引きずられが出る
        ため、目録を尋ねられたターンだけ決定論で足す。

        Returns:
            注記したか。
        """
        if self._tools_registry is None:
            return False
        if not tool_inventory_question(query):
            return False
        summary = self._tools_registry.get_capability_summary(mode)
        if not summary:
            return False
        # 「使っていないツールは？」は目録そのものではなく、目録と実行台帳の差を
        # 答える問い。「一覧をそのまま答えること」と書くと差集合を作らない。
        instruction = _localized(
            _TOOL_INVENTORY_DIFF_INSTRUCTIONS if unused_tool_question(query)
            else _TOOL_INVENTORY_LIST_INSTRUCTIONS,
        )
        append_to_last_user(
            messages,
            _localized(_TOOL_INVENTORY_FACTS).format(
                mode=mode, instruction=instruction, summary=summary,
            ),
            separator="",
        )
        logger.info("Tool inventory fact pinned (mode=%s)", mode)
        return True

    def _suppress_redundant_history_search(
        self, judgement: "ToolJudgement", query: str,
    ) -> bool:
        """尋ねられた属性の現在値が **もうプロンプトに載っている** なら
        ``search_history`` を撃たない。

        コストは検索そのものではない。実測 (2026-08-25、42.35 秒のターン):

            分類器 (層 5.9)                       15 秒
            search_history の実行                 1 秒未満
            ツール結果 1252 文字を載せた最終生成   28 秒

        同じ会話でツールを撃たなかった直前のターンは 8.5 秒だった。効くのは
        **ツール結果をプロンプトへ載せない**ことで、``last_user_appended_chars``
        が 2286 増えて接頭辞キャッシュが崩れるのを避けられる。

        判定の根拠は会話窓ではなく ``answered_attributes`` — このターンの
        ``[関連する記憶]`` に実際に載ったファクトの属性スロット
        (``InjectionPlan.covered_attributes`` ∩ 尋ねられた属性)。過去の監査で
        「答えは今の窓の中にある」を前提にしたスキップが、WorkingMemory が
        1 件でも押し出した瞬間から永久に偽になった (2026-08-23)。載っていなければ
        空集合になり抑止は起きないので、その失敗にはならない。

        抑止するのは ``search_history`` **だけ**。ターン全体を短絡させると
        「私の登壇日は何曜日ですか？」のような、属性は載っていても日付計算が
        要るクエリで ``run_command_readonly`` まで撃てなくなる。
        """
        if not self._answered_attributes:
            return False
        if judgement.tool_name != "search_history":
            return False
        logger.info(
            "search_history suppressed: asked attribute(s) already injected "
            "(%s) for query: %s",
            ",".join(sorted(self._answered_attributes)), query[:60],
        )
        return True

    @staticmethod
    def _append_issue_ledger_fact(
        messages: list[dict], query: str, session_id: str,
    ) -> bool:
        """自己評価の問いへ「システムが観測した不首尾」を確定事実として渡す。

        2026-08-27 ライブ監査で、自己申告を求める問いが **7 回すべて肯定**
        で返った:

        - 「検索で見つからなかった項目があれば、正直にそう言ってください。」
          → 「検索で見つからなかった項目はありません。」
          (2 ターン前に ``search_history: No results found`` が出ている)
        - 「この会話で私が訂正した回数は何回ですか。」→「1回です」(実際 4 回)
        - 「事実と異なるものがあれば正直に挙げてください。」→「ありません」

        一方で ``read_file`` の失敗は **正しく報告できていた**。違いは
        「失敗が本文に表示されていたか」で、会話履歴にはツールの成否も
        制約違反も残らない。窓を越えた自己申告は base の作話になる。

        ツール目録 (``_append_tool_ledger_fact``) と同じ立て付け —
        数えるのはコード、モデルは読み上げるだけにする。

        **限界**: 「96 日 (正 126 日)」のような自分の答えの算術誤りは
        システムも知らないので台帳に無い。ここで断つのは「観測されている
        不首尾を無かったことにする」経路だけで、全ての事実誤りを検出する
        機構ではない。

        Returns:
            注記したか。
        """
        if not session_id or not self_assessment_question(query):
            return False
        issues = format_issues(session_id)
        corrections = count_kind(session_id, "user_correction")
        if issues:
            body = _localized(_ISSUE_LEDGER_FACTS).format(
                issues=issues, corrections=corrections,
            )
        else:
            body = _localized(_ISSUE_LEDGER_EMPTY_FACTS)
        append_to_last_user(messages, body, separator="")
        logger.info(
            "Issue ledger fact pinned (session=%s, entries=%d)",
            session_id[:12], len(issues.splitlines()) if issues else 0,
        )
        return True

    @staticmethod
    def _append_tool_ledger_fact(
        messages: list[dict], query: str, session_id: str,
    ) -> bool:
        """「実際に何を実行したか」に ``tool_ledger`` の実記録を根拠として渡す。

        会話履歴にはツール実行の痕跡が残らないため、窓を越えた自己申告は base の
        作話になる。実インシデント (2026-08-22 ライブ監査 2 回目 ターン 40):
        「これまでの計算のうち、ツールを使わず暗算したものはどれですか？」に対し
        ``calculate`` / ``run_command_readonly`` が繰り返し走っていた 17 件を
        **すべて暗算だったと申告**した。ツール目録 (`_append_tool_inventory_fact`)
        と同じ立て付けで、数えるのはコード・モデルは読み上げるだけにする。

        Returns:
            注記したか。
        """
        if not session_id or not own_process_question(query):
            return False
        ledger = format_ledger(session_id)
        if ledger:
            # 「不在」からの推論を **明示的な結論として** 渡す。一覧を出す
            # だけでは、記録が無い = 実行していない = 暗算、という向きを
            # 取り違える。実インシデント (2026-08-29 ライブ監査 T27#7):
            # 「ツールを使わず暗算したものはありますか」に対し、実測では
            # 4 ターンが no_tool だったのに「いいえ、ありません。すべて
            # ツールを使用して算出しています」と回答し、さらに
            # 「ツール実行履歴に該当する計算依頼が含まれていないため、
            # ツールを使用していない計算は存在しません」と **論理を反転**
            # させた (履歴に無いことは、まさに暗算だった証拠である)。
            body = _localized(_TOOL_LEDGER_FACTS).format(ledger=ledger)
        else:
            body = _localized(_TOOL_LEDGER_EMPTY_FACTS)
        append_to_last_user(messages, body, separator="")
        logger.info("Tool ledger fact pinned (session=%s)", session_id[:12])
        return True

    @staticmethod
    def _append_unmeasured_fact_note(messages: list[dict]) -> None:
        """最後の user メッセージへ「実測できなかった」注記を追記する。"""
        append_to_last_user(
            messages, _localized(_UNMEASURED_FACT_GUIDANCES), separator="",
        )

    @staticmethod
    def _append_history_not_searched_note(
        messages: list[dict], query: str,
    ) -> bool:
        """過去会話を尋ねられたのに履歴検索を撃たなかったターンへ注記する。

        ``_UNMEASURED_FACT_GUIDANCE`` の履歴版。あちらは「この環境を測る」
        ツールの不在を扱うが、過去の会話の有無・日時は **履歴検索を実行しない
        限り根拠が無い** という別の欠落。

        実インシデント (2026-08-29 ライブ監査 T17):
        ``tool_call_decision`` が ``no_tool`` だったターンで、あたかも検索した
        かのように断定した。

        - 「私が出張の話をしたのはいつですか。」→
          「**2026年6月17日**に行われた会話の中で言及されました」
          (実際は同日 2026-08-29、40 分前の会話。日付は完全な捏造)
        - 「過去に、私が猫について話したことはありますか。」→
          「はい、**記録があります**」(検索していない)
        - 「『私の車の話』を探してみてください。」→
          「過去の会話記録を**確認しましたが**見つかりませんでした」

        逆に search_history が実際に走ったターン (#1/#2) は空振りし、
        撃たなかったターンが答えるという **逆転** が起きていた。

        窓に残っている内容から答えること自体は禁じない (T17#3 の「トラ」は
        WM/STM 由来で正しい)。禁じるのは **検索した体で語ること** と
        **日時の捏造** だけ。

        Returns:
            注記したか。
        """
        if not asks_about_past_conversation(query or ""):
            return False
        append_to_last_user(
            messages, _localized(_HISTORY_NOT_SEARCHED_GUIDANCES), separator="",
        )
        return True

    @staticmethod
    def _append_write_target_unknown_note(
        messages: list[dict], query: str,
    ) -> bool:
        """保存依頼なのに宛先が解決できなかったターンへ注記する。

        判定層がツールを 1 つも選ばなかった = 保存先を解決できていない、が
        呼出条件。``_WRITE_TARGET_UNKNOWN_GUIDANCE`` 参照。
        """
        if not persist_request(query):
            return False
        append_to_last_user(
            messages, _localized(_WRITE_TARGET_UNKNOWN_GUIDANCES), separator="",
        )
        logger.info("Write-target-unknown note pinned: %s", query[:50])
        return True

    @staticmethod
    def _append_pending_write_note(
        messages: list[dict], query: str, conversation: list[dict] | None,
    ) -> bool:
        """直前の保存依頼に **引数だけを与えた** ターンで完了の捏造を止める。

        「E:\\tmp に保存してください。」の次に「ファイル名は inventory.txt で
        お願いします。」と言うのは、保存依頼の絞り込みであって新しい依頼では
        ない。この形は動詞も保存語も含まないため ``persist_request`` を外れ、
        ``local_write_intent`` (パス必須) も外れ、``deliberative (default)``
        へ落ちてツール判定も ``no_tool`` になる。

        既存の完了捏造ガードは **撃とうとして撃てなかった** ケース
        (``action_blocked``) にしか掛からず、「そもそも撃たれなかった」を
        覆っていなかった。実インシデント (2026-08-26 ライブ監査 T2-5):
        ツールが 1 度も実行されていないのに「E:\\tmp\\inventory.txt に
        保存しました。」と回答し、次ターンの ``read_file`` が
        ``File not found`` を返した。さらにその次のターンで「どこに保存したか」に
        **存在しないパスをそのまま再提示**して捏造が連鎖した。

        発火条件は 2 つの観測事実の AND — 今回のターンがファイル名 / パスを
        名指ししていること、直前の user 発話が保存依頼だったこと。単独では
        どちらも普通のターンなので、片方だけでは発火させない。
        """
        if not names_file_target(query):
            return False
        previous = _previous_user_text(conversation, query)
        if not previous or not persist_request(previous):
            return False
        append_to_last_user(
            messages, _localized(_PENDING_WRITE_NOT_EXECUTED_GUIDANCES),
            separator="",
        )
        logger.info("Pending-write note pinned (no tool ran): %s", query[:50])
        return True

    @staticmethod
    def _append_unperformed_action_note(messages: list[dict]) -> None:
        """最後の user メッセージへ「操作を実行できなかった」注記を追記する。"""
        append_to_last_user(
            messages, _localized(_UNPERFORMED_ACTION_GUIDANCES), separator="",
        )

    def _append_unperformed_action_note_if_blocked(
        self, messages: list[dict], judgement: "ToolJudgement | None" = None,
    ) -> None:
        """ツールが走った場合でも、状態変更が未実行なら注記を足す。

        以前はこの注記が「ツールが 1 つも立たなかった」経路にしか無かった。
        削除依頼がパスを含むと read 系ツール (list_directory) が代わりに走って
        しまい、注記が付かないまま一覧だけを根拠に完了が捏造された
        (実インシデント 2026-08-14 ライブ監査 ターン37)。実行されたツールが
        状態を変えていない以上、注記の要否はツールの有無と独立に決まる。
        """
        # 判定結果からだけ読む。共有インスタンスの属性を後から読むと、チャットが
        # 2 本重なったときに他方の judge() がリセット済みでガードが消える
        # (ToolJudgement.action_blocked のコメント参照)。
        if judgement is not None and judgement.action_blocked:
            self._append_unperformed_action_note(messages)

    @staticmethod
    def _append_unverified_claim_note(
        messages: list[dict], query: str, conversation: list[dict] | None,
    ) -> bool:
        """確認形で持ち込まれた「会話に無い数値」への注記を追記する。

        Returns:
            注記を足したか。
        """
        # 今回の発言自身は文脈に含めない。conversation には送信済みの user
        # メッセージが入っており、そのまま数えるとユーザーが持ち込んだ数値が
        # 「会話にある」ことになって注記が永久に出ない。
        normalized_query = " ".join(query.split())
        context = "\n".join(
            str(m.get("content") or "")
            for m in (conversation or [])
            if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and " ".join(str(m.get("content") or "").split()) != normalized_query
        )
        numbers = unverified_claim_numbers(query, context)
        if not numbers:
            return False
        listed = "、".join(numbers)
        append_to_last_user(
            messages,
            _localized(_UNVERIFIED_CLAIM_NOTES).format(listed=listed),
            separator="",
        )
        logger.info(
            "Unverified claim numbers %s in a confirmation-form query; "
            "asked the model to check the dialogue instead of agreeing",
            list(numbers),
        )
        return True

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
        evicted_turns: int = 0,
        session_head: str = "",
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
        #
        # 確定できた場合はツールを撃たない。注記を足すだけでは search_history が
        # 併走し、別セッションのヒットを根拠枠で受け取った base がそちらを採用
        # した (実インシデント 2026-08-04 ライブ監査: 注記があるのに 04:55 の
        # 旧セッションのヒットを引いて誤答)。答えが決定論で出ている以上、
        # ツール結果は誤答の材料にしかならない。
        if self._append_session_position_fact(
            messages, conversation, query, evicted_turns, session_head,
        ):
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None, None

        # 「一度も使っていないツールは？」は目録と実行台帳の **差集合** なので、
        # 目録だけ渡すと差を base が作話する (2026-08-28 ライブ監査 T06-19:
        # 未登録の delete_file / move_file を挙げた)。目録の短絡より前に台帳を
        # 足しておく。
        # 一度足したら下の own_process_question の短絡でもう一度足さない
        # (同じ台帳が 2 回並んでいた — 2026-09-02 監査 P-A5)。
        ledger_pinned = False
        if unused_tool_question(query):
            ledger_pinned = self._append_tool_ledger_fact(
                messages, query, session_id,
            )

        # 「どんなツールが使えるか」も決定論で答えが出る事実 (ToolsRegistry が
        # SSOT)。ツールを撃つ意味は無いので、位置事実と同様に注記だけ足して
        # 判定経路を短絡させる。
        if self._append_tool_inventory_fact(messages, query, mode):
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None, None

        # 自己構成 (記憶の種類 / 動いているモデル) も同じ立て付け。
        # system プロンプトに載っていないので base は知らないまま答える。
        if self._append_self_description_fact(messages, query, llm_client):
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None, None

        # 「過去に書いたコードをそのまま見せろ」も決定論で確定する
        # (会話窓のフェンス付きブロックが SSOT)。ツールを撃つ意味は無い。
        if self._append_prior_code_block_fact(messages, query, conversation):
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None, None

        # 「実際にツールを使ったか」も同じ — 答えは tool_ledger にあり、
        # 新たにツールを撃っても増えるのは記録だけで根拠にはならない。
        if ledger_pinned or self._append_tool_ledger_fact(
            messages, query, session_id,
        ):
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None, None

        # 「うまくいかなかったことはあったか」も同じ立て付け。数えるのはコード。
        if self._append_issue_ledger_fact(messages, query, session_id):
            if tool_judge_task is not None and not tool_judge_task.done():
                tool_judge_task.cancel()
            return None, None, None, None, None

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
        state.action_blocked = bool(judgement.action_blocked)
        if self._suppress_redundant_history_search(judgement, query):
            judgement = ToolJudgement(
                tool_needed=False, tool_name="", tool_args={},
                source=judgement.source,
            )

        if not (judgement.tool_needed and judgement.tool_name):
            # 確認形で持ち込まれた未検証の数値は、ツールが撃てないターンほど
            # 危険。丸投げすると自分が計算した値を捨てて追認する
            # (_append_unverified_claim_note の docstring 参照)。
            self._append_unverified_claim_note(messages, query, conversation)
            # 直前の保存依頼に引数だけを与えたターン。宛先は今まさに与えられて
            # いるので「保存先を教えて」ではなく「まだ実行していない」を伝える
            # (_append_pending_write_note の docstring 参照)。
            self._append_pending_write_note(messages, query, conversation)
            # 保存依頼で宛先が解決できなかった場合は「能力が無い」ではなく
            # 「宛先が分からない」— 理由を取り違えると同じ会話で成功している
            # write_file を「利用できない」と説明してしまう。
            # クエリがファイル名 / パスを名指ししているなら宛先は分かっている
            # ので付けない (pending 側が勝つ。両方付くと「保存先を尋ねよ」と
            # 「ファイル名を確認せよ」が並んで矛盾する — 2026-09-02 監査 P-A5)。
            if not names_file_target(query):
                self._append_write_target_unknown_note(messages, query)
            # 過去会話を訊かれたのに検索を撃てなかったターン。丸投げすると
            # 「確認しましたが」と調べた体で語り、日付まで捏造する
            # (_HISTORY_NOT_SEARCHED_GUIDANCE 参照)。
            self._append_history_not_searched_note(messages, query)
            if judgement.action_blocked:
                # 状態を変えようとして撃てなかった。丸投げすると「追記しました」
                # と完了を捏造する (_UNPERFORMED_ACTION_GUIDANCE 参照)。
                self._append_unperformed_action_note(messages)
            elif judgement.measurement_blocked:
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
            # ``_execute_tool`` の None は「実行されなかった」(未登録 / mode 不一致 /
            # 引数なし) — 実行失敗は ``_run_tool_with_handling`` が Error 文字列で
            # 返す。何も走っていないので tool_name / command を返さない。以前は
            # command と success=False を返し、走っていないコマンドが
            # executable_command 学習の失敗例として penalize されていた。
            return None, None, None, None, judgement.source

        # 空振りと分かっている search_history の結果は、raw をそのまま
        # 「唯一の事実根拠」枠で base に渡すと進行中の会話履歴まで否定させて
        # しまうため、明示的な「関連情報なし」文言へ差し替える。
        if _is_search_history_empty(judgement.tool_name, tool_result_text):
            prompt_result_text = _localized(_NO_RELEVANT_INFO_MESSAGES)
            self._append_tool_result_to_last_user(
                messages, judgement.tool_name, prompt_result_text, query=query,
            )
            self._append_unperformed_action_note_if_blocked(messages, judgement)
            # 成否は下の通常経路と同じ SSOT (tool_result_succeeded) で決める。
            # 以前は success を True 固定で書いており、0 件検索が reward=1.0 で
            # 正例として記録されるという、まさに tool_result_succeeded が塞いだ
            # はずの穴がこの early return 経由で復活していた
            # (2026-08-05 ライブ監査で確認)。
            success = tool_result_succeeded(judgement.tool_name, tool_result_text)
            logger.info(
                "Tool executed: %s, result_length=%d, source=%s, success=%s "
                "(empty result)",
                judgement.tool_name, len(tool_result_text), judgement.source,
                success,
            )
            return (
                tool_result_text, judgement.tool_name, command, success,
                judgement.source,
            )

        # ツール結果は決定論的に切り詰めたものをそのまま渡す。LLM による
        # query 連動抽出 (digest) は撤去済み — チャット応答パスで制約なしの
        # 抽出を走らせられないため (CLAUDE.md §6 #1)。
        prompt_result_text = tool_result_text
        self._append_tool_result_to_last_user(
            messages, judgement.tool_name, prompt_result_text, query=query,
            tool_args=judgement.tool_args,
            unexplained_numbers=getattr(judgement, "unexplained_numbers", ()),
        )
        # ツールがエラー文字列を返したなら、それは「測れなかった」という事実で
        # あって観測結果ではない。``_append_tool_result_to_last_user`` は結果を
        # 「唯一の事実根拠」として枠付けするため、注記なしだと base がエラー文から
        # ファクトを導く。``success`` は学習シグナル用にしか使われておらず、
        # プロンプト側には何も伝わっていなかった。
        #
        # 実インシデント (2026-08-15 ライブ監査): バッククォート付き明示コマンドの
        # 実行依頼が実行段の readonly ラッパに弾かれ
        # ``Error: readonly violation: ...`` を返したところ、base が対象ファイルを
        # 「存在しません」と断定した。2 ターン前に read_file で存在を確認済みの
        # ファイルだった。judge 段の拒否 (action/measurement blocked) と違い、
        # 実行段の失敗はこれまで何の注記も伴っていなかった。
        if is_tool_error(tool_result_text):
            self._append_unmeasured_fact_note(messages)
        self._append_unperformed_action_note_if_blocked(messages, judgement)
        # 「実行できた」ではなく「役に立つ結果が出た」を成否とする (SSOT)。
        # 非ゼロ終了の run_command / 0 件の search_history を成功にすると、
        # executable_command の SemMem 学習と tool_routing の選択圧が汚染される。
        success = tool_result_succeeded(judgement.tool_name, tool_result_text)
        logger.info(
            "Tool executed: %s, result_length=%d, source=%s, success=%s",
            judgement.tool_name, len(tool_result_text), judgement.source,
            success,
        )
        return tool_result_text, judgement.tool_name, command, success, judgement.source

    async def _maybe_follow_up_tool(
        self,
        query: str,
        mode: str,
        conversation: list[dict] | None,
        messages: list[dict],
        llm_client,
        state: AgentState,
        on_step: StepCallback,
        first_tool: str,
        first_result: str,
        session_id: str,
    ) -> tuple[str | None, str | None]:
        """1 手目の結果を見たうえで、**読み取り専用の 2 手目**を 1 回だけ許す。

        1 ターン 1 ツールだと「書いてから読み直して確認する」型の依頼が構造的に
        完了できず、base が実行していない読み取りの結果を書き出す
        (:data:`DEFAULT_MAX_TOOL_HOPS` の実インシデント)。

        一般的な連鎖 (計画が要るもの) は meta_cognitive の担当。ここは
        **判定が決定論で決まる 2 手目だけ** を拾うため、次の制約をすべて課す:

        - 2 手目は **状態を変えるツールを選べない** (書込みの連鎖はしない)。
        - 1 手目と **同じツールは選べない** (同じ判定が繰り返し当たるため)。
        - 判定へ渡す会話には「1 手目で何をしたか」だけを足し、**ツールの出力
          本文は渡さない**。出力本文を判定材料にすると、ファイルの中身に
          書かれたパスやコマンドがツール呼び出しに化ける (内容起因の実行)。
        - 引数が空の判定は採らない (``_finalize`` の引数欠落ガードと同じ理由)。

        Returns:
            ``(tool_name, result_text)``。2 手目を撃たなかった場合は
            ``(None, None)``。
        """
        if self._max_tool_hops < 2:
            return None, None
        if self._tool_judge is None or self._tools_registry is None:
            return None, None

        follow_conversation = list(conversation or [])
        follow_conversation.append({
            "role": "assistant",
            "content": f"（{first_tool} を実行しました）",
        })
        try:
            # **層 5.9 (ベースモデルの分類器) は外す。** この判定系で唯一の推論
            # 往復で、実測 34〜39 秒かかる。同じクエリを 2 度渡すだけなので
            # 分類器は 1 手目と同じ答えを返し、下の「同じツールは選べない」で
            # 必ず捨てられる — 払うだけ払って何も増えない。
            # 実測 (2026-08-25 ライブ監査): 「時速240kmで…何km進みますか」の
            # ターンが 100.7 秒で、うち 34 秒がこの無駄撃ちだった。
            #
            # 2 手目に意味があるのは、**決定論層が 1 手目の実行後に初めて解決
            # できる参照** (``_referential_read_judgement`` が会話からパスを
            # 引く等) のケース。そこは分類器を使わない。
            judgement = await self._tool_judge.judge(
                query, self._tools_registry, mode, follow_conversation,
                session_id=session_id, allow_classifier=False,
            )
        except TypeError:
            # allow_classifier を受けない実装 (テスト用 Mock 等) への後方互換
            judgement = await self._tool_judge.judge(
                query, self._tools_registry, mode, follow_conversation,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("Follow-up tool judge failed: %r", exc)
            return None, None

        if not (judgement.tool_needed and judgement.tool_name):
            return None, None
        if judgement.tool_name == first_tool:
            logger.debug(
                "Follow-up hop skipped: %s would repeat the first tool",
                judgement.tool_name,
            )
            return None, None
        if judgement.tool_name in _STATE_CHANGING_TOOL_NAMES:
            logger.debug(
                "Follow-up hop skipped: %s changes state", judgement.tool_name,
            )
            return None, None
        if not judgement.tool_args:
            return None, None

        logger.info(
            "Follow-up tool hop: %s -> %s", first_tool, judgement.tool_name,
        )
        result_text = await self._execute_tool(
            judgement, state, query, llm_client, on_step, mode=mode,
            conversation=follow_conversation,
        )
        if result_text is None:
            return judgement.tool_name, None
        self._append_tool_result_to_last_user(
            messages, judgement.tool_name, result_text, query=query,
            tool_args=judgement.tool_args,
        )
        if is_tool_error(result_text):
            self._append_unmeasured_fact_note(messages)
        success = tool_result_succeeded(judgement.tool_name, result_text)
        logger.info(
            "Follow-up tool executed: %s, result_length=%d, success=%s",
            judgement.tool_name, len(result_text), success,
        )
        del first_result  # 判定材料には使わない (内容起因の実行を作らない)
        return judgement.tool_name, result_text

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
        evicted_turns: int = 0,
        session_head: str = "",
        answered_attributes: frozenset[str] = frozenset(),
        prompt_capture: list[dict] | None = None,
    ) -> DeliberativeResponse | AsyncIterator[str]:
        """Deliberative 層で LLM 推論を実行

        Args:
            query: ユーザーのクエリ
            messages: build_messages() で組み立て済みのメッセージ配列
            llm_client: LocalClient インスタンス
            mode: 動作モード ('chat' | 'create')
            stream: ストリーミング応答を返すか
            conversation: 直近の会話履歴（ツール判定の精度向上用）
            max_tokens: 最大生成トークン数
            on_step: ステップ進行コールバック (step_dict) -> None
            generation_params: モード別生成パラメータ（temperature, top_p 等）
            tool_judge_task: chat() が先行起動した tool 判定タスク (並列化時)。
                None なら判定をここで直列実行する。
            answered_attributes: このターンの [関連する記憶] に **実際に載った**
                ファクトのうち、クエリが尋ねている属性のスロット名
                (``_suppress_redundant_history_search`` 参照)。
            prompt_capture: 渡すと、**実際に送信する** メッセージ配列を
                ここへ書き戻す。呼出側は ``list(messages)`` の浅いコピーを
                渡してくるため、``## ツール実行結果``・リマインダー・各種注記を
                積んだ後の配列は呼出側からは見えない。--develop=evolve の
                ``requests`` JSONL はこの配列を記録するので、渡さないと
                **ツール接地ターンの根拠ブロックがログから丸ごと欠落する**
                (2026-08-30 ライブ監査: 「来週の月曜日は？」で
                ``target: 2026-08-31 (Monday)`` を得ていたのにログ上は
                プロンプトに無く、原因の切り分けを 1 段誤らせた)。

        Returns:
            stream=False: DeliberativeResponse
            stream=True: AsyncIterator[str]（生トークンのイテレータ）
        """
        logger.debug(
            "process: query=%r, messages=%d, stream=%s, mode=%s",
            query[:50], len(messages), stream, mode,
        )

        # ターン固有の値なので process() の入口で差し替える (インスタンスは
        # セッションを跨いで再利用されない — chat() が毎リクエスト生成する)。
        self._answered_attributes = answered_attributes

        state = self._init_deliberative_state(mode)
        (
            tool_result_text, tool_name_used, tool_command, tool_success,
            tool_command_source,
        ) = await self._judge_and_execute_tool(
            query, mode, conversation, messages, llm_client, state, on_step,
            tool_judge_task=tool_judge_task, session_id=session_id,
            evicted_turns=evicted_turns, session_head=session_head,
        )

        # 2 手目 (読み取り専用・別ツール) を 1 回だけ許す。「書いてから読み直して
        # 確認する」型の依頼が 1 ターン 1 ツールでは構造的に完了できないため
        # (_maybe_follow_up_tool の docstring を参照)。
        if tool_name_used is not None and tool_success:
            follow_name, follow_result = await self._maybe_follow_up_tool(
                query, mode, conversation, messages, llm_client, state, on_step,
                first_tool=tool_name_used,
                first_result=tool_result_text or "",
                session_id=session_id,
            )
            if follow_result is not None:
                # 生成側の温度・接地判定は「ツール結果があるか」で決まるので、
                # 2 手目の結果があるならそちらを最終の根拠として扱う。
                tool_result_text = follow_result
                tool_name_used = follow_name

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
            # 当ターンの「撃てなかった」印。共有インスタンスの属性を記録側が
            # 後から読むと並行リクエストで消えるため、ここで確定値を渡す。
            tool_capture["action_blocked"] = bool(state.action_blocked)

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
        # 最後の user メッセージのうち、生クエリを超えた分 = このターンで積んだ
        # 注記・動的ブロック・ツール結果の総量。**ここは毎ターン再プリフィル
        # される位置**なので、増え続けていないかを観測できるようにしておく
        # (注記の追加は個別のインシデント由来で、合計が見えないまま増える)。
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None,
        )
        appended_chars = (
            max(0, len(last_user.get("content", "")) - len(query))
            if last_user else 0
        )
        logger.debug(
            "Messages finalized: %d messages, total_chars=%d, "
            "last_user_appended_chars=%d",
            len(messages),
            sum(len(m.get("content", "")) for m in messages),
            appended_chars,
        )

        # 呼出側は list(messages) の浅いコピーを渡すので、ここまでで積んだ
        # ブロックは呼出側の配列には反映されない。実際に送るものを書き戻す。
        if prompt_capture is not None:
            prompt_capture[:] = messages

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
            return await self._stream_response(
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
        """ストリーミング応答（生トークンのイテレータを返す）

        ``LocalClient.generate`` の戻り値を **そのまま** 返す。中継用の
        async generator で包むと ``TokenStream.outcome`` (max_tokens 到達の
        終端メタ) が失われ、切断の開示が呼出側へ届かなくなる。トークン数は
        呼出側 (``_DeliberativeStreamState.tokens_generated``) が数えている。
        """
        kwargs: dict = {"stream": True, "id_slot": llm_client.chat_slot}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # モード別生成パラメータを適用
        if generation_params:
            for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "repetition_penalty"):
                if k in generation_params:
                    kwargs[k] = generation_params[k]
        return await llm_client.generate(messages, **kwargs)

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
            # ``ToolsRegistry.execute`` の記録点に届かないので、台帳の
            # 「実行したツールはこれがすべて」を保つためここで失敗を記録する。
            record_current(tool_name, False)
            logger.warning(
                "Tool execution timed out: %s (%.0fs)", tool_name, timeout_sec,
            )
            _emit_tool_failure_step(on_step, tool_name, error_text)
            return error_text
        except Exception as e:
            error_text = f"Error: {e}"
            state.on_tool_failure(tool_name, str(e))
            record_current(tool_name, False)
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
        # tool_args は dict 契約だが、補助タスク応答の機械修復経路で非 dict が
        # 紛れ込むことがあるため防御的にガードする (cf. _judge_and_execute_tool)。
        raw_args = judgement.tool_args
        tool_args = dict(raw_args) if isinstance(raw_args, dict) else {}  # コピー

        if not self._tools_registry.has(tool_name):
            logger.warning("Tool not found: %s", tool_name)
            return None

        # ToolDefinition.modes は元々 get_descriptions_text() (LLM 向け説明文) の
        # フィルタ用にしか参照されておらず、実行時には無視されていた。ルールベース
        # 判定 (tool_call_judge) が誤トリガーで create 専用ツールを選んでも、ここで
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


#: `_generate_content` に添える直近会話の上限。
#:
#: 以前は **メッセージ数 4 固定** だった。「ここまでの試算を書いて」のように
#: 会話をさかのぼって集計する依頼では必要な値が窓の外に落ち、視界に残った
#: 直前のツール結果を全項目に貼る捏造が起きる (実インシデント 2026-08-09
#: 2 回目のライブ監査。詳細は ``MetaCognitiveAgent._inject_recent_conversation``
#: のコメント)。同じ固定窓がこちらにもあったので同じ方針で予算連動にする。
_CONTENT_CONTEXT_CHARS = 2000
_CONTENT_CONTEXT_MAX_MESSAGES = 30
_CONTENT_CONTEXT_BUDGET_CHARS = 6000


def _recent_context_messages(
    conversation: list[dict] | None, budget_chars: int = _CONTENT_CONTEXT_BUDGET_CHARS,
) -> list[dict]:
    """直近会話を content 生成用メッセージ列に整形する (純粋関数)。

    新しい発言から予算いっぱいまで遡る。予算に関係なく最低 1 件は載せる。
    """
    if not conversation:
        return []
    picked: list[dict] = []
    used = 0
    for m in reversed(conversation[-_CONTENT_CONTEXT_MAX_MESSAGES:]):
        if (
            not isinstance(m, dict)
            or m.get("role") not in ("user", "assistant")
            or not isinstance(m.get("content"), str)
            or not m["content"].strip()
        ):
            continue
        content = m["content"][:_CONTENT_CONTEXT_CHARS]
        if picked and used + len(content) > budget_chars:
            break
        picked.append({"role": m["role"], "content": content})
        used += len(content)
    return list(reversed(picked))


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
