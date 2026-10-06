"""会話に依存する対象パスの解決と参照型の判定

「同じファイルに保存し直して」「そのファイルの全文を見せて」のように、対象が
クエリ単体では確定しない依頼を、直近の会話からパスを引いて確定させる層。
確定できなければ ``None`` を返して後続層へ委ねる (推測でパスを埋めない)。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.free.agent.tool_judge_args import (
    _extract_file_path,
    _extract_file_path_literal,
    _extract_head_line_count,
    extract_write_target_path,
)
from backend.free.agent.router import (
    _referential_destination,
    explicit_path_is_topic_only,
    file_name_cases,
    filename_is_topic_only,
    write_intent_probe,
)
from backend.free.agent.safety_patterns import strip_command_literals
from backend.free.agent.tool_judge_types import ToolJudgement
from backend.free.agent.tools_registry import ToolsRegistry
from backend.free.core.intent_vocab import (
    REFERENTIAL_WRITE_TARGET_RE,
    strip_file_reference_clauses,
    write_prohibited,
)
from backend.free.core.session_mode import is_create_mode
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.agent.tool_judge_guards import JudgeCall

logger = get_logger("agent.tool_call_judge")

#: 「同じファイルに」「そのファイルを」「保存したファイルに」等、保存先を直前の
#: 文脈に委ねる表現。ルータの宛先の証拠と同じ 1 本 (``intent_vocab`` が SSOT)。
#: 以前はここに同じ語彙の別定義があり、説明節を持たず (「保存しておいたファイル」
#: が参照にならない)、``さきほど`` / ``書き直して保存`` はこちらにしか無かった
#: (2026-09-27 レビュー、#14(a))。
_REFERENTIAL_TARGET_RE = REFERENTIAL_WRITE_TARGET_RE
#: 保存/書き出しを求める動詞 (パス無しの参照依頼を拾うための最小集合)。
#: ``追記`` / ``書き足`` / ``書[きい]て`` は 2026-08-09 に追加 (実インシデント:
#: 「そのファイルの末尾に追記して書いて」が保存動詞として認識されなかった)。
_REWRITE_VERB_RE = re.compile(
    r"保存|書き込|書き出|書き足|追記|上書き|セーブ|書[きい]て"
    r"|\bsave\b|\bwrite\b|\bappend\b|\boverwrite\b",
    re.IGNORECASE,
)
#: パス区切りを含むか (ドライブ接頭辞 / スラッシュ / バックスラッシュ)。
#: 含まない = 裸のファイル名で、書込み先としては **どのディレクトリか未確定**。
_PATH_SEPARATOR_RE = re.compile(r"[\\/]")


#: 拡張子なしで呼ばれる定番の文書名。
_WELL_KNOWN_DOC_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(README|CHANGELOG|LICENSE|CONTRIBUTING)(?![A-Za-z0-9_-]|\.[A-Za-z0-9])",
    re.IGNORECASE,
)


#: 文書の種類・書き方を問う質問 (読む依頼ではない)。
_DOC_KIND_QUESTION_RE = re.compile(
    r"とは|って何|書き方|作り方|テンプレ|ひな形|雛形|例を|違い|what is|how to write",
    re.IGNORECASE,
)
#: 製品の作業ルート (インストール根) を指す言い方。
_REPO_REFERENCE_RE = re.compile(
    r"このリポジトリ|このプロジェクト|このレポ|この repo|this (?:repo|repository|project)",
    re.IGNORECASE,
)


def _path_is_written_in_query(query: str) -> bool:
    """ディレクトリ付きのパスが **発話の本文に書かれているか** (純粋関数)。

    ``_extract_file_path`` は本文にパスが無いとき file_ledger の直近ファイルへ
    フォールバックするので、その戻り値では「本文に書かれているか」を判定でき
    ない。ここは本文の literal だけを見る。

    実インシデント (2026-08-28 ライブ監査の修正検証):
    「保存したファイルを読み出して、構文エラーがないか確認してください。」で
    暗黙参照が ``E:\\tmp\\verify2_20260828.py`` に解決された結果、本層が
    「パスは本文にある」と誤認して後続へ委ね、ルール層が chat では使えない
    ``verify_syntax`` を選んで ``no_tool`` へ降格 → ``read_file`` が撃たれない
    まま「構文エラーはありません」と答えた (ファイルは読んでいない)。
    """
    literal = _extract_file_path_literal(query)
    return bool(literal and _PATH_SEPARATOR_RE.search(literal))


def write_target_is_topic_only(query: str, *, whole_request: bool = False) -> bool:
    """書込み先の候補が話題・根拠の格にしか立たない、または書込みが禁止されているか (純粋関数)。

    「このファイル**に対する** pytest のテストを書いて」「util_fixed.py **に対する**
    テストを書いて」の名指しは何について書くかの題材で、書込みの宛先ではない。規則層は
    宛先かをルータの宛先の証拠と **同じ 1 本・同じ入力** で決める (#14 (a)、docs/f_03 §1.4):
    格と保存の動詞は依頼節 (``write_intent_probe``、create は ``whole_request``) で見る。
    本文にパスが無ければ参照表現を ``router._referential_destination`` (参照表現は発話から
    取る — 説明節の参照表現は依頼節の正規化で消えるため)、裸のファイル名 (相対パス) は
    ``router.filename_is_topic_only``、ディレクトリ付きのパスは
    ``router.explicit_path_is_topic_only``。書込みの禁止 (「ファイルには保存しないで」) も真。
    真なら ``write_file`` の宛先にしない。以前はルータだけが格を見ていたので、ルータが
    書込みでないとしたターンで規則層が ``write_file`` を選び、chat で降格されて
    ``action_blocked`` が立ち、応答が「保存するツールが利用できない」と頼まれていない
    保存を断った (2026-10-05 ライブ監査 T3 / 2026-10-06 独立レビュー HIGH-1・MED-2)。
    """
    if write_prohibited(query):
        return True
    probe = write_intent_probe(query, whole_request=whole_request)
    literal = _extract_file_path_literal(query)
    if literal:
        if _PATH_SEPARATOR_RE.search(literal):
            return explicit_path_is_topic_only(probe)
        return filename_is_topic_only(probe)
    return bool(_REFERENTIAL_TARGET_RE.search(query)) and not _referential_destination(
        query, verb_text=probe,
    )


def write_target_path(query: str) -> str:
    """規則層の書込み先 (``extract_write_target_path``)。話題の格に立つパスは選ばない。

    「E:\\x\\util.py に対するテストを書いて、E:\\x\\t.py を作成して」の util.py は題材。
    選んだパスが題材で、話題の格に立たないファイル名が他にあれば最初のそれへ替える
    (docs/f_03 §1.4、2026-10-06 独立レビュー MED-3)。他に無ければそのまま — 題材だけを
    名指した依頼が書くのは保存の動詞があるときだけで (「util.py に対するテストを書いて
    保存して」)、その判定は :func:`write_target_is_topic_only` が先に済ませている。
    暗黙参照 (台帳) で解いたパスは本文に無いのでそのまま返す。
    """
    path = extract_write_target_path(query)
    if not path:
        return ""
    cases = [(_extract_file_path_literal(token), topic) for token, topic in file_name_cases(query)]
    if path not in {p for p, topic in cases if topic}:
        return path
    return next((p for p, topic in cases if p and not topic), path)


def _resolve_referenced_path(
    query_path: str | None, conversation: list[dict] | None,
    *, for_write: bool = False,
) -> str | None:
    """書込み/読取の対象パスを会話から解決する。

    ``query_path`` の状態で 3 通りに分かれる:

    - ディレクトリを含む絶対/相対パス → そのまま採用 (解決不要)
    - 裸のファイル名 (``notes.txt``) → ``file_ledger.resolve_bare_filename``
      (読み書きの入口が共有する 1 本) で解決する。読みで曖昧なら裸の名前のまま、
      どこにも無ければ会話の同じ名前のフルパス、それも無ければ ``None``
    - ``None`` / 空 (「そのファイル」型) → 会話で最後に出たパスを採用

    裸のファイル名をそのままツールへ渡すとカレントディレクトリに着地して
    しまい、ユーザーが指した既存ファイルとは別物を作る。会話で確定している
    場合のみ解決し、確定できなければ ``None`` を返して後続層に委ねる
    (推測でパスを埋めない)。
    """
    if query_path and _PATH_SEPARATOR_RE.search(query_path):
        return query_path
    if (query_path or "").strip():
        from backend.free.agent.file_ledger import resolve_bare_filename

        resolution = resolve_bare_filename(
            query_path.strip(), conversation=conversation, for_write=for_write,
        )
        if resolution.path:
            return resolution.path
        # 読みで決められないときも読みは撃たせる — 撃たずに答えると中身を作話する。
        # 曖昧なら裸の名前のまま (レジストリが候補つきのエラーを返す)、どこにも
        # 無ければ会話に書かれた同じ名前のフルパス (「見つからない」と答えさせる)。
        if resolution.ambiguous:
            return query_path.strip()
        return resolution.mentioned
    for msg in reversed(list(conversation or [])):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        path = _extract_file_path(content)
        if not path or not _PATH_SEPARATOR_RE.search(path):
            continue
        return path
    return None


def _query_path_of(query: str, call: "JudgeCall | None") -> str | None:
    """クエリ本文のパス (memo 付き)。"""
    found = call.extract_file_path(query) if call is not None else _extract_file_path(query)
    return found or None


def _resolve_with_call(
    query_path: str | None, conversation: list[dict] | None,
    call: "JudgeCall | None", *, for_write: bool = False,
) -> str | None:
    """会話からのパス解決。``call`` があれば層 0.9 / 0.95 で 1 度だけ走査する。"""
    if call is None:
        return _resolve_referenced_path(query_path, conversation, for_write=for_write)
    return call.referenced_path(
        query_path,
        lambda qp, conv: _resolve_referenced_path(qp, conv, for_write=for_write),
        for_write=for_write,
    )


def _referential_rewrite_judgement(
    query: str, conversation: list[dict] | None, tools_registry: ToolsRegistry,
    call: "JudgeCall | None" = None,
) -> "ToolJudgement | None":
    """「同じファイルに保存し直して」型の依頼を write_file に確定させる。

    保存動詞があり、かつ書込み先がクエリだけでは確定しない (参照表現、または
    ディレクトリを伴わない裸のファイル名) 場合に、直近の会話からパスを引いて
    ``write_file`` を返す。該当しない (ディレクトリ付きパスが本文にある /
    参照も裸名も無い / 会話にパスが無い) 場合は ``None`` で後続層に委ねる。
    純粋関数 (レジストリ参照のみ)。

    裸のファイル名を拾うのは 2026-08-09 のライブ監査で判明した実害への対処:
    「inventory_notes.txt に 1 行追記してください」がどの層にも拾われず
    deliberative に落ち、ツールを 1 つも撃たないまま **フルパスを補って**
    「E:\\tmp\\inventory_notes.txt の末尾に追記しました」と報告した
    (実ファイルは無変更)。フルパスで同じ依頼をすると正常に書き込まれており、
    差はパス表記だけだった。
    """
    if not tools_registry.has("write_file"):
        return None
    # 説明節 (「保存したファイル」) の「保存」は対象の説明で、依頼された動作では
    # ない。消さずに見ると「保存したファイルの場所を教えて」が write_file になり、
    # chat で格下げされた後に list_directory('.') へ落ちた (2026-09-27 監査 C07#5)。
    # 節の SSOT はルータの ``write_intent_probe`` と同じ (docs/c_17 §3.5.1)。
    if not _REWRITE_VERB_RE.search(strip_file_reference_clauses(query)):
        return None
    if _path_is_written_in_query(query):
        return None  # ディレクトリ付きパスが本文にあるなら通常のルール層で足りる
    query_path = _query_path_of(query, call)
    if not query_path and not _REFERENTIAL_TARGET_RE.search(query):
        return None
    if write_target_is_topic_only(
        query, whole_request=call is not None and is_create_mode(call.mode),
    ):
        return None
    path = _resolve_with_call(query_path, conversation, call, for_write=True)
    if not path:
        return None
    logger.info(
        "Referential rewrite: resolved target from conversation: %s "
        "(query_path=%r)", path, query_path,
    )
    return ToolJudgement(
        tool_needed=True,
        tool_name="write_file",
        tool_args={"file_path": path},
        source="rule",
    )


#: ファイルの中身を「見せる」ことを求める表現。read_file を撃たずに答えると
#: 記憶から再構成した偽の内容を「ファイルの中身」として提示する
#: (2026-08-09 ライブ監査: 追記直後の「全文をそのまま見せて」で 3 行とも実
#: ファイルと不一致、しかも同一セッション内の誤答が中身として混入した)。
_FILE_CONTENT_DISPLAY_RE = re.compile(
    r"(?:全文|中身|内容|そのまま|中身をそのまま)"
    r".{0,20}?(?:見せ|表示|出して|教えて|確認)"
    r"|(?:見せ|表示).{0,10}?(?:全文|中身|内容)"
    # 「中身**は何**になりましたか」型 — 内容を **問う** 形。表示動詞
    # (見せ/表示/教えて) を必須にしていたため漏れていた。
    #
    # 実インシデント (2026-08-29 ライブ監査 T05#5): 直前ターンで
    # ``memo_b.txt`` への書き込みがガードでブロックされ **ファイルは未作成**
    # だったのに、「memo_b.txt の中身は何になりましたか。」が
    # ``tool_call_decision=no_tool`` (reason=no_match_in_any_layer) となり、
    # read_file を撃たないまま **「2026-08-29」と中身を捏造** した。
    # 同テーマの T05#3 (「**その**ファイルの中身を読んで、そのまま見せて」→
    # referential_read) / T05#8 (フルパス指定 → explicit_path) は発火しており、
    # **裸のファイル名 + 問いかけ形** だけがどのルールにも当たっていなかった。
    r"|(?:全文|中身|内容)(?:は|が|って)?\s*(?:何|なに|どう|どんな|いくつ)"
    r"|\bshow\s+(?:me\s+)?(?:the\s+)?(?:full\s+)?(?:content|contents|file)\b"
    r"|\b(?:display|print)\s+(?:the\s+)?(?:content|contents|file)\b",
    re.IGNORECASE,
)
#: 「ファイル」を指す語。表示要求が **ファイルに関するもの** かの絞り込みに使う。
_FILE_NOUN_RE = re.compile(r"ファイル|\bfile\b", re.IGNORECASE)

#: 「(そのファイルを) 読み出して」型の **素の読取動詞**。
#:
#: :data:`_FILE_CONTENT_DISPLAY_RE` は目的語の名詞 (``全文`` / ``中身`` /
#: ``内容`` / ``そのまま``) を必須にしているため、目的語が「ファイル」そのもの
#: である普通の言い方が漏れていた。
#:
#: 実インシデント (2026-08-28 ライブ監査 T15-15):
#: 「保存したファイルを読み出して、構文エラーがないか確認してください。」に
#: ``read_file`` が 1 度も撃たれず (``tool_call_decision=no_tool``)、
#: 直前のターンで ``write_file`` が成功しているのに
#: 「ファイル内容の読み出しや構文チェックを行うツールが利用できないため、
#: 確認できていません」と **自分の道具立てについて誤った主張** をした。
_FILE_READ_VERB_RE = re.compile(
    r"読み(?:出|込|取)|読んで|開いて"
    r"|\bread\s+(?:the\s+|that\s+|this\s+)?file\b"
    r"|\bopen\s+(?:the\s+|that\s+|this\s+)?file\b",
    re.IGNORECASE,
)

#: ファイルの計測値 (行数・文字数・サイズ) を尋ねる表現。``read_file`` の結果には
#: ``lines`` / ``chars`` のメタ行が付くので、読めば決定論で答えられる。撃たないと
#: モデルが数値を捏造する — しかも **正解が直前ターンに出ていても**捏造する
#: (実インシデント 2026-08-10 ライブ監査: 直前の read_file 出力に
#: ``lines: 10 | chars: 411`` と表示されていたのに「12 行、357 文字」と答えた)。
_FILE_METRICS_RE = re.compile(
    r"(?:行数|文字数|バイト数|何行|何文字|ファイルサイズ)"
    r"|\b(?:line|character|byte|word)\s*count\b"
    r"|\bhow\s+many\s+(?:lines|characters|bytes|words)\b",
    re.IGNORECASE,
)


def _referential_read_judgement(
    query: str, conversation: list[dict] | None, tools_registry: ToolsRegistry,
    call: "JudgeCall | None" = None,
) -> "ToolJudgement | None":
    """「そのファイルの全文を見せて」型の依頼を read_file に確定させる。

    ``_referential_rewrite_judgement`` の読取版。書込み側と同じく、対象が
    クエリだけでは確定しない (参照表現 / 裸のファイル名) 場合に会話から
    パスを引く。ディレクトリ付きパスが本文にあるなら通常のルール層で足りる。

    ファイル名詞または参照表現を要求するので、「さっきの説明の中身を見せて」の
    ような非ファイルの表示要求は拾わない。純粋関数 (レジストリ参照のみ)。
    """
    if not tools_registry.has("read_file"):
        return None
    # 計測値の問い合わせも読取で決まる (read_file が lines/chars を返す)。
    wants_metrics = bool(_FILE_METRICS_RE.search(query))
    # 素の読取動詞は、対象がファイルだと分かるときだけ受ける
    # (「さっきの説明を読んで」のような非ファイルの依頼を拾わないため)。
    plain_read = bool(
        _FILE_READ_VERB_RE.search(query)
        and (_FILE_NOUN_RE.search(query) or _REFERENTIAL_TARGET_RE.search(query)),
    )
    if not (
        _FILE_CONTENT_DISPLAY_RE.search(query) or wants_metrics or plain_read
    ):
        return None
    if _path_is_written_in_query(query):
        return None
    query_path = _query_path_of(query, call)
    if not query_path and not (
        _REFERENTIAL_TARGET_RE.search(query) or _FILE_NOUN_RE.search(query)
    ):
        return None
    path = _resolve_with_call(query_path, conversation, call)
    if not path:
        return None
    logger.info(
        "Referential read: resolved target from conversation: %s "
        "(query_path=%r)", path, query_path,
    )
    tool_args: dict = {"file_path": path}
    # 計測は全文を読まないと数えられないので範囲指定しない。
    head = None if wants_metrics else _extract_head_line_count(query)
    if head is not None:
        # ``_infer_tool`` と同じ引数形 (read_file は start/end_line を取る)。
        tool_args["start_line"] = 1
        tool_args["end_line"] = head
    return ToolJudgement(
        tool_needed=True,
        tool_name="read_file",
        tool_args=tool_args,
        source="rule",
    )


#: コードを **作る** 依頼の構造 (「〜するコードを書いて」「スクリプトを作って」「write code
#: that …」)。この形では名前は作るコードが扱う対象で、読む対象ではない (「Python で
#: config.json を読み込むコードを書いて」の「読み込む」は作るコードの説明)。書込み動詞の
#: 語彙 (計画のタスク文向け) は使わない — 利用者の発話の「書いてある」に当たる。
_CODE_CREATION_RE = re.compile(
    r"(?:コード|スクリプト|プログラム|関数|クラス|処理)\s*(?:を|が)?\s*"
    r"(?:書いて|書く|書け|作って|作る|作成|生成|実装)"
    r"|\b(?:write|create|generate|implement)\s+(?:a\s+|an\s+|the\s+|some\s+)?"
    r"(?:\w+\s+)?(?:code|script|program|function|class)\b",
    re.IGNORECASE,
)
#: ファイルの中身を述べる状態の形 (「何が書いてあるか」「書かれている」)。書込みの依頼では
#: ないので、ツールの推定 (``_infer_tool`` の書込みの語「書いて」) へ渡す前に中立の語へ置く。
_STATIVE_WRITTEN_RE = re.compile(r"書(?:いて(?:ある|あり|あっ|る)|かれて)")
#: ツールの推定へ渡すときに依頼文の裸の名前を置き換える中立の名前。
_NEUTRAL_FILENAME = "target_file.txt"


def _bare_filename_read_judgement(
    query: str,
    conversation: list[dict] | None,
    tools_registry: ToolsRegistry,
    mode: str,
    infer: Callable[[str], tuple[str, dict]],
) -> "ToolJudgement | None":
    """依頼文の裸のファイル名が利用者の名指したフォルダの **実在のファイル** に解決できたら
    read_file に確定させる。

    「buggy.py のバグを見つけて直し方を教えて」は区切りの無い名前なので明示パスの
    信号にならず、知識質問として規則層が否定し、kNN の門も 2/5 で閉じて、読まずに
    答えた (2026-10-05 ライブ監査 T4)。名前が利用者の名指したフォルダの実在のファイルに
    解決できること自体が「このファイルの話」という証拠で、近道 (知識質問の字句・kNN の
    門・「直し方 / 使い方を教えて」の教示の形) はこの証拠を上書きしない (不変則 #15)。

    探すのは利用者が名指したフォルダだけ (``named_only``: 依頼文・台帳の名指し・
    user の発話。LLM が自分で一覧した / 読んだフォルダと CWD は数えない)。次の
    ときは棄権して後続へ委ねる:

    - 名前が解決できない (どこにも無い / 曖昧)
    - コードを作る依頼の構造 (:data:`_CODE_CREATION_RE`)
    - 読み以外のツール (書込み・検索・実行) を ``infer`` (``ToolCallJudge._infer_tool``)
      が決めた — 書込みの宛先は書込みの規則で決める。中身を述べる状態の形
      (「何が書いてあるか」) は書込みと推定させない

    ファイルシステムを見るので、呼出側はイベントループの外 (executor) で呼ぶ。
    """
    if not tools_registry.is_available("read_file", mode):
        return None
    stripped = strip_command_literals(query)
    literal = _extract_file_path_literal(stripped)
    if literal and _PATH_SEPARATOR_RE.search(literal):
        import os

        # フォルダだけが書かれた依頼 (「X フォルダの README を読んで」) は名前の判定へ進む。
        if not os.path.isdir(literal):
            return None
        literal = ""
    if _CODE_CREATION_RE.search(query):
        logger.debug("Bare filename read abstained (code creation): %s", query[:60])
        return None
    from backend.free.agent.file_ledger import resolve_bare_filename

    resolution = None
    if literal:
        resolution = resolve_bare_filename(
            literal, query=query, conversation=conversation, named_only=True,
        )
    else:
        # 拡張子の無い定番の文書名 (「README を読んで」) は、実在する綴りへ解決する。
        # 拡張子が無いと名前として抽出されず、一覧だけ取って「中身が無い」と答えていた。
        # 探すのは利用者が名指ししたフォルダだけ (named_only) で、作業ルートは
        # 「このリポジトリ」と指したときだけ加える — 利用者が README と言うとき、
        # 製品自身のインストール根の README を指すとは限らない。文書の種類を問う質問
        # (「README とは」「README の書き方」) は読む依頼ではない。
        m = _WELL_KNOWN_DOC_RE.search(stripped)
        if m and not _DOC_KIND_QUESTION_RE.search(stripped):
            import os

            for ext in (".md", ".txt", ".rst", ""):
                name = m.group(1) + ext
                cand = resolve_bare_filename(
                    name, query=query, conversation=conversation, named_only=True,
                )
                if not cand.path and _REPO_REFERENCE_RE.search(stripped):
                    root = os.path.join(os.getcwd(), name)
                    if os.path.isfile(root):
                        cand = type(cand)(path=root)
                if cand.path:
                    literal, resolution = name, cand
                    break
    if resolution is None or not resolution.path:
        return None
    # 名前の綴りは依頼の語ではない (「run.py の使い方」の run を実行の語と読まない)。
    # 名前は中立の名前へ置いて、依頼の形 (書込み・検索・実行) だけを推定させる。
    neutral = _STATIVE_WRITTEN_RE.sub("記載", query.replace(literal, _NEUTRAL_FILENAME))
    inferred_tool, _ = infer(neutral)
    if inferred_tool and inferred_tool != "read_file":
        return None
    tool_args: dict = {"file_path": resolution.path}
    head = _extract_head_line_count(query)
    if head is not None:
        tool_args["start_line"] = 1
        tool_args["end_line"] = head
    logger.info(
        "Bare filename in the query resolved to an existing file: %s -> %s",
        literal, resolution.path,
    )
    return ToolJudgement(
        tool_needed=True,
        tool_name="read_file",
        tool_args=tool_args,
        source="rule",
    )
