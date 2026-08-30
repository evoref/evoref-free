"""会話に依存する対象パスの解決と参照型の判定

「同じファイルに保存し直して」「そのファイルの全文を見せて」のように、対象が
クエリ単体では確定しない依頼を、直近の会話からパスを引いて確定させる層。
確定できなければ ``None`` を返して後続層へ委ねる (推測でパスを埋めない)。
"""

from __future__ import annotations

import re

from backend.free.agent.tool_judge_args import (
    _extract_file_path,
    _extract_file_path_literal,
    _extract_head_line_count,
)
from backend.free.agent.tool_judge_types import ToolJudgement
from backend.free.agent.tools_registry import ToolsRegistry
from backend.log_config import get_logger

logger = get_logger("agent.tool_call_judge")

#: 「同じファイルに」「そのファイルを」等、保存先を直前の文脈に委ねる表現。
#: ``さきほど`` (ひらがな) は 2026-08-09 に追加。``先ほど`` / ``さっき`` しか
#: 無く、「さきほど作った notes.txt に追記して」が参照表現として認識されず
#: 書込みが 1 度も走らないまま完了を捏造していた。
_REFERENTIAL_TARGET_RE = re.compile(
    r"(?:同じ|その|この|先ほどの?|さきほどの?|さっきの?)\s*(?:ファイル|ところ|場所)"
    r"|保存し直|上書き|書き直して保存|同じ場所に"
    r"|\b(?:same|that)\s+file\b|\boverwrite\b",
    re.IGNORECASE,
)
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


def _resolve_referenced_path(
    query_path: str | None, conversation: list[dict] | None,
) -> str | None:
    """書込み/読取の対象パスを会話から解決する (純粋関数)。

    ``query_path`` の状態で 3 通りに分かれる:

    - ディレクトリを含む絶対/相対パス → そのまま採用 (解決不要)
    - 裸のファイル名 (``notes.txt``) → 会話に **同じ basename** の
      フルパスがあればそれを採用。無ければ ``None``
    - ``None`` / 空 (「そのファイル」型) → 会話で最後に出たパスを採用

    裸のファイル名をそのままツールへ渡すとカレントディレクトリに着地して
    しまい、ユーザーが指した既存ファイルとは別物を作る。会話で確定している
    場合のみ解決し、確定できなければ ``None`` を返して後続層に委ねる
    (推測でパスを埋めない)。
    """
    if query_path and _PATH_SEPARATOR_RE.search(query_path):
        return query_path
    want = (query_path or "").strip().lower() or None
    for msg in reversed(list(conversation or [])):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        path = _extract_file_path(content)
        if not path or not _PATH_SEPARATOR_RE.search(path):
            continue
        if want and _PATH_SEPARATOR_RE.split(path)[-1].lower() != want:
            continue
        return path
    return None


def _referential_rewrite_judgement(
    query: str, conversation: list[dict] | None, tools_registry: ToolsRegistry,
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
    if not _REWRITE_VERB_RE.search(query):
        return None
    if _path_is_written_in_query(query):
        return None  # ディレクトリ付きパスが本文にあるなら通常のルール層で足りる
    query_path = _extract_file_path(query) or None
    if not query_path and not _REFERENTIAL_TARGET_RE.search(query):
        return None
    path = _resolve_referenced_path(query_path, conversation)
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
    query_path = _extract_file_path(query) or None
    if not query_path and not (
        _REFERENTIAL_TARGET_RE.search(query) or _FILE_NOUN_RE.search(query)
    ):
        return None
    path = _resolve_referenced_path(query_path, conversation)
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
