"""この会話で直近に触れたファイルのパスをセッション単位で保持する。

2026-08-27 ライブ監査で、**暗黙参照の保存先/読み先が解決できず依頼が実行
されない**事象を 2 回観測した (再現率 2/2)::

    T13-6 「完成したコードを E:\\tmp\\...\\list_old_logs.py に保存してください。」
          → 書き込み成功
    T13-7 「保存したファイルを読んで、構文エラーがないか確認してください。」
          → 「ファイルの内容を確認できません。」(238 秒、read_file は **1 回も
             撃たれていない**)

    T15-5 「E:\\tmp\\...\\summary.md にまとめて保存してください。」→ 書き込み成功
    T15-7 「その中身を見せてください。」→ 「ファイルの内容を表示できません。」

いずれも実ファイルは存在し、明示パスを与えた T05 では ``read_file`` が正常に
動いていた。原因は ``tool_judge_args._extract_file_path`` が

    「保存したファイルを読んで、構文エラーがないか確認してください。」→ ""
    「その中身を見せてください。」                                  → ""

と空を返すこと。パスが決まらないのでツールが選ばれない。

``_extract_file_path`` はクエリの **文字列から** パスを取る。暗黙参照には
文字列としてのパスが無いので、いくら正規表現を足しても解けない。解けるのは
「**直前に何を書いたか**」という観測事実の側で、それは ``ToolsRegistry``
が実行時に知っている。``tool_ledger`` と同じ立て付けで記録する。
"""

from __future__ import annotations

import re
from collections import OrderedDict, deque
from contextvars import ContextVar

from backend.log_config import get_logger

logger = get_logger("agent.file_ledger")

__all__ = [
    "file_ledger_scope",
    "last_file_path",
    "record_current_file",
    "record_file",
    "references_recent_file",
    "reset",
]

#: 1 セッションあたりの保持件数 (新しい方を残す)。
MAX_ENTRIES_PER_SESSION = 12

#: 保持するセッション数。
MAX_SESSIONS = 16

_ledger: "OrderedDict[str, deque[str]]" = OrderedDict()


def _bucket(session_id: str) -> "deque[str]":
    existing = _ledger.get(session_id)
    if existing is not None:
        _ledger.move_to_end(session_id)
        return existing
    created: "deque[str]" = deque(maxlen=MAX_ENTRIES_PER_SESSION)
    _ledger[session_id] = created
    while len(_ledger) > MAX_SESSIONS:
        _ledger.popitem(last=False)
    return created


def record_file(session_id: str, path: str) -> None:
    """このセッションで触れたファイルのパスを記録する。

    同じパスを重ねて記録しない (最後に触れた順を保つため、既存を消して
    末尾へ積み直す)。
    """
    cleaned = (path or "").strip().strip("\"'")
    if not session_id or not cleaned:
        return
    bucket = _bucket(session_id)
    while cleaned in bucket:
        bucket.remove(cleaned)
    bucket.append(cleaned)


#: 現在のリクエストの ``session_id``。``tool_ledger`` と同じ理由で contextvar
#: に置く — 記録点は ``ToolsRegistry.execute`` の 1 つだが、宛先は呼出側が
#: 決めるため。
_current_session: ContextVar[str | None] = ContextVar(
    "file_ledger_session", default=None,
)


def file_ledger_scope(session_id: str):
    """``record_current_file`` の宛先を設定する。"""
    return _current_session.set(session_id or "")


def record_current_file(path: str) -> None:
    """現在のリクエストの宛先へファイルパスを記録する。"""
    session_id = _current_session.get()
    if session_id:
        record_file(session_id, path)


def last_file_path(session_id: str) -> str:
    """このセッションで最後に触れたファイルのパス (無ければ空文字)。"""
    bucket = _ledger.get(session_id)
    return bucket[-1] if bucket else ""


def reset(session_id: str | None = None) -> None:
    """台帳を消す (``None`` で全消去)。"""
    if session_id is None:
        _ledger.clear()
        return
    _ledger.pop(session_id, None)


#: 直近のファイルを指す **指示** の形。
#:
#: 語彙でファイルの種類を数えない。見るのは指示詞という閉じた文法クラスと、
#: 「保存した/書いた」という **過去の自分の操作** への参照だけ。
#: 「何を指すか」は「直前にファイルを書いた」という観測事実が決める。
_IMPLICIT_FILE_REF_RE = re.compile(
    r"(?:その|それ|この|これ|さっきの|先ほどの|いまの|今の|上記の"
    r"|保存した|書き込んだ|書いた|作った|作成した|出力した"
    r"|that|the same)",
)

#: ファイルそのものを対象にしていることを示す語。指示詞だけだと直前の
#: 「文章」「計算」など別の対象まで拾う。
_FILE_OBJECT_RE = re.compile(
    r"(?:ファイル|中身|内容|中身|保存先|パス|file|contents?)",
)


def references_recent_file(query: str) -> bool:
    """``query`` が直近に触れたファイルを **明示パス無しで** 指しているか。

    **呼出側は「パスが抽出できなかった」ことを既に確認している** 前提。
    その上で、この発話がファイルを対象にした指示参照かを見る。

    「保存したファイルを読んで」「その中身を見せて」の 2 つが実際に落ちた形
    (モジュール docstring 参照)。どちらも指示/過去参照 + ファイル対象語。
    """
    if not query:
        return False
    return bool(
        _IMPLICIT_FILE_REF_RE.search(query) and _FILE_OBJECT_RE.search(query),
    )
