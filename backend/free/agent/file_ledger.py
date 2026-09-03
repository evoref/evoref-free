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

import os
import re
from collections import OrderedDict, deque
from contextvars import ContextVar

from backend.log_config import get_logger

logger = get_logger("agent.file_ledger")

__all__ = [
    "file_ledger_scope",
    "forget_current_file",
    "forget_file",
    "last_file_path",
    "record_current_file",
    "record_file",
    "references_recent_file",
    "resolve_against_recent_dir",
    "resolve_current_against_recent_dir",
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


def forget_file(session_id: str, path: str) -> bool:
    """記録済みのパスを取り消す (取り消せたら True)。

    ``ToolsRegistry.execute`` は戻り値だけで ``write_file`` の成功を記録するため、
    書込後の読み戻し突合で失敗と分かった時点では、壊れたファイルが「直近に触れた
    ファイル」として残っている。そのままだと次ターンの「保存したファイルを読んで」
    が壊れたファイルへ向く。呼出側 (meta 経路の ``_write_file``) が失敗確定直後に
    呼び、その前に触れていたファイルを直近へ戻す。
    """
    cleaned = (path or "").strip().strip("\"'")
    bucket = _ledger.get(session_id)
    if not bucket or cleaned not in bucket:
        return False
    while cleaned in bucket:
        bucket.remove(cleaned)
    return True


def forget_current_file(path: str) -> bool:
    """現在のリクエストの宛先から ``path`` の記録を取り消す。"""
    session_id = _current_session.get()
    if not session_id:
        return False
    return forget_file(session_id, path)


def last_file_path(session_id: str) -> str:
    """このセッションで最後に触れたファイルのパス (無ければ空文字)。"""
    bucket = _ledger.get(session_id)
    return bucket[-1] if bucket else ""


def resolve_against_recent_dir(session_id: str, path: str) -> str:
    """裸のファイル名を「この会話で使っているディレクトリ」へ寄せる。

    ``note2.md`` のようにディレクトリを伴わない名前は、そのまま渡すと
    バックエンドプロセスの cwd (= リポジトリ直下) に落ちる。会話の文脈では
    「直前に扱ったファイルと同じ場所」を指しているので、台帳の最新エントリの
    親ディレクトリへ寄せる。

    実インシデント 2026-09-03 ライブ監査 T06#7:
    「note1.txt の内容を…別ファイル note2.md に保存して」で
    ``E:\\tmp\\audit_20260903\\note1.txt`` を読んだ直後の書込みが
    **リポジトリ直下の note2.md** になり、ユーザーの作業ツリーを汚した。
    さらに次ターンは存在しない ``E:\\tmp\\audit_20260903\\note2.md`` を
    読んだと答えた (台帳に相対パスのまま入るため突合もできない)。

    寄せるのは **区切りを 1 つも含まない名前だけ**。``sub/a.txt`` のような
    相対パスはユーザーが構造を書いているので触らない。
    """
    cleaned = (path or "").strip().strip("\"'")
    if not cleaned or not session_id:
        return path
    if os.path.isabs(cleaned) or "/" in cleaned or "\\" in cleaned:
        return path
    recent = last_file_path(session_id)
    if not recent:
        return path
    parent = os.path.dirname(recent)
    if not parent:
        return path
    resolved = os.path.join(parent, cleaned)
    logger.info(
        "Resolved bare filename against the conversation's directory: "
        "%s -> %s", cleaned, resolved,
    )
    return resolved


def resolve_current_against_recent_dir(path: str) -> str:
    """現在のリクエストの宛先で :func:`resolve_against_recent_dir` を掛ける。"""
    session_id = _current_session.get()
    if not session_id:
        return path
    return resolve_against_recent_dir(session_id, path)


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
