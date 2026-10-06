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
import threading
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import dataclass

from backend.free.core.intent_vocab import EN_DESTINATION_BEFORE_RE, PAST_FILE_OPERATION_PATTERN
from backend.i18n_helper import template_line_patterns
from backend.log_config import get_logger

logger = get_logger("agent.file_ledger")

__all__ = [
    "FileStatus",
    "Resolution",
    "current_session",
    "file_ledger_scope",
    "file_statuses",
    "folder_role",
    "invalidate_resolutions",
    "is_bare_filename",
    "ledger_folders",
    "named_file_paths",
    "named_folder_paths",
    "record_current_dir",
    "record_dir",
    "record_file_status",
    "record_listed_folder",
    "forget_current_file",
    "forget_file",
    "last_file_path",
    "last_written_path",
    "record_current_file",
    "record_file",
    "references_recent_file",
    "resolution_error_places",
    "resolve_bare_filename",
    "session_has_file",
    "reset",
    "restore_from_conversation",
    "starts_with_write_report",
    "user_texts",
    "written_paths_in_conversation",
    "written_paths_in_text",
]

#: 1 セッションあたりのファイルの記録の保持件数 (新しい方を残す)。
MAX_ENTRIES_PER_SESSION = 12
#: 1 セッションあたりのフォルダの記録の保持件数。ファイルとは別の枠 — 同じ枠だと
#: フォルダを一覧するたびに「そのファイル」が指すファイルの記録が押し出される。
MAX_DIRS_PER_SESSION = 12

#: 保持するセッション数。
MAX_SESSIONS = 16

#: 台帳の構造 (``_ledger`` / ``_dirs`` / ``_status``) の排他。記録点
#: (``ToolsRegistry.execute``) はスレッド (``asyncio.to_thread``) から書く。
_lock = threading.RLock()


@dataclass(frozen=True, slots=True)
class _Entry:
    """台帳のファイルの 1 件。

    ``named`` は **記録した時点で** そのパスが依頼文由来の根 (依頼文が挙げた
    フォルダ / 名指しの記録の親) の配下にあったか。書込みゲート (docs/f_03 §4.y)
    は ``named`` の記録の親フォルダだけを「書いてよい根」に足す — LLM が自分で
    選んで読んだパスを、次のターンの書込みの根にしないため。
    """

    path: str
    named: bool = False
    #: 最後に **書いた** 順番 (0 は書いていない = 読んだだけ)。「保存したファイル」の
    #: 宛先は最後に書いたファイルで、最後に触れたファイルではない
    #: (2026-09-28 レビュー H1: notes.md を書いた後に config.yaml を読むと、宛先が
    #: config.yaml になっていた)。
    written_seq: int = 0
    #: 最後に触れた順番 (フォルダの記録と並べて新しい順にするため)。
    touched_seq: int = 0


@dataclass(frozen=True, slots=True)
class _DirEntry:
    """台帳のフォルダの 1 件 (依頼文が挙げた / ``list_directory`` / ``search_code`` の対象)。

    裸のファイル名の **読み** の探し場所にだけ使う。書込みの宛先・書込みゲートの根には
    しない — 一覧しただけのフォルダへ書かせない。
    """

    path: str
    named: bool = False
    touched_seq: int = 0


_ledger: "OrderedDict[str, deque[_Entry]]" = OrderedDict()
_dirs: "OrderedDict[str, deque[_DirEntry]]" = OrderedDict()
#: ``written_seq`` の採番 (プロセス内で単調増加)。
_write_counter = 0
#: ``touched_seq`` の採番 (プロセス内で単調増加)。
_touch_counter = 0
#: 台帳の世代 (記録が変わるたびに上がる)。解決のキャッシュの鍵に入れる。
_generation = 0


def _bump() -> int:
    """触れた順番を採番し、台帳の世代を上げる (``_lock`` の中で呼ぶ)。"""
    global _touch_counter, _generation
    _touch_counter += 1
    _generation += 1
    return _touch_counter


def invalidate_resolutions() -> None:
    """解決のキャッシュを無効にする (台帳の世代を上げる)。

    ツールの実行はファイルを作りうる (``run_command`` が書いたファイルを、同じ
    リクエストの後の読みが「見つからない」と覚えたままにしない)。``ToolsRegistry.execute``
    がツールを実行するたびに呼ぶ。
    """
    global _generation
    with _lock:
        _generation += 1


def _session_deque(store: OrderedDict, session_id: str, maxlen: int) -> deque:
    existing = store.get(session_id)
    if existing is not None:
        store.move_to_end(session_id)
        return existing
    created: deque = deque(maxlen=maxlen)
    store[session_id] = created
    while len(store) > MAX_SESSIONS:
        store.popitem(last=False)
    return created


def _bucket(session_id: str) -> "deque[_Entry]":
    return _session_deque(_ledger, session_id, MAX_ENTRIES_PER_SESSION)


def _remove(bucket: deque, path: str) -> tuple[bool, int]:
    """``path`` の記録をすべて外す。``(どれかが named か, 最後に書いた順番)``。"""
    named = False
    written_seq = 0
    for entry in [e for e in bucket if e.path == path]:
        named = named or entry.named
        written_seq = max(written_seq, getattr(entry, "written_seq", 0))
        bucket.remove(entry)
    return named, written_seq


def record_file(
    session_id: str, path: str, *, named: bool = False, written: bool = False,
) -> None:
    """このセッションで触れたファイルのパスを記録する。

    同じパスを重ねて記録しない (最後に触れた順を保つため、既存を消して
    末尾へ積み直す)。一度 ``named`` で記録したパスは、後で名指し無しに触れても
    ``named`` のまま (「同じファイルに追記して」の次のターンも書ける)。
    ``written`` (書き込んだ) の記録は最後に書いた順番を更新し、読んだだけの
    記録はそれを保つ (:func:`last_written_path`)。
    """
    global _write_counter
    cleaned = (path or "").strip().strip("\"'")
    if not session_id or not cleaned:
        return
    with _lock:
        bucket = _bucket(session_id)
        was_named, written_seq = _remove(bucket, cleaned)
        if written:
            _write_counter += 1
            written_seq = _write_counter
        bucket.append(_Entry(cleaned, was_named or named, written_seq, _bump()))


#: 現在のリクエストの ``session_id``。``tool_ledger`` と同じ理由で contextvar
#: に置く — 記録点は ``ToolsRegistry.execute`` の 1 つだが、宛先は呼出側が
#: 決めるため。
_current_session: ContextVar[str | None] = ContextVar(
    "file_ledger_session", default=None,
)
#: 現在のリクエストの会話 (裸のファイル名の読みの探し場所。user の発話だけを見る)。
_current_conversation: ContextVar[tuple | None] = ContextVar(
    "file_ledger_conversation", default=None,
)
#: 現在のリクエストの解決のキャッシュ (``file_ledger_scope`` がリクエストごとに作る)。
#: 判定器の層は同じ名前を何度も解決し、そのたびに複数のフォルダを stat していた。
_resolution_cache: ContextVar[dict | None] = ContextVar(
    "file_ledger_resolution_cache", default=None,
)


def file_ledger_scope(session_id: str, conversation=None):
    """``record_current_file`` の宛先を設定する。

    ``conversation`` (このリクエストの会話) は裸のファイル名の探し場所に使う
    (:func:`resolve_bare_filename`)。戻り値はセッションの contextvar のトークン。
    """
    _current_conversation.set(tuple(conversation) if conversation else None)
    _resolution_cache.set({})
    return _current_session.set(session_id or "")


def record_current_file(path: str, *, named: bool = False, written: bool = False) -> None:
    """現在のリクエストの宛先へファイルパスを記録する。"""
    session_id = _current_session.get()
    if session_id:
        record_file(session_id, path, named=named, written=written)


def forget_file(session_id: str, path: str) -> bool:
    """記録済みのパスを取り消す (取り消せたら True)。

    ``ToolsRegistry.execute`` は戻り値だけで ``write_file`` の成功を記録するため、
    書込後の読み戻し突合で失敗と分かった時点では、壊れたファイルが「直近に触れた
    ファイル」として残っている。そのままだと次ターンの「保存したファイルを読んで」
    が壊れたファイルへ向く。呼出側 (meta 経路の ``_write_file``) が失敗確定直後に
    呼び、その前に触れていたファイルを直近へ戻す。
    """
    cleaned = (path or "").strip().strip("\"'")
    with _lock:
        bucket = _ledger.get(session_id)
        if not bucket or not any(e.path == cleaned for e in bucket):
            return False
        _remove(bucket, cleaned)
        _bump()
        return True


def forget_current_file(path: str) -> bool:
    """現在のリクエストの宛先から ``path`` の記録を取り消す。"""
    session_id = _current_session.get()
    if not session_id:
        return False
    return forget_file(session_id, path)


def last_file_path(session_id: str, *, named_only: bool = False) -> str:
    """このセッションで最後に触れたファイルのパス (無ければ空文字)。

    ``named_only`` なら ``named`` の記録だけを見る。
    """
    with _lock:
        entries = list(_ledger.get(session_id) or ())
    for entry in reversed(entries):
        if entry.named or not named_only:
            return entry.path
    return ""


def last_written_path(session_id: str) -> str:
    """このセッションで最後に **書いた** ファイルのパス (無ければ空文字)。

    「保存したファイルに追記して」「保存したファイルの場所」の宛先はこちら
    (docs/f_03 §1.6)。読んだだけのファイルは対象にしない。会話の書込み報告からの
    復元 (:func:`restore_from_conversation`) も書込みとして記録するので、再起動の
    前後で同じファイルになる。
    """
    with _lock:
        written = [e for e in _ledger.get(session_id) or () if e.written_seq]
    if not written:
        return ""
    return max(written, key=lambda e: e.written_seq).path


def named_file_paths(session_id: str) -> list[str]:
    """このセッションの ``named`` のファイルの記録 (古い順)。"""
    with _lock:
        return [e.path for e in _ledger.get(session_id) or () if e.named]


def session_has_file(session_id: str, path: str) -> bool:
    """このセッションで ``path`` のファイルに触れた (読んだ / 書いた) 記録があるか。

    参照表現 (「このファイル」「同じファイル」) が指しうるのは会話で扱ったファイルだけ —
    書込みの対象の判定点 ``overwrite_target`` (c_17 §3.21) が使う。比較は大文字小文字と
    区切りを揃えた絶対パスで行う。
    """
    if not session_id or not path:
        return False
    key = os.path.normcase(os.path.abspath(path))
    with _lock:
        entries = list(_ledger.get(session_id) or ())
    return any(os.path.normcase(os.path.abspath(e.path)) == key for e in entries)


# ─────────────────────────────────────────────────────────────────────
# フォルダの記録と裸のファイル名の解決 (読み書きの入口が共有する 1 本)
# ─────────────────────────────────────────────────────────────────────


def record_dir(session_id: str, path: str, *, named: bool = False) -> None:
    """このセッションで扱ったフォルダを記録する (裸のファイル名の読みの探し場所)。

    依頼文が挙げたフォルダ、``list_directory`` / ``search_code`` の対象を積む。
    同じフォルダは末尾へ積み直す (新しい順を保つ)。一度 ``named`` で記録した
    フォルダは ``named`` のまま。
    """
    cleaned = (path or "").strip().strip("\"'")
    if len(cleaned) > 3:
        cleaned = cleaned.rstrip("\\/")
    if not session_id or not cleaned:
        return
    with _lock:
        bucket = _session_deque(_dirs, session_id, MAX_DIRS_PER_SESSION)
        was_named, _ = _remove(bucket, cleaned)
        bucket.append(_DirEntry(cleaned, was_named or named, _bump()))


def record_current_dir(path: str, *, named: bool = False) -> None:
    """現在のリクエストの宛先へフォルダを記録する。"""
    session_id = _current_session.get()
    if session_id:
        record_dir(session_id, path, named=named)


def ledger_folders(session_id: str, *, named_only: bool = False) -> list[str]:
    """台帳のフォルダ (フォルダの記録と、記録したファイルの親) を新しい順に返す。

    ``named_only`` なら依頼文由来 (``named``) の記録だけ。
    """
    with _lock:
        entries = [
            (e.touched_seq, e.path, e.named)
            for e in _dirs.get(session_id) or ()
        ] + [
            (e.touched_seq, os.path.dirname(e.path), e.named)
            for e in _ledger.get(session_id) or ()
        ]
    out: list[str] = []
    seen: set[str] = set()
    for _seq, folder, named in sorted(entries, key=lambda t: t[0], reverse=True):
        if not folder or (named_only and not named):
            continue
        key = _folder_key(folder)
        if key not in seen:
            seen.add(key)
            out.append(folder)
    return out


def named_folder_paths(session_id: str | None = None) -> list[str]:
    """利用者が名指したフォルダ (書込みゲートの根、docs/f_03 §4.y)、新しい順。

    ``named`` のフォルダの記録 (依頼文が挙げたフォルダ) と ``named`` のファイルの記録の
    親。ツールが自分で一覧しただけのフォルダ (``named=False``) は含めない — 「E:\\work の
    中を見て」の後の「summary.md に保存して」は E:\\work へ書けるが、LLM が選んで一覧した
    フォルダへは書かない。``session_id`` を省くと現在のリクエストの宛先を見る。
    """
    sid = session_id if session_id is not None else _current_session.get()
    if not sid:
        return []
    return ledger_folders(sid, named_only=True)


def _folder_key(path: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError):
        return os.path.normcase(path)


def _is_file(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except (OSError, ValueError):
        return False


def _is_unc(path: str) -> bool:
    """UNC (ネットワーク共有) のパスか。"""
    text = str(path or "")
    return text.startswith(("\\\\", "//"))


_DRIVE_QUALIFIED_RE = re.compile(r"^[A-Za-z]:")


def is_bare_filename(path: str) -> bool:
    """区切りを 1 つも含まない名前か (``staff.csv``)。

    ``sub/a.txt`` は構造を書いている。ドライブ付きの ``C:foo.txt`` (ドライブ相対) も
    裸の名前ではない。
    """
    cleaned = (path or "").strip().strip("\"'")
    return bool(cleaned) and not (
        os.path.isabs(cleaned) or "/" in cleaned or "\\" in cleaned
        or _DRIVE_QUALIFIED_RE.match(cleaned)
    )


def _message_text(msg) -> str:
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    return content if isinstance(content, str) else ""


def _message_role(msg) -> str:
    role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
    return role if isinstance(role, str) else ""


def user_texts(conversation) -> list[str]:
    """会話の user の発話 (新しい順)。

    探し場所・素材の名指しは利用者が書いた文だけから取る — assistant の発話 (要約した
    ページ・読んだファイルの中身を含む) に書かれたパスを探しに行くと、注入された
    ``\\\\host\\share`` を stat して資格情報を送る (2026-10-05 レビュー H2)。
    """
    return [
        _message_text(m) for m in reversed(list(conversation or ()))
        if _message_role(m) == "user" and _message_text(m)
    ]


def _conversation_folders(conversation) -> list[str]:
    """user の発話に書かれたフォルダ (新しい発話から、発話の中は出現順)。"""
    from backend.free.agent.write_gate import request_named_folders

    out: list[str] = []
    for text in user_texts(conversation):
        # 過去の発話の UNC は抜き出す前に消す (抜き出しの is_dir も共有へ届く)
        text = _UNC_TOKEN_RE.sub(" ", text)
        out.extend(str(folder) for folder in request_named_folders(text))
    return out


#: 文中の UNC / ``//host/share`` の語 (過去の発話から探し場所を取るときに消す)。
_UNC_TOKEN_RE = re.compile(r"(?:\\\\|//)[^\s\"'「」『』()（）]+")


def _conversation_paths_named(conversation, name: str) -> list[str]:
    """user の発話に書かれた ``name`` と同じ basename のフルパス (新しい発話から)。"""
    from backend.free.agent.tool_judge_args import _extract_file_path_literal

    want = name.lower()
    out: list[str] = []
    for text in user_texts(conversation):
        found = _extract_file_path_literal(_UNC_TOKEN_RE.sub(" ", text))
        if found and ("\\" in found or "/" in found) and not _is_unc(found):
            if re.split(r"[\\/]", found)[-1].lower() == want:
                out.append(found)
    return out


@dataclass(frozen=True, slots=True)
class Resolution:
    """裸のファイル名の解決結果。

    ``path`` は採用したパス (決められなければ ``None``)。``candidates`` は実在した
    候補 (2 つ以上で ``path`` が ``None`` なら曖昧)、``searched`` は探したフォルダ
    (優先順)。区切りを含む名前は解決の対象外で、``bare=False`` / ``path`` はそのまま。
    """

    path: str | None
    candidates: tuple[str, ...] = ()
    searched: tuple[str, ...] = ()
    bare: bool = True
    #: 読みで実在の候補が無いとき、user の発話に書かれた同じ basename のフルパス (無ければ
    #: ``None``)。「memo_b.txt の中身は何」の memo_b.txt が未作成でも、読みを撃って
    #: 「見つからない」と答えさせる材料 (撃たないと中身を作話する、2026-08-29)。
    mentioned: str | None = None

    @property
    def ambiguous(self) -> bool:
        """複数のフォルダに同じ名前があって決められなかったか。"""
        return self.path is None and len(self.candidates) > 1

    def error_message(self, name: str) -> str:
        """解決できなかったときのツール結果 (``Error: ...``、探した場所 / 候補つき)。"""
        from backend.free.constants import FILE_AMBIGUOUS_PREFIX, FILE_NOT_FOUND_PREFIX

        if self.ambiguous:
            return (
                f"{FILE_AMBIGUOUS_PREFIX}: {name} "
                f"(candidates: {'; '.join(self.candidates)})"
            )
        return f"{FILE_NOT_FOUND_PREFIX}: {name} (searched: {'; '.join(self.searched)})"


#: :meth:`Resolution.error_message` の末尾の一覧の見出し。
_RESOLUTION_LIST_HEADS = ("(searched: ", "(candidates: ")


def resolution_error_places(text: str) -> tuple[str, ...]:
    """:meth:`Resolution.error_message` の末尾の一覧 (探した場所 / 候補) を取り出す。

    自前の形式だけを見る (字句の鍵、不変則 #14)。一覧が無ければ空。
    """
    text = (text or "").rstrip()
    for head in _RESOLUTION_LIST_HEADS:
        at = text.rfind(head)
        if at >= 0 and text.endswith(")"):
            body = text[at + len(head):-1]
            return tuple(p for p in body.split("; ") if p)
    return ()


def _dedupe(folders: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for folder in folders:
        if not folder:
            continue
        key = _folder_key(folder)
        if key not in seen:
            seen.add(key)
            out.append(folder)
    return out


def resolve_bare_filename(
    name: str,
    *,
    query: str | None = None,
    conversation=None,
    for_write: bool = False,
    session_id: str | None = None,
    named_only: bool = False,
) -> Resolution:
    """裸のファイル名を、この会話の文脈で確定しているフォルダへ解決する (読み書き共通の 1 本)。

    裸の名前をそのままツールへ渡すとプロセスの CWD (= インストール根) を見る。
    2026-10-05 ライブ監査: フォルダを一覧した直後の「staff.csv を読んで」が
    ``File not found: staff.csv`` になり、続くターンで給与の数字と人名を作話した。
    以前は読みの入口 3 つ (meta の ``resolve_read_path`` / 判定器の参照解決 /
    レジストリ) がそれぞれ別の探し場所を持っていた。

    読み (``for_write=False``) の探す順:

    1. 依頼文が挙げたフォルダ
    2. 台帳のフォルダ (フォルダの記録・触れたファイルの親、新しい順)
    3. user の発話に書かれたフォルダ (新しい発話から — 再起動で台帳が空でも解ける)
    4. プロセスの CWD

    依頼文のフォルダにあればそれ、実在する候補が 1 つならそれ、複数のフォルダに
    あれば曖昧 (``path=None``、``candidates`` に列挙)、無ければ ``path=None``。
    CWD は 1〜3 のどこにも無いときだけ候補にする (曖昧さに数えない)。UNC のフォルダは
    **今の依頼文が挙げたときだけ** 探す (過去の発話・台帳の UNC を stat しない)。
    ``named_only`` は依頼文由来のフォルダだけ (台帳の ``named`` と user の発話、CWD は
    含めない)。

    書き (``for_write=True``) は :func:`_resolve_for_write` (依頼文のフォルダのうち
    出力の名前に付いたもの / 名指しのファイルの親 / user の発話の同じ名前のフルパス、
    どれも無ければ ``None`` = 呼出側が ``outputs_dir`` へ寄せる)。

    ``query`` / ``conversation`` / ``session_id`` を省くと現在のリクエストの値を使う。
    結果はリクエストごとに (台帳の世代つきで) キャッシュする。
    """
    cleaned = (name or "").strip().strip("\"'")
    if not is_bare_filename(cleaned):
        return Resolution(path=name, bare=False)
    if query is None:
        from backend.free.agent.tool_ledger import current_query

        query = current_query()
    if conversation is None:
        conversation = _current_conversation.get()
    sid = session_id if session_id is not None else (_current_session.get() or "")
    if not sid:
        from backend.free.agent.tool_ledger import current_session_id

        sid = current_session_id()

    cache = _resolution_cache.get()
    cache_key = (
        cleaned.lower(), query or "", for_write, named_only, sid,
        hash(tuple(user_texts(conversation))), _generation,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    result = _resolve_uncached(cleaned, query or "", conversation, for_write, sid, named_only)
    if cache is not None:
        cache[cache_key] = result
    return result


def _resolve_uncached(
    cleaned: str, query: str, conversation, for_write: bool, sid: str, named_only: bool,
) -> Resolution:
    from backend.free.agent.write_gate import request_named_folders

    query_dirs = _dedupe([str(p) for p in request_named_folders(query)])
    if for_write:
        return _resolve_for_write(cleaned, query, query_dirs, sid, conversation)

    query_keys = {_folder_key(d) for d in query_dirs}
    cwd = os.getcwd()
    others = _dedupe(
        (ledger_folders(sid, named_only=named_only) if sid else [])
        + _conversation_folders(conversation)
        + ([] if named_only else [cwd]),
    )
    # UNC は今の依頼文が挙げたときだけ (台帳・過去の発話の共有へ資格情報を送らない)
    searched = _dedupe(
        query_dirs
        + [f for f in others if not _is_unc(f) or _folder_key(f) in query_keys],
    )
    hits: list[str] = []
    hit_keys: set[str] = set()
    for folder in searched:
        if hits and _folder_key(folder) == _folder_key(cwd):
            # CWD (インストール根) は最後の手段 — 会話のフォルダで見つかっていれば
            # README.md / config.yaml のような名前で曖昧にしない。
            continue
        candidate = os.path.join(folder, cleaned)
        if not _is_file(candidate):
            continue
        try:
            key = os.path.normcase(os.path.realpath(candidate))
        except (OSError, ValueError):
            key = _folder_key(candidate)
        if key not in hit_keys:
            hit_keys.add(key)
            hits.append(candidate)
    in_query = [h for h in hits if _folder_key(os.path.dirname(h)) in query_keys]
    if in_query:
        chosen: str | None = in_query[0]
    elif len(hits) == 1:
        chosen = hits[0]
    else:
        chosen = None
    if chosen:
        logger.info("Resolved bare filename: %s -> %s", cleaned, chosen)
    elif hits:
        logger.info(
            "Bare filename is ambiguous: %s (candidates=%d)", cleaned, len(hits),
        )
    mentioned = None
    if not hits:
        same = _conversation_paths_named(conversation, cleaned)
        mentioned = same[0] if same else None
    return Resolution(
        path=chosen, candidates=tuple(hits), searched=tuple(searched), mentioned=mentioned,
    )


def folder_role(query: str, folder: str) -> str:
    """依頼文の中でのフォルダの役 (``destination`` / ``source`` / ``""``)。

    フォルダの直後の助詞と直前の前置詞だけを見る (構造の鍵):

    - 宛先: 「<フォルダ> に (保存/置いて/作って/書いて …)」「<フォルダ> へ」「to / into <フォルダ>」
      (「にある」「に入っている」「に置かれた」は所在なので宛先ではない)
    - 元: 「<フォルダ> の <ファイル>」「<フォルダ> から」「from / in <フォルダ>」
    """
    text = (query or "").replace("/", "\\")
    needle = str(folder).replace("/", "\\").rstrip("\\")
    lowered = text.lower()
    # フォルダの名前で終わる出現だけ (親フォルダが子のパスの途中に当たらないように)
    ends = [
        m.start() for m in re.finditer(re.escape(needle.lower()), lowered)
        if not re.match(r"\\?[A-Za-z0-9_.\-]", lowered[m.end():m.end() + 2])
    ]
    if not ends:
        return ""
    at = ends[-1]
    after = text[at + len(needle):].lstrip("\\").lstrip()
    # 「<フォルダ> フォルダに」— 置き場の名詞を挟んでも同じ役 (2026-10-05 独立レビュー H1)
    after = _FOLDER_NOUN_RE.sub("", after, count=1)
    before = text[:at].rstrip()
    if _DESTINATION_AFTER_RE.match(after) or _DESTINATION_BEFORE_RE.search(before):
        return "destination"
    if _SOURCE_AFTER_RE.match(after) or _SOURCE_BEFORE_RE.search(before):
        return "source"
    return ""


#: フォルダのパスの直後の置き場の名詞 (役はその後ろの助詞で読む)。
_FOLDER_NOUN_RE = re.compile(r"^(?:フォルダー?|ディレクトリ)\s*")
#: 宛先の印 (フォルダの直後)。所在の「にある / に入って / に置かれ / に格納され」は除く。
_DESTINATION_AFTER_RE = re.compile(r"(?:へ|に(?!ある|あった|あります|入っ|置かれ|格納され))")
#: 宛先の印 (フォルダの直前、英語)。
_DESTINATION_BEFORE_RE = EN_DESTINATION_BEFORE_RE
#: 元の印 (フォルダの直後)。
_SOURCE_AFTER_RE = re.compile(r"(?:の|から|にある|にあった|に入っ)")
#: 元の印 (フォルダの直前、英語)。
_SOURCE_BEFORE_RE = re.compile(r"(?:\bfrom|\bin)$", re.IGNORECASE)


def _resolve_for_write(
    name: str, query: str, query_dirs: list[str], session_id: str, conversation,
) -> Resolution:
    """書込みの裸の名前 (:func:`resolve_bare_filename` の ``for_write=True``)。

    依頼文のフォルダを見るのは、出力の名前が依頼文に書かれているときだけ。依頼文が
    フォルダを 1 つだけ挙げたらそこ (既にあるファイルなら追記・上書きの宛先)。
    複数なら **宛先の印** の付いたフォルダ (「E:\\B に保存」「E:\\B へ」「to E:\\B」) で、
    元の印の付いたフォルダ (「E:\\A の data.csv」「E:\\A から」) は宛先にしない。既にある
    ファイルを優先しない (元のフォルダの同じ名前を上書きしない)。宛先の印が 1 つに
    決まらなければ決めない (呼出側が ``outputs_dir`` へ寄せる)。依頼文にフォルダが無ければ
    利用者が名指したフォルダ (既にあるファイルを優先し、無ければ最新)、user の発話の
    同じ名前のフルパスの順。ツールが一覧しただけのフォルダは宛先にしない。
    """
    named = [f for f in named_folder_paths(session_id) if not _is_unc(f)] if session_id else []
    # 依頼文のフォルダが決めるのは、依頼文が書いた出力の名前だけ。名前が依頼文に無い
    # 書込み (LLM が自分で付けた名前) は従来どおり名指しの記録か ``outputs_dir`` へ。
    if name.lower() not in (query or "").lower():
        query_dirs = []
    searched = tuple(_dedupe(query_dirs + named))
    if len(query_dirs) == 1:
        return Resolution(os.path.join(query_dirs[0], name), searched=searched)
    if query_dirs:
        roles = {folder: folder_role(query, folder) for folder in query_dirs}
        marked = [f for f, role in roles.items() if role == "destination"]
        if len(marked) == 1:
            return Resolution(os.path.join(marked[0], name), searched=searched)
        return Resolution(None, searched=searched)
    for folder in named:
        if _is_file(os.path.join(folder, name)):
            return Resolution(os.path.join(folder, name), searched=searched)
    if named:
        return Resolution(os.path.join(named[0], name), searched=searched)
    same = _conversation_paths_named(conversation, name)
    if same:
        return Resolution(same[0], searched=searched)
    return Resolution(None, searched=searched)


# ─────────────────────────────────────────────────────────────────────
# ファイルごとの状態 (一覧した / 読めた / 読めなかった)
# ─────────────────────────────────────────────────────────────────────

#: 状態の語彙 (閉じた集合)。
FILE_LISTED = "listed"
FILE_READ_OK = "read_ok"
FILE_READ_FAILED = "read_failed"
FILE_WRITTEN = "written"

#: 1 セッションあたりの状態の保持件数。
MAX_STATUS_PER_SESSION = 64
#: 一覧したフォルダから状態へ積むファイルの上限 (1 フォルダあたり)。
_MAX_LISTED_FILES = 32
#: 上限を超えたときに先に押し出す状態の順 (一覧しただけ → 読めなかった → 読めた / 書いた)。
#: 読めた / 書いたの記録が押し出されると「中身を読めたファイル: なし」と偽りを言う。
_EVICTION_ORDER = (FILE_LISTED, FILE_READ_FAILED, FILE_READ_OK, FILE_WRITTEN)


@dataclass(frozen=True, slots=True)
class FileStatus:
    """このセッションでのファイル 1 件の状態。"""

    path: str
    status: str
    error_kind: str = ""
    searched: tuple[str, ...] = ()


_status: "OrderedDict[str, OrderedDict[str, FileStatus]]" = OrderedDict()


def _evict(bucket: "OrderedDict[str, FileStatus]") -> None:
    """上限を超えた分を、押し出す順 (:data:`_EVICTION_ORDER`) の古いものから外す。"""
    for status in _EVICTION_ORDER:
        while len(bucket) > MAX_STATUS_PER_SESSION:
            victim = next((k for k, v in bucket.items() if v.status == status), None)
            if victim is None:
                break
            del bucket[victim]


def record_file_status(
    session_id: str, path: str, status: str, *,
    error_kind: str = "", searched: tuple[str, ...] = (),
) -> None:
    """ファイルの状態を記録する。

    ``listed`` は既に読んだ / 読めなかった記録を上書きしない。読めた (書いた) 記録は、
    同じ名前の読めなかった記録 (裸の名前で失敗した分) を消す。
    """
    cleaned = (path or "").strip().strip("\"'")
    if not session_id or not cleaned:
        return
    with _lock:
        bucket = _status.get(session_id)
        if bucket is None:
            bucket = OrderedDict()
            _status[session_id] = bucket
            while len(_status) > MAX_SESSIONS:
                _status.popitem(last=False)
        key = os.path.normcase(cleaned)
        existing = bucket.get(key)
        if status == FILE_LISTED and existing is not None and existing.status != FILE_LISTED:
            return
        if status in (FILE_READ_OK, FILE_WRITTEN):
            base = os.path.basename(key)
            for other_key in [
                k for k, v in bucket.items()
                if v.status == FILE_READ_FAILED and os.path.basename(k) == base
            ]:
                del bucket[other_key]
        bucket.pop(key, None)
        bucket[key] = FileStatus(cleaned, status, error_kind, tuple(searched))
        _evict(bucket)


def record_listed_folder(session_id: str, folder: str) -> int:
    """一覧したフォルダ直下のファイルを ``listed`` として記録する (記録した件数)。"""
    try:
        names = sorted(
            entry.name for entry in os.scandir(folder) if entry.is_file()
        )[:_MAX_LISTED_FILES]
    except (OSError, ValueError):
        return 0
    for name in names:
        record_file_status(session_id, os.path.join(folder, name), FILE_LISTED)
    return len(names)


def file_statuses(session_id: str) -> list[FileStatus]:
    """このセッションのファイルの状態 (古い順)。"""
    with _lock:
        return list((_status.get(session_id) or {}).values())


def current_session() -> str:
    """現在のリクエストの宛先 (無ければ空文字)。"""
    return _current_session.get() or ""


def reset(session_id: str | None = None) -> None:
    """台帳を消す (``None`` で全消去)。"""
    with _lock:
        _bump()
        if session_id is None:
            _ledger.clear()
            _dirs.clear()
            _status.clear()
            return
        _ledger.pop(session_id, None)
        _dirs.pop(session_id, None)
        _status.pop(session_id, None)


#: 直近のファイルを指す **指示** の形。
#:
#: 語彙でファイルの種類を数えない。見るのは指示詞という閉じた文法クラスと、
#: 「保存した/書いた」という **過去の自分の操作** への参照だけ。
#: 「何を指すか」は「直前にファイルを書いた」という観測事実が決める。
#: 過去の操作の語形は説明節の SSOT (``intent_vocab.PAST_FILE_OPERATION_PATTERN``、
#: docs/c_17 §3.5.1) を合成する — 「保存しておいた」「作成していただいた」
#: 「保存済みの」がここにだけ無く、ルータ・参照解決と判定が割れていた。
_IMPLICIT_FILE_REF_RE = re.compile(
    r"(?:その|それ|この|これ|さっきの|先ほどの|いまの|今の|上記の"
    rf"|{PAST_FILE_OPERATION_PATTERN}"
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


# ─────────────────────────────────────────────────────────────────────
# 会話履歴の書込み報告からの復元
# ─────────────────────────────────────────────────────────────────────
#
# 台帳はプロセス内の dict なので、再起動すると空になる。一方、書込みターンの
# 応答はシステム自身が i18n の定型文 (``agent.files_written`` 「{paths} に
# 書き込みました。」/ ``agent.written_content_header`` 「{path} に書き込んだ
# 内容:」) で出しており、会話履歴に残る。**文面の SSOT は i18n** なので、正規
# 表現はそこから組む (語彙を二重に持たない)。

#: パスを運ぶ定型文のキーと、その中のパスの差込み名。
_WRITE_REPORT_KEYS: tuple[tuple[str, str], ...] = (
    ("agent.files_written", "paths"),
    ("agent.files_written_with_failures", "paths"),
    ("agent.written_content_header", "path"),
)
#: 複数のパスの区切り (``chat_stream_meta`` は「、」で連結する)。
_PATHS_SEPARATOR_RE = re.compile(r"、|,\s+")


def _write_report_patterns() -> tuple[re.Pattern[str], ...]:
    """読み込み済みの全 locale の書込み報告の文面から 1 行分の正規表現を組む。"""
    return template_line_patterns(_WRITE_REPORT_KEYS, group="paths")


#: パスらしさ: 区切り文字を含むか、拡張子で終わる 1 語。報告の文面 (英語の
#: 「Wrote {paths}.」) は普通の文にも当たる — 「Wrote a short poem about autumn.」
#: を path "a short poem about autumn" と読んでいた (2026-09-27 レビュー M4)。
_PATH_LIKE_RE = re.compile(r"[\\/]|^[^\s]+\.[A-Za-z0-9]{1,6}$")


def _report_line_paths(line: str) -> list[str] | None:
    """1 行が書込み報告ならその中のパス (報告でなければ ``None``)。

    パスらしくない語 (「メモ帳」「a short poem about autumn」) しか無い行は
    報告とみなさない。
    """
    for pattern in _write_report_patterns():
        m = pattern.match(line)
        if m:
            paths = [
                p.strip().strip("\"'`")
                for p in _PATHS_SEPARATOR_RE.split(m.group("paths"))
                if p.strip()
            ]
            paths = [p for p in paths if _PATH_LIKE_RE.search(p)]
            if paths:
                return paths
    return None


def written_paths_in_text(text: str) -> list[str]:
    """応答本文中のシステム自身の書込み報告が挙げるパス (出現順、純粋関数に近い)。"""
    out: list[str] = []
    for line in (text or "").splitlines():
        paths = _report_line_paths(line)
        if paths:
            out.extend(paths)
    return out


def starts_with_write_report(text: str) -> bool:
    """応答の最初の行がシステム自身の書込み報告か (「…に書き込みました。」/
    「…に書き込んだ内容:」+ 本文)。

    書込みターンの応答は、本文を添えていても **見出し行とフェンスが付いた報告** で、
    そのまま別のファイルの本文にすると見出しとフェンスごと書かれる。素材探しは
    これを候補にしない (docs/f_03 §1.2)。
    """
    for line in (text or "").splitlines():
        if line.strip():
            return _report_line_paths(line) is not None
    return False


def written_paths_in_conversation(
    conversation: list[dict] | None, *, existing_only: bool = False,
) -> list[str]:
    """会話履歴の assistant 発話から、書き込んだパスを古い順に返す (重複は最後の位置)。

    ``existing_only`` なら実在するファイルだけ。書込み報告の文面は LLM の応答にも
    現れる (捏造の「E:\\tmp\\memo.txt に書き込みました。」) ので、事実として使う
    読み手 (台帳の復元 / 所在の事実注記) は実在を確かめる。
    """
    ordered: list[str] = []
    for msg in conversation or ():
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for path in written_paths_in_text(str(msg.get("content") or "")):
            # 報告の文面は LLM の応答にも現れる。UNC は stat しない (SMB の認証を送らない)
            if existing_only and (_is_unc(path) or not _is_existing_file(path)):
                continue
            if path in ordered:
                ordered.remove(path)
            ordered.append(path)
    return ordered


def _is_existing_file(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except (OSError, ValueError):
        return False


def restore_from_conversation(session_id: str, conversation: list[dict] | None) -> int:
    """台帳が空のセッションを、会話履歴の書込み報告から復元する (復元した件数)。

    台帳はプロセス内だけなので、再起動後の「保存したファイルの場所」「保存した
    ファイルに追記」は台帳からは解けない。会話履歴に残っている書込み報告を
    古い順に記録し直す。台帳に既に **書込みの** 記録があるセッションは触らない
    (実行時の記録の方が新しい)。読んだだけの記録しか無いとき (再起動後にファイルを
    読んでから「保存したファイルに追記して」) は、書込み報告から書込みを復元する。
    """
    if not session_id or last_written_path(session_id):
        return 0
    paths = written_paths_in_conversation(conversation, existing_only=True)
    for path in paths:
        record_file(session_id, path, written=True)
    if paths:
        logger.info(
            "File ledger restored from the conversation's write reports "
            "(session=%s, files=%d)", session_id[:12], len(paths),
        )
    return len(paths)
