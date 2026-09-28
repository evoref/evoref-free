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
from dataclasses import dataclass

from backend.free.core.intent_vocab import PAST_FILE_OPERATION_PATTERN
from backend.i18n_helper import template_line_patterns
from backend.log_config import get_logger

logger = get_logger("agent.file_ledger")

__all__ = [
    "current_named_file_paths",
    "file_ledger_scope",
    "forget_current_file",
    "forget_file",
    "last_file_path",
    "last_written_path",
    "named_file_paths",
    "record_current_file",
    "record_file",
    "references_recent_file",
    "resolve_against_recent_dir",
    "resolve_current_against_recent_dir",
    "reset",
    "restore_from_conversation",
    "starts_with_write_report",
    "written_paths_in_conversation",
    "written_paths_in_text",
]

#: 1 セッションあたりの保持件数 (新しい方を残す)。
MAX_ENTRIES_PER_SESSION = 12

#: 保持するセッション数。
MAX_SESSIONS = 16



@dataclass(frozen=True, slots=True)
class _Entry:
    """台帳の 1 件。

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


_ledger: "OrderedDict[str, deque[_Entry]]" = OrderedDict()
#: ``written_seq`` の採番 (プロセス内で単調増加)。
_write_counter = 0


def _bucket(session_id: str) -> "deque[_Entry]":
    existing = _ledger.get(session_id)
    if existing is not None:
        _ledger.move_to_end(session_id)
        return existing
    created: "deque[_Entry]" = deque(maxlen=MAX_ENTRIES_PER_SESSION)
    _ledger[session_id] = created
    while len(_ledger) > MAX_SESSIONS:
        _ledger.popitem(last=False)
    return created


def _remove(bucket: "deque[_Entry]", path: str) -> tuple[bool, int]:
    """``path`` の記録をすべて外す。``(どれかが named か, 最後に書いた順番)``。"""
    named = False
    written_seq = 0
    for entry in [e for e in bucket if e.path == path]:
        named = named or entry.named
        written_seq = max(written_seq, entry.written_seq)
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
    bucket = _bucket(session_id)
    was_named, written_seq = _remove(bucket, cleaned)
    if written:
        _write_counter += 1
        written_seq = _write_counter
    bucket.append(_Entry(cleaned, was_named or named, written_seq))


#: 現在のリクエストの ``session_id``。``tool_ledger`` と同じ理由で contextvar
#: に置く — 記録点は ``ToolsRegistry.execute`` の 1 つだが、宛先は呼出側が
#: 決めるため。
_current_session: ContextVar[str | None] = ContextVar(
    "file_ledger_session", default=None,
)


def file_ledger_scope(session_id: str):
    """``record_current_file`` の宛先を設定する。"""
    return _current_session.set(session_id or "")


def record_current_file(path: str, *, named: bool = False, written: bool = False) -> None:
    """現在のリクエストの宛先へファイルパスを記録する。"""
    session_id = _current_session.get()
    if session_id:
        record_file(session_id, path, named=named, written=written)


def named_file_paths(session_id: str) -> list[str]:
    """このセッションの ``named`` の記録 (古い順)。"""
    return [e.path for e in _ledger.get(session_id) or () if e.named]


def current_named_file_paths() -> list[str]:
    """現在のリクエストの宛先の ``named`` の記録 (宛先が無ければ空)。"""
    session_id = _current_session.get()
    return named_file_paths(session_id) if session_id else []


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
    if not bucket or not any(e.path == cleaned for e in bucket):
        return False
    _remove(bucket, cleaned)
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
    for entry in reversed(_ledger.get(session_id) or ()):
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
    written = [e for e in _ledger.get(session_id) or () if e.written_seq]
    if not written:
        return ""
    return max(written, key=lambda e: e.written_seq).path


def resolve_against_recent_dir(session_id: str, path: str, *, named_only: bool = False) -> str:
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

    書込み (``named_only=True``) では ``named`` の記録だけを見る。LLM が自分で
    選んで読んだファイルの隣は書込みゲート (docs/f_03 §4.y) が断るので、そこへ
    寄せずに ``outputs_dir`` へ任せる。
    """
    cleaned = (path or "").strip().strip("\"'")
    if not cleaned or not session_id:
        return path
    if os.path.isabs(cleaned) or "/" in cleaned or "\\" in cleaned:
        return path
    recent = last_file_path(session_id, named_only=named_only)
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


def resolve_current_against_recent_dir(path: str, *, named_only: bool = False) -> str:
    """現在のリクエストの宛先で :func:`resolve_against_recent_dir` を掛ける。"""
    session_id = _current_session.get()
    if not session_id:
        return path
    return resolve_against_recent_dir(session_id, path, named_only=named_only)


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
            if existing_only and not _is_existing_file(path):
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
