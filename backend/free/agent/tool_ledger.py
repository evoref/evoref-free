"""セッション単位の「実際に実行したツール」の台帳。

チャットの会話履歴には **ツールを実行した痕跡が一切残らない**。ツール結果は
そのターンの ``## ツール実行結果`` ブロックとしてプロンプトへ差し込まれるだけで、
次ターン以降の履歴には整形済みの本文しか載らない。その結果、自分の処理経路を
尋ねられると base は事前知識で埋めてしまう。

実インシデント (2026-08-22 ライブ監査 2 回目):

- ターン 40 「これまでの計算のうち、ツールを使わず暗算したものはどれですか？」
  → 実行済みの計算 17 件を **すべて暗算だったと申告**。実際は ``calculate`` と
  ``run_command_readonly`` が繰り返し走っていた。
- ターン 100 「この一連のやり取りで、あなたが実際に文字数を数えた場面は
  ありましたか？」→「いいえ、ありません。」実際はターン 64 で決定論の
  文字数注記が入り、正答している。

同じターン内の話なら会話本文から読めるので当たる (ターン 138 は正確だった) —
外すのは **窓を越えた自己申告**だけで、これは記録が無い以上どう促しても直らない。
``ToolsRegistry`` の目録 (``deliberative._append_tool_inventory_fact``) と同じ
立て付けで、数えるのはコード・モデルは読み上げるだけにする。

台帳はプロセス内メモリのみ (再起動で消えて構わない — 履歴に残らない値を
永続化する要件は無い)。セッション数・エントリ数とも上限付きで、
``_cancel_flags`` (chat_stream_deliberative) と同じくモジュールスコープに置く。
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: 1 セッションあたり保持するツール実行の件数。長い監査会話 (100 ターン超) でも
#: 「この会話で何を実行したか」に答えられる程度に取る。
MAX_ENTRIES_PER_SESSION = 80

#: 同時に台帳を保持するセッション数 (LRU で溢れた古いものから捨てる)。
MAX_SESSIONS = 16

#: プロンプトへ載せる最大件数。これを超える場合は新しい方を残す。
MAX_RENDERED_ENTRIES = 40


@dataclass(frozen=True, slots=True)
class ToolUse:
    """1 回のツール実行。"""

    tool_name: str
    success: bool
    query_head: str


_ledger: "OrderedDict[str, deque[ToolUse]]" = OrderedDict()


def _bucket(session_id: str) -> "deque[ToolUse]":
    existing = _ledger.get(session_id)
    if existing is not None:
        _ledger.move_to_end(session_id)
        return existing
    created: "deque[ToolUse]" = deque(maxlen=MAX_ENTRIES_PER_SESSION)
    _ledger[session_id] = created
    while len(_ledger) > MAX_SESSIONS:
        _ledger.popitem(last=False)
    return created


def record_tool_use(
    session_id: str, tool_name: str | None, success: bool, query: str,
) -> None:
    """ツール実行を台帳へ追記する。``tool_name`` が空なら no-op。"""
    if not session_id or not tool_name:
        return
    _bucket(session_id).append(
        ToolUse(
            tool_name=tool_name,
            success=bool(success),
            query_head=(query or "")[:40],
        ),
    )


#: 現在のリクエストの ``(session_id, query)``。``ToolsRegistry.execute`` が
#: 台帳へ落とすときの宛先で、``ledger_scope`` が設定する。
#:
#: なぜ contextvar か: ツール実行は 5 箇所 (deliberative のツールループ /
#: meta_cognitive のファストパス 3 種 / タスク実行ループ) に
#: 分かれており、記録を **呼出側に配る** と必ず取りこぼす。実インシデント
#: (2026-08-23 ライブ監査セット 2): 記録は deliberative の 1 箇所にしか無く、
#: meta_cognitive 経由の ``write_file`` 3 回が台帳に入らなかった。台帳を
#: 「実行したツールはこれがすべて」と断定してプロンプトへ載せているため、
#: モデルは「ファイルの書き込みに使ったツールはありません」と答えた
#: (実際には spec.txt / 日本語名メモ.txt が書かれ、追記も成功していた)。
#:
#: 記録は実行の唯一の合流点 (``ToolsRegistry.execute``) で行い、宛先だけを
#: contextvar で運ぶ。contextvar は ``asyncio.create_task`` / ``to_thread`` を
#: 越えて伝播するので、同期ツールのスレッド実行でも失われない。
_current_target: ContextVar[tuple[str, str] | None] = ContextVar(
    "tool_ledger_target", default=None,
)


def set_ledger_target(session_id: str, query: str) -> None:
    """このリクエスト (asyncio タスク) のツール実行の記録先を設定する。

    リクエストハンドラから呼ぶ。ストリーミング応答は関数が返った **後** に
    ジェネレータが回るため ``with`` で囲むと早すぎる時点で解除される。
    contextvar はタスクごとのコピーなので、明示的に解除しなくてもリクエスト
    タスクの終了とともに破棄される。
    """
    _current_target.set((session_id or "", query or ""))


@contextmanager
def ledger_scope(session_id: str, query: str) -> Iterator[None]:
    """スコープ内のツール実行を ``session_id`` の台帳へ記録する (テスト / 同期用)。"""
    token = _current_target.set((session_id or "", query or ""))
    try:
        yield
    finally:
        _current_target.reset(token)


def record_current(tool_name: str | None, success: bool) -> None:
    """``ledger_scope`` が設定した宛先へツール実行を記録する。

    スコープ外 (sleep-time / 学習ジョブ等) の実行は宛先が無いので no-op。
    """
    target = _current_target.get()
    if target is None:
        return
    session_id, query = target
    record_tool_use(session_id, tool_name, success, query)


def mark_last_failed(tool_name: str | None = None) -> bool:
    """現在のリクエストの台帳で **直前の 1 件** を失敗に書き換える。

    ``ToolsRegistry.execute`` は戻り値の文字列だけで成否を決めて記録するが、
    ``write_file`` は実行後に呼出側 (meta_cognitive の ``_verify_written_file``)
    が実ファイルを読み戻して初めて失敗と分かることがある。その時点で台帳に
    「成功」が残っていると、自己申告が実態とずれる。検証で失敗が判明した
    直後に呼ぶ。``tool_name`` を渡した場合は直前の 1 件がそのツールのときだけ
    書き換える (別ツールの成功を巻き込まない)。

    Returns:
        書き換えたら True。宛先なし / 台帳が空 / ツール名不一致なら False。
    """
    target = _current_target.get()
    if target is None:
        return False
    bucket = _ledger.get(target[0])
    if not bucket:
        return False
    last = bucket[-1]
    if tool_name and last.tool_name != tool_name:
        return False
    bucket[-1] = ToolUse(
        tool_name=last.tool_name, success=False, query_head=last.query_head,
    )
    return True


def current_session_id() -> str:
    """現在のリクエストの ``session_id`` (未設定なら空文字)。

    宛先を既に contextvar で運んでいるので、``session_id`` を引数で回せない
    純粋関数側 (``tool_judge_args._extract_file_path``) が参照するための
    読み出し口。
    """
    target = _current_target.get()
    return target[0] if target else ""


def format_ledger(session_id: str) -> str:
    """台帳を「確定事実」ブロック向けのテキストへ整形する。

    Returns:
        1 件も無ければ空文字列 (呼出側が「1 度も実行していない」と述べる)。
    """
    from backend.i18n_helper import msg

    entries = list(_ledger.get(session_id) or ())
    if not entries:
        return ""
    tail = entries[-MAX_RENDERED_ENTRIES:]
    lines = [
        msg(
            "agent.tool_ledger.entry",
            index=i,
            tool=e.tool_name,
            status=msg(
                "agent.tool_ledger.success" if e.success
                else "agent.tool_ledger.failure",
            ),
            query=e.query_head,
        )
        for i, e in enumerate(tail, start=len(entries) - len(tail) + 1)
    ]
    if len(entries) > len(tail):
        lines.insert(
            0, msg("agent.tool_ledger.omitted", count=len(entries) - len(tail)),
        )
    return "\n".join(lines)


def reset(session_id: str | None = None) -> None:
    """テスト用。``session_id`` 指定でそのセッションだけ、無指定で全消去。"""
    if session_id is None:
        _ledger.clear()
        return
    _ledger.pop(session_id, None)
