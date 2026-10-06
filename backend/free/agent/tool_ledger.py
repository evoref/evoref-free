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

import os
from collections import OrderedDict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from backend.free.core.verifier_events import record_tool_use_event

#: 1 セッションあたり保持するツール実行の件数。長い監査会話 (100 ターン超) でも
#: 「この会話で何を実行したか」に答えられる程度に取る。
MAX_ENTRIES_PER_SESSION = 80

#: 同時に台帳を保持するセッション数 (LRU で溢れた古いものから捨てる)。
MAX_SESSIONS = 16

#: プロンプトへ載せる最大件数。これを超える場合は新しい方を残す。
MAX_RENDERED_ENTRIES = 40

#: 成功したツールの結果の本文を 1 セッションあたり何件・1 件何文字まで持つか
#: (:func:`record_current_result`)。
MAX_RESULTS_PER_SESSION = 20
MAX_RESULT_CHARS = 20_000

#: 結果がモデル自身の生成した下書きのツール (外から取った事実ではない)。
GENERATED_DRAFT_TOOLS = frozenset({"draft_document", "summarize", "translate"})


@dataclass(frozen=True, slots=True)
class ToolUse:
    """1 回のツール実行。"""

    tool_name: str
    success: bool
    query_head: str
    #: 失敗理由の識別子 (``timeout`` / ``error`` / ``invalid_args``)。
    #: 空なら理由を添えずに「失敗」とだけ表示する。
    reason: str = ""


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
    reason: str = "",
) -> None:
    """ツール実行を台帳へ追記する。``tool_name`` が空なら no-op。"""
    if not session_id or not tool_name:
        return
    _bucket(session_id).append(
        ToolUse(
            tool_name=tool_name,
            success=bool(success),
            query_head=(query or "")[:40],
            reason=reason,
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


#: このリクエストで書いたファイル (正規化した絶対パス)。``set_ledger_target`` /
#: ``ledger_scope`` がリクエストごとに空の集合を置き、``ToolsRegistry.execute`` が
#: 書込みの成功を積む。書込みゲートの上書きの規則 (docs/f_03 §4.y) が「計画の前段が
#: 作り、後段が追記する」ファイルを依頼の対象として扱うのに使う。集合は可変で、
#: ``create_task`` / ``to_thread`` が写した文脈からも同じ集合に積まれる。
_request_writes: ContextVar[set[str] | None] = ContextVar(
    "tool_ledger_request_writes", default=None,
)


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def set_ledger_target(session_id: str, query: str) -> None:
    """このリクエスト (asyncio タスク) のツール実行の記録先を設定する。

    リクエストハンドラから呼ぶ。ストリーミング応答は関数が返った **後** に
    ジェネレータが回るため ``with`` で囲むと早すぎる時点で解除される。
    contextvar はタスクごとのコピーなので、明示的に解除しなくてもリクエスト
    タスクの終了とともに破棄される。セッションのターン番号もここで 1 つ進める
    (取得した資料の持ち越しの窓、:func:`recent_observations`)。
    """
    _current_target.set((session_id or "", query or ""))
    _current_turn.set(_next_turn(session_id))
    _request_writes.set(set())


@contextmanager
def ledger_scope(session_id: str, query: str) -> Iterator[None]:
    """スコープ内のツール実行を ``session_id`` の台帳へ記録する (テスト / 同期用)。

    :func:`set_ledger_target` と同じく、入るたびにセッションのターン番号を進める。
    """
    token = _current_target.set((session_id or "", query or ""))
    turn_token = _current_turn.set(_next_turn(session_id))
    writes_token = _request_writes.set(set())
    try:
        yield
    finally:
        _request_writes.reset(writes_token)
        _current_turn.reset(turn_token)
        _current_target.reset(token)


#: 現在のリクエストの、そのセッションでのターン番号 (1 始まり。宛先が無ければ 0)。
_current_turn: ContextVar[int] = ContextVar("tool_ledger_turn", default=0)
#: セッションごとに最後に発番したターン番号 (プロセス内のみ)。
_turn_counters: "OrderedDict[str, int]" = OrderedDict()


def _next_turn(session_id: str) -> int:
    """``session_id`` の次のターン番号を発番する (セッションが空なら 0)。"""
    if not session_id:
        return 0
    turn = _turn_counters.pop(session_id, 0) + 1
    _turn_counters[session_id] = turn
    while len(_turn_counters) > MAX_SESSIONS:
        evicted, _ = _turn_counters.popitem(last=False)
        # 番号が 1 から振り直されるので、旧番号で積んだ資料も一緒に捨てる
        # (残すと新しい番号と衝突して、古い資料が直近の取得に見える)。
        _observations.pop(evicted, None)
    return turn


def record_request_write(path: str) -> None:
    """このリクエストで ``path`` へ書いたことを記録する (スコープ外は no-op)。"""
    writes = _request_writes.get()
    if writes is not None and path:
        writes.add(_path_key(path))


def written_in_request(path: str) -> bool:
    """``path`` をこのリクエストで既に書いたか。"""
    writes = _request_writes.get()
    return bool(writes and path and _path_key(path) in writes)


def record_current(tool_name: str | None, success: bool, reason: str = "") -> None:
    """``ledger_scope`` が設定した宛先へツール実行を記録する。

    スコープ外 (sleep-time / 学習ジョブ等) の実行は宛先が無いので no-op。
    ``reason`` は失敗時のみ意味を持つ (``timeout`` / ``error`` /
    ``invalid_args``)。
    """
    # 根拠台帳 (経験の ``signals.tool_uses``) へも同じ合流点から積む。こちらは
    # request scope なので、セッション台帳の宛先が無くても記録される。
    if tool_name:
        record_tool_use_event(tool_name, success, reason)
    target = _current_target.get()
    if target is None:
        return
    session_id, query = target
    record_tool_use(session_id, tool_name, success, query, reason)


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
        reason=last.reason,
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


def current_query() -> str:
    """現在のリクエストの依頼文 (ユーザーの発話。未設定なら空文字)。

    書込みゲート (``write_gate``) が「依頼が挙げたフォルダ」を決める材料。
    """
    target = _current_target.get()
    return target[1] if target else ""


#: 失敗理由 -> i18n キー。未知の理由は理由なしの「失敗」へ縮退させる。
_REASON_KEYS: dict[str, str] = {
    "timeout": "agent.tool_ledger.failure_timeout",
    "error": "agent.tool_ledger.failure_error",
    "invalid_args": "agent.tool_ledger.failure_invalid_args",
}


def _status_key(entry: ToolUse) -> str:
    """1 件の成否 (と失敗理由) に対応する i18n キー。"""
    if entry.success:
        return "agent.tool_ledger.success"
    return _REASON_KEYS.get(entry.reason, "agent.tool_ledger.failure")


def latest_use(session_id: str) -> ToolUse | None:
    """セッションで最後に記録したツール実行 (無ければ ``None``)。"""
    entries = _ledger.get(session_id)
    if not entries:
        return None
    return entries[-1]


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
            status=msg(_status_key(e)),
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
        _results.clear()
        _observations.clear()
        _turn_counters.clear()
        return
    _ledger.pop(session_id, None)
    _results.pop(session_id, None)
    _observations.pop(session_id, None)
    _turn_counters.pop(session_id, None)


#: セッションごとの成功したツールの結果の本文 (:func:`record_current_result`)。
_results: "OrderedDict[str, deque[str]]" = OrderedDict()


def record_current_result(tool_name: str | None, result: str) -> None:
    """成功したツールの結果の本文を ``ledger_scope`` の宛先セッションへ積む。

    後のターンの出力検査が「応答の人名はこの会話で取った中身にあるか」を確かめる
    根拠 (``deliberative._append_session_files_fact``)。積むのはツールの **結果** で、
    それを受けた答えではない — 答えを根拠にすると、そのターンで作った名前が後の
    ターンで正当化される。下書きのツール (:data:`GENERATED_DRAFT_TOOLS`) の結果は
    モデルの生成なので積まない。スコープ外・失敗は呼出側が呼ばない / no-op。
    """
    target = _current_target.get()
    if target is None or not tool_name or tool_name in GENERATED_DRAFT_TOOLS:
        return
    session_id = target[0]
    bucket = _results.get(session_id)
    if bucket is None:
        bucket = _results[session_id] = deque(maxlen=MAX_RESULTS_PER_SESSION)
        while len(_results) > MAX_SESSIONS:
            _results.popitem(last=False)
    else:
        _results.move_to_end(session_id)
    bucket.append(str(result or "")[:MAX_RESULT_CHARS])


def session_results(session_id: str) -> list[str]:
    """セッションで成功したツールの結果の本文 (古い順)。"""
    return list(_results.get(session_id, ()))


# ─────────────────────────────────────────────────────────────────────
# 取得した資料の持ち越し (ファイル / URL の本文)
# ─────────────────────────────────────────────────────────────────────
#
# 会話履歴には応答の本文しか残らないので、ファイルを読んで答えた次のターンの
# 「地域別ではどちらが多いですか？」をツール無しで答えると、資料を見ないまま作話する
# (2026-10-05 ライブ監査: 実在しない North / South の金額を挙げた。実際の地域は
# 東 / 西)。取った本文を、どの依頼で取ったかと一緒に持ち、後のターンへ持ち越す
# 材料にする (docs/f_03 §3.5 の ``_append_recent_observations``)。

#: 本文を持ち越す対象のツール (名指しした資料を取るもの)。
OBSERVATION_TOOLS = frozenset({"read_file", "fetch_url"})
#: 1 セッションあたり保持する取得の件数 (新しい方を残す)。
MAX_OBSERVATIONS_PER_SESSION = 8
#: 取得したターンから何ターン先まで持ち越すか (このセッションのターン番号で数える)。
RECENT_OBSERVATION_TURNS = 3


@dataclass(frozen=True, slots=True)
class Observation:
    """取得した資料 1 件。"""

    tool_name: str
    #: 資料の所在 (ファイルのパス / URL)。
    source: str
    #: 本文 (``MAX_RESULT_CHARS`` で切った写し)。
    content: str
    #: 取得したターンの番号 (:func:`set_ledger_target` が発番、持ち越しの窓をこれで数える)。
    turn: int
    #: 切る前の本文の文字数。
    total_chars: int = 0
    #: ファイルの ``(st_mtime_ns, st_size)`` (読む **前** に取った値)。取得後の変更を
    #: 見分ける。URL・取れなかったときは ``None``。
    fingerprint: tuple[int, int] | None = None
    #: private のターンで取ったか。private でないターンへは持ち越さない。
    private: bool = False

    @property
    def source_key(self) -> str:
        """同じ資料を見分ける鍵 (ファイルは大文字小文字を畳む、URL はそのまま)。"""
        return os.path.normcase(self.source) if self.tool_name == "read_file" else self.source


_observations: "OrderedDict[str, deque[Observation]]" = OrderedDict()


def record_current_observation(
    tool_name: str, source: str, content: str,
    fingerprint: tuple[int, int] | None = None,
) -> None:
    """取得した資料を ``ledger_scope`` の宛先セッションへ積む (宛先が無ければ no-op)。"""
    from backend.trace_context import is_private

    target = _current_target.get()
    if target is None or not target[0] or tool_name not in OBSERVATION_TOOLS:
        return
    if not (source or "").strip():
        return
    session_id = target[0]
    bucket = _observations.get(session_id)
    if bucket is None:
        bucket = _observations[session_id] = deque(maxlen=MAX_OBSERVATIONS_PER_SESSION)
        while len(_observations) > MAX_SESSIONS:
            _observations.popitem(last=False)
    else:
        _observations.move_to_end(session_id)
    text = str(content or "")
    bucket.append(Observation(
        tool_name=tool_name, source=source.strip(),
        content=text[:MAX_RESULT_CHARS], turn=_current_turn.get(),
        total_chars=len(text), fingerprint=fingerprint, private=is_private(),
    ))


def recent_observations(
    session_id: str, *, max_turns: int = RECENT_OBSERVATION_TURNS,
) -> list[tuple[Observation, int]]:
    """持ち越す資料と、何ターン前に取得したか (新しい順、資料ごとに最新の 1 件)。

    今のターン (:func:`set_ledger_target` が発番した番号) より前の、``max_turns``
    ターン以内に取得したものだけを返す。ターンは番号で数える — 依頼文の一致で
    数えると、同じ文を 2 度送った会話で取り違える。別の会話・再起動前の取得は
    台帳に無いので持ち越さない。private のターンで取った資料は private でない
    ターンへ持ち越さない (ログへ本文が出る)。
    """
    from backend.trace_context import is_private

    bucket = _observations.get(session_id) if session_id else None
    current = _current_turn.get()
    if not bucket or not current:
        return []
    private_turn = is_private()
    out: list[tuple[Observation, int]] = []
    seen: set[str] = set()
    # 実行中のツールが別スレッドから積みうるので、写しを走査する。
    for obs in reversed(list(bucket)):
        if obs.private and not private_turn:
            continue
        if obs.source_key in seen:
            continue
        seen.add(obs.source_key)
        age = current - obs.turn
        if 1 <= age <= max_turns:
            out.append((obs, age))
    return out


def file_fingerprint(path: str) -> tuple[int, int] | None:
    """ファイルの ``(st_mtime_ns, st_size)`` (読めなければ ``None``)。"""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return st.st_mtime_ns, st.st_size


def observation_is_stale(obs: Observation) -> bool:
    """取得した後にファイルが変わった / 消えたか (URL など指紋の無い取得は偽)。"""
    if obs.fingerprint is None:
        return False
    return file_fingerprint(obs.source) != obs.fingerprint
