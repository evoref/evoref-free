"""レスポンス記録（メモリ・デバッグログ・フィードバック・履歴）"""

from __future__ import annotations

import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from backend.app_state import AppState
from backend.free.core.turn_text import TOOL_RESULT_HEADER
from backend.free.core.text_quality import strip_system_notes
from backend.free.api.chat._artifact import remember_artifact
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.history import history_manager as _history_manager_module
from backend.free.history.history_manager import (
    active_base_model_key,
    active_base_model_name,
    get_history_manager,
)
from backend.free.history.utils import parse_iso
from backend.free.core.text_quality import extract_calculate_result, extract_measured_values
from backend.io.readonly import DataReadonlyError
from backend.io.writer_thread import default_writer
from backend.log_config import get_logger
from backend.trace_context import get_trace_id
from backend.utils import format_utc, utc_now_dt

if TYPE_CHECKING:
    from backend.free.learning.level0_instant import GenerationConfigRef

logger = get_logger("api.chat.recorder")

# 文書系の出力先拡張子。これらへの出力依頼で content_type=code が返るのは
# ルーティング誤り (long_form_success の判定材料)。
_DOC_TARGET_EXT_RE = re.compile(r"\.(?:md|txt|csv)\b", re.IGNORECASE)

def read_llama_prompt_tokens(state: AppState) -> tuple[int | None, int | None]:
    """直近ストリームの ``(prompt_tokens, cached_prompt_tokens)`` を返す。

    llama-server の ``usage.prompt_tokens_details.cached_tokens`` を
    :class:`~backend.free.llm.local_client.LocalClient` が ``_last_timings``
    (``prompt_n`` = 再評価分 / ``cache_n`` = 再利用分) へ畳んでいる。
    ``prompt_tokens`` はその合計。

    **プロンプト側コストの唯一の一次情報**なので読み手をここに集約する
    (``requests.jsonl`` の timing 畳み込みと Level 0 の経験記録の両方が使う)。
    取得できない構成 (クライアント未接続 / usage 非対応) では ``(None, None)``
    を返し、呼出側は「消費ゼロ」ではなく「未計測」として扱う。

    注意: クライアント単位の直近値なので、並行チャット中は別ターンの値を読み
    うる。既存の timing 畳み込みが元から持っていた制約と同じで、個々のターンの
    厳密値ではなく統計量としての利用を前提とする。
    """
    # **実際にストリームしたクライアント**から読む。チャットは
    # ``ensure_llm_client`` が返す ``state.llm_client`` で生成しており、
    # ``state.gen.llm_client`` は配線時の参照で lazy-connect / モード切替後に
    # 別オブジェクトになりうる。2026-09-03 監査: KV 行 (op=kv_cache) は 101 件
    # あるのに timing 側の prompt_n は 0/102 ターンで、プロンプト側コストが
    # timing から一切追えなかった (c_07 §3.2.1)。
    timings = None
    for client in (
        getattr(state, "llm_client", None),
        getattr(getattr(state, "gen", None), "llm_client", None),
    ):
        candidate = getattr(getattr(client, "local", client), "_last_timings", None)
        if isinstance(candidate, dict):
            timings = candidate
            break
    if not isinstance(timings, dict):
        return None, None
    prompt_n = timings.get("prompt_n")
    cache_n = timings.get("cache_n")
    if not isinstance(prompt_n, int) or not isinstance(cache_n, int):
        return None, None
    return prompt_n + cache_n, cache_n


def is_content_type_mismatch(metrics: dict, user_query: str) -> bool:
    """文書拡張子への出力依頼なのに ``content_type=code`` を返したか。

    長文ルーティング自体の誤検出 (= ``long_form_false_positive``) の判定材料。
    """
    return (
        bool(_DOC_TARGET_EXT_RE.search(user_query))
        and str(metrics.get("content_type") or "") == "code"
    )


def judge_long_form_success(
    metrics: dict, user_query: str, delivered: str | None = None,
) -> bool:
    """長文生成ターンの成否を判定する (Level 0 記録 / MDP episode 共通)。

    条件:
      - ``units_completed > 0`` (1 ユニット以上生成)
      - ``validation_errors == 0`` (CODE は AST 検証、TEXT は残 review issue /
        目標文字数比 / 文重複率のゲート。``orchestrator._validate_generated_text``)
      - 要求成果物と ``content_type`` が矛盾しない (文書拡張子への出力依頼なのに
        code 生成 = ルーティング誤り。2026-07-15 に Python コードを .md へ書いた
        訂正ターンが「成功」として正例学習され誤ルーティングを増幅した)

    失敗時は ``long_form_success=False`` となり、learned_patterns への boost も
    新規追加も走らない。record 側と agent_trace 側で式が食い違うと同じターンが
    別々の成否で二重学習されるため、両者はこの関数を共有する。
    """
    # ユーザーへ 1 文字も届かなかったターンは成功ではない。
    #
    # 実インシデント (2026-08-27、WS2 検証中に 1 回観測): 長文生成が 605 秒
    # かけて **空応答** を返した。units_completed だけを見ていると、内部で
    # ユニットを組み立てた形跡があるかぎり「成功」として正例学習される。
    # 画面には何も出ていないので、ユーザーから見れば完全な失敗。
    if delivered is not None and not delivered.strip():
        logger.warning(
            "Long-form turn delivered no text (units=%s, elapsed metrics=%s); "
            "recording it as a failure",
            metrics.get("units_completed"), metrics.get("budget_used_pct"),
        )
        return False
    units_completed = int(metrics.get("units_completed", 0) or 0)
    validation_errors = int(metrics.get("validation_errors", 0) or 0)
    return (
        units_completed > 0
        and validation_errors == 0
        and not is_content_type_mismatch(metrics, user_query)
    )


@dataclass
class SessionLedger:
    """セッション寿命の台帳 (応答パス側)。WM の窓とは独立に会話全体を持つ。

    以前は開始時刻 / 全ターン / private 印 / 出典 の 4 つが別々のモジュール辞書で、
    片付けの経路 (明示終了 / LRU 押し出し) ごとに漏れが出ていた。1 つの
    レコードにまとめ、寿命は :func:`clear_session_data` (= WM 台帳の
    ``on_drop``) だけが握る。
    """

    #: 初回リクエスト時の開始時刻 (ISO)。保存先ファイル名を決める。
    started_at: str | None = None
    #: 全ターンの蓄積 (WM のエビクションに依存しない完全な履歴)。
    turns: list[dict] = field(default_factory=list)
    #: private ターンを 1 度でも含んだか (``memory.private.history_storage: skip`` 用)。
    had_private: bool = False
    #: 会話単位の根拠台帳 (``[参考情報]`` に注入した資料の所在、ターン順)。
    sources: list[dict] = field(default_factory=list)
    #: ``turns`` のうち履歴の追記ログへ出し済みの件数 (次に出すターンの ``seq``)。
    persisted: int = 0


_ledgers: dict[str, SessionLedger] = {}


def _ledger(session_id: str, *, create: bool = True) -> SessionLedger | None:
    led = _ledgers.get(session_id)
    if led is None and create:
        led = _ledgers[session_id] = SessionLedger()
    return led


def _recorded_body(response: str) -> str:
    """記憶へ積む本文 (システムの開示注記を除いたもの)。

    開示そのものは必要だが、**記憶に残す本文ではない**。注記込みで保存すると
    次のターンでモデルがそれを自分が書いた文の一部として読む。実インシデント
    (2026-08-27 ライブ監査 T09-2): 本文 45 文字 + 注記 34 文字を保存した結果、
    「いま書いた文章は何文字でしたか。」に **81 文字** と答えた。

    「制約を破った」という信号は issue 台帳が持つので、履歴から落としても
    失われない。
    """
    return strip_system_notes(response)


def _accumulate_turn(
    session_id: str, role: str, content: str, *, private: bool = False,
    turn_id: str = "", meta: dict | None = None,
) -> None:
    """セッションのターンを蓄積

    WorkingMemory はターン数・トークン数上限で古いターンを押し出すため、
    履歴保存用に全ターンを独立して蓄積する。

    ``private=True`` のターンはディスク永続化対象から
    除外する (memory_only)。蓄積バッファ自体に追加しない。

    ``turn_id`` は ``WorkingMemory.add_turn`` が発行した ID。``meta`` は
    そのターンで確定した付帯情報 (訂正フラグ / ツール実行の結果 / 根拠に
    使ったチャンク・ファクト等)。以前はここが role / content / timestamp の
    3 キーだけで、**履歴から回帰タスクを組み直せなかった** (2026-09-05 監査)。
    """
    if private:
        _ledger(session_id).had_private = True
        logger.debug(
            "accumulate skipped (private turn): role=%s, session=%s, len=%d",
            role, session_id, len(content),
        )
        return
    entry: dict = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
    }
    if turn_id:
        entry["turn_id"] = turn_id
    if trace_id := get_trace_id():
        entry["trace_id"] = trace_id
    if meta:
        entry["meta"] = {k: v for k, v in meta.items() if v not in (None, "", [], {})}
    _ledger(session_id).turns.append(entry)


def accumulate_user_turn(
    session_id: str, user_query: str, *, private: bool = False,
    turn_id: str = "", meta: dict | None = None,
) -> None:
    """user 発話を蓄積バッファへ積む (**冪等**)。

    応答パスの入口 (``prepare_memory_context`` が WM へ積んだ直後) から呼ぶ。
    以前は ``record_*`` の末尾でしか積んでいなかったため、生成が失敗 /
    タイムアウトしたターンは WM には居るのに履歴 (台帳の ``turns``) には
    無い、という食い違いが起きていた。

    冪等性: 直前に積まれたターンが同じ user 発話なら二重に積まない。
    ``record_*`` も同じ経路を通るので、入口で積んだ後に record が走っても
    1 回しか数えない。同じ文面を連続 2 回送って 1 回目が失敗したケースは
    1 回に畳まれる (許容)。
    """
    # 新しいターンの入口。前ターンの注入 id を必ず落とす (c_16 §5.5)。
    open_turn(session_id)
    if private:
        _accumulate_turn(session_id, "user", user_query, private=True)
        return
    _ensure_session_restored(session_id)
    led = _ledger(session_id, create=False)
    turns = led.turns if led is not None else None
    if turns and turns[-1].get("role") == "user" and turns[-1].get("content") == user_query:
        return
    _accumulate_turn(
        session_id, "user", user_query, turn_id=turn_id, meta=meta,
    )


def _ensure_session_restored(session_id: str, mgr=None) -> bool:
    """再起動を跨いだセッションの開始時刻とターン列を索引 / ファイルから戻す。

    保存先ファイル名は開始時刻から決まる (``HistoryManager._resolve_session_path``)。
    プロセスが再起動すると 台帳 (``SessionLedger``) は空になり、
    同じ session_id の続きが **別ファイル** に書かれ、索引は session_id で置換
    されるため旧ファイルが孤児になっていた。既知のセッションなら開始時刻と
    既存ターンを引き継ぎ、同じファイルへ追記する形にする。

    ``mgr=None`` のときは **既に構築済みのシングルトンだけ** を使う
    (``get_history_manager`` の初回構築は checkpoint 昇格などの副作用を持つ
    ので、応答パスの入口からは起こさない)。保存側は自分の ``mgr`` を渡す。

    Returns:
        ``False`` = 索引には在るのにファイルが読めなかった (一過性の I/O 失敗
        等)。このとき開始時刻は **記録しない** — 以前は先に記録していたため、
        読めなかったターンは二度と復元を試みず、次の保存が同じファイルを
        新しいターンだけで上書きして旧ターンを失っていた。呼出側 (保存) は
        このターンの保存を見送り、次のターンで復元からやり直す。
    """
    led = _ledger(session_id)
    if led.started_at:
        return True
    if mgr is None:
        mgr = _history_manager_module._manager_cache
        if mgr is None:
            return True
    try:
        started = mgr.get_session_started_at(session_id)
    except Exception as exc:
        logger.debug("history lookup failed for %s: %s", session_id, exc)
        return True
    if not isinstance(started, str) or not started:
        return True
    try:
        session = mgr.get_session(session_id)
    except Exception as exc:
        logger.warning(
            "history load failed for %s (save deferred to the next turn): %s",
            session_id, exc,
        )
        return False
    led.started_at = started
    if session is None or not session.turns:
        return True
    restored: list[dict] = []
    for t in session.turns:
        entry = {"role": t.get("role", "user"), "content": t.get("content", "")}
        ts = parse_iso(str(t.get("timestamp") or ""))
        entry["timestamp"] = ts.timestamp() if ts else 0.0
        # 復元でも turn_id / trace_id / meta を落とさない。落とすと、圧縮済みの
        # 印 (``compressed`` / ``original_length``) ごと消えて次の保存で
        # 「切り詰められているのに印の無いターン」になり、二重に切られる。
        for key in ("turn_id", "trace_id", "meta", "compressed", "original_length"):
            if (value := t.get(key)) not in (None, ""):
                entry[key] = value
        restored.append(entry)
    led.turns = restored + list(led.turns)
    # 復元したターンはディスクにある (追記ログの seq はこの続きから)。
    led.persisted += len(restored)
    logger.info(
        "Restored %d turn(s) of session %s from history (started_at=%s)",
        len(restored), session_id, started,
    )
    return True


def session_turn_count(session_id: str) -> int:
    """このセッションの累計ターン数 (user + assistant)。**進行中の user 発話を含む**。

    ``WorkingMemory`` は窓を越えた分を押し出すので、会話全体を数えられるのは
    こちらの蓄積バッファだけ。実インシデント (2026-08-27 ライブ監査 T19-4):
    148 ターン目に「50ターン目です」と答えた (窓に入っている分だけを数えた)。

    user 発話は ``accumulate_user_turn`` で messages を組む前に積まれるため、
    応答パスから読むと「いまのターン」がすでに 1 と数えられている
    (:func:`count_term_in_session` と同じ契約)。
    """
    led = _ledger(session_id, create=False)
    return len(led.turns) if led is not None else 0


def count_term_in_session(session_id: str, term: str) -> int:
    """このセッションの **これまでの** ターン本文に ``term`` が現れた回数。

    実インシデント (2026-08-27 ライブ監査 T08-7): 「これまでの会話に「横浜」は
    何回出てきましたか。」に「5回」と答えた (実際 4 回)。ツールを使わず数を
    断定していた。

    進行中の user 発話 (問い自身) は **数えない**。問いは数える語を引用して
    いるので、含めると「私が「ファイルに書かないで」と言ったのは何回ですか」が
    一度も言っていない語で「1 回」になる (2026-09-10 ライブ監査 (i) I-05)。
    蓄積バッファには問いが既に積まれている (:func:`session_turn_count` の契約)
    ので、末尾の user ターンを外す。
    """
    if not term:
        return 0
    led = _ledger(session_id, create=False)
    turns = list(led.turns) if led is not None else []
    if turns and str(turns[-1].get("role") or "") == "user":
        turns = turns[:-1]
    return sum(str(turn.get("content") or "").count(term) for turn in turns)


def clear_session_data(session_id: str) -> None:
    """セッション終了 / 台帳の LRU 押し出し時にセッション固有データをクリーンアップ"""
    _ledgers.pop(session_id, None)
    rec = _turn_var.get()
    if rec is not None and rec.session_id == session_id:
        _turn_var.set(None)


@dataclass
class TurnRecord:
    """このターン (= 1 リクエスト) の間だけ生きる観測の受け皿。

    プロンプト組み立て側 (``chat._build_messages_with_search`` / 軽量パス) が
    注入した Evidence id / 検索の副情報 / few-shot 例の id を置き、``record_*``
    が :func:`_active_gen_config` 経由で経験へ刻む。kwarg で 10 箇所の
    ``record_*`` 呼出へ通さないのは、注入の決まる場所と記録の場所の間に、
    経路ごとに違う 5 種類のディスパッチが挟まるため。

    以前はセッション id をキーにしたモジュール辞書で、同一セッションで
    2 ターンが並走すると互いに上書きした。contextvar に置くことでリクエスト
    (asyncio のタスク文脈) に閉じ、後始末も要らない。``session_id`` を持つのは
    テストや直接呼出 (1 つの文脈で複数セッションを扱う) との互換のため —
    別セッションの読み書きは別のレコードとして扱う。
    """

    session_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    rag_meta: dict | None = None
    fewshot_ids: list[str] = field(default_factory=list)


_turn_var: ContextVar[TurnRecord | None] = ContextVar("chat_turn_record", default=None)


def open_turn(session_id: str) -> TurnRecord:
    """新しいターンの受け皿を置く (ターンの入口、``accumulate_user_turn``)。"""
    rec = TurnRecord(session_id=session_id)
    _turn_var.set(rec)
    return rec


def _turn(session_id: str, *, create: bool) -> TurnRecord | None:
    rec = _turn_var.get()
    if rec is not None and rec.session_id == session_id:
        return rec
    return open_turn(session_id) if create else None


def set_turn_evidence_ids(session_id: str, evidence_ids: list[str]) -> None:
    """このターンで注入した Evidence id を置く (c_16 §5.5)。"""
    _turn(session_id, create=True).evidence_ids = list(evidence_ids)


def turn_evidence_ids(session_id: str) -> list[str]:
    """このターンで注入した Evidence id (未設定は空)。"""
    rec = _turn(session_id, create=False)
    return list(rec.evidence_ids) if rec is not None else []


def set_turn_rag_meta(
    session_id: str, *, corpus_gated: bool, pseudo_derived: int,
    lexical_candidate_ids: list[str] | None = None,
    corpus_starved: bool = False,
) -> None:
    """このターンの検索の副情報を置く (``GenerationConfigRef`` へ写す)。

    ``lexical_candidate_ids`` は抑止応答の turn で取りこぼした問いの種
    (f_01 §6.4 の misses) に使う。
    """
    _turn(session_id, create=True).rag_meta = {
        "corpus_gated": bool(corpus_gated), "pseudo_derived": int(pseudo_derived),
        "lexical_candidate_ids": list(lexical_candidate_ids or []),
        "corpus_starved": bool(corpus_starved),
    }


def record_pq_misses_if_abstained(
    state: AppState, entry: object, session_id: str, user_query: str,
) -> int:
    """抑止応答 (``signals.rag_abstained``) か、corpus の候補があったのに棒で
    全部落ちた turn (``corpus_starved``) なら、取りこぼした問いを疑似クエリの
    種にする (f_01 §6.4 の misses)。積んだチャンク数を返す。"""
    signals = getattr(entry, "signals", None)
    meta = turn_rag_meta(session_id) if session_id else None
    abstained = bool(signals) and getattr(signals, "rag_abstained", None) is True
    starved = bool((meta or {}).get("corpus_starved"))
    if not (abstained or starved):
        return 0
    ids = list((meta or {}).get("lexical_candidate_ids") or [])
    manager = getattr(state, "cartridge_manager", None)
    if not ids or manager is None or not hasattr(manager, "record_pq_misses"):
        return 0
    try:
        n = int(manager.record_pq_misses(ids, user_query))
    except Exception as e:  # noqa: BLE001 — 観測のための記録で応答を止めない
        logger.warning("Failed to record pseudo-query misses: %s", e)
        return 0
    if n:
        logger.info(
            "Retrieval miss: the answer abstained on injected material; queued %d "
            "lexical candidate(s) for pseudo-query generation with the question as a hint",
            n,
        )
    return n


def turn_rag_meta(session_id: str) -> dict | None:
    """このターンの検索の副情報 (検索を通っていなければ ``None``)。"""
    rec = _turn(session_id, create=False)
    meta = rec.rag_meta if rec is not None else None
    return dict(meta) if meta else None


#: 会話単位の根拠台帳 (``SessionLedger.sources``) の上限。古いものから落とす。
#: 注入はモデルの履歴に残らず UI の出典フレームにしか出ないため、「この会話で
#: 参照した資料は」に答える材料はここにしか無い (2026-09-12 ライブ監査 T06/5)。
_SESSION_SOURCES_CAP = 200


def session_user_turn_count(session_id: str) -> int:
    """このセッションの **ユーザー発話** の通し番号 (進行中の発話を含む)。

    「今の 4 つの回答は…」のようにユーザーは質問で数える。user + assistant を
    合わせた :func:`session_turn_count` (1, 3, 5, 7 …) を台帳に振ると、モデルが
    「ターン 3」を 3 問目と読んで節を付け替えた (2026-09-12 (b) T04/5)。
    """
    led = _ledger(session_id, create=False)
    turns = led.turns if led is not None else ()
    return sum(1 for e in turns if e.get("role") == "user")


def record_turn_sources(session_id: str, items: list[dict]) -> None:
    """このターンで注入した資料を台帳へ積む (出典フレームと同じ item 形)。

    ``turn`` はユーザー発話の通し番号 (:func:`session_user_turn_count`)。
    """
    if not session_id or not items:
        return
    turn = session_user_turn_count(session_id)
    ledger = _ledger(session_id).sources
    seen = {(e["turn"], e["id"]) for e in ledger}
    for item in items:
        key = (turn, str(item.get("id") or ""))
        if key in seen:
            continue
        seen.add(key)
        ledger.append({
            "turn": turn,
            "id": str(item.get("id") or ""),
            "store": str(item.get("store") or ""),
            "package_name": str(item.get("package_name") or item.get("package_id") or ""),
            "doc_id": str(item.get("doc_id") or ""),
            "heading": str(item.get("heading") or ""),
        })
    if len(ledger) > _SESSION_SOURCES_CAP:
        del ledger[: len(ledger) - _SESSION_SOURCES_CAP]


def session_sources(session_id: str) -> list[dict]:
    """このセッションで注入した資料の台帳 (ターン順)。"""
    led = _ledger(session_id, create=False)
    return [dict(e) for e in (led.sources if led is not None else ())]


def set_turn_fewshot_ids(session_id: str, example_ids: list[str]) -> None:
    """このターンで注入した few-shot 例の id を置く (f_04 §3.2.2)。

    以前の ``gen_config.fewshot_ids`` はプール全体 (50 件) を刻んでおり、
    手本へ成否を帰属する道が無かった。
    """
    _turn(session_id, create=True).fewshot_ids = list(example_ids)


def turn_fewshot_ids(session_id: str) -> list[str]:
    """このターンで注入した few-shot 例の id (未設定は空)。"""
    rec = _turn(session_id, create=False)
    return list(rec.fewshot_ids) if rec is not None else []


def _submit_history_persist(
    mgr, ledger: SessionLedger, history_turns: list[dict], meta: dict,
) -> None:
    """未出力のターンを履歴の追記ログへ出す (c_05 §2.1 / §0.5.9)。

    ループで行うのは行の直列化と enqueue だけ (書き手スレッドが書く)。以前は
    毎ターンセッション JSON 全体と index.json を書き直していた。要約・昇格印は
    セッション JSON の側にあり、畳むときに引き継がれる (追記ログは触らない) ので、
    sleep-time の要約器が書いた値を次のターンで消さない。

    ``history_turns`` は ``ledger.turns[ledger.persisted:]`` を履歴の形にしたもの。
    """
    contents = tuple(
        (str(t.get("role", "")), str(t.get("content", ""))) for t in ledger.turns
    )
    mgr.append_turns(
        meta["session_id"], history_turns, meta,
        first_seq=ledger.persisted, contents=contents,
    )
    ledger.persisted += len(history_turns)
    logger.debug(
        "Session turns queued for history: %s (+%d, %d total)",
        meta["session_id"], len(history_turns), ledger.persisted,
    )


def _save_session_to_history(
    state: AppState, session_id: str, mode: str,
) -> None:
    """蓄積した全ターンを HistoryManager で保存する

    レスポンス完了後に呼ばれ、会話履歴をディスクに永続化する。
    同一 session_id のファイルは上書きされるため冪等。WorkingMemory ではなく
    台帳の全ターンを使い、WM のエビクションで古いターンが失われる問題を回避する。

    このスレッド (イベントループ) では **まだ出していないターンとメタ情報の
    行を組んで enqueue するだけ** にし、ファイル / 索引の読み書きは書き手スレッド
    (:mod:`backend.io.writer_thread`) に閉じる。

    ``state`` からは ``current_project_id`` だけを読む (create モードの
    プロジェクト紐付け。以前は書かれておらず全セッションが ``None`` だった)。
    """
    ledger = _ledger(session_id, create=False)
    if ledger is None or not ledger.turns:
        return
    project_id = getattr(state, "current_project_id", None)

    try:
        from backend.config import get_config
        cfg = get_config()

        # memory.private.history_storage:
        #   memory_only (既定) — private ターンのみディスクから除外し、
        #                        同席した通常ターンはセッションファイルに残す
        #   skip            — private ターンを含んだセッションは丸ごと永続化しない
        private_cfg = ((cfg.get("memory") or {}).get("private") or {})
        if (
            private_cfg.get("history_storage", "memory_only") == "skip"
            and ledger.had_private
        ):
            logger.info(
                "history save skipped (history_storage=skip, session had private turns): %s",
                session_id,
            )
            return

        from backend.io.readonly import is_readonly

        if is_readonly():
            logger.debug("History not saved: data root is read-only")
            return
        mgr = get_history_manager()

        # 再起動を跨いだ続きなら、索引の開始時刻と既存ターンを先に引き継ぐ
        # (同じファイルへ追記する形にする)。読めなかったら今回は書かない —
        # 旧ターン抜きで同じファイルを上書きしない。
        if not _ensure_session_restored(session_id, mgr):
            return
        turns = list(ledger.turns)

        # 開始時刻を記録（初回のみ）
        if not ledger.started_at:
            first_ts = turns[0].get("timestamp")
            # c_05 §0.5 の 1 形式 (ISO 8601 UTC μs ``Z``)。以前は ``isoformat()``
            # (``+00:00``) で、同じレコードの turns (``format_utc``) と形式が
            # 割れていた。読み手は ``parse_iso`` なので旧形式も読める。
            if first_ts:
                ledger.started_at = format_utc(
                    datetime.fromtimestamp(first_ts, tz=timezone.utc),
                )
            else:
                ledger.started_at = format_utc(utc_now_dt())

        # まだ出していないターンだけを履歴用フォーマットに変換
        history_turns = []
        for t in turns[ledger.persisted:]:
            entry = {"role": t["role"], "content": t["content"]}
            ts = t.get("timestamp")
            if ts:
                entry["timestamp"] = format_utc(
                    datetime.fromtimestamp(ts, tz=timezone.utc),
                )
            for key in ("turn_id", "trace_id", "meta", "compressed", "original_length"):
                if (value := t.get(key)) not in (None, ""):
                    entry[key] = value
            history_turns.append(entry)

        meta = {
            "session_id": session_id,
            "started_at": ledger.started_at,
            "ended_at": format_utc(utc_now_dt()),
            "mode": mode,
            "instance_name": cfg.get("instance", {}).get("name", "evoref"),
            "base_model": active_base_model_name(cfg),
            "model_key": active_base_model_key(),
            "project_id": project_id,
        }
        _submit_history_persist(mgr, ledger, history_turns, meta)
    except DataReadonlyError:
        logger.debug("History not saved: data root is read-only")
    except Exception as e:
        logger.warning("Failed to save session to history: %s", e)


def release_session_turns(session_id: str) -> None:
    """セッション終了時の後始末 (蓄積バッファの掃除)。

    ノートの生成は会話履歴を入力にする sleep-time の仕事になったので
    (c_16 §4.1)、ここで記憶層へ流すものは無い。窓は呼出側が落とす。
    """
    clear_session_data(session_id)


def end_session(state: AppState, session_id: str) -> bool:
    """明示のセッション終了: 窓を台帳から外し、セッション別カウンタを畳む。

    セッション解除 API (``DELETE /api/sessions/{id}``) から呼ぶ。台帳に
    無ければ ``False``。会話は履歴に残っているので、まだノート化されて
    いないターンは次の sleep-time が拾う (c_16 §4.1)。
    """
    registry = getattr(state, "working_memory_registry", None)
    dropped = False
    if registry is not None:
        dropped = registry.drop(session_id) is not None
    release_session_turns(session_id)
    # 閉じたセッションの追記ログをセッション JSON へ畳む (書き手スレッドで、待たない)。
    mgr = _history_manager_module._manager_cache
    if mgr is not None:
        try:
            mgr.close_session(session_id)
        except Exception as exc:
            logger.warning("Failed to close history session %s: %s", session_id, exc)
    tracker = getattr(state, "judge_tracker", None)
    if tracker is not None:
        tracker.reset_session(session_id)
    return dropped


def _wm_correction_flag(
    state: AppState, user_query: str, session_id: str | None = None,
) -> bool | None:
    """``prepare_memory_context`` が WM の user ターンに立てた訂正の印を読む。

    ``restates_a_value`` は応答パスの入口で 1 回判定し、結果を turn dict の
    ``is_correction`` に置いている。record 側で再判定せずそれを読む
    (同じ述語を 1 ターンに 3 回走らせていた)。該当ターンが窓から落ちて
    いる / WM が無い場合は ``None`` (呼出側で判定にフォールバック)。
    """
    registry = getattr(state, "working_memory_registry", None)
    if registry is not None and session_id and registry.peek(session_id) is None:
        # 台帳に無いセッションの窓を読み出しのために再生しない (LRU 落ち / 終了済み)
        return None
    mem_sys = (
        state.get_memory_system(session_id)
        if hasattr(state, "get_memory_system") else None
    )
    if not mem_sys:
        return None
    wm = mem_sys[0]
    turns = getattr(wm, "turns", None)
    if not isinstance(turns, list):
        return None
    for turn in reversed(turns):
        if turn.get("role") == "user" and turn.get("content") == user_query:
            return bool(turn.get("is_correction", False))
    return None


def _schedule_sleep_time(
    state: AppState, user_query: str, private: bool,
    *, correction: bool | None = None,
) -> None:
    """sleep-time update をスケジュールする (record_* 3 経路の共通処理)。

    ``correction`` は入口 (``prepare_memory_context``) で判定済みの
    ``restates_a_value`` の結果。``None`` ならここで判定する。

    訂正ターンでは Full を **前倒し** する。ファクト抽出 (Step 8) と競合解決
    (Step 6B) は Full にしか無く、Light は Step 1-5.5
    (埋め込み / タグ / スコア / eviction) だけ。そのため
    既定 (アイドル 10 分 / 繰り延べ上限 30 分) では、ユーザーが訂正しても
    SemMem に反映されるまで最大 30 分かかる。訂正は反映が遅れると意味が薄れる
    ので、そのターンだけ待ち時間を下限まで縮める。

    **訂正でない更新** (引っ越し / 転職) も同じ理由で前倒しする。単値スロットの
    旧値を畳むのは Step 8 だけなので、Full が走るまでの間は

    - 旧値: SemMem ファクト (Tier 1)
    - 新値: STM ノートだけ (Tier 2)

    となり、**Tier の序列上どうやっても旧値が勝つ**。実測 (2026-08-29 ライブ監査
    F38): 「転職してデータサイエンティストになりました」の直後の新セッションで
    「インフラエンジニアです」と旧値を返し、自己検査も「古い情報は含まれて
    いません」と保証した。``restates_a_value`` は「〜ではなく〜」型の言い直しを
    拾う述語なので、この種の更新には掛からない。

    前倒しの誤爆は「Full が少し早く走る」だけで、正しい値を消す方向の失敗が
    無い — 窓を縮める側に倒す。

    private ターンは SemMem へ書かない契約なので前倒ししない。
    """
    scheduler = state.sleep_scheduler
    if scheduler is None:
        return
    if not private and user_query:
        try:
            from backend.free.agent.feedback import restates_a_value
            from backend.free.memory.notes.note_builder import (
                states_single_valued_attribute,
            )

            if correction is None:
                correction = restates_a_value(user_query)
            if correction:
                scheduler.request_full_soon("value_restated")
            elif states_single_valued_attribute(user_query):
                scheduler.request_full_soon("single_valued_attribute_stated")
        except Exception as exc:
            logger.warning(
                "correction-triggered full request skipped: %s", exc,
            )
    scheduler.on_response_sent()


def _tool_result_text_in_prompt(messages: list[ChatMessage]) -> str:
    """最後の user メッセージへ注入済みのツール実行結果ブロック (無ければ空)。

    ``_calculate_result_in_prompt`` と同じ理由でプロンプトから読み戻す。
    日付演算の ``target:`` と応答本文の日付を突き合わせる
    (``core.response_dates.ignores_date_result``、2026-09-09 監査 G-06)。

    返すのは ``TOOL_RESULT_HEADER`` 以降の **ブロックだけ**。以前は user
    メッセージ全文を返しており、``target:`` を探す消費者には害が無かったが、
    F-09 が足した ``tool_grounded = bool(tool_result_text)`` が **全ターンで
    真** になり、few-shot プールが 50 ターンで 0 件になった (2026-09-10
    ライブ監査 (g) G-06)。「ツール結果がプロンプトに有ったか」は見出しの
    有無で決める (見出しは ``turn_text.split_last_user`` と同じ境界)。
    """
    if not messages:
        return ""
    content = str(messages[-1].get("content") or "")
    idx = content.find(TOOL_RESULT_HEADER)
    return content[idx:] if idx >= 0 else ""


def _calculate_result_in_prompt(messages: list[ChatMessage]) -> float | None:
    """最後の user メッセージへ注入済みの calculate 結果 (無ければ ``None``)。

    ``_turn_contradiction_inputs`` の実測値と同じく、注入したのはこのプロセス
    なのでプロンプトから読み戻す。
    """
    if not messages:
        return None
    return extract_calculate_result(str(messages[-1].get("content") or ""))


def _stated_context(messages: list[ChatMessage]) -> str:
    """応答の「数え直し」判定に渡す **ユーザー側の本文** をまとめる (純粋関数)。

    プロンプトに載った user メッセージ (会話履歴 + ``[関連する記憶]`` /
    ``[参考情報]`` / ツール結果のブロック) を連結する。ここに現れない人数を
    応答が述べていれば、本人が言っていない数を補ったことになる
    (:func:`~backend.free.core.text_quality.fabricated_household_count`)。
    assistant メッセージは入れない — 自分の過去の数え直しを根拠にすると、
    一度補った数がそのまま正当化されて固定する。
    """
    return "\n".join(
        str(m.get("content") or "")
        for m in (messages or [])
        if str(m.get("role") or "") == "user"
    )


def _turn_contradiction_inputs(
    state: AppState,  # noqa: ARG001 - 呼出面の互換 (判定器を後から覗く経路は撤去)
    messages: list[ChatMessage],
    action_blocked: bool | None = None,
) -> tuple[bool, dict[str, set[int]]]:
    """``turn_outcome`` の矛盾検出に渡す 2 つの入力を集める。

    どちらも「システムが既に知っていること」で、応答本文と突き合わせると
    真偽の推定なしに矛盾を検出できる (``_derive_turn_outcome`` 参照)。

    - ``action_blocked``: 状態を変える依頼なのに撃てるツールが無かったか。
      当ターンの ``ToolJudgement.action_blocked`` (deliberative が注記の要否を
      決めるのに読むのと同じ値) を呼出側が渡す。
    - ``measured_values``: ``[システム計測]`` として最後の user メッセージへ
      注入した実測値。注入したのはこのプロセスなので、プロンプトから読み戻す
      (新しい引数を 5 つの層へ通す代わりに、注入結果そのものを見る)。
    """
    # 判定結果が渡されない経路 (reactive 即応答等) はツール判定を経ていない
    # ので「撃てなかった」も無い。``ToolCallJudge`` はターン固有の値を保持
    # しない (プロセス唯一の共有インスタンスで、後から属性を読むとチャットが
    # 2 本重なったときに他方の値を読む — ``ToolJudgement.action_blocked`` の
    # コメント参照) ため、判定器を後から覗く経路は撤去した。
    measured: dict[str, set[int]] = {}
    if messages:
        measured = extract_measured_values(str(messages[-1].get("content") or ""))
    return bool(action_blocked), measured


#: 成果物として保持する最小の応答長 (文字)。履歴予算 (実測 1612 トークン
#: ≒ 日本語 2000 文字強) の半分を目安にする — 他のターンと合わさると
#: この程度から落ち始める。短い応答まで保持すると、直後の相槌で本物の
#: 成果物を上書きしてしまう。
ARTIFACT_MIN_CHARS = 1200


def tool_routing_signals(
    tool_calls: list[dict] | None,
) -> tuple[bool, bool]:
    """ツール実行結果から ``(tool_routing_success, tool_routing_false_positive)`` を導く。

    3 経路が別々の式で書いていた同じ規則を 1 箇所にする:

    - **deliberative**: 判定層が撃った run_command 1 件 (``tool_command`` /
      ``tool_command_success``)。:func:`command_tool_calls` で 1 要素のリストに
      して渡す。success が None (実行されなかった) なら要素を作らない。
    - **meta_cognitive**: ``resp.tool_calls`` (複数)。1 件でも成功なら誘導は
      妥当 (success)、全部失敗なら誤検出 (false_positive)。
    - **long_form**: ツールを撃たないので常に ``(False, False)``。

    ``success is None`` の要素は「実行されなかった」なので数えない。呼ばれた
    ツールが 1 件も無ければ両方 False (未使用は誤検出ではない)。
    """
    calls = [
        tc for tc in (tool_calls or [])
        if isinstance(tc, dict) and tc.get("success") is not None
    ]
    if not calls:
        return False, False
    ok = any(bool(tc.get("success")) for tc in calls)
    return ok, not ok


def command_tool_calls(
    tool_command: str | None, tool_command_success: bool | None,
) -> list[dict]:
    """deliberative の単一 run_command を :func:`tool_routing_signals` の入力へ。"""
    if tool_command is None or tool_command_success is None:
        return []
    return [{"tool": "run_command", "success": bool(tool_command_success)}]


def _log_request_debug(
    dl, tokens_generated: int, messages: list[ChatMessage], response: str,
    *, private: bool,
) -> None:
    """``requests`` JSONL へ記録する。private ターンは本文を残さない。

    件数 (トークン数 / メッセージ数) は残し、本文だけを伏せる。private の
    契約は「ディスクへ本文を残さない」で、evolve の requests JSONL も
    ディスクなので例外にしない (2026-09-02 監査 R-A6)。
    """
    if not private:
        dl.log_request(tokens_generated, messages, response)
        return
    redacted = [
        {"role": m.get("role", ""), "content": "[REDACTED: private turn]"}
        for m in messages
    ]
    dl.log_request(tokens_generated, redacted, "[REDACTED: private turn]")


def _remember_if_artifact_sized(
    state: AppState, session_id: str, response: str, query: str, mode: str,
) -> None:
    """長い応答だけを「直前の成果物」として保持する。

    短い確認応答 (「plan.md に書き込みました。」) まで保持すると、直後の
    「その中身を見せて」で **確認応答の方** が素材になり、本物の成果物への
    参照が切れる。閾値で分けるのはそのため。
    """
    if len(response or "") < ARTIFACT_MIN_CHARS:
        return
    remember_artifact(state, session_id, text=response, query=query, mode=mode)


def _record_assistant_turn_to_memory(
    state: AppState, full_response: str, session_id: str, user_query: str,
    mode: str, *,
    private: bool,
    tool_command: str | None = None,
    tool_command_name: str | None = None,
    tool_command_success: bool | None = None,
    tool_command_source: str | None = None,
) -> str:
    """assistant 応答をワーキングメモリへ積む (record_* 3 経路の共通処理)。

    - 記憶層への書き込みはここでは起きない (c_16 §2.1: エピソード記憶の
      書き手は sleep-time だけ)。窓へ積むのと、蓄積バッファ → 会話履歴に
      残すのが応答パスの仕事。
    - WM は ``session_id`` のもの (``get_memory_system(session_id)``)。
    - ``tool_command*`` は run_command 実行ターンの learning メタ。3 経路とも
      同じ kwargs を受けるので、meta-cognitive 経路の run_command も
      sleep-time Step 8.6 (executable_command_curator) に届く。
    """
    registry = getattr(state, "working_memory_registry", None)
    if (
        registry is not None
        and session_id
        and registry.peek(session_id) is None
        and full_response
    ):
        # 台帳から落ちたセッション (LRU 押し出し / 明示終了) に遅れて応答が
        # 返った。窓を作り直しても次のターンは来ないので積まない。応答本文は
        # 蓄積バッファ経由で会話履歴に残り、sleep-time がそこからノートにする。
        logger.info(
            "record: session %s has no working memory (ended/evicted); "
            "the late assistant turn is kept in history only", session_id[:8],
        )
        return ""
    mem_sys = state.get_memory_system(session_id)
    if not mem_sys:
        return ""
    wm, _episodic = mem_sys
    turn_id = ""
    if full_response:
        body = _recorded_body(full_response)
        # 発火元のクエリをターンに確定させる。curator がノートを走査して
        # 「直前で最も近い user ノート」から推測すると、当該ターンの user
        # ノートが無い場合に別ターンのクエリと結び付く。
        tool_command_query = user_query if tool_command else None
        turn_id = wm.add_turn(
            "assistant", body,
            private=private, mode=mode, source="assistant",
            tool_command=tool_command,
            tool_command_name=tool_command_name,
            tool_command_success=tool_command_success,
            tool_command_source=tool_command_source,
            tool_command_query=tool_command_query,
        )
    return turn_id


def _active_gen_config(
    state: AppState, mode: str, session_id: str = "",
) -> "GenerationConfigRef":
    """このターンで有効だった構成 (プロンプト版 / few-shot / ポリシー / LoRA)。

    fitness の帰属先を **推測せず記録** するための素性 (c_05 §0.6)。取得
    できない要素は ``None`` / 空のままにする (0 で埋めると「未設定」と
    「値が 0」が区別できない)。
    """
    from backend.free.learning.level0_instant import GenerationConfigRef

    ref = GenerationConfigRef()
    try:
        from backend.i18n_helper import get_locale

        ref.locale = get_locale() or ""
    except Exception:
        pass
    pm = getattr(state, "prompt_manager", None) or getattr(
        getattr(state, "loop", None), "prompt_manager", None,
    )
    if pm is not None:
        try:
            # このセッションが凍結している版 (f_03 §7.1.3 #1)。記録時点の現行版を
            # 刻むと、採用後も旧版で話し続けるターンが新しい版に帰属する。
            frozen = getattr(pm, "frozen_version", None)
            version = frozen(mode, session_id) if (frozen and session_id) else None
            if not isinstance(version, int):
                version = pm.get_meta(mode).version
            ref.prompt_version = int(version)
        except Exception:
            pass
    # 実際に注入した例だけを刻む (以前はプール全体で、読み手は無かった)。
    ref.fewshot_ids = turn_fewshot_ids(session_id) if session_id else []
    sched = getattr(state, "learning_scheduler", None)
    if sched is not None:
        evolver = getattr(sched, "_policy_param_evolver", None)
        generation = getattr(evolver, "generation", None)
        if isinstance(generation, int):
            ref.policy_generation = generation
    # adapters は記録しない (空のまま): llama-server が載せたアダプタは起動時に決まり、
    # backend はその版を追跡していない (GenerationConfigRef.adapters)。
    ref.evidence_ids = turn_evidence_ids(session_id) if session_id else []
    rag_meta = turn_rag_meta(session_id) if session_id else None
    if rag_meta is not None:
        ref.corpus_gated = rag_meta["corpus_gated"]
        ref.pseudo_derived_count = rag_meta["pseudo_derived"]
    # 体裁の継承 (write_file が WriteResult.metadata へ載せた場合) / 帳票の
    # 穴埋め (書込み成功時) の来歴。session_id を持たないツール境界
    # (write_file) から運ばれるため、TurnRecord ではなく専用 contextvar 経由
    # (c_16 §4.5.2)。plan_seed (長文の構成テンプレート) はこの経路を通らない
    # 呼出元 (record_long_form_response / record_meta_cognitive_response) が
    # 明示的に上書きする。
    from backend.export.template_context import applied_template_key

    ref.template = applied_template_key()
    return ref


def _gen_config_with_template(
    state: AppState, mode: str, session_id: str, template: str,
) -> "GenerationConfigRef":
    """``_active_gen_config`` に、呼出元が明示した来歴鍵があれば上書きして返す。

    長文の構成テンプレート seed (``plan_seeded``) は ``write_file`` の
    contextvar を経由しないため、呼出元 (``record_long_form_response`` /
    ``record_meta_cognitive_response``) が ``metrics`` / ``production_metrics``
    から読んだ値をここで明示的に渡す。
    """
    ref = _active_gen_config(state, mode, session_id)
    if template:
        ref.template = template
    return ref


#: 履歴ターンの ``meta`` に載せる WM ターンのキー。
#:
#: ノート生成が会話履歴を入力にするようになったので (c_16 §4.1)、ノートの
#: フィールドを埋めるのに要るものはここを通すしかない。``tool_command`` /
#: ``tool_command_query`` は sleep-time Step 8.6
#: (``executable_command_curator``) の入力で、以前は WM ターン →
#: ``MemoryNote`` へ直に渡っていた。
_TURN_META_KEYS = (
    "mode", "source", "is_correction",
    "tool_command", "tool_command_name", "tool_command_success",
    "tool_command_source", "tool_command_query",
)


def _turn_meta_from_wm(state: AppState, session_id: str, turn_id: str) -> dict | None:
    """WM から ``turn_id`` のターンを引き、履歴に残すメタを組む。"""
    mem_sys = state.get_memory_system(session_id)
    if not mem_sys:
        return None
    wm = mem_sys[0]
    for turn in reversed(getattr(wm, "turns", [])):
        if turn.get("turn_id") != turn_id:
            continue
        return {k: turn.get(k) for k in _TURN_META_KEYS if turn.get(k) is not None}
    return None


def _finish_turn_bookkeeping(
    state: AppState, full_response: str, session_id: str, user_query: str,
    mode: str, *, private: bool,
    turn_id: str = "", turn_meta: dict | None = None,
) -> None:
    """record_* 3 経路の末尾処理: sleep-time スケジュール / 蓄積 / 履歴保存。

    ``turn_meta`` 未指定なら WM のターン本体から組み立てる (呼出元 3 経路で
    引数を並べ直すより、既に確定している 1 か所から読む方が食い違わない)。
    """
    if turn_meta is None and turn_id:
        turn_meta = _turn_meta_from_wm(state, session_id, turn_id)
    # sleep-time update をスケジュール (訂正ターンは Full を前倒し)。判定は
    # 入口で済んでいるので WM の印を読む (無ければ判定にフォールバック)。
    _schedule_sleep_time(
        state, user_query, private,
        correction=_wm_correction_flag(state, user_query, session_id),
    )

    # ターンを蓄積（WM エビクションに依存しない完全な履歴）。user 側は入口で
    # 積まれているのが通常で、ここは冪等な保険。
    accumulate_user_turn(session_id, user_query, private=private)
    if full_response:
        # 履歴へ積む本文は記憶 / 経験と同じ (開示注記を落としたもの)。
        # 生の full_response を積むと、同じターンについて履歴と記憶が別々の
        # 「真実」を持つ (2026-09-05 監査)。
        _accumulate_turn(
            session_id, "assistant", _recorded_body(full_response),
            private=private, turn_id=turn_id, meta=turn_meta,
        )

    # 会話履歴をディスクに保存 (private なら蓄積されていないので no-op)
    if not private:
        _save_session_to_history(state, session_id, mode)
    # このターンの追記 (経験・履歴) をファイルごとに 1 回 fsync する (書き手スレッド)。
    default_writer().end_turn()


def record_response(
    state: AppState, full_response: str, messages: list[ChatMessage],
    session_id: str, user_query: str, mode: str,
    tokens_generated: int,
    *,
    private: bool = False,
    tool_command: str | None = None,
    tool_command_name: str | None = None,
    tool_command_success: bool | None = None,
    tool_command_source: str | None = None,
    tool_routing_success: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    action_blocked: bool | None = None,
    sent_messages: list[ChatMessage] | None = None,
    cancelled: bool = False,
    truncated: bool = False,
    generation_failed: bool = False,
) -> None:
    """応答をメモリ・デバッグログ・経験バッファに記録する

    ``private=True`` の場合は WM/STM までの伝搬のみ行い
    会話履歴ディスク永続化と feedback collector への記録をスキップする。

    ``cancelled`` (クライアントキャンセルで途中まで) のターンはメモリ / 履歴
    の帳簿は付けるが **経験としては記録しない** — 部分応答が成功例として
    学習に入っていた (2026-09-02 監査 R-A2)。``truncated`` は
    ``finish_reason=length`` の印で経験へそのまま刻む。``generation_failed``
    (error フレームで終わった) と **空応答** は ``response=""`` の失敗経験と
    して記録する — 以前は記録自体が無く、失敗が選択圧に一件も入らなかった。

    ``sent_messages`` は **実際に llama-server へ送った** メッセージ配列。
    ``messages`` は ``build_messages()`` 直後の配列で、``DeliberativeAgent``
    には ``list(messages)`` の浅いコピーが渡るため、``## ツール実行結果`` や
    リマインダーを積んだ後の姿は入っていない。--develop=evolve の
    ``requests`` JSONL は「プロンプト起因の不具合をログから追う」ためのもの
    なので、送信版がある場合はそちらを記録する (2026-08-30 ライブ監査:
    ツール接地ターンの根拠ブロックがログから丸ごと欠けていた)。

    ``tool_command`` / ``tool_command_name`` / ``tool_command_success`` は
    run_command 実行ターンの learning メタで、assistant note に載せて
    sleep-time の executable_command_curator が参照する (それ以外は None)。
    """
    # 同一ターンの明確な失敗 = ツールをルーティングしたが失敗 → false_positive。
    _, tool_fp = tool_routing_signals(
        command_tool_calls(tool_command, tool_command_success),
    )
    _record_turn(
        state, full_response, messages, session_id, user_query, mode,
        tokens_generated,
        layer="deliberative",
        private=private,
        tool_command=tool_command,
        tool_command_name=tool_command_name,
        tool_command_success=tool_command_success,
        tool_command_source=tool_command_source,
        action_blocked=action_blocked,
        sent_messages=sent_messages,
        cancelled=cancelled,
        truncated=truncated,
        generation_failed=generation_failed,
        experience_kwargs={
            "tool_routing_success": tool_routing_success,
            "tool_routing_false_positive": tool_fp,
            "rag_used": rag_used,
            "rag_top1_score": rag_top1_score,
        },
    )


def _record_turn(
    state: AppState, full_response: str, messages: list[ChatMessage],
    session_id: str, user_query: str, mode: str,
    tokens_generated: int,
    *,
    layer: str,
    private: bool,
    tool_command: str | None,
    tool_command_name: str | None,
    tool_command_success: bool | None,
    tool_command_source: str | None,
    action_blocked: bool | None,
    sent_messages: list[ChatMessage] | None,
    cancelled: bool,
    truncated: bool,
    generation_failed: bool,
    experience_kwargs: dict,
    keep_artifact: bool = False,
    template: str = "",
) -> None:
    """``record_*`` 3 経路の共通本体 (WM → 成果物 → デバッグログ → 経験 → 帳簿)。

    経路ごとの違いは ``experience_kwargs`` (経験レコードに足す層固有の信号) と
    ``keep_artifact`` (長文経路は長さに関わらず成果物として保持) だけ。以前は
    3 経路が同じ手順を写経しており、meta_cognitive / long_form だけ経験へ
    ``turn_id`` (ID 連鎖、c_05 §0.6) を刻まず、抑止応答の疑似クエリ種
    (:func:`record_pq_misses_if_abstained`) も積んでいなかった。

    ``template`` は呼出元が明示的に確定した来歴鍵 (長文の構成テンプレート
    seed 等、``write_file`` の contextvar を経由しない経路)。指定が無ければ
    ``_active_gen_config`` が読む contextvar (体裁の継承 / 帳票の穴埋め) に
    委ねる。
    """
    # メモリに応答を記録
    assistant_turn_id = _record_assistant_turn_to_memory(
        state, full_response, session_id, user_query, mode,
        private=private,
        tool_command=tool_command,
        tool_command_name=tool_command_name,
        tool_command_success=tool_command_success,
        tool_command_source=tool_command_source,
    )

    # 履歴予算に入らない長さの応答は成果物として保持する。
    # 実測 (2026-08-27 ライブ監査) の履歴予算は 1612 トークン。これを超える
    # 出力は次ターンで落ちるため、「いま書いたコードは何行ですか」に
    # **「1行です」** (実際は 26 行) と答えていた。長文経路は WM へ積んでも
    # 次のターンには残らないので長さに関わらず保持する (同 T10: 6696 文字の
    # 計画書の次のターンで「履歴に含まれていない」と答えた)。
    if keep_artifact:
        # キャンセルされた生成の途中本文は成果物にしない (次ターンの
        # 「その計画書を保存して」が部分本文を掴む)。
        if full_response and not cancelled:
            remember_artifact(
                state, session_id,
                text=full_response, query=user_query, mode=mode,
            )
    else:
        _remember_if_artifact_sized(
            state, session_id, full_response, user_query, mode,
        )

    # デバッグログ
    dl = state.debug_logger
    if dl:
        _log_request_debug(
            dl, tokens_generated, sent_messages or messages, full_response,
            private=private,
        )

    # 経験バッファに記録 (Level 0) — private / キャンセルは学習対象外
    fc = state.feedback_collector
    if fc and cancelled and not private:
        logger.info(
            "Skipping experience record for cancelled turn (%s, session=%s)",
            layer, session_id,
        )
    elif fc and not private:
        try:
            # 記憶へ積む本文と同じもの (開示注記を落とした本文) を経験にも
            # 使う。生の full_response を渡すと注記込みの応答が手本に昇格する。
            body = _recorded_body(full_response)
            prompt_tokens, cached_tokens = read_llama_prompt_tokens(state)
            blocked, measured = _turn_contradiction_inputs(
                state, messages, action_blocked,
            )
            entry = fc.record(
                query=user_query, response=body, mode=mode,
                completion_tokens=tokens_generated,
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_tokens,
                action_blocked=blocked,
                measured_values=measured,
                stated_context=_stated_context(messages),
                calculate_result=_calculate_result_in_prompt(messages),
                tool_result_text=_tool_result_text_in_prompt(messages),
                truncated=truncated,
                generation_failed=generation_failed or not body.strip(),
                session_id=session_id,
                turn_id=assistant_turn_id,
                gen_config=_gen_config_with_template(state, mode, session_id, template),
                **experience_kwargs,
            )
            # 長文経路は経験を記録してからファイルを書くので、体裁の継承を適用したか
            # どうかはこの時点でまだ決まっていない。来歴の入れ先を預けておき、
            # write_file が適用した時点で埋めさせる (様式が無いターンは no-op)。
            from backend.export.template_context import defer_template_provenance

            gen_config = getattr(entry, "gen_config", None)
            if gen_config is not None:
                defer_template_provenance(gen_config)
            record_pq_misses_if_abstained(state, entry, session_id, user_query)
        except Exception as e:
            # 経験記録の失敗でチャットを壊さない方針は維持するが、**握り潰さない**。
            # 実インシデント (2026-08-23): 引数を 1 つ足し忘れた NameError が
            # WARNING 1 行に化け、meta_cognitive / long_form の経験記録が静かに
            # 全滅していた (テストで検出)。traceback 付き ERROR なら気づける。
            logger.error(
                "FeedbackCollector.record failed (%s): %s", layer, e,
                exc_info=True,
            )

    _finish_turn_bookkeeping(
        state, full_response, session_id, user_query, mode, private=private,
        turn_id=assistant_turn_id,
    )


def record_meta_cognitive_response(
    state: AppState, full_response: str, messages: list[ChatMessage],
    session_id: str, user_query: str, mode: str,
    tokens_generated: int, step_credits: list,
    *,
    private: bool = False,
    agent_loops: int = 0,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_routing_success: bool = False,
    tool_routing_false_positive: bool = False,
    tool_command: str | None = None,
    tool_command_name: str | None = None,
    tool_command_success: bool | None = None,
    tool_command_source: str | None = None,
    sent_messages: list[ChatMessage] | None = None,
    action_blocked: bool | None = None,
    cancelled: bool = False,
    truncated: bool = False,
    generation_failed: bool = False,
    template: str = "",
) -> None:
    """Meta-Cognitive 層の応答をメモリ・経験バッファに記録（クレジット付き）

    ``private=True`` の場合は WM/STM までの伝搬のみ。

    ``tool_command*`` / ``sent_messages`` / ``cancelled`` / ``truncated`` /
    ``generation_failed`` は :func:`record_response` と同じ意味。meta 経路で
    run_command が走ったターンも、呼出側がこれを渡せば sleep-time Step 8.6
    (executable_command_curator) の学習対象になる。

    ``action_blocked`` は deliberative の ``ToolJudgement.action_blocked`` に
    相当する印だが、meta 経路にはそれを出す判定層が無い (ツールはタスク計画
    から呼ばれ、「撃てるツールが無い」はタスク failed として現れる)。呼出側が
    導けなければ ``None`` のまま = 矛盾検出のこの入力は使わない。

    ``template`` は production stage (staged/longform、f_03 §4.4) の
    ``MetaCognitiveResponse.production_metrics`` から呼出側が読んだ来歴鍵
    (長文の構成テンプレート seed。体裁の継承 / 帳票の穴埋めは write_file の
    contextvar 経由で ``_active_gen_config`` が拾うのでここでは渡さない)。
    """
    credits_dicts = [
        {"step_index": c.step_index, "action": c.action, "credit": c.credit}
        for c in step_credits
    ] if step_credits else []
    _record_turn(
        state, full_response, messages, session_id, user_query, mode,
        tokens_generated,
        layer="meta-cognitive",
        private=private,
        tool_command=tool_command,
        tool_command_name=tool_command_name,
        tool_command_success=tool_command_success,
        tool_command_source=tool_command_source,
        action_blocked=action_blocked,
        sent_messages=sent_messages,
        cancelled=cancelled,
        truncated=truncated,
        generation_failed=generation_failed,
        template=template,
        experience_kwargs={
            "agent_loops": agent_loops,
            "rag_used": rag_used,
            "rag_top1_score": rag_top1_score,
            "tool_routing_success": tool_routing_success,
            "tool_routing_false_positive": tool_routing_false_positive,
            "step_credits": credits_dicts,
        },
    )


def record_long_form_response(
    state: AppState, full_response: str, messages: list[ChatMessage],
    session_id: str, user_query: str, mode: str,
    tokens_generated: int, metrics: dict,
    *,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
    tool_command: str | None = None,
    tool_command_name: str | None = None,
    tool_command_success: bool | None = None,
    tool_command_source: str | None = None,
    sent_messages: list[ChatMessage] | None = None,
    action_blocked: bool | None = None,
    cancelled: bool = False,
    truncated: bool = False,
    generation_failed: bool = False,
) -> None:
    """長文生成の応答をメモリ・経験バッファに記録

    ``private=True`` の場合は WM/STM までの伝搬のみ。``tool_command*`` /
    ``sent_messages`` / ``cancelled`` / ``truncated`` / ``generation_failed``
    は :func:`record_response` と同じ意味。

    ``action_blocked`` は長文経路では導出できない (ツール判定層を通らず、
    状態を変える操作はユニット生成の外で write_file が担う) ため、呼出側は
    渡さない = 矛盾検出のこの入力は使わない。
    """
    units_completed = int(metrics.get("units_completed", 0) or 0)
    validation_errors = int(metrics.get("validation_errors", 0) or 0)
    long_form_success = judge_long_form_success(
        metrics, user_query, _recorded_body(full_response),
    )
    _record_turn(
        state, full_response, messages, session_id, user_query, mode,
        tokens_generated,
        layer="long-form",
        private=private,
        tool_command=tool_command,
        tool_command_name=tool_command_name,
        tool_command_success=tool_command_success,
        tool_command_source=tool_command_source,
        action_blocked=action_blocked,
        sent_messages=sent_messages,
        cancelled=cancelled,
        truncated=truncated,
        generation_failed=generation_failed,
        keep_artifact=True,
        # 構成テンプレートで seed した場合の来歴 (f_08 §3.1.1)。orchestrator が
        # plan_seeded 時だけ ``last_metrics["template"]`` を持つ (c_05 §0.6)。
        template=str(metrics.get("template") or ""),
        experience_kwargs={
            "long_form_used": True,
            "long_form_content_type": metrics.get("content_type"),
            "long_form_strategy": metrics.get("strategy"),
            "long_form_units_total": metrics.get("units_total", 0),
            "long_form_units_completed": units_completed,
            "long_form_validation_errors": validation_errors,
            "long_form_budget_used_pct": metrics.get("budget_used_pct"),
            "long_form_success": long_form_success,
            # 長文経路に入ったが 1 ユニットも生成できなかった、または要求
            # 成果物と content_type が矛盾 = 長文分類の明確な誤検出
            # → false_positive (パターン重み decay の対象)。units>0 で
            # validation_errors のみのケースは長文ルーティング自体は妥当なので除外。
            "long_form_false_positive": (
                units_completed == 0
                or is_content_type_mismatch(metrics, user_query)
            ),
            "rag_used": rag_used,
            "rag_top1_score": rag_top1_score,
        },
    )
