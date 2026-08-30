"""レスポンス記録（メモリ・デバッグログ・フィードバック・履歴）"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from backend.app_state import AppState
from backend.free.core.text_quality import strip_system_notes
from backend.free.api.chat._artifact import remember_artifact
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.core.text_quality import is_query_echo, strip_echoed_query
from backend.free.history.history_manager import (
    SessionData,
    active_base_model_name,
    get_history_manager,
)
from backend.free.core.text_quality import extract_measured_values
from backend.log_config import get_logger
from backend.utils import utc_now_dt

if TYPE_CHECKING:
    from backend.free.memory.stores.working import WorkingMemory
    from backend.free.memory.stores.short_term import ShortTermMemory

logger = get_logger("api.chat.recorder")

# 文書系の出力先拡張子。これらへの出力依頼で content_type=code が返るのは
# ルーティング誤り (long_form_success の判定材料)。
_DOC_TARGET_EXT_RE = re.compile(r"\.(?:md|txt|csv)\b", re.IGNORECASE)

def read_llama_prompt_tokens(state: AppState) -> tuple[int | None, int | None]:
    """直近ストリームの ``(prompt_tokens, cached_prompt_tokens)`` を返す。

    llama-server の ``usage.prompt_tokens_details.cached_tokens`` を
    :class:`~backend.free.llm.local_client.LocalLLMClient` が ``_last_timings``
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
    client = getattr(getattr(state, "gen", None), "llm_client", None)
    timings = getattr(getattr(client, "local", client), "_last_timings", None)
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


# セッション別の開始時刻（初回リクエスト時に記録）
_session_started: dict[str, str] = {}

# セッション別の全ターン蓄積（WM のエビクションに依存しない完全な履歴）
_session_turns: dict[str, list[dict]] = {}

# private ターンを 1 度でも含んだセッション ID
# (``memory.private.history_storage: skip`` でセッションごと永続化を落とす判定に使う)
_session_had_private: set[str] = set()


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
) -> None:
    """セッションのターンを蓄積

    WorkingMemory はターン数・トークン数上限で古いターンを押し出すため、
    履歴保存用に全ターンを独立して蓄積する。

    ``private=True`` のターンはディスク永続化対象から
    除外する (memory_only)。蓄積バッファ自体に追加しない。
    """
    if private:
        _session_had_private.add(session_id)
        logger.debug(
            "accumulate skipped (private turn): role=%s, session=%s, len=%d",
            role, session_id, len(content),
        )
        return
    if session_id not in _session_turns:
        _session_turns[session_id] = []
    _session_turns[session_id].append({
        "role": role,
        "content": content,
        "timestamp": time.time(),
    })


def session_turn_count(session_id: str) -> int:
    """このセッションの累計ターン数 (user + assistant)。

    ``WorkingMemory`` は窓を越えた分を押し出すので、会話全体を数えられるのは
    こちらの蓄積バッファだけ。実インシデント (2026-08-27 ライブ監査 T19-4):
    148 ターン目に「50ターン目です」と答えた (窓に入っている分だけを数えた)。
    """
    return len(_session_turns.get(session_id) or ())


def count_term_in_session(session_id: str, term: str) -> int:
    """このセッションの全ターン本文に ``term`` が現れた回数。

    実インシデント (2026-08-27 ライブ監査 T08-7): 「これまでの会話に「横浜」は
    何回出てきましたか。」に「5回」と答えた (実際 4 回)。ツールを使わず数を
    断定していた。
    """
    if not term:
        return 0
    return sum(
        str(turn.get("content") or "").count(term)
        for turn in (_session_turns.get(session_id) or ())
    )


def clear_session_data(session_id: str) -> None:
    """セッション切替時にセッション固有データをクリーンアップ"""
    _session_started.pop(session_id, None)
    _session_turns.pop(session_id, None)
    _session_had_private.discard(session_id)


def _loaded_cartridge_ids(state: AppState) -> list[str]:
    """現在ロード中のカートリッジ ID を返す (degraded 時は空)。

    Level 0 経験に刻み、Level 2 のカートリッジ汚染フィルタ / 依存度メタ /
    get_cartridge_impact (自動ロールバック) が参照する。
    """
    mgr = getattr(state, "cartridge_manager", None)
    if mgr is None:
        return []
    return list(mgr.loaded)


def _existing_summary(mgr, session_id: str) -> str | None:
    """既に生成済みのセッション要約を引き継ぐ (未生成なら ``None``)。

    自動保存は毎ターン走るため、sleep-time の LLM 要約器が書いた要約を
    次ターンの保存で消さないよう索引から読み直す。
    """
    try:
        index = mgr._load_index()
    except Exception:
        return None
    entry = next(
        (e for e in index.sessions if e.session_id == session_id), None,
    )
    return entry.summary if entry is not None else None


def _save_session_to_history(
    state: AppState, session_id: str, mode: str,  # noqa: ARG001
) -> None:
    """蓄積した全ターンを HistoryManager で保存する

    レスポンス完了後に呼ばれ、会話履歴をディスクに永続化する。
    同一 session_id のファイルは上書きされるため冪等。
    WorkingMemory ではなく _session_turns を使用し、
    WM のエビクションで古いターンが失われる問題を回避する。
    """
    turns = _session_turns.get(session_id, [])
    if not turns:
        return

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
            and session_id in _session_had_private
        ):
            logger.info(
                "history save skipped (history_storage=skip, session had private turns): %s",
                session_id,
            )
            return

        mgr = get_history_manager()

        # 開始時刻を記録（初回のみ）
        if session_id not in _session_started:
            first_ts = turns[0].get("timestamp")
            if first_ts:
                _session_started[session_id] = datetime.fromtimestamp(
                    first_ts, tz=timezone.utc,
                ).isoformat()
            else:
                _session_started[session_id] = utc_now_dt().isoformat()

        started_at = _session_started[session_id]
        now_iso = utc_now_dt().isoformat()

        # ターンを履歴用フォーマットに変換
        history_turns = []
        for t in turns:
            entry = {"role": t["role"], "content": t["content"]}
            ts = t.get("timestamp")
            if ts:
                entry["timestamp"] = datetime.fromtimestamp(
                    ts, tz=timezone.utc,
                ).isoformat()
            history_turns.append(entry)

        instance_name = cfg.get("instance", {}).get("name", "evoref")
        # summary は書かない (None のまま残す)。
        #
        # 以前はユーザーの最初のメッセージを検索用 summary として毎ターン
        # 書き込んでいたが、sleep-time の LLM 要約器
        # (memory.sleep.summarize.summarize_unsummarized_sessions) は
        # ``summary is None`` のセッションだけを対象にするため、summary が
        # 一度も None にならず要約器が永久に発火しない状態になっていた
        # (2026-07-26 実測: 31 セッション全件が「最初のユーザ発話そのまま」で、
        # LLM 要約は 0 件)。さらに要約器が書いた要約も次ターンの自動保存で
        # 最初の発話へ上書きされる構造だった。
        #
        # 検索は index の ``search_text`` (summary + 全ターン結合) が担うので、
        # summary を空にしても search_history のヒット率は落ちない。一覧見出しは
        # index の ``first_user_preview`` (最初のユーザ発話) で代替される。
        # 既に要約が付いているセッションは上書きせず引き継ぐ。
        summary = _existing_summary(mgr, session_id)

        session = SessionData(
            session_id=session_id,
            started_at=started_at,
            ended_at=now_iso,
            mode=mode,
            modes_used=[mode],
            instance_name=instance_name,
            base_model=active_base_model_name(cfg),
            source="auto",
            turns=history_turns,
            turn_count=len(history_turns),
            summary=summary,
        )

        path = mgr.save_session(session)
        if path:
            logger.debug("Session saved to history: %s (%d turns)", session_id, len(history_turns))
    except Exception as e:
        logger.warning("Failed to save session to history: %s", e)


def drain_evicted_to_stm(
    wm: WorkingMemory, stm: ShortTermMemory, session_id: str,
) -> None:
    """WorkingMemory から押し出されたターンを ShortTermMemory に吸収

    直前のユーザー発言を逐語コピーしただけの assistant 応答は、記憶として
    保存しない。保存すると同じ問いで想起されて再生産され、繰り返し回数が
    増えていく自己増幅ループになる (実インシデント 2026-08-04 ライブ監査:
    「今日は何曜日ですか。」が 5 回繰り返され答えが出ない状態まで悪化。
    汚染ノートを除去したら 5/5 で解消した)。エコー部分を落として中身が
    残ればその中身だけを吸収し、何も残らなければ丸ごと捨てる。
    """
    _absorb_turns_to_stm(wm.drain_evicted(), stm, session_id, origin="evicted")


def snapshot_wm_to_stm(
    wm: WorkingMemory, stm: ShortTermMemory, session_id: str,
) -> None:
    """WorkingMemory の未転送ターンを **非破壊で** ShortTermMemory へ写す。

    f_02 §1.2 経路 (c)。sleep-time Full の直前に呼ばれ、押し出しが起きていない
    進行中セッションでも Step 8 抽出の入力を用意する。WM のターンは残るので
    会話 context は壊れない。エコー落としは押し出し経路と同じ規則を使う。
    """
    _absorb_turns_to_stm(
        wm.snapshot_unabsorbed(), stm, session_id, origin="snapshot",
    )


def _absorb_turns_to_stm(
    turns: list[dict], stm: ShortTermMemory, session_id: str, *, origin: str,
) -> None:
    """ターン列を STM へ吸収する共通処理 (押し出し / スナップショット共用)。

    直前のユーザー発言を逐語コピーしただけの assistant 応答は、記憶として
    保存しない。保存すると同じ問いで想起されて再生産され、繰り返し回数が
    増えていく自己増幅ループになる (実インシデント 2026-08-04 ライブ監査:
    「今日は何曜日ですか。」が 5 回繰り返され答えが出ない状態まで悪化。
    汚染ノートを除去したら 5/5 で解消した)。エコー部分を落として中身が
    残ればその中身だけを吸収し、何も残らなければ丸ごと捨てる。
    """
    last_user = ""
    dropped = 0
    for turn in turns:
        content = turn.get("content") or ""
        if turn.get("role") == "user" or turn.get("source") == "user":
            last_user = content
        elif last_user and content:
            if is_query_echo(content, last_user):
                dropped += 1
                continue
            cleaned = strip_echoed_query(content, last_user)
            if cleaned != content:
                turn = {**turn, "content": cleaned}
        stm.absorb(turn, session_id)
    if dropped:
        logger.info(
            "Dropped %d echo-only assistant turn(s) before STM absorb (session=%s)",
            dropped, session_id,
        )
    if turns:
        total_chars = sum(len(t.get("content", "")) for t in turns)
        logger.debug(
            "Absorbed %d %s turns to STM: total_chars=%d, session=%s",
            len(turns), origin, total_chars, session_id,
        )
        logger.info("Absorbed %d %s turns to STM", len(turns), origin)


def _schedule_sleep_time(state: AppState, user_query: str, private: bool) -> None:
    """sleep-time update をスケジュールする (record_* 3 経路の共通処理)。

    訂正ターンでは Full を **前倒し** する。ファクト抽出 (Step 8) と競合解決
    (Step 6B) は Full にしか無く (``memory.facts.trigger: idle_full_only``)、
    Light は Step 1-5.5 (埋め込み / タグ / スコア / eviction) だけ。そのため
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

            if restates_a_value(user_query):
                scheduler.request_full_soon("value_restated")
            elif states_single_valued_attribute(user_query):
                scheduler.request_full_soon("single_valued_attribute_stated")
        except Exception as exc:
            logger.warning(
                "correction-triggered full request skipped: %s", exc,
            )
    scheduler.on_response_sent()


def _turn_contradiction_inputs(
    state: AppState,
    messages: list[ChatMessage],
    action_blocked: bool | None = None,
) -> tuple[bool, dict[str, set[int]]]:
    """``turn_outcome`` の矛盾検出に渡す 2 つの入力を集める。

    どちらも「システムが既に知っていること」で、応答本文と突き合わせると
    真偽の推定なしに矛盾を検出できる (``_derive_turn_outcome`` 参照)。

    - ``action_blocked``: 状態を変える依頼なのに撃てるツールが無かったか。
      ``ToolCallJudge`` が当ターンの ``judge()`` 冒頭でリセットし、以後この
      ターンの間は保持する。deliberative が注記の要否を決めるのに読むのと
      同じ値をここでも読む。
    - ``measured_values``: ``[システム計測]`` として最後の user メッセージへ
      注入した実測値。注入したのはこのプロセスなので、プロンプトから読み戻す
      (新しい引数を 5 つの層へ通す代わりに、注入結果そのものを見る)。
    """
    # 当ターンの判定結果から渡された値を優先する。``ToolCallJudge`` は
    # プロセス唯一の共有インスタンスなので、後から属性を読むとチャットが
    # 2 本重なったときに他方の judge() がリセット済みで値が消える
    # (``ToolJudgement.action_blocked`` のコメント参照)。渡されない経路
    # (reactive 即応答等) のみ従来どおり判定器を見る。
    if action_blocked is None:
        judge = getattr(state, "tool_call_judge", None)
        action_blocked = bool(getattr(judge, "action_blocked", False))
    measured: dict[str, set[int]] = {}
    if messages:
        measured = extract_measured_values(str(messages[-1].get("content") or ""))
    return action_blocked, measured


#: 成果物として保持する最小の応答長 (文字)。履歴予算 (実測 1612 トークン
#: ≒ 日本語 2000 文字強) の半分を目安にする — 他のターンと合わさると
#: この程度から落ち始める。短い応答まで保持すると、直後の相槌で本物の
#: 成果物を上書きしてしまう。
ARTIFACT_MIN_CHARS = 1200


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
) -> None:
    """応答をメモリ・デバッグログ・経験バッファに記録する

    ``private=True`` の場合は WM/STM までの伝搬のみ行い
    会話履歴ディスク永続化と feedback collector への記録をスキップする。

    ``tool_command`` / ``tool_command_name`` / ``tool_command_success`` は
    run_command 実行ターンの learning メタで、assistant note に載せて
    sleep-time の executable_command_curator が参照する (それ以外は None)。
    """
    # メモリに応答を記録
    mem_sys = state.get_memory_system()
    if mem_sys and full_response:
        wm, stm, _ltm = mem_sys
        wm.add_turn(
            "assistant", _recorded_body(full_response),
            private=private, mode=mode, source="assistant",
            tool_command=tool_command,
            tool_command_name=tool_command_name,
            tool_command_success=tool_command_success,
            tool_command_source=tool_command_source,
            # 発火元のクエリを note に確定させる。curator が STM を走査して
            # 「直前で最も近い user note」から推測すると、当該ターンの user note が
            # 吸収されていない場合に別ターンのクエリと結び付く。
            tool_command_query=user_query if tool_command else None,
        )
        drain_evicted_to_stm(wm, stm, session_id)

    # 履歴予算に入らない長さの応答は成果物として保持する。
    # 実測 (2026-08-27 ライブ監査) の履歴予算は 1612 トークン。これを超える
    # 出力は次ターンで落ちるため、「いま書いたコードは何行ですか」に
    # **「1行です」** (実際は 26 行) と答えていた。
    _remember_if_artifact_sized(state, session_id, full_response, user_query, mode)

    # デバッグログ
    dl = state.debug_logger
    if dl:
        dl.log_request(tokens_generated, messages, full_response)

    # 経験バッファに記録 (Level 0) — private は学習対象外
    fc = state.feedback_collector
    if fc and full_response and not private:
        try:
            # 同一ターンの明確な失敗 = ツールをルーティングしたが失敗 → false_positive。
            tool_fp = tool_command is not None and tool_command_success is False
            prompt_tokens, cached_tokens = read_llama_prompt_tokens(state)
            blocked, measured = _turn_contradiction_inputs(
                state, messages, action_blocked,
            )
            fc.record(
                query=user_query, response=full_response, mode=mode,
                tool_routing_success=tool_routing_success,
                tool_routing_false_positive=tool_fp,
                cartridge_ids=_loaded_cartridge_ids(state),
                rag_used=rag_used, rag_top1_score=rag_top1_score,
                completion_tokens=tokens_generated,
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_tokens,
                action_blocked=blocked,
                measured_values=measured,
            )
        except Exception as e:
            # 経験記録の失敗でチャットを壊さない方針は維持するが、**握り潰さない**。
            # 実インシデント (2026-08-23): 引数を 1 つ足し忘れた NameError が
            # WARNING 1 行に化け、meta_cognitive / long_form の経験記録が静かに
            # 全滅していた (テストで検出)。traceback 付き ERROR なら気づける。
            logger.error(
                "FeedbackCollector.record failed: %s", e, exc_info=True,
            )

    # sleep-time update をスケジュール (訂正ターンは Full を前倒し)
    _schedule_sleep_time(state, user_query, private)

    # ターンを蓄積（WM エビクションに依存しない完全な履歴）
    _accumulate_turn(session_id, "user", user_query, private=private)
    if full_response:
        _accumulate_turn(session_id, "assistant", full_response, private=private)

    # 会話履歴をディスクに保存 (private なら蓄積されていないので no-op)
    if not private:
        _save_session_to_history(state, session_id, mode)


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
) -> None:
    """Meta-Cognitive 層の応答をメモリ・経験バッファに記録（クレジット付き）

    ``private=True`` の場合は WM/STM までの伝搬のみ
    """
    # メモリに応答を記録
    mem_sys = state.get_memory_system()
    if mem_sys and full_response:
        wm, stm, _ltm = mem_sys
        wm.add_turn(
            "assistant", _recorded_body(full_response),
            private=private, mode=mode, source="assistant",
        )
        drain_evicted_to_stm(wm, stm, session_id)

    # デバッグログ
    dl = state.debug_logger
    if dl:
        dl.log_request(tokens_generated, messages, full_response)

    # 経験バッファに記録 (Level 0) — ステップクレジット付き / private は対象外
    fc = state.feedback_collector
    if fc and full_response and not private:
        try:
            credits_dicts = [
                {"step_index": c.step_index, "action": c.action, "credit": c.credit}
                for c in step_credits
            ] if step_credits else []
            prompt_tokens, cached_tokens = read_llama_prompt_tokens(state)
            blocked, measured = _turn_contradiction_inputs(
                state, messages,
            )
            fc.record(
                query=user_query,
                response=full_response,
                mode=mode,
                action_blocked=blocked,
                measured_values=measured,
                agent_loops=agent_loops,
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
                tool_routing_success=tool_routing_success,
                tool_routing_false_positive=tool_routing_false_positive,
                cartridge_ids=_loaded_cartridge_ids(state),
                step_credits=credits_dicts,
                completion_tokens=tokens_generated,
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_tokens,
            )
        except Exception as e:
            logger.error(
                "FeedbackCollector.record failed (meta-cognitive): %s", e,
                exc_info=True,
            )

    # sleep-time update をスケジュール (訂正ターンは Full を前倒し)
    _schedule_sleep_time(state, user_query, private)

    # ターンを蓄積（WM エビクションに依存しない完全な履歴）
    _accumulate_turn(session_id, "user", user_query, private=private)
    if full_response:
        _accumulate_turn(session_id, "assistant", full_response, private=private)

    # 会話履歴をディスクに保存 (private は no-op)
    if not private:
        _save_session_to_history(state, session_id, mode)


def record_long_form_response(
    state: AppState, full_response: str, messages: list[ChatMessage],
    session_id: str, user_query: str, mode: str,
    tokens_generated: int, metrics: dict,
    *,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> None:
    """長文生成の応答をメモリ・経験バッファに記録

    ``private=True`` の場合は WM/STM までの伝搬のみ
    """
    # メモリに応答を記録
    mem_sys = state.get_memory_system()
    if mem_sys and full_response:
        wm, stm, _ltm = mem_sys
        wm.add_turn(
            "assistant", _recorded_body(full_response),
            private=private, mode=mode, source="assistant",
        )
        drain_evicted_to_stm(wm, stm, session_id)

    # 成果物として保持する。WM へ積んでも **次のターンには残らない** —
    # 実測 (2026-08-27 ライブ監査 T10) で 6696 文字の計画書を出した次のターンが
    # ``_trim_history: 5/5 turns kept, 812 estimated tokens (max=1612)`` で、
    # 履歴予算に入らず落ちていた。その結果「いまの計画書は何章?」に
    # 「履歴に含まれていないので、計画書のテキストを共有いただければ」と
    # 答えていた (ユーザーが 1 ターン前に受け取った本文を貼り直せ、という要求)。
    if full_response:
        remember_artifact(
            state, session_id,
            text=full_response, query=user_query, mode=mode,
        )

    # デバッグログ
    dl = state.debug_logger
    if dl:
        dl.log_request(tokens_generated, messages, full_response)

    # 経験バッファに記録 (Level 0) — 長文生成メトリクス付き / private は対象外
    fc = state.feedback_collector
    if fc and full_response and not private:
        try:
            units_completed = int(metrics.get("units_completed", 0) or 0)
            validation_errors = int(metrics.get("validation_errors", 0) or 0)
            long_form_success = judge_long_form_success(
                metrics, user_query, full_response,
            )
            prompt_tokens, cached_tokens = read_llama_prompt_tokens(state)

            blocked, measured = _turn_contradiction_inputs(
                state, messages,
            )
            fc.record(
                query=user_query,
                response=full_response,
                mode=mode,
                action_blocked=blocked,
                measured_values=measured,
                long_form_used=True,
                long_form_content_type=metrics.get("content_type"),
                long_form_strategy=metrics.get("strategy"),
                long_form_units_total=metrics.get("units_total", 0),
                long_form_units_completed=units_completed,
                long_form_validation_errors=validation_errors,
                long_form_budget_used_pct=metrics.get("budget_used_pct"),
                long_form_success=long_form_success,
                # 長文経路に入ったが 1 ユニットも生成できなかった、または要求
                # 成果物と content_type が矛盾 = 長文分類の明確な誤検出
                # → false_positive (パターン重み decay の対象)。units>0 で
                # validation_errors のみのケースは長文ルーティング自体は妥当なので除外。
                long_form_false_positive=(
                    units_completed == 0
                    or is_content_type_mismatch(metrics, user_query)
                ),
                cartridge_ids=_loaded_cartridge_ids(state),
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
                completion_tokens=tokens_generated,
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_tokens,
            )
        except Exception as e:
            logger.error(
                "FeedbackCollector.record failed (long-form): %s", e,
                exc_info=True,
            )

    # sleep-time update をスケジュール (訂正ターンは Full を前倒し)
    _schedule_sleep_time(state, user_query, private)

    # ターンを蓄積（WM エビクションに依存しない完全な履歴）
    _accumulate_turn(session_id, "user", user_query, private=private)
    if full_response:
        _accumulate_turn(session_id, "assistant", full_response, private=private)

    # 会話履歴をディスクに保存 (private は no-op)
    if not private:
        _save_session_to_history(state, session_id, mode)
