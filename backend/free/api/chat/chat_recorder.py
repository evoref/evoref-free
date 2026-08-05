"""レスポンス記録（メモリ・デバッグログ・フィードバック・履歴）"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from backend.app_state import AppState
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.history.history_manager import (
    SessionData,
    active_base_model_name,
    get_history_manager,
)
from backend.log_config import get_logger
from backend.utils import utc_now_dt

if TYPE_CHECKING:
    from backend.free.memory.stores.working import WorkingMemory
    from backend.free.memory.stores.short_term import ShortTermMemory

logger = get_logger("api.chat.recorder")

# 文書系の出力先拡張子。これらへの出力依頼で content_type=code が返るのは
# ルーティング誤り (long_form_success の判定材料)。
_DOC_TARGET_EXT_RE = re.compile(r"\.(?:md|txt|csv)\b", re.IGNORECASE)

def is_content_type_mismatch(metrics: dict, user_query: str) -> bool:
    """文書拡張子への出力依頼なのに ``content_type=code`` を返したか。

    長文ルーティング自体の誤検出 (= ``long_form_false_positive``) の判定材料。
    """
    return (
        bool(_DOC_TARGET_EXT_RE.search(user_query))
        and str(metrics.get("content_type") or "") == "code"
    )


def judge_long_form_success(metrics: dict, user_query: str) -> bool:
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


def _attach_rag_judge_answer(
    state: AppState, user_query: str, full_response: str,
) -> None:
    """RAG 判定イベントへ、確定した応答本文を紐付ける。

    necessity/quality の assist 判定はターンの先頭で走るため、記録時点では
    応答本文が存在しない。ここで結びつけておかないと sleep-time のキュレータが
    STM を引き直すことになり、light サイクルの eviction 済みターンでは答えが
    見つからず学習信号が落ちる (2026-08-01 プロファイリング: 生成 181 件に対し
    world_fact 化は 3 件)。

    全ての ``record_*_response`` から呼ぶ。網羅は
    ``test_chat_recorder.py::TestRagJudgeAnswerAttachment`` が静的に強制する。
    """
    log = getattr(state, "rag_judge_assist_log", None)
    if log is None or not full_response:
        return
    try:
        log.attach_answer(user_query, full_response)
    except Exception:
        # 学習用の付随処理。応答パスを壊さない。
        logger.debug("rag judge answer attachment failed", exc_info=True)


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
    """WorkingMemory から押し出されたターンを ShortTermMemory に吸収"""
    evicted = wm.drain_evicted()
    for turn in evicted:
        stm.absorb(turn, session_id)
    if evicted:
        total_chars = sum(len(t.get("content", "")) for t in evicted)
        logger.debug(
            "Drained %d turns to STM: total_chars=%d, session=%s",
            len(evicted), total_chars, session_id,
        )
        logger.info("Drained %d evicted turns to STM", len(evicted))


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
) -> None:
    """応答をメモリ・デバッグログ・経験バッファに記録する

    ``private=True`` の場合は WM/STM までの伝搬のみ行い
    会話履歴ディスク永続化と feedback collector への記録をスキップする。

    ``tool_command`` / ``tool_command_name`` / ``tool_command_success`` は
    run_command 実行ターンの learning メタで、assistant note に載せて
    sleep-time の executable_command_curator が参照する (それ以外は None)。
    """
    _attach_rag_judge_answer(state, user_query, full_response)

    # メモリに応答を記録
    mem_sys = state.get_memory_system()
    if mem_sys and full_response:
        wm, stm, _ltm = mem_sys
        wm.add_turn(
            "assistant", full_response,
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
            fc.record(
                query=user_query, response=full_response, mode=mode,
                tool_routing_success=tool_routing_success,
                tool_routing_false_positive=tool_fp,
                cartridge_ids=_loaded_cartridge_ids(state),
                rag_used=rag_used, rag_top1_score=rag_top1_score,
            )
        except Exception as e:
            logger.warning("FeedbackCollector.record failed: %s", e)

    # sleep-time update をスケジュール
    scheduler = state.sleep_scheduler
    if scheduler:
        scheduler.on_response_sent()

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
    _attach_rag_judge_answer(state, user_query, full_response)

    # メモリに応答を記録
    mem_sys = state.get_memory_system()
    if mem_sys and full_response:
        wm, stm, _ltm = mem_sys
        wm.add_turn(
            "assistant", full_response,
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
            fc.record(
                query=user_query,
                response=full_response,
                mode=mode,
                agent_loops=agent_loops,
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
                tool_routing_success=tool_routing_success,
                tool_routing_false_positive=tool_routing_false_positive,
                cartridge_ids=_loaded_cartridge_ids(state),
                step_credits=credits_dicts,
            )
        except Exception as e:
            logger.warning("FeedbackCollector.record failed (meta-cognitive): %s", e)

    # sleep-time update をスケジュール
    scheduler = state.sleep_scheduler
    if scheduler:
        scheduler.on_response_sent()

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
    _attach_rag_judge_answer(state, user_query, full_response)

    # メモリに応答を記録
    mem_sys = state.get_memory_system()
    if mem_sys and full_response:
        wm, stm, _ltm = mem_sys
        wm.add_turn(
            "assistant", full_response,
            private=private, mode=mode, source="assistant",
        )
        drain_evicted_to_stm(wm, stm, session_id)

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
            long_form_success = judge_long_form_success(metrics, user_query)

            fc.record(
                query=user_query,
                response=full_response,
                mode=mode,
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
            )
        except Exception as e:
            logger.warning("FeedbackCollector.record failed (long-form): %s", e)

    # sleep-time update をスケジュール
    scheduler = state.sleep_scheduler
    if scheduler:
        scheduler.on_response_sent()

    # ターンを蓄積（WM エビクションに依存しない完全な履歴）
    _accumulate_turn(session_id, "user", user_query, private=private)
    if full_response:
        _accumulate_turn(session_id, "assistant", full_response, private=private)

    # 会話履歴をディスクに保存 (private は no-op)
    if not private:
        _save_session_to_history(state, session_id, mode)
