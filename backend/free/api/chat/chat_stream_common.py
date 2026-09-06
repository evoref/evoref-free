"""ストリーミング層の共通基盤

SSE フレームビルダー / セッション別キャンセルフラグ / 計測とアウトカム記録
など、reactive・meta_cognitive・long_form・deliberative・staged のどの層からも
使うものだけを置く。層固有の処理は各 ``chat_stream_*`` へ。
"""

from __future__ import annotations

import time

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator
from backend.app_state import AppState
from backend.aux_telemetry import aux_failure_signals, current_aux_failures
from backend.exceptions import EvorefError
from backend.free.agent.issue_ledger import record_current_issue
from backend.free.core.verifier_events import current_turn_outcome
from backend.free.api.chat.chat_constants import MAX_STEP_QUEUE_SIZE
from backend.free.api.chat.chat_recorder import (
    read_llama_prompt_tokens,
    record_response,
    tool_routing_signals,
)
from backend.free.api.chat.chat_service import make_token_info
from backend.free.api.chat.chat_types import ChatMessage, StepCallback
from backend.free.api.schemas import ChatResponse, TokenInfo
from backend.free.core.sse import SSEFrameBuilder
from backend.i18n_helper import msg
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer


logger = get_logger("api.chat.streaming")

# SSE フレームビルダー（モジュールレベルの共有インスタンス）
sse = SSEFrameBuilder()


def meta_tool_routing_success(resp) -> bool:
    """meta-cognitive 応答でツールが 1 件以上実行成功したか (tool_routing 正例)。

    規則は :func:`~backend.free.api.chat.chat_recorder.tool_routing_signals`
    (3 経路共通) に集約。meta は複数ツールを呼ぶため any (1 件でも成功なら
    誘導は妥当)。tool_calls 空 (= ツール未使用) は False。
    """
    if resp is None:
        return False
    ok, _ = tool_routing_signals(getattr(resp, "tool_calls", None))
    return ok


def meta_last_command_call(resp) -> dict:
    """meta-cognitive 応答で最後に実行された run_command 系ツールを recorder 向けに要約する。

    deliberative の ``tool_command*`` と同じ kwargs (``tool_command`` /
    ``tool_command_name`` / ``tool_command_success``) を返す。meta 経路の
    コマンド実行も STM ノートへ載せ、Step 8.6 の executable command 索引の
    学習対象にするため (未使用なら空 dict)。判定層は meta では固定でないので
    ``tool_command_source`` は付けない。
    """
    for tc in reversed(getattr(resp, "tool_calls", None) or []):
        name = tc.get("tool", "")
        if name not in ("run_command", "run_command_readonly"):
            continue
        command = (tc.get("args") or {}).get("command")
        if not command:
            continue
        return {
            "tool_command": str(command),
            "tool_command_name": name,
            "tool_command_success": bool(tc.get("success")),
        }
    return {}


def meta_tool_routing_false_positive(resp) -> bool:
    """meta-cognitive 応答でツールを呼んだが全て失敗したか (tool_routing 誤検出)。

    規則は :func:`~backend.free.api.chat.chat_recorder.tool_routing_signals`
    (3 経路共通) に集約。tool_calls 空 (未使用) は False。
    """
    if resp is None:
        return False
    _, fp = tool_routing_signals(getattr(resp, "tool_calls", None))
    return fp


def _record_failed_generation(
    state: AppState,
    *,
    query: str,
    messages: list[ChatMessage],
    session_id: str,
    mode: str,
    private: bool,
    agent_layer: str,
    tokens_generated: int = 0,
) -> None:
    """error フレームで終わったターンを ``response=""`` の失敗経験として記録する。

    ストリーム層の ``except Exception`` から呼ぶ。以前は例外経路では
    ``record_*`` が一切走らず、失敗ターンは経験バッファに 1 件も無かった
    (2026-09-02 監査 R-A4)。メモリ / 履歴の帳簿 (user 発話の蓄積、押し出し
    済みターンの STM 転送、sleep-time スケジュール) も同時に付く。記録の
    失敗で error フレームの後始末を壊さないよう例外は握って ERROR ログに出す。
    """
    try:
        record_response(
            state, "", messages, session_id, query, mode, tokens_generated,
            private=private, generation_failed=True,
        )
    except Exception:
        logger.error(
            "%s: failed to record the errored turn as an experience",
            agent_layer, exc_info=True,
        )


def rag_signals_from_chunks(
    scored: list[tuple[str, float, str]] | None,
    raw_top_score: float | None = None,
) -> tuple[bool, float | None]:
    """``scored_chunks`` から Level 0 経験記録用の ``(rag_used, rag_top1_score)`` を導出。

    ``scored_chunks`` は ``(chunk_id, score, content)`` の salience 降順リスト。
    空 / None なら ``(False, None)`` (RAG 未使用)。

    ``raw_top_score`` は ``SearchResult.top_raw_score`` (採用チャンクの **生スコア**
    最大値、cosine スケール)。``scored`` 側のスコアは ``rag.score_normalization``
    適用後で、``minmax`` では先頭が定義上 1.0 に固定されるため、記録される
    ``rag_top1_score`` が観測値として死ぬ (実機 2026-08-13: RAG 使用 7 ターン全てが
    厳密に 1.0、embed_instruction の初期集団 5 候補も全て fitness 1.0000)。
    渡された場合はそちらを採用する。``None`` (検索結果を持ち回れない経路 / 旧
    呼出) は従来どおり正規化スコアへフォールバックする。
    """
    if not scored:
        return False, None
    if raw_top_score is not None:
        return True, raw_top_score
    return True, scored[0][1]


def _emit_timing(
    state: AppState, timer: StageTimer | None,
    agent_layer: str, tokens_generated: int, mode: str = "",
) -> None:
    """StageTimer の計測結果をデバッグログに出力し、直近メトリクスを更新する"""
    if timer is None:
        return
    timing = timer.to_dict()

    # デバッグオーバーレイ用に直近メトリクスを AppState に保存
    from backend.app_state import LastRequestMetrics
    ttft_ms = timing.get("llm_first_token_ms")
    llm_total_ms = timing.get("llm_total_ms")
    tok_per_sec: float | None = None
    if llm_total_ms and llm_total_ms > 0 and tokens_generated > 0:
        tok_per_sec = round(tokens_generated / (llm_total_ms / 1000), 1)
    state.last_request_metrics = LastRequestMetrics(
        ttft_ms=ttft_ms,
        tok_per_sec=tok_per_sec,
        updated_at=time.monotonic(),
    )

    # llama-server の timings から接頭辞 KV キャッシュの効きを timing へ畳み込む。
    # prompt_n = 再評価したトークン / cache_n = 再利用できたトークン。
    # これが requests.jsonl に無いと、キャッシュの効きは llama-base.stderr.log の
    # 行をチャットターンへ手で突き合わせるしかなく、aux と取り違えやすい。
    total, cache_n = read_llama_prompt_tokens(state)
    if total is not None and cache_n is not None:
        timing["prompt_n"] = total - cache_n
        timing["cache_n"] = cache_n
        if total > 0:
            timing["cache_hit_pct"] = round(100.0 * cache_n / total, 1)

    dl = state.debug_logger
    if dl is None:
        return
    if timing:
        dl.log_request_timing(
            timing, agent_layer=agent_layer,
            tokens_generated=tokens_generated, mode=mode,
        )


def _capture_stream_outcome(token_stream: Any, state: Any) -> None:
    """``TokenStream.outcome`` から切断メタを ``state`` へ吸い上げる。

    ``state`` は ``truncated`` / ``truncated_tokens`` / ``truncated_max_tokens``
    を持つ層別ストリーム状態 (deliberative / long_form)。``outcome`` を持たない
    イテレータ (テストの mock / 中継 generator / orchestrator) は素通しする —
    切断が分からないだけで、本文は従来どおり流れる。

    以前は deliberative だけがこれを持ち、他の層では ``finish_reason=length``
    が SSE に一切出なかった。再生成 (0 トークン再試行 / 制約修復) が再度通る
    場合、切断の有無は **最後に画面へ出した生成** で上書きする。
    """
    outcome = getattr(token_stream, "outcome", None)
    if outcome is None:
        return
    state.truncated = bool(getattr(outcome, "truncated", False))
    state.truncated_tokens = int(getattr(outcome, "tokens_generated", 0) or 0)
    state.truncated_max_tokens = getattr(outcome, "max_tokens", None)


async def _close_token_stream(token_stream: Any) -> None:
    """キャンセルで早期 break したストリームの基底 generator を閉じる。

    ``TokenStream.aclose`` は存在していたが誰も呼んでいなかった。閉じないと
    ``_generate_stream`` の ``async with httpx.AsyncClient`` が GC まで生き、
    llama-server 側の生成は接続が切れるまで続く (キャンセル後もスロットを
    占有する)。閉じる際の例外は握る — キャンセルの後始末で応答を壊さない。
    """
    aclose = getattr(token_stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:  # pragma: no cover - 後始末の失敗は本流へ出さない
        logger.debug("token stream aclose failed", exc_info=True)


async def _emit_stream_error(
    state: AppState,
    exc: BaseException,
    *,
    timer: StageTimer | None,
    agent_layer: str,
    mode: str,
    tokens_generated: int = 0,
) -> AsyncIterator[str]:
    """ストリーム層の ``except Exception`` 末尾の共通処理 (計測停止 → error → done)。

    4 層 (deliberative / reactive 軽量 / meta_cognitive / long_form) が同じ
    ブロックを写経し、ログの書式 (``%s`` / ``%r``) と timing の tokens が
    揺れていた。型付き例外 (:class:`EvorefError`) は ``code`` 付きフレームで
    送り、ユーザー向け文言は i18n キーが解決できればそれを、できなければ
    例外メッセージ (英語) を使う。
    """
    logger.error("%s stream error: %r", agent_layer, exc)
    if timer:
        timer.stop("llm_total_ms")
    _emit_timing(state, timer, agent_layer, tokens_generated, mode=mode)
    if isinstance(exc, EvorefError):
        text = msg(exc.i18n_key, **{k: v for k, v in exc.context.items() if isinstance(v, str | int | float)})
        if text == exc.i18n_key or "{" in text:
            text = str(exc) or exc.i18n_key
        yield sse.error_with_code(exc.code, text)
    else:
        yield sse.error(str(exc))
    yield sse.done()


# ---------------------------------------------------------------------------
# セッション別キャンセルフラグ（chat.py の cancel エンドポイントからも参照）
# ---------------------------------------------------------------------------
_cancel_flags: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# cancel_scope — finally ブロックのクリーンアップを共通化
# ---------------------------------------------------------------------------

@asynccontextmanager
async def cancel_scope(session_id: str):
    """キャンセルフラグのスコープ管理

    ストリーミング関数の try/finally パターンを統一する。
    """
    _cancel_flags[session_id] = False
    try:
        yield
    finally:
        _cancel_flags.pop(session_id, None)


def _make_step_queue_callback(
    step_queue: list[dict],
) -> StepCallback:
    """`step_queue` に要素を追加する on_step コールバックを構築する。

    `MAX_STEP_QUEUE_SIZE` を超えたら古い要素を破棄する (BUG-10 対策)。
    """

    def _on_step(step_data: dict) -> None:
        if len(step_queue) >= MAX_STEP_QUEUE_SIZE:
            logger.debug("Step queue overflow, discarding oldest event")
            step_queue.pop(0)
        step_queue.append(step_data)

    return _on_step


# ---------------------------------------------------------------------------
# ターンの結末記録 / 同期応答の組み立て
# ---------------------------------------------------------------------------

def _log_chat_outcome(
    state: AppState,
    *,
    started_at: float,
    success: bool,
    tokens_out: int,
    signals: dict,
    cancelled: bool = False,
) -> None:
    """チャット 1 ターンの結末を outcome.jsonl へ記録する。

    ``kind`` と経過時間の取り方を層ごとに持つと、outcome JSONL を層をまたいで
    突き合わせられなくなる。5 つの層 (meta_cognitive / long_form /
    deliberative / reactive 軽量パスのストリーム・同期) が同じ形を書き写して
    いたのをここへ集約する。層ごとの違いは ``signals`` と ``success`` だけ。

    ``cancelled`` は「エラーではないのに成功で終わっていない」= クライアント
    切断の印。evolve の fitness がユーザーキャンセルを失敗として計上しないよう
    区別できるようにする。

    ``debug_logger`` 未配線 (evolve レベル以外) では no-op。
    """
    dl = getattr(state, "debug_logger", None)
    if dl is None:
        return
    if cancelled:
        signals = {**signals, "cancelled": True}
    # ターン中に縮退した補助判定を結末へ持ち上げる。``success`` は「SSE を
    # 最後まで届けられたか」= 配送の成否なので、**中身が縮退したターンも
    # success=true になる**。この信号が無いと、記憶想起もツール判定も落ちた
    # ターンと健全なターンが事後に区別できない (2026-09-03 監査:
    # chat_response 105/105 が success=true)。
    failures = current_aux_failures()
    signals = {**signals, **aux_failure_signals(failures)}
    if failures:
        signals["degraded"] = True
        # 自己申告 (「今日の回答で自信が持てなかったものは?」) の材料にもする。
        # 結末 JSONL にだけ残しても、モデルはそれを読めない。
        record_current_issue(
            "aux_degraded",
            ", ".join(sorted({f["purpose"] for f in failures})),
        )
    # 規則台帳の計数 (f_03 §3.5.1): このターンで発火した検証器を、対応する
    # 規則の harmful に、発火しなかった規則の helpful に写す。これが
    # 「削ってよい規則」の唯一の根拠。計数の失敗で結末記録を止めない。
    signals = {**signals, **_account_rule_outcomes(state)}
    # 経験記録が導出した成否 (本文の決定論的な破綻) を結末へ反映する。
    # ``success`` が配送の成否だけだと、計算の破綻や自己矛盾のターンが
    # 100/100 success で evolve の fitness に入る (2026-09-05 監査 F-11)。
    turn_outcome, outcome_reason = current_turn_outcome()
    if turn_outcome:
        signals["turn_outcome"] = turn_outcome
        if outcome_reason:
            signals["turn_outcome_reason"] = outcome_reason
        if turn_outcome == "failed" and not cancelled:
            success = False
    dl.log_outcome(
        kind="chat_response",
        success=success,
        duration_ms=(time.monotonic() - started_at) * 1000,
        tokens_out=tokens_out,
        quality_signals=signals,
    )


def _account_rule_outcomes(state: AppState) -> dict:
    """検証器の発火 → 規則 id の計数へ (純粋でない: 台帳を更新し保存する)。"""
    try:
        from backend.free.agent.prompt_ledger import record_rule_outcome, rule_ids_for_verifiers
        from backend.free.core.verifier_events import current_verifier_hits, current_verifier_mode
        from backend.utils import utc_now

        mode = current_verifier_mode()
        pm = getattr(state, "prompt_manager", None)
        if mode is None or pm is None or not hasattr(pm, "get_ledger"):
            return {}
        hits = set(current_verifier_hits())
        ledger = pm.get_ledger(mode)
        violated = rule_ids_for_verifiers(ledger, hits)
        record_rule_outcome(ledger, violated, fired_at=utc_now())
        pm.save_ledger_counts(mode)
        out: dict = {"verifier_hits": sorted(hits)}
        if violated:
            out["rule_violations"] = sorted(violated)
        return out
    except Exception:  # noqa: BLE001 - 計数は結末記録の付随物
        logger.debug("rule outcome accounting skipped", exc_info=True)
        return {}


def _sync_chat_response(
    state: AppState,
    timer: StageTimer | None,
    *,
    agent_layer: str,
    text: str,
    tokens: int,
    messages: list[ChatMessage],
    session_id: str,
    instance_name: str,
    context_size: int,
    mode: str = "",
) -> ChatResponse:
    """同期応答の共通末尾: 計測の記録 → token_info の算出 → ``ChatResponse``。

    4 つの ``sync_*`` が同じ 3 手順を書き写していた。``agent_layer`` は timing の
    ラベルと応答フィールドの双方に使うので、1 箇所で受けて食い違わせない。
    """
    _emit_timing(state, timer, agent_layer, tokens, mode=mode)
    token_info = make_token_info(messages, tokens, context_size, instance_name)
    return ChatResponse(
        response=text,
        token_info=TokenInfo(**token_info),
        session_id=session_id,
        agent_layer=agent_layer,
    )
