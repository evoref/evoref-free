"""ストリーミング層の共通基盤

SSE フレームビルダー / セッション別キャンセルフラグ / 計測とアウトカム記録
など、reactive・meta_cognitive・long_form・deliberative・staged のどの層からも
使うものだけを置く。層固有の処理は各 ``chat_stream_*`` へ。
"""

from __future__ import annotations

import asyncio
import json
import time

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, AsyncIterator
from fastapi import HTTPException

from backend.app_state import AppState
from backend.trace_context import get_trace_id
from backend.aux_telemetry import aux_failure_signals, current_aux_failures
from backend.exceptions import EvorefError
from backend.free.agent.issue_ledger import record_current_issue
from backend.free.core.verifier_events import (
    current_grounding,
    current_rag_signals,
    current_tool_uses,
    current_turn_outcome,
)
from backend.free.api.chat.chat_constants import (
    DEFAULT_KEEPALIVE_INTERVAL_SEC,
    MAX_STEP_QUEUE_SIZE,
)
from backend.free.api.chat.chat_recorder import (
    read_llama_prompt_tokens,
    record_response,
    tool_routing_signals,
)
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

#: トークン待ちの間にキャンセル要求を見に行く間隔 (秒)。keepalive 間隔とは別。
_CANCEL_POLL_SEC = 1.0


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
    最大値、cosine スケール)。``scored`` 側のスコアは順位式 (c_16 §7.2)
    適用後の salience で cosine スケールではない (旧 ``rag.score_normalization`` の
    minmax 時代は先頭が定義上 1.0 に固定され、記録される ``rag_top1_score`` が
    観測値として死んでいた — 実機 2026-08-13: RAG 使用 7 ターン全てが厳密に 1.0)。
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


async def iter_tokens_with_keepalive(
    token_stream: Any,
    session_id: str,
    *,
    interval: float = DEFAULT_KEEPALIVE_INTERVAL_SEC,
) -> AsyncIterator[str | None]:
    """トークン列を ``interval`` 秒の無音ごとに ``None`` を挟んで yield する。

    ``None`` は「keepalive を送る番」の合図で、呼出側が ``sse.keepalive()`` を
    出す (トークンの有無に関わらずフロントの chunk timeout を防ぐ)。
    セッションのキャンセルフラグが立てば止まる。

    deliberative / long_form が同じ ``asyncio.wait`` ループを写経していたのを
    1 本にする。ループを抜けるときは **必ず** 基底ストリームを閉じる — 従来は
    キャンセルフラグの経路でしか閉じておらず、クライアント切断
    (``yield`` に CancelledError が届く) では ``__anext__`` を待つタスクが
    孤児になり、httpx のストリームは GC まで開いたまま llama-server 側の
    生成が続いていた。閉じる順序も直す: 先に ``__anext__`` の完了を待ってから
    ``aclose`` を呼ぶ (走行中の async generator へ ``aclose`` を投げると
    ``RuntimeError: aclose(): asynchronous generator is already running`` で
    握り潰されていた)。
    """
    aiter = token_stream.__aiter__()
    pending: asyncio.Task | None = None
    # キャンセルの判定は keepalive 間隔 (15 秒) より細かく回す。long_form の
    # 計画段階のようにトークンが 60〜100 秒出ない区間では、フラグを見るのが
    # この待ちのタイムアウト時だけなので、キャンセルの反映が最大 15 秒遅れて
    # いた (2026-09-17 ライブ監査)。keepalive フレーム自体は従来どおり
    # ``interval`` 秒の無音ごとに 1 回だけ出す。
    poll = min(interval, _CANCEL_POLL_SEC)
    silent_since = time.monotonic()
    try:
        while True:
            if cancel_requested(session_id):
                return
            if pending is None:
                pending = asyncio.create_task(aiter.__anext__())
            # ``asyncio.wait`` はタイムアウト時にタスクをキャンセルしないため、
            # keepalive 送出後も同じ ``__anext__()`` 呼び出しを継続できる。
            done, _ = await asyncio.wait({pending}, timeout=poll)
            if pending not in done:
                now = time.monotonic()
                if now - silent_since >= interval:
                    silent_since = now
                    yield None
                continue
            try:
                token = pending.result()
            except StopAsyncIteration:
                pending = None
                return
            pending = None
            silent_since = time.monotonic()
            yield token
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await _close_token_stream(token_stream)


def _finish_stream_outcome(
    state: AppState,
    session_id: str,
    *,
    started_at: float,
    completed: bool,
    errored: bool,
    tokens_out: int,
    signals: dict,
    success: bool | None = None,
) -> None:
    """ストリーム層の ``finally`` から結末を記録する (4 層共通)。

    ユーザーキャンセル (``/api/chat/cancel`` のフラグ) は途中まで流した本文で
    終端処理まで到達するので ``outcome_success`` が True になる。これをそのまま
    書くと **キャンセルした部分応答が成功として evolve の fitness に入る**
    (``record_response`` 側は ``cancelled=True`` で経験を落としているのに、結末
    JSONL だけ success だった)。フラグは ``cancel_scope`` の後始末より先に
    読めるので、ここで成否へ畳む。クライアント切断 (例外でも完走でもない) も
    ``cancelled`` として区別する。

    ``completed`` は「ストリームを終端まで届けたか」、``success`` は層が持つ
    品質判定 (long_form の validation_errors / meta の failed_tasks)。2 つを
    1 つの真偽に畳んで渡すと、完走したが品質で落ちたターンが「完走していない
    = 切断」と誤分類される (2026-09-17 ライブ監査: validation_errors=1 の
    long_form が cancelled=true になった)。
    """
    user_cancelled = cancel_requested(session_id)
    quality_ok = True if success is None else bool(success)
    _log_chat_outcome(
        state,
        started_at=started_at,
        success=completed and quality_ok and not user_cancelled,
        tokens_out=tokens_out,
        signals=signals,
        cancelled=user_cancelled or (not completed and not errored),
    )


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
    logger.error("%s stream error: %r", agent_layer, exc, exc_info=True)
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

#: セッション → そのセッションで進行中のリクエストのキャンセルキー。
#: ``/api/chat/cancel`` が ``request_id`` を伴わないとき (旧クライアント /
#: CLI) はセッションの全リクエストを止める。
_session_requests: dict[str, set[str]] = {}

#: このターンのキャンセルキー (``cancel_scope`` が置く)。リクエスト
#: (trace_id) 単位で、trace の無い呼出 (テスト / 直接呼出) ではセッション id。
_cancel_key_var: ContextVar[str | None] = ContextVar("chat_cancel_key", default=None)
#: 切断後も制作を続ける (detached) タスクが握っているキャンセルキー。
#: ``cancel_scope`` の finally はここにある鍵を pop しない (タスクの完了で解放)。
_retained_cancel_keys: set[str] = set()


def retain_cancel_scope(session_id: str, task: "asyncio.Task") -> None:
    """detached タスクのためにこのターンのキャンセルキーを scope 終了後も残す。

    クライアント切断で SSE の generator が閉じると ``cancel_scope`` が鍵を pop し、
    走り続ける制作を ``/api/chat/cancel`` で止められなくなる (f_10 §3、Phase 3b)。
    鍵はタスクの完了コールバックで解放する。
    """
    key = _cancel_key(session_id)
    _retained_cancel_keys.add(key)

    def _release(_task: "asyncio.Task") -> None:
        _retained_cancel_keys.discard(key)
        _cancel_flags.pop(key, None)
        keys = _session_requests.get(session_id)
        if keys is not None:
            keys.discard(key)
            if not keys:
                _session_requests.pop(session_id, None)

    task.add_done_callback(_release)


def _cancel_key(session_id: str) -> str:
    return _cancel_key_var.get() or session_id


def current_request_id() -> str | None:
    """このターンの request_id (= trace_id)。フロントがキャンセルに使う。"""
    return get_trace_id() or None


def cancel_requested(session_id: str) -> bool:
    """このターンにキャンセルが要求されたか。

    リクエスト単位のキーを見る。``session_id`` 直指定のフラグも読む —
    テスト / 直接呼出 (trace 無し) はセッション id をキーにするため。
    """
    key = _cancel_key(session_id)
    return bool(_cancel_flags.get(key)) or (
        key != session_id and bool(_cancel_flags.get(session_id))
    )


def request_cancel(session_id: str, request_id: str | None = None) -> bool:
    """キャンセルを要求する (``/api/chat/cancel``)。止められたものがあれば True。

    ``request_id`` があればそのリクエストだけ、無ければセッションの進行中
    リクエスト全部。同一セッションで 2 本走っているとき (2 タブ / staged →
    long_form の入れ子) にセッション id 1 本のフラグでは互いに干渉していた。
    """
    if request_id:
        # 明示された id が既に終わっている / 別セッションのものなら何もしない
        # (セッション全体へ倒すと、隣で走っている別リクエストを巻き込む)。
        if request_id in _cancel_flags:
            _cancel_flags[request_id] = True
            return True
        return False
    keys = _session_requests.get(session_id) or set()
    hit = False
    for key in list(keys):
        if key in _cancel_flags:
            _cancel_flags[key] = True
            hit = True
    if not hit and session_id in _cancel_flags:
        _cancel_flags[session_id] = True
        hit = True
    return hit


@asynccontextmanager
async def cancel_scope(session_id: str):
    """キャンセルフラグのスコープ管理

    ストリーミング関数の try/finally パターンを統一する。キーはリクエスト
    (trace_id) 単位。trace の無い呼出はセッション id をキーにする (従来互換)。
    """
    key = get_trace_id() or session_id
    # 入れ子 (staged → long_form フォールバック) では外側のスコープが所有者。
    # 内側で False に戻したり pop したりすると、その間に届いたキャンセルが消える。
    nested = key in _cancel_flags
    token = _cancel_key_var.set(key)
    if not nested:
        _cancel_flags[key] = False
        _session_requests.setdefault(session_id, set()).add(key)
    try:
        yield
    finally:
        if not nested and key not in _retained_cancel_keys:
            _cancel_flags.pop(key, None)
            keys = _session_requests.get(session_id)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    _session_requests.pop(session_id, None)
        _cancel_key_var.reset(token)


def agent_layer_frame(layer: str) -> str:
    """``agent_layer`` フレーム (このターンの ``request_id`` 付き)。"""
    return sse.agent_layer(layer, request_id=current_request_id())


def create_run_frame(run_id: str, session_id: str) -> str:
    """``create_run`` フレーム (staged create run の再接続用 run_id 通知、f_10 §7)。"""
    return sse.create_run(run_id, session_id)


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
    # RAG の便益 (rag_used / rag_abstained / rag_cited、経験記録が導出)。
    signals = {**signals, **current_rag_signals()}
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
    # 根拠台帳 (f_04 §2.2): 実行したツールと接地の疑義を結末へ写す。未判定は載せない。
    tool_uses = current_tool_uses()
    if tool_uses:
        signals["tool_uses"] = [u["tool"] for u in tool_uses]
    unexplained, issues, date_math = current_grounding()
    if unexplained:
        signals["unexplained_numbers"] = unexplained
    if issues:
        signals["expression_issues"] = issues
    if date_math:
        signals["unexplained_date_math"] = True
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


async def collect_chat_response(
    frames: AsyncIterator[str], *, session_id: str,
) -> ChatResponse:
    """SSE フレーム列を飲み干して非ストリーミング応答 (``ChatResponse``) に畳む。

    非ストリーミング API (``stream=False``) の実装はこれ 1 つ。以前は層ごとに
    ``sync_*`` を別実装しており、記録 / 結末 / 継続解除 / 失敗経験の有無が層に
    よって揺れていた。ストリーム実装が唯一の経路になるので、非対称は構造的に
    起きない。

    写像規則:

    - ``token`` の連結が本文。本文が空で ``task_result`` があれば最後の
      ``detail`` を本文にする (long_form のファイル出力は本文を流さず
      書込み結果だけを step で出す)
    - ``editor_code`` (partial でない) はコードフェンスとして本文末尾に畳む
      (非ストリームにはエディタチャネルが無い)
    - ``error`` は 503 (``HTTPException``) に写す。ストリーム側で結末と
      失敗経験は記録済み
    - ``token_info`` / ``agent_layer`` はそのまま応答へ
    - ``template_hint`` はそのまま応答へ (無ければ ``None``、c_17 §3.8)
    """
    text_parts: list[str] = []
    token_info: dict | None = None
    agent_layer = "reactive"
    error: str | None = None
    last_result: str | None = None
    editor_blocks: list[str] = []
    template_hint: dict | None = None
    async for frame in frames:
        if not frame.startswith("data: "):
            continue  # keepalive コメント
        body = frame[len("data: "):].strip()
        if body == "[DONE]":
            continue
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if "token" in obj:
            text_parts.append(str(obj["token"]))
        elif "token_info" in obj and isinstance(obj["token_info"], dict):
            token_info = obj["token_info"]
        elif "agent_layer" in obj:
            agent_layer = str(obj["agent_layer"])
        elif "error" in obj:
            err = obj["error"]
            if isinstance(err, dict):
                error = str(err.get("message") or err.get("code") or err)
            else:
                error = str(err)
        elif "step" in obj and isinstance(obj["step"], dict):
            step = obj["step"]
            if step.get("type") == "task_result" and step.get("detail"):
                last_result = str(step["detail"])
        elif "editor_code" in obj and isinstance(obj["editor_code"], dict):
            ec = obj["editor_code"]
            if not ec.get("partial"):
                lang = ec.get("language") or ""
                editor_blocks.append(
                    "```" + lang + "\n" + str(ec.get("content") or "") + "\n```",
                )
        elif "template_hint" in obj and isinstance(obj["template_hint"], dict):
            template_hint = obj["template_hint"]
    if error is not None:
        raise HTTPException(status_code=503, detail=error)
    text = "".join(text_parts)
    if not text.strip() and last_result:
        text = last_result
    if editor_blocks:
        blocks = "\n\n".join(editor_blocks)
        text = (text + "\n\n" + blocks) if text else blocks
    info = token_info or {"used": 0, "limit": 0, "pct": 0, "instance_name": ""}
    return ChatResponse(
        response=text,
        token_info=TokenInfo(**info),
        session_id=session_id,
        agent_layer=agent_layer,
        template_hint=template_hint,
    )
