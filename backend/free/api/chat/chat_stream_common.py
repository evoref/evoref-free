"""ストリーミング層の共通基盤

SSE フレームビルダー / セッション別キャンセルフラグ / 計測とアウトカム記録
など、reactive・meta_cognitive・long_form・deliberative・staged のどの層からも
使うものだけを置く。層固有の処理は各 ``chat_stream_*`` へ。
"""

from __future__ import annotations

import time

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from backend.app_state import AppState
from backend.free.api.chat.chat_constants import MAX_STEP_QUEUE_SIZE
from backend.free.api.chat.chat_recorder import read_llama_prompt_tokens
from backend.free.api.chat.chat_service import make_token_info
from backend.free.api.chat.chat_types import ChatMessage, StepCallback
from backend.free.api.schemas import ChatResponse, TokenInfo
from backend.free.core.sse import SSEFrameBuilder
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer


logger = get_logger("api.chat.streaming")

# SSE フレームビルダー（モジュールレベルの共有インスタンス）
sse = SSEFrameBuilder()


def meta_tool_routing_success(resp) -> bool:
    """meta-cognitive 応答でツールが 1 件以上実行成功したか (tool_routing 正例)。

    deliberative の ``tool_command_success is True`` と同義 (ツールが呼ばれ成功)。
    meta は複数ツールを呼ぶため any (1 件でも成功なら誘導は妥当)。tool_calls 空
    (= ツール未使用) は False。phase7 がクエリ単位の弱い正例として消費する。
    """
    if resp is None:
        return False
    return any(tc.get("success") for tc in (getattr(resp, "tool_calls", None) or []))


def meta_tool_routing_false_positive(resp) -> bool:
    """meta-cognitive 応答でツールを呼んだが全て失敗したか (tool_routing 誤検出)。

    deliberative の ``tool_command is not None and tool_command_success is False``
    と同義 (ルーティングしたが結果が伴わなかった同一ターンの明確な失敗)。tool_calls
    空 (未使用) は False。phase7 + パターン decay がクエリ単位で消費する。
    """
    if resp is None:
        return False
    tool_calls = getattr(resp, "tool_calls", None) or []
    return bool(tool_calls) and not any(tc.get("success") for tc in tool_calls)


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
    dl.log_outcome(
        kind="chat_response",
        success=success,
        duration_ms=(time.monotonic() - started_at) * 1000,
        tokens_out=tokens_out,
        quality_signals=signals,
    )


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
