"""Reactive / Meta-Cognitive 層のストリーミング・同期応答"""

from __future__ import annotations

import asyncio
import re
import time

from typing import (
    AsyncIterator,
    TYPE_CHECKING,
)
from backend.app_state import AppState
from backend.free.api.chat.chat_recorder import record_meta_cognitive_response
from backend.free.api.chat.chat_service import make_token_info
from backend.free.api.chat.chat_types import (
    ChatMessage,
    GenerationParams,
)
from backend.free.api.schemas import (
    ChatResponse,
    TokenInfo,
)
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.meta_cognitive_utils import (
    looks_like_task_log_residue,
    strip_task_log_scaffold,
)
from backend.free.llm.local_client import LocalClient
from backend.i18n_helper import msg
from backend.utils import estimate_tokens as _estimate_tokens
from fastapi import HTTPException

from backend.free.api.chat.chat_stream_common import (
    _cancel_flags,
    _emit_timing,
    _log_chat_outcome,
    _sync_chat_response,
    cancel_scope,
    logger,
    meta_tool_routing_false_positive,
    meta_tool_routing_success,
    sse,
)

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer


# ---------------------------------------------------------------------------
# Reactive ストリーミング
# ---------------------------------------------------------------------------

async def stream_reactive(
    content: str, instance_name: str, context_size: int,
) -> AsyncIterator[str]:
    """Reactive 層の応答を SSE ストリーミングで返す"""
    yield sse.agent_layer("reactive")
    yield sse.token(content)
    token_info = {"used": 0, "limit": context_size, "pct": 0, "instance_name": instance_name}
    yield sse.token_info(token_info)
    yield sse.done()


# ---------------------------------------------------------------------------
# Meta-Cognitive ストリーミング / 同期
# ---------------------------------------------------------------------------

def _build_meta_cognitive_agent_runner(
    agent: MetaCognitiveAgent,
    *,
    query: str,
    system_prompt: str,
    conversation: list[ChatMessage],
    client: LocalClient,
    state: AppState,
    session_id: str,
    mode: str,
    generation_params: GenerationParams | None,
    step_queue: "asyncio.Queue[dict | None]",
    result_holder: dict,
    output_target: str = "file",
):
    """MetaCognitive agent.process() をバックグラウンド実行するコルーチンを生成。

    ステップは step_queue に push され、完了・例外時に None で終端を通知する。
    結果または例外は result_holder に格納して呼び出し側に返す。
    """
    async def on_step(step_data: dict) -> None:
        await step_queue.put(step_data)

    async def _run_agent() -> None:
        try:
            resp = await agent.process(
                query=query,
                system_prompt=system_prompt,
                conversation=conversation,
                llm_client=client,
                tools_registry=state.tools_registry,
                on_step=on_step,
                generation_params=generation_params,
                session_id=session_id,
                mode=mode,
                output_target=output_target,
            )
            result_holder["resp"] = resp
        except Exception as e:
            result_holder["error"] = e
        finally:
            await step_queue.put(None)

    return _run_agent


async def _drain_meta_cognitive_steps(
    step_queue: "asyncio.Queue[dict | None]",
    session_id: str,
    keepalive_interval: float,
):
    """step_queue から step フレームを逐次 yield する（keepalive / cancel 対応）。"""
    while True:
        try:
            step_data = await asyncio.wait_for(
                step_queue.get(), timeout=keepalive_interval,
            )
        except asyncio.TimeoutError:
            yield sse.keepalive()
            continue

        if step_data is None:
            return

        if _cancel_flags.get(session_id):
            return

        yield sse.step(step_data)


def _meta_cognitive_body_text(resp) -> str:
    """MetaCognitive 応答のうち、チャット本文として出すべきテキストを返す。

    ``resp.content`` は ``_build_final_response`` が組み立てたタスク進捗ノート
    (「- [done] ... / Written N bytes to ...」) であることが多く、これは
    task_result step として別途送出済みなので本文には出さない。ノート行を
    取り除いて残る実質的な本文 (チャット出力向けの生成コード等) があれば
    それを、無ければタスクの成否をまとめた 1 文を返す。

    これが無いと、ファイル書き込みのようにタスクだけで完結するターンで
    assistant バブルが空になる (実インシデント 2026-07-27 ライブ検証:
    「note.txt に保存してください」への応答が step 行のみで本文なし)。
    """
    body = strip_task_log_scaffold(resp.content or "").strip()
    # 行頭アンカーの行単位除去では落ちない断片 (ノート行が他のテキストと
    # 1 行に連結された形) が残ることがある。残骸を本文として出すくらいなら
    # タスクの成否をまとめた 1 文の方がユーザーには有用。
    if body and looks_like_task_log_residue(body):
        logger.warning(
            "MetaCognitive body looks like task-log residue; falling back to "
            "the task summary: %r", body[:120],
        )
        body = ""
    if body:
        return body
    done = sum(1 for t in resp.tasks if t.status == "done")
    failed = sum(1 for t in resp.tasks if t.status == "failed")
    if failed and not done:
        return msg("agent.tasks_all_failed")
    if failed:
        return msg("agent.tasks_partially_done", done=done, failed=failed)
    # 書込みで完結したターンは「何を書いたか」を出す。「タスクを完了しました。」
    # だけだと、ユーザーは書き込まれた先を確認する手掛かりが本文に無い
    # (2026-08-09 ライブ監査で指摘)。
    written = _written_paths(resp.tasks)
    if written:
        return msg("agent.files_written", paths="、".join(written))
    if done == 1:
        return msg("agent.task_done")
    return msg("agent.tasks_all_done", count=done)


#: ``write_file`` の戻り値 (``Written 158 bytes to E:\tmp\a.txt``) から書込み先を拾う。
_WRITTEN_PATH_RE = re.compile(r"Written\s+\d+\s+bytes?\s+to\s+(.+?)\s*$", re.MULTILINE)

#: task_result ステップ見出しに載せるタスク記述の最大長。
_STEP_DESCRIPTION_MAX_CHARS = 48


def _truncate_step_description(description: str) -> str:
    """ステップ見出し用にタスク記述を短く畳む (純粋関数)。"""
    text = " ".join(str(description or "").split())
    if len(text) <= _STEP_DESCRIPTION_MAX_CHARS:
        return text
    return text[:_STEP_DESCRIPTION_MAX_CHARS] + "…"


def _written_paths(tasks) -> list[str]:
    """完了タスクの結果から書込み先パスを重複なく取り出す (純粋関数)。"""
    paths: list[str] = []
    for task in tasks or []:
        if getattr(task, "status", None) != "done":
            continue
        for match in _WRITTEN_PATH_RE.finditer(str(getattr(task, "result", "") or "")):
            path = match.group(1).strip()
            if path and path not in paths:
                paths.append(path)
    return paths


async def _emit_meta_cognitive_result_frames(resp) -> AsyncIterator[str]:
    """MetaCognitive 応答から最終フレーム（task_result と本文 token）を yield する。"""
    if resp is None:
        return

    # エディタ経路: 生成コードを専用チャネルで送出 (チャット本文には混ぜない)
    # editor_artifacts は dataclass の field(default_factory=list) で常にリスト
    if resp.editor_artifacts:
        for art in resp.editor_artifacts:
            yield sse.editor_code(art.content, language=art.language, filename=art.filename)

    if resp.tasks:
        logger.debug(
            "MetaCognitive final: sending %d task results as step events",
            len(resp.tasks),
        )
        for task in resp.tasks:
            # タスク記述はプラン生成が失敗すると **生のユーザー発言そのもの**
            # になる。そのまま出すと UI の折りたたみ見出しが依頼文まるごとに
            # なり、実際の結果が読めない (2026-08-09 ライブ監査:
            # 「E:\tmp に inventory_notes.txt というファイルを作って、ここまでの
            # 試算結果を3行で書いてください。 Written 158 bytes to ...」)。
            # 見出しは短く保ち、結果を主役にする。
            detail = _truncate_step_description(task.description)
            if task.result:
                detail += f" — {task.result[:500]}"
            logger.debug(
                "MetaCognitive task_result: status=%s, detail=%s",
                task.status, detail[:120],
            )
            yield sse.step({"type": "task_result", "detail": detail, "status": task.status})
        # step だけでは本文が空のままになるため、本文テキストも必ず送る。
        yield sse.token(_meta_cognitive_body_text(resp))
    else:
        logger.debug(
            "MetaCognitive final: no tasks, sending content as token (%d chars)",
            len(resp.content),
        )
        yield sse.token(resp.content)


def _finalize_meta_cognitive_stream(
    resp,
    *,
    state: AppState,
    messages: list[ChatMessage],
    session_id: str,
    query: str,
    mode: str,
    instance_name: str,
    context_size: int,
    timer: "StageTimer | None",
    t_start: float,
    private: bool = False,
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> TokenInfo:
    """MetaCognitive ストリーム完了時の記録・タイミング計測を行い TokenInfo を返す。"""
    if timer:
        timer.stop("llm_total_ms")

    elapsed = time.monotonic() - t_start
    steps = resp.steps if resp else 0
    tool_calls_count = len(resp.tool_calls) if resp else 0
    logger.info(
        "MetaCognitive stream complete: steps=%d, tool_calls=%d, elapsed=%.2fs",
        steps, tool_calls_count, elapsed,
    )

    content = resp.content if resp else ""
    step_credits = resp.step_credits if resp else []
    estimated_tokens = max(1, _estimate_tokens(content))
    record_meta_cognitive_response(
        state, content, messages, session_id,
        query, mode, estimated_tokens, step_credits,
        private=private,
        agent_loops=steps,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
        tool_routing_success=meta_tool_routing_success(resp),
        tool_routing_false_positive=meta_tool_routing_false_positive(resp),
    )

    _emit_timing(state, timer, "meta_cognitive", estimated_tokens, mode=mode)
    return make_token_info(messages, estimated_tokens, context_size, instance_name)


async def stream_meta_cognitive(
    agent: MetaCognitiveAgent, query: str, system_prompt: str,
    conversation: list[ChatMessage], client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    messages: list[ChatMessage], mode: str,
    *, generation_params: GenerationParams | None = None,
    keepalive_interval: float = 15.0,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    rag_used: bool = False,
    rag_top1_score: float | None = None,
):
    """Meta-Cognitive 層の SSE ストリーミング（ステップフレーム付き）

    agent.process() をバックグラウンドタスクで実行し、on_step コールバック
    からのステップ通知をリアルタイムで SSE フレームとして送信する。
    定期的に keepalive コメントを送信してクライアントのタイムアウトを防止する。
    """
    async with cancel_scope(session_id):
        t_start = time.monotonic()
        step_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        result_holder: dict = {"resp": None, "error": None}
        outcome_success = False

        run_agent = _build_meta_cognitive_agent_runner(
            agent,
            query=query, system_prompt=system_prompt,
            conversation=conversation, client=client, state=state,
            session_id=session_id, mode=mode,
            generation_params=generation_params,
            step_queue=step_queue, result_holder=result_holder,
            output_target=output_target,
        )

        try:
            yield sse.agent_layer("meta_cognitive")
            # ``MetaCognitiveAgent._plan`` が同じ ``type="plan"`` を最初に
            # emit する。ここでも出すと UI のステップ一覧に英語と日本語の
            # plan が 2 行並ぶ (2026-08-09 ライブ監査で確認)。エージェント側の
            # 1 本に任せる。

            if timer:
                timer.start("llm_total_ms")
            agent_task = asyncio.create_task(run_agent())

            async for frame in _drain_meta_cognitive_steps(
                step_queue, session_id, keepalive_interval,
            ):
                yield frame

            await agent_task

            if result_holder["error"] is not None:
                raise result_holder["error"]

            resp = result_holder["resp"]

            if not _cancel_flags.get(session_id):
                async for frame in _emit_meta_cognitive_result_frames(resp):
                    yield frame

            ti = _finalize_meta_cognitive_stream(
                resp,
                state=state, messages=messages, session_id=session_id,
                query=query, mode=mode, instance_name=instance_name,
                context_size=context_size, timer=timer, t_start=t_start,
                private=private,
                rag_used=rag_used,
                rag_top1_score=rag_top1_score,
            )
            yield sse.token_info(ti)
            yield sse.done()
            outcome_success = True

        except Exception as e:
            logger.error("MetaCognitive stream error: %s", e)
            if timer:
                timer.stop("llm_total_ms")
            _emit_timing(state, timer, "meta_cognitive", 0, mode=mode)
            yield sse.error(str(e))
            yield sse.done()
        finally:
            resp_obj = result_holder.get("resp")
            tokens_out = (
                int(getattr(resp_obj, "tokens", 0) or 0)
                if resp_obj is not None else 0
            )
            # SSE 完走 = success ではなくタスク成否を反映する。ファイル未作成の
            # 失敗ターンが success=True で記録され、負例が学習に伝播しない
            # 問題 (2026-07-15) への対策。
            quality_signals: dict = {"agent_layer": "meta_cognitive"}
            task_list = list(getattr(resp_obj, "tasks", None) or [])
            if task_list:
                failed_tasks = sum(
                    1 for t in task_list
                    if getattr(t, "status", "") == "failed"
                )
                writes = sum(
                    1 for tc in (getattr(resp_obj, "tool_calls", None) or [])
                    if tc.get("tool") == "write_file" and tc.get("success")
                )
                quality_signals.update({
                    "tasks": len(task_list),
                    "failed_tasks": failed_tasks,
                    "writes": writes,
                })
                if failed_tasks:
                    outcome_success = False
            _log_chat_outcome(
                state,
                started_at=t_start,
                success=outcome_success,
                tokens_out=tokens_out,
                signals=quality_signals,
            )


async def sync_meta_cognitive(
    agent: MetaCognitiveAgent, query: str, system_prompt: str,
    conversation: list[ChatMessage], client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    messages: list[ChatMessage], mode: str,
    *, generation_params: GenerationParams | None = None,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> ChatResponse:
    """Meta-Cognitive 層の同期応答"""
    try:
        if timer:
            timer.start("llm_total_ms")
        tools_registry = state.tools_registry
        resp = await agent.process(
            query=query,
            system_prompt=system_prompt,
            conversation=conversation,
            llm_client=client,
            tools_registry=tools_registry,
            generation_params=generation_params,
            session_id=session_id,
            mode=mode,
            output_target=output_target,
        )

        if timer:
            timer.stop("llm_total_ms")

        # 非ストリームではエディタチャネルが無いため、エディタ経路の生成コードは
        # コードブロックとして応答本文に畳み込む (CLI 等で内容を失わない)。
        response_text = resp.content
        editor_artifacts = getattr(resp, "editor_artifacts", None)
        if editor_artifacts:
            blocks = "\n\n".join(
                f"```{art.language}\n{art.content}\n```" for art in editor_artifacts
            )
            response_text = blocks if not response_text else f"{response_text}\n\n{blocks}"

        estimated_tokens = max(1, _estimate_tokens(response_text))
        record_meta_cognitive_response(
            state, response_text, messages, session_id,
            query, mode, estimated_tokens, resp.step_credits,
            private=private,
            agent_loops=resp.steps,
            rag_used=rag_used,
            rag_top1_score=rag_top1_score,
            tool_routing_success=meta_tool_routing_success(resp),
            tool_routing_false_positive=meta_tool_routing_false_positive(resp),
        )

        return _sync_chat_response(
            state, timer,
            agent_layer="meta_cognitive",
            text=response_text,
            tokens=estimated_tokens,
            messages=messages,
            session_id=session_id,
            instance_name=instance_name,
            context_size=context_size,
            mode=mode,
        )
    except Exception as e:
        logger.error("MetaCognitive error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
