"""Reactive / Meta-Cognitive 層のストリーミング・同期応答"""

from __future__ import annotations

import asyncio
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
    TokenInfo,
)
from backend.free.agent.meta_cognitive import MetaCognitiveAgent
from backend.free.agent.output_format import WRITTEN_PATH_RE
from backend.free.agent.meta_cognitive_utils import (
    looks_like_task_log_residue,
    strip_task_log_scaffold,
)
from backend.free.llm.local_client import LocalClient
from backend.i18n_helper import msg
from backend.utils import estimate_tokens as _estimate_tokens

from backend.free.api.chat.chat_constants import DEFAULT_KEEPALIVE_INTERVAL_SEC
from backend.free.api.chat.chat_stream_common import (
    agent_layer_frame,
    cancel_requested,
    create_run_frame,
    _emit_stream_error,
    _emit_timing,
    _finish_stream_outcome,
    _record_failed_generation,
    cancel_scope,
    logger,
    meta_last_command_call,
    meta_tool_routing_false_positive,
    meta_tool_routing_success,
    retain_cancel_scope,
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
    yield agent_layer_frame("reactive")
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
    private: bool = False,
):
    """MetaCognitive agent.process() をバックグラウンド実行するコルーチンを生成。

    ステップは step_queue に push され、完了・例外時に None で終端を通知する。
    結果または例外は result_holder に格納して呼び出し側に返す。``private`` は
    MDP エピソードの begin イベントに刻まれる (エピソード記憶へ昇格させない)。
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
                private=private,
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
    agent_task: "asyncio.Task | None" = None,
):
    """step_queue から step フレームを逐次 yield する（keepalive / cancel 対応）。

    キャンセルを検知したら ``agent_task`` も cancel する。以前は drain を抜ける
    だけで、エージェント (ツール実行 / LLM 生成) はそのまま完走していた —
    ユーザーが止めたのに llama-server のスロットとツールが動き続ける。
    """
    while True:
        try:
            step_data = await asyncio.wait_for(
                step_queue.get(), timeout=keepalive_interval,
            )
        except asyncio.TimeoutError:
            if cancel_requested(session_id):
                _cancel_agent_task(agent_task, session_id)
                return
            yield sse.keepalive()
            continue

        if step_data is None:
            return

        if cancel_requested(session_id):
            _cancel_agent_task(agent_task, session_id)
            return

        if step_data.get("type") == "create_run":
            yield create_run_frame(
                str(step_data.get("run_id", "")), str(step_data.get("session_id", "")),
            )
            continue

        yield sse.step(step_data)


#: detached で走らせている制作ターン (GC からの保護。完了で自動的に外れる)。
_DETACHED_TASKS: set["asyncio.Task"] = set()


def _should_detach_on_disconnect(
    agent, agent_task: "asyncio.Task | None", session_id: str,
) -> bool:
    """切断時に agent_task を cancel せず完走させるか (create ∧ 制作ステージあり)。"""
    return (
        agent_task is not None
        and not agent_task.done()
        and getattr(agent, "_production_stage", None) is not None
        and not cancel_requested(session_id)
    )


def _detach_agent_task(
    agent_task: "asyncio.Task", session_id: str, on_done=None,
) -> None:
    """切断後も制作を続けるタスクを登録する。``on_done`` は完走時に 1 回呼ぶ
    (履歴 / 経験の記録。SSE の終端処理は相手がいないので走らない、f_10 §3)。"""
    _DETACHED_TASKS.add(agent_task)
    agent_task.add_done_callback(_DETACHED_TASKS.discard)
    retain_cancel_scope(session_id, agent_task)
    if on_done is not None:
        def _run_on_done(_task: "asyncio.Task") -> None:
            try:
                on_done()
            except Exception as exc:  # noqa: BLE001 - 記録の失敗で例外を漏らさない
                logger.warning(
                    "MetaCognitive: detached turn finalize failed (session=%s): %s",
                    session_id, exc,
                )
        agent_task.add_done_callback(_run_on_done)
    logger.info(
        "MetaCognitive: client disconnected; create turn continues detached "
        "(session=%s)", session_id,
    )


def _cancel_agent_task(agent_task: "asyncio.Task | None", session_id: str) -> None:
    if agent_task is not None and not agent_task.done():
        logger.info("MetaCognitive: cancelling agent task (session=%s)", session_id)
        agent_task.cancel()


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
#: パターンは agent 層 (``output_format.WRITTEN_PATH_RE``) が SSOT — 最終応答の
#: 本文提示 (``_written_content_block``) も同じ形式を読むため、書き写すと
#: 片方だけが形式追随に失敗する。
_WRITTEN_PATH_RE = WRITTEN_PATH_RE

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


def _meta_truncation_frame(resp) -> str | None:
    """内部生成が ``finish_reason=length`` で切れていれば開示フレームを返す。

    deliberative の ``_truncation_frame`` と同じく **本文の外** (SSE フレーム)
    に出す。メタ経路はエージェントがストリームを内部で消費するため、
    切断は ``MetaCognitiveResponse.truncated`` 経由で知る。継続待ち
    (「続けて」の継続生成) は武装しない — 本文はタスク結果の要約であって
    切れた生成そのものではないため。
    """
    if resp is None or not getattr(resp, "truncated", False):
        return None
    return sse.output_truncated(
        int(getattr(resp, "truncated_tokens", 0) or 0),
        getattr(resp, "truncated_max_tokens", None),
    )


def meta_cognitive_recorded_text(resp) -> str:
    """履歴・WM へ記録する本文 — **UI に出したのと同じテキスト** を返す。

    以前は表示だけ ``_meta_cognitive_body_text`` で浄化し、記録は
    ``resp.content`` (= タスク進捗ノートの生文) を使っていた。結果、履歴には
    ``- [done] Create … / Written 373 bytes to …`` が残り、次のターンで
    ベースがその形式を **模倣** する。模倣は deliberative 層から出るので
    ここの浄化を通らず、そのままユーザーへ届く。

    実インシデント 2026-08-22 ライブ監査 (ターン77-78): ファイル書込みの次に
    「そのファイルを削除してください」と頼むと、削除ツールは存在せず
    ``tool_call_judge`` も ``Action blocked: file deletion requested but no tool
    can delete`` を出していたのに、回答は
    ``[done] Delete the file E:\\tmp\\bs_audit.py``。続く「本当に削除できたか
    確認してください」も ``[done] Confirm the file … has been deleted``。
    実行していない操作を 2 ターン続けて完了として提示していた
    (ファイルは実際に残っていた)。
    """
    if resp is None:
        return ""
    if resp.tasks:
        return _meta_cognitive_body_text(resp)
    return resp.content or ""


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
    cancelled: bool = False,
) -> TokenInfo:
    """MetaCognitive ストリーム完了時の記録・タイミング計測を行い TokenInfo を返す。

    ``cancelled`` (ユーザーキャンセルで途中終了) は経験記録をスキップする
    (``record_meta_cognitive_response`` の同名引数)。
    """
    if timer:
        timer.stop("llm_total_ms")

    elapsed = time.monotonic() - t_start
    steps = resp.steps if resp else 0
    tool_calls_count = len(resp.tool_calls) if resp else 0
    logger.info(
        "MetaCognitive stream complete: steps=%d, tool_calls=%d, elapsed=%.2fs",
        steps, tool_calls_count, elapsed,
    )

    content = meta_cognitive_recorded_text(resp)
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
        cancelled=cancelled,
        # 内部生成の切断は経験へ刻む (deliberative の ``truncated=`` と同じ)。
        truncated=bool(getattr(resp, "truncated", False)),
        **meta_last_command_call(resp),
    )

    _emit_timing(state, timer, "meta_cognitive", estimated_tokens, mode=mode)
    return make_token_info(messages, estimated_tokens, context_size, instance_name)


async def stream_meta_cognitive(
    agent: MetaCognitiveAgent, query: str, system_prompt: str,
    conversation: list[ChatMessage], client: LocalClient, state: AppState,
    session_id: str, instance_name: str, context_size: int,
    messages: list[ChatMessage], mode: str,
    *, generation_params: GenerationParams | None = None,
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL_SEC,
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
        # genuine error (except Exception) と client cancel を区別する。
        errored = False
        # 終端処理 (record) まで到達したか。例外経路の失敗記録と二重にしない。
        recorded = False
        # finally が参照する (クライアント切断で generator が閉じられたときに
        # エージェントを孤児にしない)。
        agent_task: asyncio.Task | None = None

        run_agent = _build_meta_cognitive_agent_runner(
            agent,
            query=query, system_prompt=system_prompt,
            conversation=conversation, client=client, state=state,
            session_id=session_id, mode=mode,
            generation_params=generation_params,
            step_queue=step_queue, result_holder=result_holder,
            output_target=output_target,
            private=private,
        )

        try:
            yield agent_layer_frame("meta_cognitive")
            # ``MetaCognitiveAgent._plan`` が同じ ``type="plan"`` を最初に
            # emit する。ここでも出すと UI のステップ一覧に英語と日本語の
            # plan が 2 行並ぶ (2026-08-09 ライブ監査で確認)。エージェント側の
            # 1 本に任せる。

            if timer:
                timer.start("llm_total_ms")
            agent_task = asyncio.create_task(run_agent())

            async for frame in _drain_meta_cognitive_steps(
                step_queue, session_id, keepalive_interval, agent_task,
            ):
                yield frame

            try:
                await agent_task
            except asyncio.CancelledError:
                # 自分が (ユーザーキャンセルで) 止めたタスクなら終端処理へ進む。
                # 外側のタスク自体のキャンセル (クライアント切断) は伝播させる。
                if not (agent_task.cancelled() and cancel_requested(session_id)):
                    raise

            if result_holder["error"] is not None:
                raise result_holder["error"]

            resp = result_holder["resp"]

            if not cancel_requested(session_id):
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
                cancelled=bool(cancel_requested(session_id)),
            )
            recorded = True
            truncation = _meta_truncation_frame(resp)
            if truncation and not cancel_requested(session_id):
                yield truncation
            yield sse.token_info(ti)
            yield sse.done()
            outcome_success = True

        except Exception as e:
            errored = True
            async for frame in _emit_stream_error(
                state, e, timer=timer, agent_layer="meta_cognitive", mode=mode,
            ):
                yield frame
            if not recorded:
                _record_failed_generation(
                    state, query=query, messages=messages,
                    session_id=session_id, mode=mode, private=private,
                    agent_layer="meta_cognitive",
                )
        finally:
            # クライアント切断 (yield に CancelledError) でも計画 / ツールループ /
            # LLM 生成を走らせ続けない。明示キャンセルは drain 側が既に止めている。
            # 例外は create の制作ステージ (production_stage) を持つターン —
            # 切断で止めず detached で完走させる (f_10 §3、Phase 3b)。制作物は
            # meta の write 経路 / /api/create/runs/<id>/artifacts から取れる。
            if _should_detach_on_disconnect(agent, agent_task, session_id):
                def _finalize_detached() -> None:
                    # 完走した detached ターンの履歴 / 経験は、通常経路と同じ
                    # 終端処理で記録する (token_info / done は相手がいないので出さない)。
                    resp_done = result_holder.get("resp")
                    if resp_done is None or result_holder.get("error") is not None:
                        return
                    _finalize_meta_cognitive_stream(
                        resp_done,
                        state=state, messages=messages, session_id=session_id,
                        query=query, mode=mode, instance_name=instance_name,
                        context_size=context_size, timer=timer, t_start=t_start,
                        private=private, rag_used=rag_used,
                        rag_top1_score=rag_top1_score,
                        cancelled=bool(cancel_requested(session_id)),
                    )
                _detach_agent_task(agent_task, session_id, on_done=_finalize_detached)
            else:
                _cancel_agent_task(agent_task, session_id)
            resp_obj = result_holder.get("resp")
            # ``MetaCognitiveResponse`` はトークン数を持たない (以前は存在しない
            # 属性を読んで常に 0 だった)。記録側と同じ本文の見積りを使う。
            tokens_out = (
                max(1, _estimate_tokens(meta_cognitive_recorded_text(resp_obj)))
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
            # production_stage (staged/longform、f_03 §4.4) の metrics を
            # Level 0 経験記録へ載せる (3a-2: 旧 staged 側の record_long_form_response
            # 直接呼出しに代わる 1 本化した記録経路)。
            production_metrics = dict(
                getattr(resp_obj, "production_metrics", None) or {},
            )
            if production_metrics:
                quality_signals["production_metrics"] = production_metrics
            _finish_stream_outcome(
                state, session_id,
                started_at=t_start,
                completed=outcome_success,
                # タスク成否は完走とは別の軸 (切断と混同しない)
                success=not quality_signals.get("failed_tasks"),
                errored=errored,
                tokens_out=tokens_out,
                signals=quality_signals,
            )


