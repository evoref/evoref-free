"""Long-form (長文生成) 層のストリーミング・同期応答"""

from __future__ import annotations

import asyncio
import time

from dataclasses import dataclass
from typing import (
    AsyncIterator,
    TYPE_CHECKING,
)
from pathlib import Path
from backend.app_state import AppState
from backend.free.api.chat.chat_constants import DEFAULT_KEEPALIVE_INTERVAL_SEC
from backend.free.api.chat._long_form_intent import (
    LongFormMode,
    detect_long_form_mode,
)
from backend.free.api.chat.chat_recorder import (
    judge_long_form_success,
    record_long_form_response,
)
from backend.free.api.chat.chat_service import make_token_info
from backend.free.api.chat.chat_types import ChatMessage
from backend.free.api.schemas import ChatResponse
from backend.free.agent.tool_call_judge import _extract_file_path
from backend.free.llm.editor_filename import derive_editor_filename_stem
from backend.free.generation.orchestrator import LongFormOrchestrator
from backend.free.generation.validators import remove_code_fences
from fastapi import HTTPException

from backend.free.api.chat.chat_stream_common import (
    _cancel_flags,
    _capture_stream_outcome,
    _close_token_stream,
    _emit_stream_error,
    _emit_timing,
    _log_chat_outcome,
    _make_step_queue_callback,
    _record_failed_generation,
    _sync_chat_response,
    cancel_scope,
    logger,
    rag_signals_from_chunks,
    sse,
)

from backend.free.api.chat.chat_stream_output import (
    _WRITE_HINT_RE,
    _infer_output_extension,
    _normalize_editor_text,
    _resolve_editor_output_format,
    long_form_write_file,
    split_write_index,
    split_write_single_unit,
)

if TYPE_CHECKING:
    from backend.free.core.stage_timer import StageTimer


# ---------------------------------------------------------------------------
# Long-form ストリーミング / 同期
# ---------------------------------------------------------------------------

@dataclass
class _LongFormStreamState:
    """`stream_long_form` のループ mutable 状態を集約。"""

    tokens_generated: int = 0
    full_response: str = ""
    first_token_recorded: bool = False
    #: 品質ゲートが検出した問題数。ユーザーには警告ステップとして見せている
    #: ため、成否シグナルにも同じ判断を反映させる (下の log_outcome 参照)。
    validation_errors: int = 0
    #: 生成ストリームが ``finish_reason=length`` で切れたか (``TokenStream.outcome``
    #: を公開するストリームのみ観測できる)。開示は本文の外 (``sse.output_truncated``)。
    truncated: bool = False
    truncated_tokens: int = 0
    truncated_max_tokens: int | None = None
    #: ``record_long_form_response`` まで到達したか (例外経路の失敗記録と二重にしない印)。
    recorded: bool = False


async def _emit_long_form_init_steps(
    query: str, file_output_mode: bool,
) -> AsyncIterator[str]:
    """長文生成開始時の初期 SSE フレームを yield する。

    `agent_layer` → (file_output_mode 時) `long_form_file_mode` → `long_form_plan`。
    """
    yield sse.agent_layer("meta_cognitive")
    if file_output_mode:
        file_path = _extract_file_path(query)
        yield sse.step({
            "type": "long_form_file_mode",
            "detail": f"ファイル出力モード: {file_path}",
            "status": "running",
        })
    yield sse.step({
        "type": "long_form_plan",
        "detail": "Generating plan...",
        "status": "running",
    })


async def _flush_step_queue_to_sse(
    step_queue: list[dict],
) -> AsyncIterator[str]:
    """`step_queue` の蓄積フレームを順次 yield して空にする。"""
    while step_queue:
        step_data = step_queue.pop(0)
        yield sse.step(step_data)


async def _flush_step_queue_split_aware(
    step_queue: list[dict],
    *,
    long_form_mode: LongFormMode,
    base_path: str,
    used_paths: set[str],
    written: list[dict],
    state: AppState,
    extension: str = ".txt",
) -> AsyncIterator[str]:
    """SPLIT モード対応の step_queue flush。

    ``long_form_unit_file`` イベントを per-unit ファイル書込みに変換し、
    結果を ``written`` リストに追記する。それ以外のイベントは
    :func:`_flush_step_queue_to_sse` と同じく SSE フレームとして yield する。

    SPLIT 以外のモードでは ``_flush_step_queue_to_sse`` と完全等価に動作する。
    """
    while step_queue:
        step_data = step_queue.pop(0)
        if (
            long_form_mode == LongFormMode.SPLIT
            and step_data.get("type") == "long_form_unit_file"
            and base_path
        ):
            idx = int(step_data.get("idx", 0))
            total = int(step_data.get("total", 0))
            heading = str(step_data.get("heading", ""))
            file_name = step_data.get("file_name")
            content = str(step_data.get("content", ""))
            written_path = await split_write_single_unit(
                base_path=base_path,
                idx=idx,
                total=total,
                heading=heading,
                file_name=file_name,
                content=content,
                state=state,
                used_paths=used_paths,
                extension=extension,
            )
            if written_path:
                written.append({
                    "path": written_path, "heading": heading, "idx": idx,
                })
                yield sse.step({
                    "type": "long_form_file_written",
                    "detail": (
                        f"[{idx + 1}/{total}] {heading} → {Path(written_path).name}"
                    ),
                    "status": "done",
                })
            else:
                yield sse.step({
                    "type": "long_form_file_written",
                    "detail": f"[{idx + 1}/{total}] {heading} → write failed",
                    "status": "error",
                })
            continue
        yield sse.step(step_data)


def _emit_long_form_episode(
    sess_state: AppState,
    *,
    session_id: str,
    mode: str,
    query: str,
    delivered: str,
    metrics: dict,
    private: bool,
) -> None:
    """長文生成ターンを MDP episode として agent_trace へ記録する。

    cogwriter/recurrent 経路は AgentTracer を経由しないため、MDP ingest の
    decision/failure ファクトから長文ターンが欠落していた。単一ステップの
    episode として task / 成果メトリクス / 成否を残す。
    """
    tracer = getattr(sess_state, "agent_tracer", None)
    if tracer is None or private:
        return
    try:
        from backend.free.agent.agent_tracer import MDPStep

        units_completed = int(metrics.get("units_completed", 0) or 0)
        validation_errors = int(metrics.get("validation_errors", 0) or 0)
        # Level 0 記録側 (chat_recorder) と同一式を共有する。式が食い違うと
        # 同じターンが reward=1.0 と long_form_success=False で二重学習される。
        success = judge_long_form_success(metrics, query, delivered)
        episode_id = tracer.begin_episode(session_id, mode)
        tracer.record_step(episode_id, MDPStep(
            step_index=0,
            state={
                "task": query[:200],
                "layer": "long_form",
                "strategy": str(metrics.get("strategy") or ""),
                "content_type": str(metrics.get("content_type") or ""),
                "units_total": int(metrics.get("units_total", 0) or 0),
                "units_completed": units_completed,
                "validation_errors": validation_errors,
            },
            action="long_form_generate",
            observation=(delivered or "")[:200],
            reward=1.0 if success else 0.0,
        ))
        tracer.end_episode(episode_id, "success" if success else "failure")
        tracer.cleanup_episode(episode_id)
    except Exception as e:
        logger.debug("long_form episode emit skipped: %s", e)


async def _finalize_long_form_stream(
    state: _LongFormStreamState,
    sess_state: AppState,
    orchestrator: LongFormOrchestrator,
    query: str,
    messages: list[ChatMessage],
    session_id: str,
    mode: str,
    instance_name: str,
    context_size: int,
    file_output_mode: bool,
    timer: StageTimer | None,
    t_start: float,
    private: bool = False,
    long_form_mode: LongFormMode = LongFormMode.CONTINUE,
    split_written: list[dict] | None = None,
    output_target: str = "file",
    rag_used: bool = False,
    rag_top1_score: float | None = None,
) -> AsyncIterator[str]:
    """長文生成終端処理: timer 停止 + record + write_file + token_info + done。

    SPLIT モード時は単一ファイル書込みをスキップし、INDEX.md を生成する。
    ``split_written`` は ``stream_long_form`` 内で `long_form_unit_file` を
    受けて書き込んだ各ファイルの ``[{"path", "heading", "idx"}]``。

    ``output_target == "editor"`` の場合はディスク書込みをスキップし、
    `sse.editor_code` で生成本文をエディタペインへ送出する (create モードの
    ``_dispatch_meta_cognitive`` 経路の挙動に揃える)。
    """
    if timer:
        timer.stop("llm_total_ms")
    elapsed = time.monotonic() - t_start
    logger.info(
        "Long-form stream complete: strategy=%s, tokens=%d, elapsed=%.2fs, session=%s, file_mode=%s, lf_mode=%s, output_target=%s",
        orchestrator.strategy_name, state.tokens_generated, elapsed,
        session_id, file_output_mode, long_form_mode.value, output_target,
    )
    metrics = getattr(orchestrator, "last_metrics", {})
    # create の editor/file 出力は orchestrator が検証・修正した assembled
    # (last_code_output) を配信する。生ストリーム (full_response) は review の
    # revise トークンが二重追記されるため、コード出力の確定本文には使わない。
    is_code = getattr(orchestrator, "last_content_type", None) == "code"
    code_output = getattr(orchestrator, "last_code_output", None)
    text_output = getattr(orchestrator, "last_text_output", None)
    if is_code and isinstance(code_output, str) and code_output:
        delivered = code_output
    elif (not is_code) and isinstance(text_output, str) and text_output:
        # document_quality モード: 改稿済みユニットから組んだ確定本文。生ストリーム
        # は revise トークンを二重追記するため file 出力には使わない (CODE と対称)。
        delivered = text_output
    else:
        delivered = state.full_response
    record_long_form_response(
        sess_state, delivered, messages, session_id,
        query, mode, state.tokens_generated, metrics,
        private=private,
        rag_used=rag_used,
        rag_top1_score=rag_top1_score,
        # キャンセルで途中まで流した本文は経験に採らない / 切断は経験へ刻む。
        cancelled=bool(_cancel_flags.get(session_id)),
        truncated=state.truncated,
    )
    state.recorded = True

    # 長文経路も MDP episode を残す (agent_trace 互換)。従来は agent_trace を
    # 経由しないため MDP ingest (decision/failure ファクト) から長文ターンが
    # 全欠落していた (2026-07-15: 最重要の問題経路 10 ターンが学習素材にならず)。
    _emit_long_form_episode(
        sess_state, session_id=session_id, mode=mode, query=query,
        delivered=delivered, metrics=metrics, private=private,
    )

    # 検証エラーを含む生成物がエディタ/ディスクへそのまま出力されると、ユーザは
    # 破損に気づけない (validate は事後計測でブロックしない)。
    # validation_errors > 0 のときは警告ステップを surface して認識させる。
    # 文面は content_type で分ける (TEXT にも品質ゲートが入ったため、散文に対して
    # 「構文エラー」と表示していた 2026-07-25 の誤表示を解消)。
    validation_errors = int(metrics.get("validation_errors", 0) or 0)
    state.validation_errors = validation_errors
    if validation_errors > 0:
        logger.warning(
            "Long-form output has %d validation error(s) (content_type=%s, "
            "session=%s); surfacing warning to user",
            validation_errors, metrics.get("content_type"), session_id,
        )
        if str(metrics.get("content_type") or "") == "code":
            detail = (
                f"⚠ 生成コードに構文エラーが {validation_errors} 件検出されました。"
                "そのまま実行する前に内容を確認してください。"
            )
        else:
            detail = (
                f"⚠ 生成文の品質チェックで {validation_errors} 件の問題を検出しました"
                "(反復・目標文字数からの乖離・未対応のレビュー指摘)。内容を確認してください。"
            )
        yield sse.step({
            "type": "task_result",
            "detail": detail,
            "status": "failed",
        })

    if getattr(orchestrator, "last_needed_clarification", False) is True:
        # 主題不明で確認質問を返しただけの応答。delivered は確認質問文なので、
        # そのまま file 書込み/editor 送出してしまうと確認質問がドキュメントに
        # なってしまう (2026-07-22 発見のトピック混入バグの対策で追加)。
        write_result = None
    elif output_target == "editor":
        # エディタ経路: ディスクには書込みせず生成本文を editor_code フレームで送出。
        # SPLIT モードでもエディタ送出を優先 (per-unit ファイルは on_step 側で完了済み)。
        ext, language = _resolve_editor_output_format(query, is_code)
        # コード生成時は確定本文 (delivered = 検証・修正済み assembled) を使い、
        # markdown コードフェンスを除去する。
        body = remove_code_fences(delivered) if is_code else delivered
        # 空行を含む連続改行を単一 \n に圧縮 (markdown 見出しは保持)。
        editor_text = _normalize_editor_text(body)
        # 指示文と言語から ASCII snake_case のファイル名を導出する
        # (日本語見出しをそのまま流用するとタブ名が日本語化するため)。
        stem = derive_editor_filename_stem(hint=query, language=language)
        filename = f"{stem}{ext}"
        yield sse.editor_code(
            editor_text, language=language, filename=filename,
        )
        write_result = None
    elif long_form_mode == LongFormMode.SPLIT and split_written is not None:
        # SPLIT: per-unit 書込みは既に on_step で完了。INDEX.md だけ生成。
        base_path = _extract_file_path(query) or ""
        if base_path:
            index_path = await split_write_index(
                base_path=base_path,
                written=split_written,
                state=sess_state,
            )
            if index_path:
                yield sse.step({
                    "type": "task_result",
                    "detail": (
                        f"{len(split_written)} files written; index: {index_path}"
                    ),
                    "status": "done",
                })
            else:
                yield sse.step({
                    "type": "task_result",
                    "detail": f"{len(split_written)} files written (no index)",
                    "status": "done",
                })
        write_result = None
    else:
        write_result = await long_form_write_file(
            query, delivered, sess_state,
        )
    if write_result:
        logger.debug("Long-form file write result: %s", write_result[:120])
        yield sse.step({
            "type": "task_result", "detail": write_result, "status": "done",
        })
    _emit_timing(sess_state, timer, "long_form", state.tokens_generated, mode=mode)
    if state.truncated:
        # 切断の開示は本文の外 (deliberative と同じ扱い)。本文へ注記を混ぜると
        # 履歴 / STM へ保存され次ターンで復唱される。
        yield sse.output_truncated(state.truncated_tokens, state.truncated_max_tokens)
    ti = make_token_info(
        messages, state.tokens_generated, context_size, instance_name,
    )
    yield sse.token_info(ti)
    yield sse.done()


async def stream_long_form(
    orchestrator: LongFormOrchestrator, query: str, session_id: str,
    mode: str, state: AppState,
    instance_name: str, context_size: int,
    messages: list[ChatMessage],
    existing_content: str = "",
    *,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    prefetched_rag_top_score: float | None = None,
    file_context_block: str | None = None,
):
    """長文生成の SSE ストリーミング（long_form_* ステップフレーム付き）

    ``output_target`` は create モード時の出力先 (``"file"`` / ``"editor"`` /
    ``"chat"``)。``"editor"`` の場合はトークンの逐次送出を抑止して終端で
    `sse.editor_code` を送る (`_dispatch_meta_cognitive` 経路の挙動に揃える)。
    """
    async with cancel_scope(session_id):
        t_start = time.monotonic()
        stream_state = _LongFormStreamState()
        outcome_success = False

        # ファイル出力モード判定
        file_output_mode = bool(
            _WRITE_HINT_RE.search(query) and _extract_file_path(query)
        )
        # エディタ経路: トークンは chat へ流さず終端で editor_code 送出に切替える。
        editor_output_mode = (output_target == "editor")
        # token を sse.token としてチャットへ流さない条件を共通化。
        suppress_chat_token_stream = file_output_mode or editor_output_mode
        # 出力モード判定 (EXPAND / SPLIT / CONTINUE)。
        # SPLIT/EXPAND は P2/P3 で挙動分岐するが、P1 では CONTINUE と同じ動作。
        long_form_mode = detect_long_form_mode(
            query,
            has_existing_content=bool(existing_content),
            file_output_mode=file_output_mode,
        )
        if long_form_mode in (LongFormMode.EXPAND, LongFormMode.SPLIT):
            logger.info(
                "Long-form mode detected: %s (file_output_mode=%s, existing=%d chars)",
                long_form_mode.value, file_output_mode, len(existing_content),
            )

        step_queue: list[dict] = []
        on_step = _make_step_queue_callback(step_queue)

        # SPLIT モード用の per-unit 書込み状態
        split_base_path = _extract_file_path(query) or ""
        split_used_paths: set[str] = set()
        split_written: list[dict] = []
        # ユーザー指示文から拡張子を 1 度だけ推論し、SPLIT 全 unit に共通適用。
        split_extension = _infer_output_extension(query)

        def _flush(): return _flush_step_queue_split_aware(
            step_queue,
            long_form_mode=long_form_mode,
            base_path=split_base_path,
            used_paths=split_used_paths,
            written=split_written,
            state=state,
            extension=split_extension,
        )

        async def _flush_with_editor():
            """step をフラッシュしつつ、editor 経路ではユニット完了ごとに
            累積コードを ``editor_code(partial=True)`` で逐次送出する。

            フロント側は同一タブを上書き更新し、生成途中の経過を可視化する。
            終端の確定本文は ``_finalize_long_form_stream`` が partial=False で送る。
            """
            saw_unit_done = editor_output_mode and any(
                s.get("type") == "long_form_unit_done" for s in step_queue
            )
            async for frame in _flush():
                yield frame
            if saw_unit_done and stream_state.full_response.strip():
                # content_type は generate() 開始直後に確定するため、unit 完了が
                # 見えている時点では必ず set 済み。code ならフェンス除去 + python 表示。
                is_code = getattr(orchestrator, "last_content_type", None) == "code"
                _ext, language = _resolve_editor_output_format(query, is_code)
                body = (
                    remove_code_fences(stream_state.full_response)
                    if is_code else stream_state.full_response
                )
                yield sse.editor_code(
                    _normalize_editor_text(body),
                    language=language,
                    filename=None,
                    partial=True,
                )

        try:
            async for frame in _emit_long_form_init_steps(query, file_output_mode):
                yield frame

            if timer:
                timer.start("llm_total_ms")
                timer.start("llm_first_token_ms")

            # orchestrator.generate を async iterator として取り出し、
            # `asyncio.wait` でタイムアウトを挟むことで prefill が長引いて
            # トークンが届かない間も keepalive を送出する (S3)。
            # さらに file_output_mode 時はトークンをフロントに送出しないため、
            # `last_frame_at` で実 yield 時刻を追跡し、keepalive_interval 経過時に
            # keepalive を強制送出してフロントの chunk timeout を防ぐ。
            token_gen = orchestrator.generate(
                instruction=query,
                session_id=session_id,
                mode=mode,
                on_step=on_step,
                existing_content=existing_content,
                long_form_mode=long_form_mode,
                prefetched_rag=prefetched_rag,
                file_context_block=file_context_block,
                # ドキュメント品質ゲートは実ファイル出力時のみ意味を持つ。no-file の
                # チャット表示応答にゲート/本文差し替えを及ぼさないよう file 出力確定
                # 時だけ形式を渡す (非 file は "" → is_document_format=False で非適用)。
                target_format=(
                    _infer_output_extension(query) if file_output_mode else ""
                ),
            )
            aiter = token_gen.__aiter__()
            pending: asyncio.Task[str] | None = None
            last_frame_at = time.monotonic()
            while True:
                if _cancel_flags.get(session_id):
                    if pending is not None and not pending.done():
                        pending.cancel()
                    # orchestrator の generator を閉じて下位の生成を止める
                    await _close_token_stream(token_gen)
                    break
                if pending is None:
                    pending = asyncio.create_task(aiter.__anext__())
                done, _ = await asyncio.wait(
                    {pending}, timeout=DEFAULT_KEEPALIVE_INTERVAL_SEC,
                )
                if pending not in done:
                    # 同一 `pending` を維持したまま keepalive を送出。
                    # 蓄積中の step + editor 逐次更新も流して進行を可視化する。
                    async for frame in _flush_with_editor():
                        yield frame
                    yield sse.keepalive()
                    last_frame_at = time.monotonic()
                    continue
                try:
                    token = pending.result()
                except StopAsyncIteration:
                    pending = None
                    break
                pending = None

                step_frame_yielded = False
                async for frame in _flush_with_editor():
                    yield frame
                    step_frame_yielded = True
                if step_frame_yielded:
                    last_frame_at = time.monotonic()

                if not stream_state.first_token_recorded and timer:
                    timer.stop("llm_first_token_ms")
                    stream_state.first_token_recorded = True

                stream_state.full_response += token
                stream_state.tokens_generated += 1

                # needs_clarification 時は file_output_mode/editor_output_mode に
                # 関わらず必ずチャットへ送出する。抑制したままだと、write-hint 付き
                # query (例: 元バグの再現クエリ) で確認質問がユーザーに一切届かず
                # 無応答に見えてしまう (2026-07-22 発見のトピック混入バグ対策)。
                needs_clarification = (
                    getattr(orchestrator, "last_needed_clarification", False) is True
                )
                if not suppress_chat_token_stream or needs_clarification:
                    yield sse.token(token)
                    last_frame_at = time.monotonic()
                elif time.monotonic() - last_frame_at >= DEFAULT_KEEPALIVE_INTERVAL_SEC:
                    # file_output_mode / editor_output_mode はトークンを送出しないため、
                    # 長時間 unit でフロントの chunk timeout を防ぐ keepalive を送る。
                    yield sse.keepalive()
                    last_frame_at = time.monotonic()

            # ``finish_reason=length`` の観測 (outcome を公開するストリームのみ)。
            _capture_stream_outcome(token_gen, stream_state)

            async for frame in _flush():
                yield frame

            _lf_rag_used, _lf_rag_top1 = rag_signals_from_chunks(
                prefetched_rag, prefetched_rag_top_score,
            )
            async for frame in _finalize_long_form_stream(
                stream_state, state, orchestrator, query, messages,
                session_id, mode, instance_name, context_size,
                file_output_mode, timer, t_start,
                private=private,
                long_form_mode=long_form_mode,
                split_written=split_written,
                output_target=output_target,
                rag_used=_lf_rag_used,
                rag_top1_score=_lf_rag_top1,
            ):
                yield frame
            outcome_success = True

        except Exception as e:
            logger.debug("Long-form stream error traceback", exc_info=True)
            async for frame in _emit_stream_error(
                state, e, timer=timer, agent_layer="long_form", mode=mode,
                tokens_generated=stream_state.tokens_generated,
            ):
                yield frame
            if not stream_state.recorded:
                _record_failed_generation(
                    state, query=query, messages=messages,
                    session_id=session_id, mode=mode, private=private,
                    agent_layer="long_form",
                    tokens_generated=stream_state.tokens_generated,
                )
        finally:
            # 品質ゲートが問題を出した応答をユーザーには「要確認」と見せて
            # おきながら成功として記録すると、選択圧に失敗が一件も入らない
            # (実インシデント 2026-08-01 ライブ監査: 目標 300 字に対し 530 字
            # で警告を出したターンが success=True で記録された)。
            # meta_cognitive 経路が failed_tasks を畳み込むのと同じ扱いに揃える。
            _log_chat_outcome(
                state,
                started_at=t_start,
                success=outcome_success and not stream_state.validation_errors,
                tokens_out=stream_state.tokens_generated,
                signals={
                    "agent_layer": "long_form",
                    "file_output_mode": file_output_mode,
                    "output_target": output_target,
                    "validation_errors": stream_state.validation_errors,
                },
            )


async def sync_long_form(
    orchestrator: LongFormOrchestrator, query: str, session_id: str,
    mode: str, state: AppState,
    instance_name: str, context_size: int,
    messages: list[ChatMessage],
    existing_content: str = "",
    *,
    timer: StageTimer | None = None,
    private: bool = False,
    output_target: str = "file",
    prefetched_rag: list[tuple[str, float, str]] | None = None,
    prefetched_rag_top_score: float | None = None,
    file_context_block: str | None = None,
) -> ChatResponse:
    """長文生成の同期応答

    ``output_target == "editor"`` の場合はディスク書込みをスキップし、
    生成本文をそのまま ``ChatResponse.response`` として返す (フロント側で
    エディタペインに流す前提)。
    """
    try:
        full_response = ""
        tokens_generated = 0

        # ストリーミングと同じ意図検出をかける (出力モード一貫性のため)
        file_output_mode = bool(
            _WRITE_HINT_RE.search(query) and _extract_file_path(query)
        )
        long_form_mode = detect_long_form_mode(
            query,
            has_existing_content=bool(existing_content),
            file_output_mode=file_output_mode,
        )

        if timer:
            timer.start("llm_total_ms")
            timer.start("llm_first_token_ms")
        first_token_recorded = False

        async for token in orchestrator.generate(
            instruction=query,
            session_id=session_id,
            mode=mode,
            existing_content=existing_content,
            long_form_mode=long_form_mode,
            prefetched_rag=prefetched_rag,
            file_context_block=file_context_block,
            # ドキュメント品質ゲートは実ファイル出力時のみ適用 (非 file は "")。
            target_format=(
                _infer_output_extension(query) if file_output_mode else ""
            ),
        ):
            if not first_token_recorded and timer:
                timer.stop("llm_first_token_ms")
                first_token_recorded = True
            full_response += token
            tokens_generated += 1

        if timer:
            timer.stop("llm_total_ms")

        metrics = getattr(orchestrator, "last_metrics", {})
        # create の editor/file 出力は検証・修正済み assembled (last_code_output)
        # を配信する (生ストリームの revise 二重追記を解消)。
        is_code = getattr(orchestrator, "last_content_type", None) == "code"
        code_output = getattr(orchestrator, "last_code_output", None)
        text_output = getattr(orchestrator, "last_text_output", None)
        if is_code and isinstance(code_output, str) and code_output:
            full_response = code_output
        elif (not is_code) and isinstance(text_output, str) and text_output:
            # document_quality モード: 改稿済み確定本文 (revise 二重追記の解消)。
            full_response = text_output
        _lf_rag_used, _lf_rag_top1 = rag_signals_from_chunks(
            prefetched_rag, prefetched_rag_top_score,
        )
        record_long_form_response(
            state, full_response, messages, session_id,
            query, mode, tokens_generated, metrics,
            private=private,
            rag_used=_lf_rag_used,
            rag_top1_score=_lf_rag_top1,
        )

        if getattr(orchestrator, "last_needed_clarification", False) is True:
            # 主題不明で確認質問を返しただけの応答。full_response は確認質問文
            # なので、そのまま file 書込みしてしまうと確認質問がドキュメントに
            # なってしまう (2026-07-22 発見のトピック混入バグの対策で追加)。
            write_result = None
        elif output_target == "editor":
            write_result = None
            # コード生成時は editor 表示前に markdown コードフェンスを除去。
            if is_code:
                full_response = remove_code_fences(full_response)
        else:
            write_result = await long_form_write_file(
                query, full_response, state,
            )

        return _sync_chat_response(
            state, timer,
            agent_layer="meta_cognitive",
            text=write_result if write_result else full_response,
            tokens=tokens_generated,
            messages=messages,
            session_id=session_id,
            instance_name=instance_name,
            context_size=context_size,
            mode=mode,
        )
    except Exception as e:
        logger.error("Long-form error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
