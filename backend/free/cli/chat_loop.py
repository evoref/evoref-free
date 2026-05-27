"""CLI チャットストリーミングとメインループ"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from backend.free.cli.command_parser import (
    SessionState,
    handle_async_command,
    handle_command,
    is_async_command,
    parse_command,
)
from backend.free.cli.session_persistence import (
    finalize_session,
    save_checkpoint,
)
from backend.free.cli.renderer import (
    StepSpinner,
    format_prompt,
    render_error,
    render_info,
    render_separator,
    render_shell_out_request,
    render_shell_out_result,
    render_status_line,
    render_step,
    render_user_message,
)
from backend.free.cli.split_layout import SplitLayout
from backend.error_handlers import E6001, E6004
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.chat_loop")


@dataclass
class _ChatStreamState:
    """`chat_stream` のループ状態を集約する mutable データクラス。

    - `full_response`: 受信トークンの累積文字列
    - `tokens_generated`: トークンフレーム受信回数
    - `ttft_sec`: 最初のトークン受信までの経過秒
    - `stream_start`: SSE ストリーム開始時刻
    - `file_written`: write_file ツール完了を検出済み
    - `error_received`: サーバーからエラーフレーム受信済み
    - `post_token_gap_shown`: テキスト→ステップ間の空行を表示済み (1 回のみ)
    - `shell_out_requests`: バックエンドから受信したシェルアウト要求コマンド列
    """

    full_response: str = ""
    tokens_generated: int = 0
    ttft_sec: float | None = None
    stream_start: float | None = None
    file_written: bool = False
    error_received: bool = False
    post_token_gap_shown: bool = False
    shell_out_requests: list[str] = field(default_factory=list)


def _build_chat_payload(
    message: str, sess_state: SessionState,
) -> dict:
    """`/api/chat` 用のリクエスト payload を構築する。

    `state.file_chunks` がセットされていれば file_contexts を含める。
    """
    file_contexts: list[dict] = []
    if sess_state.file_chunks:
        for p in sess_state.context_files:
            chunks = sess_state.file_chunks.get(p)
            if chunks:
                file_contexts.append({
                    "filename": Path(p).name,
                    "chunks": chunks,
                })
    payload: dict = {
        "message": message,
        "mode": sess_state.mode,
        "session_id": sess_state.session_id,
    }
    # プライベートセッション中はターン単位で private フラグを送信
    # 受信側 (`backend.free.api.chat.chat`) は `private=True` のターンを memory_only
    # で扱い、LTM/SemMem 昇格と履歴ディスク永続化をスキップする。
    if getattr(sess_state, "private_mode", False):
        payload["private"] = True
        logger.debug("chat_stream: private mode is ON")
    if file_contexts:
        payload["file_contexts"] = file_contexts
        logger.debug("chat_stream: sending %d file contexts", len(file_contexts))
    return payload


def _setup_chat_renderers(
    non_interactive: bool, console, split_layout,
) -> tuple[object | None, object | None, StepSpinner | None]:
    """`(md_renderer, think_filter, step_spinner)` を初期化する。

    非対話モード時はすべて `None`。`md_renderer.start()` も済ませた状態で返す。
    """
    if non_interactive:
        return None, None, None
    from backend.free.cli.streaming_markdown import StreamingMarkdownRenderer
    from backend.free.llm.local_client import _ReasoningFilter
    md_renderer = StreamingMarkdownRenderer(console, split_layout=split_layout)
    think_filter = _ReasoningFilter()
    md_renderer.start()
    step_spinner = StepSpinner(console)
    return md_renderer, think_filter, step_spinner


async def _handle_token_frame(
    token: str,
    state: _ChatStreamState,
    md_renderer,
    think_filter,
    step_spinner,
    non_interactive: bool,
) -> None:
    """`token` フレームを処理。スピナー停止 / TTFT 記録 / レンダリング。"""
    if step_spinner:
        await step_spinner.stop()
    state.tokens_generated += 1
    if state.ttft_sec is None and state.stream_start is not None:
        state.ttft_sec = time.monotonic() - state.stream_start
    if non_interactive:
        sys.stdout.write(token)
        sys.stdout.flush()
        state.full_response += token
        return
    # CLI 表示は <think> タグを除去してから Rich 変換
    filtered = think_filter.feed(token)
    if filtered:
        md_renderer.feed(filtered)
    state.full_response = md_renderer.full_response


async def _handle_step_frame(
    step: dict,
    state: _ChatStreamState,
    md_renderer,
    step_spinner,
    console,
) -> None:
    """`step` フレームを処理。write_file 検出 / スピナー切替 / バッファフラッシュ。"""
    detail = step.get("detail", "")
    step_type = step.get("type", "")
    if step_type == "tool_call" and "Written" in detail:
        state.file_written = True
    if md_renderer and step_type == "tool_call" and "write_file" in detail:
        md_renderer.notify_file_written()
    status = step.get("status", "running")
    if status == "running":
        if step_spinner:
            await step_spinner.stop()
            step_spinner.start(step_type, detail)
        return
    # done/failed: スピナー停止 → バッファフラッシュ → 確定表示
    if step_spinner:
        await step_spinner.stop()
    if md_renderer and state.tokens_generated > 0:
        md_renderer.flush()
        if not state.post_token_gap_shown:
            console.print()  # テキストとステップの間に空行 (1 回のみ)
            state.post_token_gap_shown = True
    render_step(console, step)
    if step_type == "long_form_plan":
        console.print()


def _handle_shell_out_frame(shell_out_data: dict, state: _ChatStreamState) -> None:
    """`shell_out` フレームを処理。コマンドを `shell_out_requests` にキュー。"""
    so_cmd = shell_out_data.get("cmd", "")
    if not so_cmd:
        return
    state.shell_out_requests.append(so_cmd)
    logger.debug(
        "chat_stream: shell_out request queued: %s", so_cmd[:100],
    )


def _handle_error_frame(
    error_text: str,
    state: _ChatStreamState,
    sess_state: SessionState,
    console,
    non_interactive: bool,
) -> None:
    """`error` フレームを処理。`state.error_received` を立てて表示。"""
    state.error_received = True
    sess_state.record_error()
    logger.debug("chat_stream: server error frame: %s", error_text)
    if not non_interactive:
        render_error(console, msg("cli.llm_error", detail=error_text))
    else:
        print(f"Error: {error_text}", file=sys.stderr)


def _handle_token_info_frame(info: dict, sess_state: SessionState) -> None:
    """`token_info` フレームを処理。`SessionState` のトークン情報を更新。"""
    sess_state.token_used = info.get("used", sess_state.token_used)
    sess_state.token_limit = info.get("limit", sess_state.token_limit)
    if "instance_name" in info:
        sess_state.instance_name = info["instance_name"]


async def _process_sse_frame(
    data: dict,
    state: _ChatStreamState,
    sess_state: SessionState,
    md_renderer,
    think_filter,
    step_spinner,
    console,
    non_interactive: bool,
) -> None:
    """1 つの SSE フレームを処理してフレーム種別ごとにディスパッチする。"""
    if "token" in data:
        await _handle_token_frame(
            data["token"], state, md_renderer, think_filter,
            step_spinner, non_interactive,
        )
    if "step" in data and not non_interactive:
        await _handle_step_frame(
            data["step"], state, md_renderer, step_spinner, console,
        )
    if "shell_out" in data and not non_interactive:
        _handle_shell_out_frame(data["shell_out"], state)
    if "error" in data:
        _handle_error_frame(
            data["error"], state, sess_state, console, non_interactive,
        )
    if "token_info" in data:
        _handle_token_info_frame(data["token_info"], sess_state)


def _render_chat_error(
    error_code: str, error_msg_text: str, console, non_interactive: bool,
) -> None:
    """エラーコード + メッセージを CLI / stderr へ出力する (DRY)。"""
    if non_interactive:
        print(f"[{error_code}] {error_msg_text}", file=sys.stderr)
    else:
        render_error(console, error_msg_text, code=error_code)


async def _send_chat_cancel_request(
    backend_url: str, session_id: str,
) -> None:
    """バックエンドにキャンセルリクエストを送信する (設計書 09 §9.5.2)。

    接続エラー / タイムアウトは無視する (best-effort)。
    """
    try:
        async with httpx.AsyncClient() as cancel_client:
            await cancel_client.post(
                f"{backend_url}/api/chat/cancel",
                json={"session_id": session_id},
                timeout=5.0,
            )
            logger.debug(
                "chat_stream: cancel request sent for session %s", session_id,
            )
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        logger.debug("chat_stream: cancel request failed (ignored)")


async def _finalize_chat_stream(
    state: _ChatStreamState,
    md_renderer,
    think_filter,
    step_spinner,
    console,
    non_interactive: bool,
) -> None:
    """`chat_stream` の `finally` ブロック処理。スピナー停止 + バッファフラッシュ。"""
    if step_spinner:
        try:
            await step_spinner.stop()
        except Exception:  # noqa: BLE001 — クリーンアップ中の例外は握る
            pass
    if non_interactive:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    if md_renderer:
        flushed = think_filter.flush()
        if flushed:
            md_renderer.feed(flushed)
        md_renderer.finish()
        state.full_response = md_renderer.full_response
        md_renderer.render_end_marker()
    else:
        console.print()  # 改行


async def chat_stream(
    message: str,
    state: SessionState,
    console,
    *,
    non_interactive: bool = False,
    split_layout=None,
) -> tuple[str, list[str], float | None]:
    """SSE ストリーミングで応答を受信し逐次表示

    non_interactive=True の場合、トークンのみをプレーンテキストで stdout に出力し、
    ステップフレーム・token_info 表示を抑制する（設計書 09 §9.5.6）。

    Returns:
        (full_response, shell_out_requests, ttft_sec):
        応答テキスト、シェルアウト要求リスト、TTFT（秒）。トークン未受信時は None。
    """
    logger.debug(
        "chat_stream: message_len=%d, backend=%s, non_interactive=%s",
        len(message), state.backend_url, non_interactive,
    )
    payload = _build_chat_payload(message, state)
    md_renderer, think_filter, step_spinner = _setup_chat_renderers(
        non_interactive, console, split_layout,
    )
    stream_state = _ChatStreamState()

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{state.backend_url}/api/chat",
                json=payload,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,   # Meta-Cognitive 層は複数回 LLM 呼び出しがあるため余裕を持つ
                    write=10.0,
                    pool=10.0,
                ),
            ) as resp:
                resp.raise_for_status()
                stream_state.stream_start = time.monotonic()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug(
                            "chat_stream: failed to parse SSE frame: %s",
                            data_str[:100],
                        )
                        continue
                    await _process_sse_frame(
                        data, stream_state, state,
                        md_renderer, think_filter, step_spinner,
                        console, non_interactive,
                    )
    except httpx.HTTPStatusError as e:
        logger.error(
            "E6001 chat_stream: backend returned status %d: %s",
            e.response.status_code, e,
        )
        state.record_error()
        if e.response.status_code == 503:
            error_msg = msg("cli.llm_not_connected")
        else:
            error_msg = msg("cli.backend_error", status=e.response.status_code)
        _render_chat_error(E6001, error_msg, console, non_interactive)
    except httpx.ConnectError:
        logger.error(
            "E6001 chat_stream: backend connection lost: %s", state.backend_url,
        )
        state.record_error()
        _render_chat_error(
            E6001, msg("cli.backend_not_running"), console, non_interactive,
        )
    except httpx.TimeoutException as e:
        logger.error("E6004 chat_stream: request timed out: %s", e)
        state.record_error()
        _render_chat_error(
            E6004, msg("error.cli.stream_timeout"), console, non_interactive,
        )
    except httpx.ReadError:
        logger.error("E6004 chat_stream: streaming response interrupted")
        state.record_error()
        _render_chat_error(
            E6004, msg("error.cli.stream_interrupted"),
            console, non_interactive,
        )
        if not non_interactive and stream_state.full_response:
            render_info(console, stream_state.full_response)
    except KeyboardInterrupt:
        if md_renderer:
            flushed = think_filter.flush()
            if flushed:
                md_renderer.feed(flushed)
            md_renderer.finish()
            stream_state.full_response = md_renderer.full_response
        logger.debug("chat_stream: interrupted by user")
        await _send_chat_cancel_request(state.backend_url, state.session_id)
        if not non_interactive:
            render_info(console, "\n" + msg("cli.generation_interrupted"))
    finally:
        await _finalize_chat_stream(
            stream_state, md_renderer, think_filter, step_spinner,
            console, non_interactive,
        )

    logger.debug(
        "chat_stream: response_len=%d, tokens_generated=%d, shell_out_requests=%d",
        len(stream_state.full_response), stream_state.tokens_generated,
        len(stream_state.shell_out_requests),
    )

    # 空応答の警告表示（エラー受信済み or ファイル出力成功時は除外）
    if (stream_state.tokens_generated == 0 and not non_interactive
            and not stream_state.file_written
            and not stream_state.error_received):
        render_error(console, msg("cli.empty_response"))

    return (
        stream_state.full_response,
        stream_state.shell_out_requests,
        stream_state.ttft_sec,
    )


async def _execute_shell_outs(shell_out_requests: list[str], console) -> None:
    """シェルアウト要求を順次実行（ユーザー確認付き）"""
    from backend.free.cli.shell_out import shell_out as execute_shell_out
    for so_cmd in shell_out_requests:
        render_shell_out_request(console, so_cmd)
        try:
            answer = input(msg("cli.shell_out_confirm")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("y", "yes"):
            render_info(console, msg("cli.shell_out_running", cmd=so_cmd))
            so_result = execute_shell_out(so_cmd)
            render_shell_out_result(console, so_result)
        else:
            render_info(console, msg("cli.shell_out_skipped"))


async def _process_post_response(
    response: str,
    shell_out_requests: list[str],
    state: SessionState,
    console,
) -> None:
    """応答後の共通処理: diff 適用、shell_out 実行、出力省略ヒント表示

    sequential/split 両モードで共通して使用する。
    """
    if response:
        state.add_turn("assistant", response)

        # diff 適用確認
        from backend.free.cli.diff_applier import process_response_diffs
        applied = await process_response_diffs(response, console)
        if applied:
            logger.debug("Applied diffs to %d file(s): %s", len(applied), applied)

        # 出力省略ヒント
        from backend.free.constants import TRUNCATION_MARKER
        if TRUNCATION_MARKER in response:
            render_info(console, msg("cli.page_hint"))

    # シェルアウト要求の処理
    if shell_out_requests:
        await _execute_shell_outs(shell_out_requests, console)


def _save_checkpoint_if_needed(state: SessionState) -> None:
    """定期チェックポイント保存（設計書 23.3.3）"""
    if (state.checkpoint_interval > 0
            and len(state.turns) > 0
            and len(state.turns) % state.checkpoint_interval == 0):
        save_checkpoint(state)


async def _main_loop(state: SessionState, console) -> None:
    """メインインタラクションループ

    Claude Code 風の全スクロールレイアウト（sequential モード）を使用。
    固定ウィジェットを持たず、入力も出力も自然にスクロールする。
    """
    await _main_loop_sequential(state, console)


async def _main_loop_sequential(state: SessionState, console) -> None:
    """逐次出力モードのメインループ

    textual 未インストール時のフォールバック。input() でユーザー入力を取得する。
    """
    ctrl_c_count = 0

    while not state.should_exit:
        prompt_value = format_prompt(no_color=console.no_color)

        try:
            text = input(prompt_value)
            ctrl_c_count = 0
        except KeyboardInterrupt:
            ctrl_c_count += 1
            if ctrl_c_count >= 2:
                # Ctrl+C×2: 自動保存して終了（設計書 23.3.1）
                finalize_session(state)
                render_info(console, "\n" + msg("cli.exit_message"))
                break
            console.print(f"\n({msg('cli.ctrl_c_hint')})")
            continue
        except EOFError:
            finalize_session(state)
            render_info(console, "\n" + msg("cli.exit_message"))
            break

        text = text.strip()
        if not text:
            continue

        # コマンド処理
        cmd, args = parse_command(text)
        if cmd:
            logger.debug("Command detected: %s, args=%r", cmd, args[:50] if args else "")
            if is_async_command(cmd):
                await handle_async_command(cmd, args, state, console)
            else:
                handle_command(cmd, args, state, console)
            if not state.should_exit:
                render_separator(console)
            continue

        # チャットメッセージ送信
        logger.debug("Sending chat message: %d chars", len(text))
        state.add_turn("user", text)
        t0 = time.monotonic()
        render_user_message(console, text, clear_echo=True)
        try:
            response, shell_out_requests, ttft = await chat_stream(text, state, console)
        except Exception as e:
            error_detail = str(e) or type(e).__name__
            logger.error("Unexpected error in chat_stream: %s", error_detail)
            state.record_error()
            render_error(console, error_detail)
            continue
        elapsed = time.monotonic() - t0
        state.record_response_time(elapsed)
        # TTFT を debug モード時に stderr へ表示
        if ttft is not None:
            logger.debug("TTFT: %.3fs", ttft)
            sys.stderr.write(f"[TTFT: {ttft:.2f}s]\n")
            sys.stderr.flush()
            state.record_ttft(ttft)
        render_status_line(
            console, state.token_used, state.token_limit,
            elapsed, state.context_files, ttft_sec=ttft,
        )
        await _process_post_response(response, shell_out_requests, state, console)
        _save_checkpoint_if_needed(state)


def _write_user_panel(layout: SplitLayout, text: str) -> None:
    """ユーザー入力メッセージを枠付きで split レイアウトの出力エリアに表示"""
    from rich.panel import Panel
    from backend.free.cli.renderer import get_cli_theme
    theme = get_cli_theme()
    panel = Panel(
        text,
        border_style=theme.border,
        style=f"on {theme.bg_surface}",
        expand=True,
        padding=(0, 1),
    )
    layout.write_rich(panel)


def _render_split_welcome(state: SessionState, split_console) -> None:
    """split レイアウト用のウェルカム + モデル情報 + ヒントを書き込む。

    ウェルカム表示はマウント前に行い、マウント後に RichLog へフラッシュされる。
    """
    from backend.edition import current_edition
    from backend.free.cli.config_loader import _find_project_root
    from backend.free.cli.main import _build_model_info
    from backend.free.cli.renderer import (
        render_model_info, render_welcome, render_welcome_hint,
    )
    from backend.version import get_runtime_version
    render_welcome(
        split_console,
        version=get_runtime_version(),
        instance_name=state.instance_name,
        edition=current_edition().name.lower(),
    )
    _mi_models, _mi_ctx = _build_model_info(_find_project_root(), None)
    render_model_info(split_console, _mi_models, _mi_ctx)
    render_welcome_hint(split_console)


async def _handle_split_command_input(
    cmd: str,
    args: str,
    text: str,
    layout: SplitLayout,
    state: SessionState,
    split_console,
    original_console,
) -> None:
    """split レイアウトでコマンド入力を処理する。

    `/page` は TUI を suspend して新規イベントループで実行、その他は
    async/sync コマンドハンドラへディスパッチ。
    """
    logger.debug(
        "Command detected: %s, args=%r",
        cmd, args[:50] if args else "",
    )
    _write_user_panel(layout, text)
    if cmd == "/page":
        logger.debug("Suspending TUI for /page command")
        with layout.app.suspend():
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)
            try:
                _loop.run_until_complete(
                    handle_async_command(cmd, args, state, original_console),
                )
            finally:
                _loop.close()
                asyncio.set_event_loop(None)
        logger.debug("TUI resumed after /page command")
    elif is_async_command(cmd):
        await handle_async_command(cmd, args, state, split_console)
    else:
        handle_command(cmd, args, state, split_console)
    layout.append_output("\n")
    layout.update_status_idle()


def _maybe_run_post_response_terminal(
    layout: SplitLayout,
    original_console,
    response: str,
    shell_out_requests: list[str],
    state: SessionState,
    split_console,
) -> None:
    """応答に diff/shell があれば TUI suspend で対話処理を実行する。"""
    if response:
        from backend.free.constants import TRUNCATION_MARKER
        diffs_exist = "```diff" in response
        shell_outs_exist = bool(shell_out_requests)
        if diffs_exist or shell_outs_exist:
            _run_terminal_interactions(
                layout, original_console, response,
                shell_out_requests, state, diffs_exist,
            )
        elif TRUNCATION_MARKER in response:
            render_info(split_console, msg("cli.page_hint"))
        return
    if shell_out_requests:
        _run_terminal_interactions(
            layout, original_console, "",
            shell_out_requests, state, False,
        )


async def _handle_split_chat_input(
    text: str,
    layout: SplitLayout,
    state: SessionState,
    split_console,
    original_console,
) -> None:
    """split レイアウトでチャット入力を処理する。

    chat_stream を呼び出し、TTFT/エラー/diff/shell 後処理まで一通り行う。
    """
    _write_user_panel(layout, text)
    layout.update_status_thinking()
    logger.debug("Sending chat message: %d chars", len(text))
    state.add_turn("user", text)
    t0 = time.monotonic()
    layout.set_streaming(True)
    try:
        response, shell_out_requests, ttft = await chat_stream(
            text, state, split_console, split_layout=layout,
        )
    except Exception as e:  # noqa: BLE001 — 例外を握って UI を継続
        error_detail = str(e) or type(e).__name__
        logger.error("Unexpected error in chat_stream: %s", error_detail)
        state.record_error()
        layout.set_streaming(False)
        layout.append_output(f"Error: {error_detail}\n")
        layout.update_status_idle()
        return
    layout.set_streaming(False)
    state.record_response_time(time.monotonic() - t0)

    if ttft is not None:
        logger.debug("TTFT: %.3fs", ttft)
        import sys as _sys
        _sys.stderr.write(f"[TTFT: {ttft:.2f}s]\n")
        _sys.stderr.flush()
        state.record_ttft(ttft)

    layout.append_output("\n")
    layout.update_status(
        state.token_used, state.token_limit, state.context_files,
    )

    if response:
        state.add_turn("assistant", response)
    _maybe_run_post_response_terminal(
        layout, original_console, response, shell_out_requests,
        state, split_console,
    )
    _save_checkpoint_if_needed(state)


async def _main_loop_split(state: SessionState, console) -> None:
    """split レイアウトモードのメインループ

    textual App による画面分割。
    上部: 出力エリア（RichLog, スクロール可能）
    中央: 入力エリア（Input, 固定）
    下部: ステータスバー（Static, 固定）
    """
    from backend.version import get_runtime_version
    layout = SplitLayout(
        instance_name=state.instance_name,
        version=get_runtime_version(),
    )

    # split 用 Console（出力を RichLog にリダイレクト）
    split_console = layout.create_console()
    _render_split_welcome(state, split_console)
    layout.update_status(state.token_used, state.token_limit)

    original_console = console  # suspend 用に保持

    async def _main_logic():
        """App 内で実行されるメインロジック"""
        await layout.app._mounted_event.wait()
        while not state.should_exit and not layout._should_exit:
            try:
                text = await layout.get_input()
            except EOFError:
                finalize_session(state)
                break

            text = text.strip()
            if not text:
                continue

            cmd, args = parse_command(text)
            if cmd:
                await _handle_split_command_input(
                    cmd, args, text, layout, state,
                    split_console, original_console,
                )
                continue

            await _handle_split_chat_input(
                text, layout, state, split_console, original_console,
            )

        finalize_session(state)
        layout.app.exit()

    # メインロジックをバックグラウンドタスクとして起動し、App を実行
    task = asyncio.create_task(_main_logic())
    try:
        await layout.app.run_async()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _run_terminal_interactions(
    layout: SplitLayout,
    original_console,
    response: str,
    shell_out_requests: list[str],
    state: SessionState,
    has_diffs: bool,
) -> None:
    """diff 確認・シェルアウト処理を TUI を一時解除して実行

    app.suspend() でターミナルを一時返却し、input() でユーザー確認を行う。
    旧 prompt_toolkit の run_in_terminal + asyncio デッドロック問題を解消。
    """
    try:
        logger.debug("Suspending TUI for terminal interactions (diffs=%s, shell_outs=%d)",
                      has_diffs, len(shell_out_requests))
        with layout.app.suspend():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    _terminal_interactions_async(
                        original_console, response, shell_out_requests, state, has_diffs,
                    ),
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        logger.debug("TUI resumed after terminal interactions")
    except Exception as e:
        logger.warning("Terminal interaction failed: %s", e)
        # フリーズ回避: diff/shell_out をスキップしてループを継続
        layout.append_output(f"\n(diff/shell_out processing skipped: {e})\n")


async def _terminal_interactions_async(
    console,
    response: str,
    shell_out_requests: list[str],
    state: SessionState,
    has_diffs: bool,
) -> None:
    """ターミナル対話処理（diff 確認・シェルアウト）

    split モードで TUI を suspend した状態で呼ばれる。
    diff 適用と shell_out の処理は _process_post_response と共通ロジック。
    ただし state.add_turn は呼び出し元で済んでいるため skip_turn=True。
    """
    if has_diffs:
        from backend.free.cli.diff_applier import process_response_diffs
        applied = await process_response_diffs(response, console)
        if applied:
            logger.debug("Applied diffs to %d file(s): %s", len(applied), applied)

    if shell_out_requests:
        await _execute_shell_outs(shell_out_requests, console)
