"""CLI /export コマンド: 会話内容をファイル形式に書出し

Usage:
    /export <path>              — 直前の応答を指定形式で書出し
    /export <path> --all        — 会話全体を書出し
    /export <path> --last <n>   — 直近 n ターンを書出し
"""

from __future__ import annotations

from pathlib import Path

from backend.export import get_writer_registry
from backend.export.base import ExportError
from backend.export.content_converter import ContentConverter
from backend.free.cli.command_parser import CommandResult, SessionState
from backend.free.cli.renderer import render_error, render_info
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.export_commands")


def _extract_content_from_turns(
    turns: list[dict],
    mode: str,
    last_n: int | None,
) -> str:
    """会話ターンからエクスポート対象の Markdown テキストを生成"""
    target_turns = turns

    if mode == "last" and last_n is not None:
        # 直近 n ターン
        target_turns = turns[-last_n:] if last_n < len(turns) else turns
    elif mode == "last_response":
        # 直前のアシスタント応答のみ
        for turn in reversed(turns):
            if turn.get("role") == "assistant":
                return turn.get("content", "")
        return ""

    # 全ターンまたは指定ターンを結合
    parts: list[str] = []
    for turn in target_turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            parts.append(f"**User**: {content}")
        elif role == "assistant":
            parts.append(content)
    return "\n\n".join(parts)


def _cmd_export(args: str, state: SessionState, console) -> CommandResult:
    """/export コマンドの実行ロジック"""
    # 引数解析
    parts = args.strip().split()
    if not parts:
        render_error(console, msg("cli.export_file_no_path"))
        return CommandResult()

    path_str = parts[0]
    flags = parts[1:]

    # モード判定
    mode = "last_response"  # デフォルト: 直前の応答
    last_n: int | None = None

    if "--all" in flags:
        mode = "all"
    elif "--last" in flags:
        mode = "last"
        idx = flags.index("--last")
        if idx + 1 < len(flags):
            try:
                last_n = int(flags[idx + 1])
            except ValueError:
                render_error(console, msg("cli.export_file_invalid_last_n"))
                return CommandResult()
        else:
            render_error(console, msg("cli.export_file_invalid_last_n"))
            return CommandResult()

    # 会話内容がない場合
    if not state.turns:
        render_error(console, msg("cli.export_file_no_content"))
        return CommandResult()

    # パス解決
    path = Path(path_str).resolve()
    ext = path.suffix.lower()

    # レジストリ確認
    import backend.free.export  # noqa: F401  Writer 登録を確実に
    registry = get_writer_registry()

    if not registry.is_supported(ext):
        supported = ", ".join(sorted(registry.supported_extensions()))
        render_error(console, msg("cli.export_file_unsupported", ext=ext, supported=supported))
        return CommandResult()

    # コンテンツ生成
    markdown = _extract_content_from_turns(state.turns, mode, last_n)
    if not markdown.strip():
        render_error(console, msg("cli.export_file_no_content"))
        return CommandResult()

    content = ContentConverter.from_markdown(markdown)

    # 書出し実行
    try:
        result = registry.write(content, path)
    except ExportError as e:
        if e.code == "missing_library":
            lib = e.args[0] if e.args else ""
            render_error(console, msg("cli.file_missing_library", library=lib))
        else:
            render_error(console, msg("cli.export_file_write_error", detail=str(e)))
        return CommandResult()

    size_kb = result.size_bytes / 1024
    render_info(console, msg("cli.export_file_done", path=str(path), size=f"{size_kb:.1f}"))
    logger.debug("/export: wrote %s (%d bytes, mode=%s)", path, result.size_bytes, mode)
    return CommandResult()
