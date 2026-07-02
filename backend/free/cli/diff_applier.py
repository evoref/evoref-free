"""CLI diff 適用: LLM 応答からの diff 検出・確認・ファイル適用

ビジネスロジック（diff パース・適用）は backend.free.services.diff_service に委譲する。
本モジュールは CLI 固有の対話処理（ユーザー確認プロンプト・エディタ編集）を提供する。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from backend.free.services.diff_service import (
    DiffBlock,
    DiffServiceError,
    Hunk,
    apply_unified_diff,
    extract_diffs,
)
from backend.log_config import get_logger

logger = get_logger("cli.diff_applier")

__all__ = [
    "DiffBlock",
    "DiffServiceError",
    "Hunk",
    "apply_unified_diff",
    "edit_diff_with_editor",
    "extract_diffs",
    "process_response_diffs",
    "resolve_editor_command",
]


async def process_response_diffs(
    response: str,
    console,
    *,
    non_interactive: bool = False,
) -> list[str]:
    """LLM 応答から diff を検出し、ユーザー確認後にファイルに適用

    Returns:
        適用されたファイルパスのリスト
    """
    import asyncio

    from backend.free.cli.renderer import (
        render_diff,
        render_diff_applied,
        render_diff_prompt_path,
        render_diff_result,
        render_info,
    )
    from backend.i18n_helper import msg

    diffs = extract_diffs(response)
    if not diffs:
        return []

    logger.debug("Found %d diff block(s) in response", len(diffs))
    applied_files: list[str] = []

    for i, diff_block in enumerate(diffs):
        if non_interactive:
            logger.debug("Non-interactive mode: skipping diff %d", i + 1)
            continue

        # ファイルパスの解決
        file_path = diff_block.file_path
        if not file_path:
            # ユーザーにファイルパスを尋ねる
            file_path = await asyncio.to_thread(render_diff_prompt_path, console)
            if not file_path:
                render_info(console, msg("cli.diff_skipped"))
                continue

        # ファイル存在チェック
        p = Path(file_path)
        if not p.exists():
            render_info(
                console,
                msg("cli.diff_file_not_found", path=file_path),
            )
            continue

        # diff 表示
        render_diff(console, diff_block.raw)

        # 確認プロンプト
        render_info(console, msg("cli.diff_target_file", path=file_path))
        answer = await asyncio.to_thread(_prompt_apply, console)

        if answer == "y":
            if not diff_block.has_hunks:
                render_info(console, msg("cli.diff_no_hunks"))
                continue
            success, message = apply_unified_diff(file_path, diff_block.raw)
            render_diff_result(console, success, message)
            if success:
                applied_files.append(file_path)
                render_diff_applied(console, file_path)
        elif answer == "e":
            applied = await _apply_edited_diff(
                diff_block.raw, file_path, console,
            )
            if applied:
                applied_files.append(file_path)
        else:
            render_info(console, msg("cli.diff_skipped"))

    return applied_files


async def _apply_edited_diff(
    raw_diff: str, file_path: str, console,
) -> bool:
    """エディタで diff を編集してから適用する

    Returns:
        適用に成功した場合 True
    """
    import asyncio

    from backend.free.cli.renderer import (
        render_diff,
        render_diff_applied,
        render_diff_result,
        render_error,
        render_info,
    )
    from backend.i18n_helper import msg

    editor_cmd = resolve_editor_command()
    render_info(
        console,
        msg("cli.diff_editor_launching", editor=" ".join(editor_cmd)),
    )

    edited = await asyncio.to_thread(edit_diff_with_editor, raw_diff, editor_cmd)
    if edited is None:
        render_error(console, msg("cli.diff_editor_failed"), level="warning")
        return False

    if edited.strip() == "":
        render_info(console, msg("cli.diff_edit_empty"))
        return False

    # 編集後 diff を再表示
    render_diff(console, edited)

    edited_block = DiffBlock(raw=edited, file_path=None)
    if not edited_block.has_hunks:
        render_error(console, msg("cli.diff_no_hunks"), level="warning")
        return False

    success, message = apply_unified_diff(file_path, edited)
    render_diff_result(console, success, message)
    if success:
        render_diff_applied(console, file_path)
        return True
    return False


def resolve_editor_command() -> list[str]:
    """エディタコマンドを解決する

    優先順位:
        1. $VISUAL 環境変数
        2. $EDITOR 環境変数
        3. OS 既定 (Windows: notepad、それ以外: vi)

    値は shlex.split でトークン化する（shell インジェクション防止のため）。
    """
    for var in ("VISUAL", "EDITOR"):
        value = os.environ.get(var, "").strip()
        if not value:
            continue
        try:
            tokens = shlex.split(value, posix=(os.name != "nt"))
        except ValueError as e:
            logger.warning(
                "Failed to parse %s=%r: %s — falling back to next option",
                var, value, e,
            )
            continue
        if tokens:
            return tokens

    if os.name == "nt":
        return ["notepad"]
    return ["vi"]


def edit_diff_with_editor(
    raw_diff: str, editor_cmd: list[str] | None = None,
) -> str | None:
    """unified diff をエディタで編集し、編集後のテキストを返す

    Args:
        raw_diff: 編集対象の diff テキスト
        editor_cmd: エディタコマンド（省略時は resolve_editor_command で解決）

    Returns:
        編集後の diff テキスト。エディタ起動失敗・非ゼロ終了時は None。

    Note:
        - 一時ファイルは ``.diff`` 拡張子で作成し、finally で削除する
        - subprocess は shell=False で起動（コマンドインジェクション防止）
        - エディタプロセスには親の stdin/stdout/stderr を継承させる
    """
    cmd = editor_cmd if editor_cmd is not None else resolve_editor_command()

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".diff",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as tmp:
            tmp.write(raw_diff)
            tmp_path = Path(tmp.name)

        try:
            result = subprocess.run(
                [*cmd, str(tmp_path)],
                shell=False,
                check=False,
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error("Failed to launch editor %r: %s", cmd, e)
            return None

        if result.returncode != 0:
            logger.warning(
                "Editor %r exited with non-zero code: %d",
                cmd[0], result.returncode,
            )
            return None

        try:
            return tmp_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Failed to read edited diff from %s: %s", tmp_path, e)
            return None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Failed to remove temp file %s: %s", tmp_path, e)


def _prompt_apply(console) -> str:  # noqa: ARG001
    """適用確認プロンプトを表示し、ユーザー入力を返す

    Returns:
        'y', 'n', or 'e'
    """
    from backend.i18n_helper import msg

    try:
        answer = input(msg("cli.diff_confirm_prompt")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"

    if answer in ("y", "yes"):
        return "y"
    if answer in ("e", "edit"):
        return "e"
    return "n"
