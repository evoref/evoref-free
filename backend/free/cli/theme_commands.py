"""/theme 対話コマンド: テーマの一覧表示・切替・詳細確認"""

from __future__ import annotations

import httpx

from backend.free.cli.cli_theme import CLITheme
from backend.free.cli.renderer import (
    render_error,
    render_info,
    render_table,
    set_cli_theme,
)
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("cli.theme")

_TIMEOUT = 10.0


def _api_get(backend_url: str, path: str, console) -> httpx.Response | None:
    """GET リクエスト。接続・HTTPエラー時はメッセージ表示済み + None 返却"""
    try:
        resp = httpx.get(f"{backend_url}{path}", timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return None
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return None


def _api_post(
    backend_url: str, path: str, json_data: dict, console,
) -> httpx.Response | None:
    """POST リクエスト。接続・HTTPエラー時はメッセージ表示済み + None 返却"""
    try:
        resp = httpx.post(f"{backend_url}{path}", json=json_data, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return None
    except httpx.HTTPStatusError as e:
        render_error(console, msg("cli.api_error", code=e.response.status_code))
        return None


async def handle_theme_command(
    args: str,
    backend_url: str,
    console,
    state,
) -> int:
    """/theme [action] [args] のディスパッチャー"""
    parts = args.strip().split(None, 1)
    action = parts[0].lower() if parts else "status"
    sub_args = parts[1].strip() if len(parts) > 1 else ""

    if action == "status":
        return _theme_status(backend_url, console)
    elif action == "list":
        return _theme_list(backend_url, console)
    elif action == "activate":
        return await _theme_activate(backend_url, console, state, sub_args)
    elif action == "info":
        return _theme_info(backend_url, console, sub_args)
    elif action == "color-mode":
        return await _theme_color_mode(backend_url, console, state, sub_args)
    else:
        render_error(console, msg("cli.theme_unknown_action", action=action))
        return 1


def _theme_status(backend_url: str, console) -> int:
    """現在のテーマ情報を表示"""
    resp = _api_get(backend_url, "/api/themes", console)
    if resp is None:
        return 1

    data = resp.json()
    active_id = data.get("active_theme_id", "")
    color_mode = data.get("color_mode", "dark")

    if not active_id:
        render_info(console, msg("cli.theme_current_none"))
        render_info(console, msg("cli.theme_color_mode", mode=color_mode))
        return 0

    # アクティブテーマの名前と CLI 対応を取得
    active_name = active_id
    has_cli = False
    for t in data.get("themes", []):
        if t.get("theme_id") == active_id:
            active_name = t.get("name", active_id)
            has_cli = t.get("has_cli_theme", False)
            break

    cli_mark = "✓" if has_cli else "✗"
    render_info(console, msg("cli.theme_current", id=active_id, name=active_name))
    render_info(console, msg("cli.theme_cli_support", status=cli_mark))
    render_info(console, msg("cli.theme_color_mode", mode=color_mode))
    return 0


def _theme_list(backend_url: str, console) -> int:
    """テーマ一覧をテーブル形式で表示"""
    resp = _api_get(backend_url, "/api/themes", console)
    if resp is None:
        return 1

    data = resp.json()
    themes = data.get("themes", [])

    if not themes:
        render_info(console, msg("cli.theme_empty"))
        return 0

    rows = []
    for t in themes:
        rows.append({
            "ID": t.get("theme_id", ""),
            msg("cli.theme_col_name"): t.get("name", ""),
            msg("cli.theme_col_version"): t.get("version", ""),
            msg("cli.theme_col_cli"): "✓" if t.get("has_cli_theme") else "✗",
            msg("cli.theme_col_status"): msg("cli.theme_col_active") if t.get("active") else "",
        })

    headers = [
        "ID",
        msg("cli.theme_col_name"),
        msg("cli.theme_col_version"),
        msg("cli.theme_col_cli"),
        msg("cli.theme_col_status"),
    ]
    render_table(console, rows, headers)
    return 0


def _build_cli_theme_from_result(result: dict) -> CLITheme:
    """activate API レスポンスから CLITheme を構築する（純粋関数）"""
    cli_theme_data = result.get("cli_theme")
    if cli_theme_data:
        return CLITheme.from_dict(cli_theme_data)
    return CLITheme.default()


async def _theme_activate(
    backend_url: str, console, state, theme_id: str,
) -> int:
    """テーマ切替 + CLITheme リロード"""
    if not theme_id:
        render_error(console, msg("cli.theme_activate_no_id"))
        return 1

    try:
        resp = httpx.post(
            f"{backend_url}/api/themes/activate",
            json={"theme_id": theme_id},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            render_error(console, msg("cli.theme_not_found", id=theme_id))
        else:
            render_error(console, msg("cli.api_error", code=e.response.status_code))
        return 1

    result = resp.json()

    # CLITheme を更新
    new_theme = _build_cli_theme_from_result(result)
    set_cli_theme(new_theme)
    state.cli_theme = new_theme

    # テーマ名は activate レスポンスの name フィールドから取得
    display_name = result.get("name", theme_id)

    render_info(console, msg("cli.theme_activated", id=theme_id, name=display_name))
    return 0


def _theme_info(backend_url: str, console, theme_id: str) -> int:
    """テーマ詳細情報を表示"""
    if not theme_id:
        render_error(console, msg("cli.theme_activate_no_id"))
        return 1

    resp = _api_get(backend_url, "/api/themes", console)
    if resp is None:
        return 1

    data = resp.json()
    theme = None
    for t in data.get("themes", []):
        if t.get("theme_id") == theme_id:
            theme = t
            break

    if theme is None:
        render_error(console, msg("cli.theme_not_found", id=theme_id))
        return 1

    render_info(console, msg("cli.theme_info_header", id=theme['theme_id'], name=theme.get('name', '')))
    render_info(console, f"  {msg('cli.theme_col_version')}: {theme.get('version', '')}")
    if theme.get("author"):
        render_info(console, f"  {msg('cli.theme_info_author')}: {theme['author']}")
    if theme.get("description"):
        render_info(console, f"  {msg('cli.theme_info_description')}: {theme['description']}")
    render_info(console, f"  {msg('cli.theme_info_builtin')}: {'✓' if theme.get('builtin') else '✗'}")
    render_info(console, f"  {msg('cli.theme_info_trusted')}: {'✓' if theme.get('trusted') else '✗'}")
    render_info(console, f"  {msg('cli.theme_info_cli')}: {'✓' if theme.get('has_cli_theme') else '✗'}")
    render_info(console, f"  {msg('cli.theme_info_cli_modules')}: {theme.get('cli_module_count', 0)}")
    render_info(console, f"  {msg('cli.theme_info_gui_components')}: {theme.get('component_count', 0)}")
    return 0


async def _theme_color_mode(
    backend_url: str, console, state, mode: str,
) -> int:
    """カラーモード切替"""
    if not mode:
        # 引数なし: 現在のモードを表示
        resp = _api_get(backend_url, "/api/themes", console)
        if resp is None:
            return 1

        data = resp.json()
        current_mode = data.get("color_mode", "dark")
        render_info(console, msg("cli.theme_color_mode_current", mode=current_mode))
        return 0

    if mode not in ("dark", "light"):
        render_error(console, msg("cli.theme_invalid_color_mode"))
        return 1

    # 現在のアクティブテーマ ID を取得（ベストエフォート）
    active_id = ""
    try:
        list_resp = httpx.get(f"{backend_url}/api/themes", timeout=_TIMEOUT)
        if list_resp.status_code == 200:
            active_id = list_resp.json().get("active_theme_id", "")
    except Exception:
        pass

    if not active_id:
        render_error(console, msg("cli.theme_color_mode_no_active"))
        return 1

    resp = _api_post(
        backend_url, "/api/themes/activate",
        {"theme_id": active_id, "color_mode": mode}, console,
    )
    if resp is None:
        return 1

    render_info(console, msg("cli.theme_color_mode_changed", mode=mode))
    return 0
