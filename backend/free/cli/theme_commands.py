"""/theme 対話コマンド: テーマの一覧表示・切替・詳細確認・インストール・削除

``theme_install`` / ``theme_uninstall`` は ``evoref theme install|uninstall`` からも使う。
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import httpx

from backend.free.cli.backend_headers import backend_headers
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
        resp = httpx.get(f"{backend_url}{path}", headers=backend_headers(), timeout=_TIMEOUT)
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
        resp = httpx.post(f"{backend_url}{path}", headers=backend_headers(), json=json_data, timeout=_TIMEOUT)
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
    elif action == "install":
        return theme_install(backend_url, console, sub_args)
    elif action == "uninstall":
        return theme_uninstall(backend_url, console, sub_args)
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
            headers=backend_headers(),
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
    backend_url: str, console, state, mode: str,  # noqa: ARG001
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
        list_resp = httpx.get(f"{backend_url}/api/themes", headers=backend_headers(), timeout=_TIMEOUT)
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


#: URL 指定は backend がダウンロード (最大 30 秒) してから展開する。
_INSTALL_TIMEOUT = 120.0


def _response_detail(resp: httpx.Response) -> str:
    """非 2xx 応答の理由 (``detail.message`` / 文字列 / 検証エラーの一覧)。"""
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return f"HTTP {resp.status_code}"
    if isinstance(detail, dict) and detail.get("message"):
        return str(detail["message"])
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, list) and detail:
        return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return f"HTTP {resp.status_code}"


def _is_url(target: str) -> bool:
    return target.lower().startswith(("http://", "https://"))


def theme_install(backend_url: str, console, target: str) -> int:
    """ZIP ファイルまたは URL からテーマをインストールする。

    インストールしたテーマは常に未信頼 (信頼は ``evoref theme trust`` だけが付ける、
    docs/c_11 §1)。JS コンポーネントと CLI モジュールは信頼するまで無効のまま。
    """
    target = target.strip()
    if not target:
        render_error(console, msg("cli.theme_install_usage"))
        return 1
    try:
        if _is_url(target):
            resp = httpx.post(
                f"{backend_url}/api/themes/install-url",
                headers=backend_headers(), json={"url": target}, timeout=_INSTALL_TIMEOUT,
            )
        else:
            path = Path(target).expanduser()
            if not path.is_file():
                render_error(console, msg("cli.theme_install_not_found", path=target))
                return 1
            with path.open("rb") as fh:
                resp = httpx.post(
                    f"{backend_url}/api/themes/install",
                    headers=backend_headers(),
                    files={"file": (path.name, fh, "application/zip")},
                    timeout=_INSTALL_TIMEOUT,
                )
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPError as e:
        render_error(console, msg("cli.theme_install_failed", detail=str(e)))
        return 1

    if resp.status_code == 409:
        render_error(console, msg("cli.theme_install_exists", detail=_response_detail(resp)))
        return 1
    if resp.status_code not in (200, 201):
        render_error(console, msg("cli.theme_install_failed", detail=_response_detail(resp)))
        return 1

    result = resp.json()
    theme_id = result.get("theme_id", "")
    logger.info("Theme installed from CLI: id=%s", theme_id)
    render_info(console, msg(
        "cli.theme_install_done",
        id=theme_id, name=result.get("name", theme_id), version=result.get("version", ""),
    ))
    apis = (result.get("widget_manifest") or {}).get("required_apis") or []
    if apis:
        render_info(console, msg("cli.theme_install_required_apis", apis=", ".join(apis)))
    render_error(console, msg("cli.theme_install_untrusted_hint", id=theme_id), level="hint")
    return 0


def theme_uninstall(backend_url: str, console, theme_id: str) -> int:
    """テーマを削除する (信頼リストからも外れる。有効なテーマなら残りのテーマへ切り替わる)。"""
    theme_id = theme_id.strip()
    if not theme_id:
        render_error(console, msg("cli.theme_uninstall_usage"))
        return 1
    try:
        resp = httpx.delete(
            f"{backend_url}/api/themes/{quote(theme_id, safe='')}",
            headers=backend_headers(), timeout=_TIMEOUT,
        )
    except httpx.ConnectError:
        render_error(console, msg("cli.backend_not_running"))
        return 1
    except httpx.HTTPError as e:
        render_error(console, msg("cli.theme_uninstall_failed", detail=str(e)))
        return 1

    if resp.status_code == 404:
        render_error(console, msg("cli.theme_not_found", id=theme_id))
        return 1
    if resp.status_code not in (200, 204):
        render_error(console, msg("cli.theme_uninstall_failed", detail=_response_detail(resp)))
        return 1
    logger.info("Theme uninstalled from CLI: id=%s", theme_id)
    render_info(console, msg("cli.theme_uninstall_done", id=theme_id))
    return 0
