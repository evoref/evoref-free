"""CLI テーマローダー

CLI 起動時にアクティブテーマの cli-theme.json を読み込む。
バックエンド API → ローカルファイル → デフォルト の順にフォールバックする。
"""

from __future__ import annotations

from pathlib import Path

import httpx

from backend.free.cli.cli_theme import CLITheme
from backend.log_config import get_logger

logger = get_logger("cli.theme_loader")


async def load_cli_theme(
    backend_url: str,
    themes_dir: Path | None = None,
) -> CLITheme:
    """アクティブテーマの CLI テーマを読み込む。

    1. GET /api/themes/active-cli でバックエンドから取得（優先）
    2. 失敗時: themes_dir から config.yaml の active テーマの cli-theme.json を直接読込み
    3. 全失敗時: CLITheme.default()

    Args:
        backend_url: バックエンド URL (例: "http://localhost:8000")
        themes_dir: テーマディレクトリのパス（フォールバック用）
    """
    # 1. バックエンド API 経由
    theme = await _load_from_api(backend_url)
    if theme is not None:
        return theme

    # 2. ローカルファイルからフォールバック
    if themes_dir is not None:
        theme = _load_from_local(themes_dir)
        if theme is not None:
            return theme

    # 3. デフォルト
    logger.debug("Using default CLI theme")
    return CLITheme.default()


async def _load_from_api(backend_url: str) -> CLITheme | None:
    """バックエンド API からアクティブテーマの CLI テーマを取得"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{backend_url}/api/themes/active-cli")
            if resp.status_code == 200:
                data = resp.json()
                cli_theme_data = data.get("cli_theme", {})
                logger.debug(
                    "Loaded CLI theme from API: theme_id=%s",
                    data.get("theme_id"),
                )
                return CLITheme.from_dict(cli_theme_data)
    except Exception as e:
        logger.debug("Failed to load CLI theme from API: %s", e)
    return None


def _get_active_theme_id(themes_dir: Path) -> str:
    """config.yaml からアクティブテーマ ID を取得する。未設定時は空文字列"""
    import yaml

    config_path = themes_dir.parent.parent / "config.yaml"
    if not config_path.exists():
        return ""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("theme", {}).get("active", "")


def _get_cli_theme_filename(theme_dir: Path) -> str:
    """theme.json から cli_theme フィールドを取得する。未設定時は 'cli-theme.json'"""
    import json

    theme_json_path = theme_dir / "theme.json"
    if not theme_json_path.exists():
        return "cli-theme.json"
    meta = json.loads(theme_json_path.read_text(encoding="utf-8"))
    return meta.get("cli_theme", "cli-theme.json")


def _load_from_local(themes_dir: Path) -> CLITheme | None:
    """ローカルのテーマディレクトリから cli-theme.json を読み込む。

    config.yaml の active テーマを参照し、そのテーマディレクトリ内の
    cli-theme.json を読み込む。
    """
    try:
        active_theme_id = _get_active_theme_id(themes_dir)
        if not active_theme_id:
            logger.debug("No active theme configured; using default CLI theme")
            return None

        theme_dir = themes_dir / active_theme_id
        if not theme_dir.is_dir():
            logger.debug("Theme directory not found: %s", active_theme_id)
            return None

        cli_theme_filename = _get_cli_theme_filename(theme_dir)
        cli_theme_path = theme_dir / cli_theme_filename
        if not cli_theme_path.exists():
            logger.debug("cli-theme.json not found: %s", cli_theme_path)
            return None

        theme = CLITheme.from_file(cli_theme_path)
        logger.debug(
            "Loaded CLI theme from local: theme_id=%s, path=%s",
            active_theme_id,
            cli_theme_path,
        )
        return theme
    except Exception as e:
        logger.debug("Failed to load CLI theme from local: %s", e)
        return None
