"""CLITheme データクラス

cli-theme.json から読み込んだ CLI テーマ設定を保持する。
テーマファイルが存在しない・不正な場合はデフォルト値にフォールバックする。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("cli.theme")


@dataclass
class CLITheme:
    """cli-theme.json から読み込んだ CLI テーマ設定"""

    # === Prompt ===
    prompt_marker: str = "❯ "
    step_indent: int = 2

    # === Colors (rich style strings) ===
    accent: str = "#a78bfa"
    accent_light: str = "#c4b5fd"
    accent_dark: str = "#7c3aed"
    bg_primary: str = "#111118"
    bg_surface: str = "#1a1a28"
    bg_header: str = "#252540"
    border: str = "#363658"
    text_muted: str = "#8585ad"
    text_glow: str = "#e0dffe"
    step_done: str = "#6bcb77"
    step_failed: str = "#ff6b6b"
    step_running: str = "#8585ad"
    error: str = "bold #ff6b6b"
    warning: str = "#ffd93d"
    hint: str = "#a78bfa"
    info: str = "#8585ad"
    success: str = "#6bcb77"
    prompt_style: str = "bold #a78bfa"
    status_line: str = "#8585ad"
    separator: str = "#363658"
    token_low: str = "#6bcb77"
    token_mid: str = "#ffd93d"
    token_high: str = "bold #ff6b6b"
    code_theme: str = "dracula"

    # === Status Line ===
    show_status_line: bool = True
    status_format: str = "{used:,} / {limit:,} tokens · {elapsed}s"

    # === Welcome ===
    show_welcome: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> CLITheme:
        """cli-theme.json の辞書からパースする。

        ネスト構造（prompt/colors/status/welcome）をフラットなフィールドに展開する。
        不足フィールドはデフォルト値を使用する。
        型が不正な値は無視してデフォルト値にフォールバックする。

        Args:
            data: cli-theme.json の内容、または API レスポンスの cli_theme 辞書
        """
        kwargs: dict = {}

        # prompt セクション
        prompt = data.get("prompt", {})
        if isinstance(prompt, dict):
            if "marker" in prompt and isinstance(prompt["marker"], str):
                kwargs["prompt_marker"] = prompt["marker"]
            if "step_indent" in prompt and isinstance(prompt["step_indent"], int):
                kwargs["step_indent"] = prompt["step_indent"]

        # colors セクション
        colors = data.get("colors", {})
        if isinstance(colors, dict):
            # colors.prompt → prompt_style にマッピング
            color_field_map = {
                "prompt": "prompt_style",
            }
            # CLITheme のカラーフィールド一覧（デフォルトインスタンスから取得せず、直接列挙）
            color_fields = {
                "accent", "accent_light", "accent_dark",
                "bg_primary", "bg_surface", "bg_header",
                "border", "text_muted", "text_glow",
                "step_done", "step_failed", "step_running",
                "error", "warning", "hint", "info", "success",
                "status_line", "separator",
                "token_low", "token_mid", "token_high",
                "code_theme",
            }
            for json_key, value in colors.items():
                if not isinstance(value, str):
                    logger.warning(
                        "Ignoring non-string color value: %s=%r", json_key, value,
                    )
                    continue
                if json_key in color_field_map:
                    kwargs[color_field_map[json_key]] = value
                elif json_key in color_fields:
                    kwargs[json_key] = value

        # status セクション
        status = data.get("status", {})
        if isinstance(status, dict):
            if "show" in status and isinstance(status["show"], bool):
                kwargs["show_status_line"] = status["show"]
            if "format" in status and isinstance(status["format"], str):
                kwargs["status_format"] = status["format"]

        # welcome セクション
        welcome = data.get("welcome", {})
        if isinstance(welcome, dict):
            if "show" in welcome and isinstance(welcome["show"], bool):
                kwargs["show_welcome"] = welcome["show"]

        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Path) -> CLITheme:
        """cli-theme.json ファイルから読み込む。

        ファイル不存在・不正 JSON の場合はデフォルト値にフォールバックする。
        """
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            return cls.from_dict(data)
        except FileNotFoundError:
            logger.debug("CLI theme file not found, using defaults: %s", path)
            return cls.default()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load CLI theme file, using defaults: %s", e)
            return cls.default()

    @classmethod
    def default(cls) -> CLITheme:
        """デフォルトテーマを返す"""
        return cls()
