"""CLI 設定読込みユーティリティ"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.error_handlers import E6002
from backend.log_config import get_logger

logger = get_logger("cli.config_loader")


def _setup_encoding() -> bool:
    """UTF-8 を強制（Windows 対応）

    Returns:
        True: 成功、False: エンコード設定失敗
    """
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stdin.encoding and sys.stdin.encoding.lower() != "utf-8":
            sys.stdin.reconfigure(encoding="utf-8")
        return True
    except (UnicodeEncodeError, UnicodeDecodeError, OSError) as e:
        logger.error("E6002 UTF-8 encoding error: %s", e)
        print(f"[{E6002}] UTF-8 encoding setup failed: {e}", file=sys.stderr)
        return False


def _find_project_root() -> Path:
    """プロジェクトルートを探索（config.yaml が存在するディレクトリ）

    カレントディレクトリを優先し、見つからない場合はパッケージ配置元にフォールバック。
    """
    cwd = Path.cwd()
    if (cwd / "config.yaml").exists():
        return cwd
    # フォールバック: backend/cli/main.py → backend/cli → backend → project_root
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "config.yaml").exists():
        return candidate
    return cwd


def _load_yaml_section(project_root: Path, section: str) -> dict:
    """config.yaml から指定セクションを読込み"""
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get(section, {})
    except (OSError, ValueError, KeyError) as e:
        logger.debug("Failed to load %s config: %s", section, e)
        return {}


def _load_history_config(project_root: Path) -> dict:
    """config.yaml から history セクションを読込み"""
    return _load_yaml_section(project_root, "history")


def _load_cli_config(project_root: Path) -> dict:
    """config.yaml から cli セクションを読込み"""
    return _load_yaml_section(project_root, "cli")
