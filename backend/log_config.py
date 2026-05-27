"""ログ設定

develop モードを 3 段階 (``debug`` / ``investigate`` / ``evolve``)
に再設計。``setup_logging(develop_level, project_root)`` は
``develop_level`` から backend.log のレベルとコンソール出力 ON/OFF を導出
する。``debug.enabled`` 設定キーは廃止。

| develop_level | backend.log | console (stderr) |
|---------------|-------------|------------------|
| off           | INFO+       | OFF              |
| debug         | DEBUG+      | ON               |
| investigate   | DEBUG+      | ON               |
| evolve        | DEBUG+      | OFF (loop 実行を妨げない) |
"""

from __future__ import annotations

import io
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

# develop モードレベル (SSOT)。``app_state.DevelopLevel`` と
# 同じ Literal だが、``log_config`` は循環 import を避けるため独立で型を
# 持つ (``app_state.py`` は ``fastapi.Request`` を import するため
# ``log_config`` 側へ落とさない)。``app_state.DevelopLevel`` は本モジュール
# を re-export する形で同一型を共有する。
DevelopLevel = Literal["off", "debug", "investigate", "evolve"]


def _make_utf8_stream() -> io.TextIOWrapper:
    """stderr を UTF-8 でラップしたストリームを返す（Windows cp932 文字化け対策）"""
    try:
        return open(
            sys.stderr.fileno(), "w",
            encoding="utf-8", errors="replace", closefd=False,
        )
    except (AttributeError, OSError):
        return sys.stderr  # type: ignore[return-value]


def _has_rotating_file_handler(lg: logging.Logger, target_path: Path) -> bool:
    """指定パスを baseFilename とする RotatingFileHandler が既に登録済みかを判定する"""
    return any(
        isinstance(h, RotatingFileHandler)
        and Path(getattr(h, "baseFilename", "")).resolve() == target_path
        for h in lg.handlers
    )


def _has_plain_stream_handler(lg: logging.Logger) -> bool:
    """RotatingFileHandler 以外の StreamHandler（コンソール等）が登録済みかを判定する"""
    return any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, RotatingFileHandler)
        for h in lg.handlers
    )


def setup_logging(
    develop_level: DevelopLevel = "off",
    project_root: Path | None = None,
) -> None:
    """``develop_level`` に基づきログハンドラを構築する

    Args:
        develop_level: ``"off"`` は通常起動 (backend.log INFO+ 固定、コン
            ソール OFF)、``"debug"`` / ``"investigate"`` は backend.log
            DEBUG+ + コンソール ON、``"evolve"`` は backend.log DEBUG+ だが
            コンソール OFF (loop 実行を妨げないため)。
        project_root: ログディレクトリ (`local/logs/`) のベース。
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    # develop_level に応じた level / コンソール出力 ON/OFF
    # ``debug.enabled`` 設定キーは廃止 (config.yaml レベルで完全削除)。
    if develop_level == "off":
        level = logging.INFO          # 通常時: 起動完了等の動作確認 INFO ログを出す
        console_enabled = False
    else:
        level = logging.DEBUG
        # evolve は loop 自己学習向けで console を出すと loop 実行を妨げる
        console_enabled = develop_level in ("debug", "investigate")

    # ログディレクトリ作成
    log_dir = project_root / "local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ログフォーマット
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # ルートロガー設定
    root = logging.getLogger("backend")
    root.setLevel(level)

    # backend.log ハンドラ（複数回呼び出しでも重複登録しない）
    backend_log_path = (log_dir / "backend.log").resolve()
    if not _has_rotating_file_handler(root, backend_log_path):
        backend_handler = RotatingFileHandler(
            log_dir / "backend.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        backend_handler.setFormatter(formatter)
        root.addHandler(backend_handler)

    # コンソール出力（debug / investigate のみ）— UTF-8 強制
    # 非 RotatingFileHandler の StreamHandler が既に登録済みなら追加しない
    if console_enabled and not _has_plain_stream_handler(root):
        console = logging.StreamHandler(stream=_make_utf8_stream())
        console.setFormatter(formatter)
        console.setLevel(level)
        root.addHandler(console)

    # learning.log ハンドラ（学習サイクル Level 0.5〜2 専用）
    # 設計書 c_07_error_handling.md §6.2 に従い独立ファイルへ出力する
    _setup_learning_logging(log_dir, formatter, level)


# 学習サイクル系ロガー（学習サイクル Level 0.5〜2 を learning.log に集約する対象）
# - backend.learning.*       : Free 版の学習サイクル
# - backend.pro.learning.*   : Pro 版の Level 2 トレーナー等
# - backend.memory.sleep_update : Sleep-time update（Level 0.5）
_LEARNING_LOGGER_NAMES: tuple[str, ...] = (
    "backend.learning",
    "backend.pro.learning",
    "backend.memory.sleep_update",
)


def _setup_learning_logging(
    log_dir: Path,
    formatter: logging.Formatter,
    level: int,
) -> None:
    """learning.log ハンドラを学習サイクル系ロガーに登録する

    backend.log への重複出力を避けるため対象ロガーは ``propagate=False`` とする。
    複数回呼び出しても同一ファイルへのハンドラが重複しないようガードする。
    """
    target_path = (log_dir / "learning.log").resolve()

    target_loggers = [logging.getLogger(n) for n in _LEARNING_LOGGER_NAMES]
    if all(_has_rotating_file_handler(lg, target_path) for lg in target_loggers):
        return  # 既に全対象に登録済み（ハンドラ重複回避）

    learning_handler = RotatingFileHandler(
        log_dir / "learning.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    learning_handler.setFormatter(formatter)
    learning_handler.setLevel(level)

    for target_logger in target_loggers:
        if _has_rotating_file_handler(target_logger, target_path):
            continue
        target_logger.setLevel(level)
        target_logger.addHandler(learning_handler)
        target_logger.propagate = False


def setup_cli_logging(
    project_root: Path | None = None,
    debug: bool = False,
) -> None:
    """CLI 専用ロガーを初期化し cli.log に出力する

    backend.cli 名前空間のログを local/logs/cli.log へ書き込む。
    propagate=False でバックエンド側 (backend.log) への混在を防ぐ。
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    log_dir = project_root / "local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S%z"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # backend.free.cli ロガー設定（重複呼び出しガード）
    cli_logger = logging.getLogger("backend.free.cli")
    cli_logger.setLevel(logging.DEBUG)
    cli_logger.propagate = False

    # cli.log ハンドラ (5MB × 3世代) — 同一パスが既に登録済みなら追加しない
    cli_log_path = (log_dir / "cli.log").resolve()
    if not _has_rotating_file_handler(cli_logger, cli_log_path):
        cli_handler = RotatingFileHandler(
            log_dir / "cli.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        cli_handler.setFormatter(formatter)
        cli_handler.setLevel(logging.DEBUG)
        cli_logger.addHandler(cli_handler)

    # デバッグモード時は stderr にも出力 — UTF-8 強制
    if debug and not _has_plain_stream_handler(cli_logger):
        console = logging.StreamHandler(stream=_make_utf8_stream())
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        cli_logger.addHandler(console)


def get_logger(name: str) -> logging.Logger:
    """名前付きロガー取得"""
    return logging.getLogger(f"backend.{name}")
