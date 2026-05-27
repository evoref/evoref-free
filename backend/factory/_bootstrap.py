"""bootstrap (config / logging / i18n / local_dirs) ヘルパー

含まれる関数:

- :func:`_init_config`     : config.yaml 読込み + Pydantic バリデーション
- :func:`_init_logging`    : log_config + DebugLogger 初期化
- :func:`_init_i18n`       : i18n locale / fallback 設定
- :func:`_init_local_dirs` : local 配下のディレクトリ作成 + PolicyInterpreter 構築

純粋な move であり、関数本体・引数・default 値は変更していない。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.app_state import AppState
from backend.config import load_config, get_path_resolver
from backend.i18n_helper import init_i18n
from backend.log_config import setup_logging, get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("factory.bootstrap")


def _init_config(project_root: Path) -> dict[str, Any]:
    """1. config.yaml 読込み + バリデーション"""
    from pydantic import ValidationError

    try:
        cfg = load_config(project_root=project_root)
    except ValidationError as e:
        logger.error("Config validation failed. Fix config.yaml and restart.\n%s", e)
        sys.exit(1)
    logger.info("Config loaded and validated")
    return cfg


_VALID_DEVELOP_LEVELS: frozenset[str] = frozenset(
    {"off", "debug", "investigate", "evolve"},
)
# Develop 限定レベル。Free / Pro では拒否する
# (設計書 docs/e_04_user_feature_matrix.md §15 と整合)。
_DEVELOP_ONLY_DEVELOP_LEVELS: frozenset[str] = frozenset(
    {"evolve"},
)


def _resolve_develop_level() -> str:
    """環境変数から develop_level を解決する

    - ``EVOREF_DEVELOP=1`` (旧フラグ) は即 reject — sys.exit(1)。
    - ``EVOREF_DEVELOP_LEVEL`` が ``debug`` / ``investigate`` / ``evolve``
      のいずれかなら採用。
    - 未設定なら ``"off"``。
    - 不正値は sys.exit(1)。
    - ``evolve`` を Free / Pro 環境で指定された場合は sys.exit(1)
      (Develop エディション必須)。
    """
    if os.environ.get("EVOREF_DEVELOP") == "1":
        logger.error(
            "EVOREF_DEVELOP=1 is no longer supported. "
            "Use --develop=<level> CLI flag (debug | investigate | evolve) instead.",
        )
        sys.exit(1)

    raw = os.environ.get("EVOREF_DEVELOP_LEVEL", "")
    if raw == "":
        return "off"
    if raw not in _VALID_DEVELOP_LEVELS or raw == "off":
        # "off" は env var で明示する設計ではないため不正値扱い
        logger.error(
            "Invalid EVOREF_DEVELOP_LEVEL=%r (expected debug | investigate | evolve)",
            raw,
        )
        sys.exit(1)

    if raw in _DEVELOP_ONLY_DEVELOP_LEVELS:
        from backend.edition import is_develop
        if not is_develop():
            logger.error(
                "--develop=%s requires Develop edition. "
                "Use --develop=debug or --develop=investigate instead.",
                raw,
            )
            sys.exit(1)

    return raw


def _resolve_learning_disabled() -> bool:
    """環境変数 ``EVOREF_LEARNING_DISABLED`` から自己学習無効化フラグを解決する

    ``EVOREF_LEARNING_DISABLED=1`` (CLI ``--no-learning`` 経由で設定) の場合に
    True を返す。未設定または ``""`` 以外の値は False。
    """
    return os.environ.get("EVOREF_LEARNING_DISABLED") == "1"


def _init_logging(state: AppState, cfg: dict[str, Any], project_root: Path) -> "DebugLogger":
    """2. ログ設定 + 2b. DebugLogger 初期化"""
    develop_level = _resolve_develop_level()
    state.develop_level = develop_level  # type: ignore[assignment]
    state.learning_disabled = _resolve_learning_disabled()
    if state.learning_disabled:
        logger.info(
            "Self-learning disabled by --no-learning flag "
            "(Level 0/1/2 + Pro assist injection are no-op; reads continue)",
        )

    # Pro 同梱時のみ data isolation 等の overrides を適用
    if develop_level != "off":
        try:
            from backend.pro.develop_mode import apply_develop_overrides  # type: ignore[import-not-found]
            apply_develop_overrides(cfg, develop_level)  # type: ignore[arg-type]
            logger.info("Develop mode active (Pro): level=%s", develop_level)
        except ImportError:
            logger.info("Develop mode active (Free): level=%s", develop_level)

    setup_logging(develop_level, project_root)  # type: ignore[arg-type]

    # 2b. DebugLogger 初期化
    from backend.debug_logger import DebugLogger
    debug_logger = DebugLogger(
        develop_level=develop_level,  # type: ignore[arg-type]
        project_root=project_root,
    )
    state.debug_logger = debug_logger
    if debug_logger.enabled:
        logger.info(
            "DebugLogger enabled: level=%s log_dir=%s",
            develop_level, debug_logger.log_dir,
        )
    return debug_logger


def _init_i18n(cfg: dict[str, Any]) -> None:
    """3. i18n 初期化"""
    i18n_cfg = cfg.get("i18n", {})
    init_i18n(
        locale=i18n_cfg.get("locale", "ja"),
        fallback=i18n_cfg.get("fallback", "ja"),
    )
    logger.info("i18n initialized: locale=%s", i18n_cfg.get("locale", "ja"))


def _init_local_dirs(
    state: AppState, project_root: Path, cfg: dict[str, Any] | None = None,
) -> tuple[Any, "PolicyInterpreter"]:
    """4. ローカルディレクトリ作成 + 4b. PolicyInterpreter 初期化"""
    resolver = get_path_resolver()
    resolver.ensure_local_dirs()
    logger.info("Local directories ensured")

    # 4b. PolicyInterpreter 初期化
    # learning.policy.source を反映。SemMem ストアは
    # メモリ層初期化後に ``apply_semmem_policy_overrides`` で注入する。
    # 旧 ``harness:`` セクションから ``learning.policy.*`` へ移行済
    from backend.free.core.policy_interpreter import PolicyInterpreter

    learning_policy_cfg = ((cfg or {}).get("learning") or {}).get("policy") or {}
    policy_source = learning_policy_cfg.get("source", "yaml")
    min_conf = float(learning_policy_cfg.get("activation_min_confidence", 0.7))

    policies_dir = project_root / "local" / "policies"
    policy_interpreter = PolicyInterpreter(
        policies_dir,
        policy_source=policy_source,
        policy_activation_min_confidence=min_conf,
        debug_logger=state.debug_logger,
    )
    state.policy_interpreter = policy_interpreter
    logger.info(
        "PolicyInterpreter initialized: %s (policy_source=%s, min_conf=%.2f)",
        policies_dir, policy_source, min_conf,
    )
    return resolver, policy_interpreter
