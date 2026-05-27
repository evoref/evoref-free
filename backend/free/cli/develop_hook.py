"""CLI develop モードの hook 抽象

Free 版 CLI が develop モードを起動するための抽象境界。
Pro 版が import 時に `register_develop_hook(ProDevelopHook())` で
自身を登録し、Free CLI は `get_develop_hook()` を介してのみ参照する。

``--develop`` を 3 段階化 (``debug`` / ``investigate`` /
``evolve``)。

- ``debug``: Free / Pro / Develop 共通。即時 BUG 解析向け。requests.jsonl のみ。
- ``investigate``: Free / Pro / Develop 共通。挙動把握向け。
  requests/rag/memory/long_form。
- ``evolve``: Develop 限定。loop 自己学習向け。既存 6 系統 + decision/outcome。
  シナリオハーネス連携を前提にした自己進化パイプラインの全カテゴリ出力。

Free / Pro 環境で ``evolve`` を指定すると ``apply_develop_overrides`` が
``sys.exit(1)`` で起動を拒否する (Develop エディションが必要)。
``--isolate-data`` は Pro 専用のため、Free フォールバックでは受け付けても no-op。

レイヤー責務:
- `backend/free/cli/develop_hook` — 抽象 + Free 基本実装 (本モジュール)
- `backend/pro/cli/develop_mode`  — Pro 拡張 (debug / investigate / evolve)
- `backend/free/cli/{service_manager,main}` — `get_develop_hook()` のみ参照
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Protocol

from backend.i18n_helper import msg
from backend.log_config import DevelopLevel

# develop モードレベル一覧。argparse の choices としても使用する
# ``DevelopLevel`` Literal と同期させる。
DEVELOP_LEVEL_CHOICES: tuple[str, ...] = ("debug", "investigate", "evolve")
# Free エディションで利用可能なレベル (debug + investigate)。Develop 限定レベル
# (= ``evolve``) を Free で指定された場合は ``apply_develop_overrides`` 内で
# sys.exit する。
FREE_ALLOWED_LEVELS: frozenset[str] = frozenset({"debug", "investigate"})
# Develop 限定レベル。Free / Pro 環境では拒否される。
# ``backend.factory._bootstrap._DEVELOP_ONLY_DEVELOP_LEVELS`` と同期。
# 設計書 docs/e_04_user_feature_matrix.md §15 と整合 (evolve は Develop 限定)。
DEVELOP_ONLY_DEVELOP_LEVELS: frozenset[str] = frozenset({"evolve"})


class DevelopHook(Protocol):
    """Develop モードの実装が満たす契約。

    Pro 版 (`ProDevelopHook`) と Free 版 (`_FreeDevelopHook`) の双方が実装する。
    """

    def is_available(self) -> bool:
        """`--develop` が利用可能か (Free / Pro どちらも True)。"""

    def extend_parser(self, parser: argparse.ArgumentParser) -> None:
        """`--develop` / `--isolate-data` フラグを argparse パーサーに追加する。"""

    def setup_develop_env(self, level: DevelopLevel) -> None:
        """`EVOREF_DEVELOP_LEVEL=<level>` 環境変数を設定する"""

    def setup_isolate_data_env(self) -> None:
        """`EVOREF_ISOLATE_DATA=1` を設定する (Pro 拡張)。"""

    def print_develop_banner(self, level: DevelopLevel) -> None:
        """Develop モード起動バナーを表示する"""

    def apply_develop_overrides(
        self, cfg: dict, level: DevelopLevel,
    ) -> dict:
        """config 辞書に develop モードのオーバーライドを適用する。

        Free では ``evolve`` を指定された場合に sys.exit (Develop 限定レベル拒否)。
        Pro では全 3 レベルを受け付け、必要に応じて data isolation 等を適用する。
        """


class _FreeDevelopHook:
    """Free 配布での `--develop` 基本実装

    Free で許容されるのは ``debug`` / ``investigate``。``evolve`` は Develop
    エディション限定のため、起動時に sys.exit で拒否する。
    """

    def is_available(self) -> bool:
        return True

    def extend_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--develop",
            choices=list(DEVELOP_LEVEL_CHOICES),
            default=None,
            metavar="LEVEL",
            help=(
                "Develop mode level (required value): "
                "debug (Free/Pro/Develop) | investigate (Pro/Develop) | "
                "evolve (Develop only)"
            ),
        )
        parser.add_argument(
            "--isolate-data", action="store_true",
            help="Isolate local/ data dir (Pro only, no-op in Free)",
        )

    def setup_develop_env(self, level: DevelopLevel) -> None:
        os.environ["EVOREF_DEVELOP_LEVEL"] = level

    def setup_isolate_data_env(self) -> None:
        # Free では no-op (Pro 拡張が上書き登録される)
        return None

    def print_develop_banner(self, level: DevelopLevel) -> None:
        print(msg("cli.develop_banner", level=level), file=sys.stderr)

    def apply_develop_overrides(
        self, cfg: dict, level: DevelopLevel,
    ) -> dict:
        # Free エディションは debug / investigate を受け付け、Develop 限定レベル
        # (= evolve) は即拒否。
        if level not in FREE_ALLOWED_LEVELS:
            print(
                msg("cli.develop_level_develop_only", level=level),
                file=sys.stderr,
            )
            sys.exit(1)
        return cfg


# ──────────────────────────────────────────────────────────────────────────
# レジストリ
# ──────────────────────────────────────────────────────────────────────────

_hook: DevelopHook | None = None
_bootstrapped: bool = False


def register_develop_hook(hook: DevelopHook) -> None:
    """Pro 実装を登録する。Pro CLI モジュール側 (`backend.pro.cli.develop_mode`)
    が import 時に呼び出すことで Free CLI から参照可能になる。

    後勝ち (上書き可) — テストで stub に差し替えるのを許容するため。
    """
    global _hook
    _hook = hook


def reset_develop_hook() -> None:
    """テスト用: 登録状態をクリアして bootstrap をやり直せる状態に戻す。"""
    global _hook, _bootstrapped
    _hook = None
    _bootstrapped = False


def get_develop_hook() -> DevelopHook:
    """現在登録されている DevelopHook を返す。

    初回呼び出し時に Pro CLI develop_mode モジュールを **唯一の lazy import**
    として試行し、Pro 側の自己登録を発火させる。これにより Free CLI 本体には
    Pro への直接 import 文を残さず、Pro 依存箇所を本モジュール 1 ヶ所に集約する。

    Pro が利用不可 (Free エディション or import 失敗) の場合は
    `_FreeDevelopHook` を返す。
    """
    global _hook, _bootstrapped
    if _hook is not None:
        return _hook
    if not _bootstrapped:
        _bootstrapped = True
        try:
            # Pro 側 develop_mode の import が register_develop_hook を呼ぶ。
            # ImportError は Free エディションの正常系。
            import backend.pro.cli.develop_mode  # noqa: F401
        except ImportError:
            pass
    if _hook is None:
        _hook = _FreeDevelopHook()
    return _hook
