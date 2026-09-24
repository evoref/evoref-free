"""CLI の develop モード環境セットアップ共通化。

``service_manager`` (serve) と ``main`` (gui / interactive) で AST 一致して
いた ``_setup_develop_mode`` / ``_setup_develop_mode_async`` の本体を集約する。
``--develop=<level>`` の環境設定・``--isolate-data``・``--no-learning``・
``--data-root`` の伝播を一元化する (``investigate`` / ``evolve`` の Pro 限定判定や
``--isolate-data`` の実効果は ``develop_hook`` 側に閉じる)。
"""

from __future__ import annotations

import argparse
import os
import sys

from backend.data_root import ALLOW_UNSAFE_ENV, DataRootError, export_data_root, resolve_data_root
from backend.free.cli.develop_hook import get_develop_hook
from backend.free.cli.renderer import render_error
from backend.i18n_helper import msg


def add_data_root_flag(parser: argparse.ArgumentParser) -> None:
    """``--data-root`` フラグを追加する (serve / chat / gui 共通、c_05 §0.2)。"""
    parser.add_argument(
        "--data-root", default=None, metavar="PATH",
        help=(
            "Data root directory (default: EVOREF_DATA_ROOT or <install_root>/userdata). "
            "Relative paths are anchored at the install root. Propagated to child "
            "processes via EVOREF_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-data-root", action="store_true",
        help=(
            "Start even if the data root is on a network drive or OneDrive "
            "(rename/fsync ordering is not guaranteed there)."
        ),
    )


def setup_develop_mode(args: argparse.Namespace, console) -> int | None:
    """``--develop`` / ``--isolate-data`` / ``--no-learning`` / ``--data-root`` の環境セットアップ。

    ``--data-root`` は config を読む前に解決し ``EVOREF_DATA_ROOT`` へ書く
    (子プロセス = llama-server 起動・uvicorn へ伝えるため)。

    Returns:
        成功時 ``None``。``--isolate-data`` を develop なしで指定した・
        データ根の指定が不正等のエラー時は終了コード ``1``。
    """
    hook = get_develop_hook()

    develop_level = getattr(args, "develop", None)
    data_root_arg = getattr(args, "data_root", None)
    if getattr(args, "allow_unsafe_data_root", False):
        os.environ[ALLOW_UNSAFE_ENV] = "1"  # 子プロセス (uvicorn) の起動ゲートへ伝える
    if data_root_arg:
        if develop_level is not None and getattr(args, "isolate_data", False):
            render_error(console, msg("cli.data_root_conflicts_isolate"))
            return 1
        try:
            export_data_root(resolve_data_root(data_root_arg))
        except DataRootError as e:
            render_error(console, msg("cli.data_root_invalid", detail=str(e)))
            return 1

    if develop_level is not None:
        hook.setup_develop_env(develop_level)
        if getattr(args, "isolate_data", False):
            hook.setup_isolate_data_env()
        hook.print_develop_banner(develop_level)

    if getattr(args, "no_learning", False):
        os.environ["EVOREF_LEARNING_DISABLED"] = "1"
        print(msg("cli.learning_disabled_banner"), file=sys.stderr)

    if develop_level is not None:
        return None
    if getattr(args, "isolate_data", False):
        render_error(console, msg("cli.isolate_data_requires_develop"))
        return 1
    return None
