"""CLI の develop モード環境セットアップ共通化。

``service_manager`` (serve) と ``main`` (gui / interactive) で AST 一致して
いた ``_setup_develop_mode`` / ``_setup_develop_mode_async`` の本体を集約する。
``--develop=<level>`` の環境設定・``--isolate-data``・``--no-learning`` の
伝播を一元化する (``investigate`` / ``evolve`` の Pro 限定判定や
``--isolate-data`` の実効果は ``develop_hook`` 側に閉じる)。
"""

from __future__ import annotations

import argparse
import os
import sys

from backend.free.cli.develop_hook import get_develop_hook
from backend.free.cli.renderer import render_error
from backend.i18n_helper import msg


def setup_develop_mode(args: argparse.Namespace, console) -> int | None:
    """``--develop`` / ``--isolate-data`` / ``--no-learning`` の環境セットアップ。

    Returns:
        成功時 ``None``。``--isolate-data`` を develop なしで指定した等の
        エラー時は終了コード ``1``。
    """
    hook = get_develop_hook()

    develop_level = getattr(args, "develop", None)
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
