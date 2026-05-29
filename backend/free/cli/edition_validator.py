"""CLI の ``--edition`` 整合性検証の共通化。

``service_manager`` (``_validate_edition_arg``) と ``main``
(``_validate_edition_arg_async``) で AST 一致していた検証ロジックを集約する。
両者の唯一の差は「検証後に ``EVOREF_EDITION`` 環境変数を設定するか」で、
``set_env_var`` フラグで吸収する。
"""

from __future__ import annotations

import argparse
import os
import sys

from backend.free.cli.renderer import render_error
from backend.i18n_helper import msg


def validate_edition_arg(
    args: argparse.Namespace,
    console,
    *,
    set_env_var: bool = False,
) -> int | None:
    """``--edition`` の整合性を検証する。継続可なら ``None``、不整合なら ``1``。

    Args:
        set_env_var: ``True`` のとき検証通過後に ``EVOREF_EDITION`` を設定し
            上書き通知を stderr へ出す (frontend CLI 互換。serve 経路は False)。
    """
    if args.edition is None:
        return None
    if getattr(args, "develop", None) is None:
        render_error(console, msg("cli.edition_requires_develop"))
        return 1

    from backend.edition import develop_available, pro_available

    if args.edition == "pro" and not pro_available():
        render_error(console, msg("cli.edition_pro_unavailable"))
        return 1
    if args.edition == "develop" and not develop_available():
        render_error(console, msg("cli.edition_develop_unavailable"))
        return 1

    # backend lifespan の深部で sys.exit(1) する代わりに CLI 側で先に拒否する
    # (親プロセスからは "process died during startup" としか見えないため)。
    # - --edition=free|pro + evolve : Develop 必須
    from backend.free.cli.develop_hook import DEVELOP_ONLY_DEVELOP_LEVELS

    if (
        args.edition in ("free", "pro")
        and args.develop in DEVELOP_ONLY_DEVELOP_LEVELS
    ):
        render_error(
            console, msg("cli.develop_level_develop_only", level=args.develop),
        )
        return 1

    if set_env_var:
        os.environ["EVOREF_EDITION"] = args.edition
        print(msg("cli.edition_override", edition=args.edition), file=sys.stderr)
    return None
