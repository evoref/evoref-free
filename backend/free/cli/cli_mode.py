"""CLI モードのエディション別デフォルト解決

`backend/free/cli/` 内で `"mode": "coding"` 等のリテラルをハードコードする代わりに、
本モジュールの :func:`default_cli_mode` を介してエディションに応じた既定モードを
解決する。

設計指針:

- **Free**: チャットメイン (デフォルト = ``"chat"``)。`--mode coding` 指定は
  warning + ``"chat"`` フォールバック (`coerce_cli_mode`)。
- **Pro / Develop**: コーディング含む (デフォルト = ``"coding"``)。
  `--mode chat`/`coding` いずれもそのまま受け入れ。Develop は Pro の上位
  互換 (Develop ⊇ Pro) のため Pro と同じ扱い。

Pro 判定優先順位:

1. ``EVOREF_EDITION`` 環境変数 (``"free"`` / ``"pro"`` / ``"develop"``) を最優先。
   CLI 起動時の ``--edition`` 上書き (`main._validate_edition_arg_async`) や
   テストでの差し替えが効く。
2. それ以外は :func:`backend.edition.pro_available` /
   :func:`backend.edition.develop_available` で
   ``backend.pro`` / ``backend.develop`` パッケージの存在を判定。

CLI クライアント側では FastAPI の Pro / Develop pillar が初期化されないため
``is_pro_or_above()`` (``_pro_handlers`` ベース) は常に False を返してしまう。
本モジュールでは「配布物として Pro / Develop が同梱されているか」をベースに判定する。
"""

from __future__ import annotations

import os

from backend.edition import develop_available, pro_available

VALID_CLI_MODES: frozenset[str] = frozenset({"chat", "coding"})


def is_cli_pro_edition() -> bool:
    """CLI 実行時のエディションが Pro 以上 (Pro / Develop) かを返す。

    判定優先順位は本モジュール docstring を参照。Develop は Pro の上位互換
    のため、本関数は Pro / Develop の両方で True を返す。
    """
    edition_env = os.environ.get("EVOREF_EDITION", "").strip().lower()
    if edition_env == "free":
        return False
    if edition_env in ("pro", "develop"):
        return True
    return pro_available() or develop_available()


def default_cli_mode() -> str:
    """エディションに応じた CLI のデフォルトモードを返す。

    Returns:
        ``"coding"`` (Pro) / ``"chat"`` (Free)。
    """
    return "coding" if is_cli_pro_edition() else "chat"


def coerce_cli_mode(requested: str | None) -> tuple[str, bool]:
    """ユーザー指定モードをエディションに応じて補正する。

    Args:
        requested: ``--mode`` で指定された値。``None`` ならデフォルトを返す。

    Returns:
        ``(resolved_mode, downgraded)`` — ``downgraded=True`` の場合、Free 環境で
        ``"coding"`` 指定 → ``"chat"`` フォールバックが発生したことを示す。呼び出し
        元 (`main`) で warning を表示する。
    """
    if requested is None:
        return default_cli_mode(), False
    if requested not in VALID_CLI_MODES:
        # argparse の choices で弾かれる想定だが、防御的に default にフォールバック。
        return default_cli_mode(), False
    if requested == "coding" and not is_cli_pro_edition():
        return "chat", True
    return requested, False
