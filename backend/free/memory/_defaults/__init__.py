"""EvorefMem 同梱のデフォルト辞書 (shipped defaults)

pin / fact / classify の各トリガ辞書は、
``backend/free/memory/_defaults/triggers/<name>.yaml`` を同梱 default とし、
ユーザーがチューニングしたい場合は ``local/triggers/<name>.yaml`` に
同名ファイルを置くことで上書きする 2 層構造で解決する。

``local/`` 配下は ``.gitignore`` で除外されているため、ユーザー編集は
リポジトリ差分に現れない。本 package (shipped default) はリポジトリに
commit されており、fresh clone 直後から辞書が利用可能な状態を保つ。

Module 間で共通の解決ロジックを :func:`resolve_trigger_file` に集約する。
EvorefMem pillar 内部でのみ import される想定。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEFAULT_TRIGGERS_DIR", "resolve_trigger_file"]


#: 同梱デフォルト辞書ディレクトリ (package-relative)。
DEFAULT_TRIGGERS_DIR: Path = Path(__file__).resolve().parent / "triggers"


def resolve_trigger_file(
    name: str,
    triggers_dir: Path | str | None = None,
) -> Path:
    """トリガ辞書ファイルの解決。

    ``triggers_dir/<name>`` が存在すれば user override として返し、
    さもなくば package 同梱の default パスを返す。

    Args:
        name: ファイル名 (例: ``"pin_triggers.yaml"``)。
        triggers_dir: user override の配置ディレクトリ (通常
            ``PathResolver.resolve_local("triggers_dir")``)。``None`` または
            同名ファイルが存在しない場合は package default に fall back する。

    Returns:
        解決された ``Path``。default に fall back した場合、ファイルが存在
        しないことは想定されない (同梱 default はリポジトリに含まれる) が、
        念のためローダ側で empty 扱いできる仕様とする。
    """
    if triggers_dir is not None:
        override = Path(triggers_dir) / name
        if override.exists():
            return override
    return DEFAULT_TRIGGERS_DIR / name
