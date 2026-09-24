"""データ根 ``data_root`` の解決 (c_05 §0.2、c_03 §10.1)。

G1 の全データは ``data_root`` の下に置く。決め方は 3 段だけ:

1. ``--data-root`` (CLI。相対ならインストール根基準で絶対化)
2. 環境変数 ``EVOREF_DATA_ROOT`` (絶対パスのみ)
3. 既定 ``<install_root>/userdata``

設定キーにはしない (設定ファイル自身の置き場が循環するため)。決めた値は
``EVOREF_DATA_ROOT`` に入れて全子プロセス (llama-server 起動・uvicorn・reset の
再起動経路) へ伝える。

拒否する指定: ``local/`` (G0) や ``models/`` の中、それらを内側に含む指定、
インストール根そのもの。G1 は ``local/`` を開かない・動かさない・消さない (c_05 §0.3)。
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "EVOREF_DATA_ROOT"
#: ``--allow-unsafe-data-root`` を子プロセスの起動ゲートへ伝える環境変数。
ALLOW_UNSAFE_ENV = "EVOREF_ALLOW_UNSAFE_DATA_ROOT"
DEFAULT_DIRNAME = "userdata"
#: ``--isolate-data`` の別根の置き場 (本番根の外、c_05 §0.2)。
ISOLATED_DIRNAME = "userdata-isolated"

_FORBIDDEN_CHILDREN = ("local", "models")


class DataRootError(ValueError):
    """``data_root`` の指定が不正 (起動を止める)。"""


def install_root() -> Path:
    """インストール根 (= リポジトリ / 配布物の根)。"""
    return Path(__file__).resolve().parents[1]


def _normalize(path: Path) -> Path:
    return Path(os.path.normcase(os.path.abspath(path)))


def _is_within(child: Path, parent: Path) -> bool:
    try:
        _normalize(child).relative_to(_normalize(parent))
    except ValueError:
        return False
    return True


def validate_data_root(path: Path, root: Path) -> Path:
    """``path`` を検査して絶対パスで返す。不正なら :class:`DataRootError`。"""
    resolved = Path(os.path.abspath(path))
    if _normalize(resolved) == _normalize(root):
        raise DataRootError(f"data_root must not be the install root itself: {resolved}")
    for name in _FORBIDDEN_CHILDREN:
        guarded = root / name
        if _is_within(resolved, guarded):
            raise DataRootError(f"data_root must not be inside {guarded}: {resolved}")
        if _is_within(guarded, resolved):
            raise DataRootError(f"data_root must not contain {guarded}: {resolved}")
    return resolved


def resolve_data_root(
    cli_value: str | Path | None = None,
    *,
    root: Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """3 段の規則で ``data_root`` を決める (副作用なし)。"""
    base = root or install_root()
    environ = os.environ if env is None else env
    if cli_value:
        candidate = Path(cli_value)
        if not candidate.is_absolute():
            candidate = base / candidate
        return validate_data_root(candidate, base)
    raw = environ.get(ENV_VAR, "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise DataRootError(f"{ENV_VAR} must be an absolute path: {raw!r}")
        return validate_data_root(candidate, base)
    return validate_data_root(base / DEFAULT_DIRNAME, base)


def isolated_data_root(name: str = "develop", *, root: Path | None = None) -> Path:
    """``--isolate-data`` の別根 ``<install_root>/userdata-isolated/<name>``。"""
    base = root or install_root()
    return validate_data_root(base / ISOLATED_DIRNAME / name, base)


def export_data_root(path: Path) -> None:
    """決めた ``data_root`` を子プロセスへ伝える (``EVOREF_DATA_ROOT``)。"""
    os.environ[ENV_VAR] = str(path)


__all__ = [
    "ALLOW_UNSAFE_ENV",
    "DEFAULT_DIRNAME",
    "ENV_VAR",
    "ISOLATED_DIRNAME",
    "DataRootError",
    "export_data_root",
    "install_root",
    "isolated_data_root",
    "resolve_data_root",
    "validate_data_root",
]
