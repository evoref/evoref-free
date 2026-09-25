"""データ根 ``data_root`` の解決 (c_05 §0.2、c_03 §10.1)。

G1 の全データは ``data_root`` の下に置く。決め方は 3 段だけ:

1. ``--data-root`` (CLI。相対ならインストール根基準で絶対化)
2. 環境変数 ``EVOREF_DATA_ROOT`` (絶対パスのみ)
3. 既定 ``<install_root>/userdata``

設定キーにはしない (設定ファイル自身の置き場が循環するため)。決めた値は
``EVOREF_DATA_ROOT`` に入れて全子プロセス (llama-server 起動・uvicorn・reset の
再起動経路) へ伝える。

スキーマ世代 (c_05 §0.2): 形式に縛られる ``store/`` と ``cache/`` は世代フォルダ
``<data_root>/g<N>/`` の下に置き、世代に依存しない ``logs/`` ``outputs/`` ``themes/``
``profiles/`` ``bk/`` ``run/`` ``tmp/`` はデータ根の直下に置く。形式台帳の
``path_key`` (``store/...`` ``cache/...``) は世代フォルダからの相対 (:func:`data_path`)。

拒否する指定: ``models/`` の中、それを内側に含む指定、インストール根そのもの。
"""

from __future__ import annotations

import os
from pathlib import Path

# このリリースのスキーマ世代 (SSOT はリリース定数、c_05 §0.2)
from backend.free.__version__ import DATA_GENERATION

ENV_VAR = "EVOREF_DATA_ROOT"
#: ``--allow-unsafe-data-root`` を子プロセスの起動ゲートへ伝える環境変数。
ALLOW_UNSAFE_ENV = "EVOREF_ALLOW_UNSAFE_DATA_ROOT"
DEFAULT_DIRNAME = "userdata"
#: ``--isolate-data`` の別根の置き場 (本番根の外、c_05 §0.2)。
ISOLATED_DIRNAME = "userdata-isolated"

#: 世代フォルダの名前 (``<data_root>/g<N>/``)。
GENERATION_DIRNAME = f"g{DATA_GENERATION}"
#: 世代フォルダの下に置く最上位のディレクトリ (形式に縛られるもの)。
GENERATION_SCOPED_DIRS = ("store", "cache")

_FORBIDDEN_CHILDREN = ("models",)


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


def generation_root(data_root: Path) -> Path:
    """このリリースの世代フォルダ ``<data_root>/g<N>/``。"""
    return Path(data_root) / GENERATION_DIRNAME


def data_path(data_root: Path, rel: str) -> Path:
    """データ根からの論理パス ``rel`` (``PathResolver.LAYOUT`` / ``path_key``) の実パス。

    最上位が ``store`` / ``cache`` なら世代フォルダの下、それ以外はデータ根の直下。
    """
    top = rel.replace("\\", "/").split("/", 1)[0]
    base = generation_root(data_root) if top in GENERATION_SCOPED_DIRS else Path(data_root)
    return base / rel


def store_root(data_root: Path) -> Path:
    """このリリースの ``store/`` (``<data_root>/g<N>/store``)。"""
    return data_path(data_root, "store")


def generation_dirs(data_root: Path) -> dict[int, Path]:
    """データ根にある世代フォルダ ``g<N>/`` (世代 → パス)。"""
    found: dict[int, Path] = {}
    root = Path(data_root)
    if not root.is_dir():
        return found
    for entry in root.iterdir():
        name = entry.name
        if entry.is_dir() and len(name) > 1 and name[0] == "g" and name[1:].isdigit():
            found[int(name[1:])] = entry
    return found


def export_data_root(path: Path) -> None:
    """決めた ``data_root`` を子プロセスへ伝える (``EVOREF_DATA_ROOT``)。"""
    os.environ[ENV_VAR] = str(path)


__all__ = [
    "ALLOW_UNSAFE_ENV",
    "DATA_GENERATION",
    "DEFAULT_DIRNAME",
    "ENV_VAR",
    "GENERATION_DIRNAME",
    "GENERATION_SCOPED_DIRS",
    "ISOLATED_DIRNAME",
    "DataRootError",
    "data_path",
    "export_data_root",
    "generation_dirs",
    "generation_root",
    "install_root",
    "isolated_data_root",
    "resolve_data_root",
    "store_root",
    "validate_data_root",
]
