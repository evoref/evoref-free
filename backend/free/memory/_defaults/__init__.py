"""EvorefMem 同梱のデフォルト辞書 (shipped defaults)

pin / fact / classify の各トリガ辞書は、
``backend/free/memory/_defaults/triggers/<name>.yaml`` を同梱 default とし、
ユーザーがチューニングしたい場合は ``<data_root>/g1/store/overrides/triggers/<name>.yaml`` に
同名ファイルを置くことで上書きする 2 層構造で解決する。

データ根 (``userdata/``) の中身は ``.gitignore`` で除外されているため、ユーザー編集は
リポジトリ差分に現れない。本 package (shipped default) はリポジトリに
commit されており、fresh clone 直後から辞書が利用可能な状態を保つ。

Module 間で共通の解決ロジックを :func:`resolve_trigger_file` に集約する。
EvorefMem pillar 内部でのみ import される想定。

上書きの有無と解決後のパスは **プロセス内で 1 度だけ** 調べて覚える
(ノート / 文ごとに呼ばれるので、毎回 ``exists`` と ``Path.resolve`` を打つと
1 ターンで数百 ms になる)。辞書の中身もローダ側でプロセス内キャッシュ済み
なので、上書きファイルを置く・書き換える・消すのは **再起動で反映** される。
テストは各ローダの ``reset_*_cache()`` (内部で :func:`reset_trigger_path_cache`
を呼ぶ) で落とす。
"""

from __future__ import annotations

import threading
from pathlib import Path

from backend.io.format_registry import FormatSpec, register_format

__all__ = [
    "DEFAULT_TRIGGERS_DIR",
    "TRIGGER_OVERRIDE_FORMAT",
    "reset_trigger_path_cache",
    "resolve_trigger_file",
    "trigger_cache_key",
]


#: 同梱デフォルト辞書ディレクトリ (package-relative)。
DEFAULT_TRIGGERS_DIR: Path = Path(__file__).resolve().parent / "triggers"

#: 利用者が置く上書き (``PathResolver.LAYOUT["triggers_dir"]``、c_05 §0.4.9)。
TRIGGER_OVERRIDE_FORMAT = register_format(FormatSpec(
    format_id="overrides.triggers",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key="store/overrides/triggers/<name>.yaml",
    retention="written by the user; never pruned",
    export=True,
    encodings=("yaml",),
))

_CACHE_LOCK = threading.Lock()
#: ``(triggers_dir, name)`` → 解決したファイル。
_RESOLVED_FILES: dict[tuple[str, str], Path] = {}
#: 呼出側が渡したパス文字列 → ``resolve()`` 済みの鍵。
_CACHE_KEYS: dict[str, str] = {}


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

    絶対パスの ``triggers_dir`` は結果をプロセス内で覚える (モジュール
    docstring 参照。上書きの追加・削除は再起動で反映)。
    """
    if triggers_dir is None:
        return DEFAULT_TRIGGERS_DIR / name
    base = Path(triggers_dir)
    key = (str(base), name)
    cached = _RESOLVED_FILES.get(key)
    if cached is not None:
        return cached
    override = base / name
    resolved = override if override.exists() else DEFAULT_TRIGGERS_DIR / name
    if base.is_absolute():
        with _CACHE_LOCK:
            _RESOLVED_FILES[key] = resolved
    return resolved


def trigger_cache_key(path: Path | str) -> str:
    """ローダのキャッシュ鍵 (``str(Path(path).resolve())``) を覚えて返す。

    ``Path.resolve`` は Windows では対象を開いて最終パスを問い合わせるので
    1 回 100µs 級になる。絶対パスは結果が CWD に依らないので覚える。
    """
    raw = str(path)
    cached = _CACHE_KEYS.get(raw)
    if cached is not None:
        return cached
    as_path = Path(path)
    key = str(as_path.resolve())
    if as_path.is_absolute():
        with _CACHE_LOCK:
            _CACHE_KEYS[raw] = key
    return key


def reset_trigger_path_cache() -> None:
    """テスト用: 解決結果の記憶を消す (上書きファイルを置き換えたテストが使う)。"""
    with _CACHE_LOCK:
        _RESOLVED_FILES.clear()
        _CACHE_KEYS.clear()
