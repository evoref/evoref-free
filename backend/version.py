"""エディション別バージョン情報の解決

Free / Pro のバージョンを独立して管理するためのヘルパー。

- `backend/free/__version__.py` は必須 (Free 配布物に必ず含まれる)
- `backend/pro/__version__.py` は Pro 配布のみ存在
- データ世代は共有形式の ``DATA_GENERATION`` (Free) と Pro 専用形式の
  ``PRO_DATA_GENERATION`` (Pro) の 2 本 (リリース用の定数、ディスクには持たない)

`backend.edition` の `pro_available()` と協調し、Pro 未同梱の Free 配布でも
安全に動作する。

バージョン文字列の一括更新は ``scripts/bump_version.py`` を使う。
SSOT は本ヘルパーが import する 2 つの ``__version__.py`` モジュール。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from backend.edition import (
    Edition,
    current_edition,
    pro_available,
)
from backend.free.__version__ import (
    DATA_GENERATION,
    __build__ as FREE_BUILD,
    __version__ as FREE_VERSION,
)


@dataclass(frozen=True)
class VersionInfo:
    """Free / Pro バージョンのスナップショット

    Attributes:
        free: Free 配布のバージョン文字列
        pro: Pro 配布のバージョン文字列 (未同梱時 None)
        generation: 共有形式のデータ世代 (``DATA_GENERATION``)
        pro_generation: Pro 専用形式のデータ世代 (未同梱時 None)
        edition: 現在の実行エディション ("free" | "pro" | "develop")
        free_build: Free のビルド識別子 (任意)
        pro_build: Pro のビルド識別子 (未同梱時 None)
    """

    free: str
    pro: str | None
    generation: int
    pro_generation: int | None
    edition: str
    free_build: str
    pro_build: str | None

    def to_dict(self) -> dict:
        """JSON 化用の dict 表現を返す"""
        return asdict(self)


def get_version_info() -> VersionInfo:
    """現在のエディションに応じたバージョン情報を返す

    Pro モジュールが存在し、かつ ``__version__.py`` を import できた場合のみ
    ``pro`` フィールドが設定される。Free 配布では ``pro`` は ``None`` になる。

    エディション判定は ``current_edition()`` (= setup_pro 登録結果) を優先し、
    ``EVOREF_EDITION=free`` で Pro を無効化した場合は Pro バージョンを返さない。
    Pro モジュールがディスク上に存在しても、ランタイムが Free として動作して
    いれば Free として報告する。

    edition 文字列は ``current_edition()`` の小文字名 (``"free"`` /
    ``"pro"`` / ``"develop"``) を返す。
    """
    cur = current_edition()

    pro_ver: str | None = None
    pro_build: str | None = None
    pro_generation: int | None = None
    if cur >= Edition.PRO and pro_available():
        try:
            from backend.pro.__version__ import (  # type: ignore[import-not-found]
                PRO_DATA_GENERATION as _PRO_GENERATION,
                __build__ as _PRO_BUILD,
                __version__ as _PRO_VERSION,
            )
            pro_ver = _PRO_VERSION
            pro_build = _PRO_BUILD
            pro_generation = _PRO_GENERATION
        except Exception:
            pro_ver = None
            pro_build = None

    return VersionInfo(
        free=FREE_VERSION,
        pro=pro_ver,
        generation=DATA_GENERATION,
        pro_generation=pro_generation,
        edition=cur.name.lower(),
        free_build=FREE_BUILD,
        pro_build=pro_build,
    )


def get_runtime_version() -> str:
    """現在のエディションに対応した「表示用バージョン」を返す

    CLI welcome 表示や UI ヘッダで使う、エディション特化のバージョン文字列。
    Pro > Free の優先順で、対応バージョンが取得できなければ Free バージョンに
    フォールバックする。
    """
    info = get_version_info()
    if info.edition == "pro" and info.pro is not None:
        return info.pro
    return info.free
