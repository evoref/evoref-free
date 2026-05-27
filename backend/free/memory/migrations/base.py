"""EvorefMem schema migration 基底クラス

`SchemaMigrator` が連鎖実行する単一 step 分のマイグレーション仕様を定義する。

## 位置付け

EvorefMem は `local/memory/semantic/SCHEMA_VERSION` マーカーを唯一の版番号
source of truth とし、版の差異は以下の 2 手段で吸収する:

1. **本モジュール (`SchemaMigrator`)** — **in-place rewrite** 型のマイグレーション。
   `DEFAULT_MIGRATIONS` に登録された `Migration` 群から `current → SCHEMA_VERSION`
   への連鎖を解決し、`semantic/` サブツリーを書き換える
2. `initialize_evorefmem` — destructive re-init (`semantic/` 以外を退避 + 削除)

v1 のみの現状では登録インスタンスは 0 件だが、subject rename /
FactType 統廃合 / 索引再構造化等、将来の破壊的変更を無痛で取り込める足場になる。
現行版以外のマーカーは本モジュール → destructive init の 2 段階で処理される。

## 設計原則

- 各 Migration は (from_version, to_version, component) の 3 つ組で一意
- `dry_run` は副作用禁止 (対象ファイル列挙 + 推定件数のみ返す)
- `migrate` は backup が別途退避されていることを前提に in-place rewrite を行う
- `rollback` は `SchemaMigrator` が退避した backup dir からの復元責務を負う
- pillar 境界は EvorefMem 内部に閉じる (他 pillar 参照不可)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

ComponentKind = Literal["fact", "subject_ns", "embedding", "index"]


@dataclass(frozen=True)
class MigrationPlan:
    """`dry_run` の実行計画.

    副作用なしで得られる「どのファイルに / 何件 / どんな変換をかける予定か」の
    サマリ。CLI の `--dry-run` 出力や SchemaMigrator.plan() の結果に用いる。
    """

    from_version: int
    to_version: int
    component: str
    description: str
    affected_files: tuple[Path, ...] = ()
    estimated_records: int = 0


@dataclass
class MigrationResult:
    """`migrate` 実行結果サマリ."""

    from_version: int
    to_version: int
    component: str
    processed_records: int = 0
    backup_dir: Path | None = None
    elapsed_ms: float = 0.0
    details: dict[str, object] = field(default_factory=dict)


class Migration(ABC):
    """単一 step 分のスキーマ変換を表す抽象基底.

    サブクラスは以下の class 変数をオーバーライドする:

    - ``from_version`` / ``to_version``: この Migration が繋ぐ版番号
    - ``component``: 対象領域 ("fact" / "subject_ns" / "embedding" / "index")

    クラス変数は `__init_subclass__` で必須チェックし、未設定のまま
    `SchemaMigrator` に登録された場合は即座に `TypeError` を送出する。
    """

    from_version: ClassVar[int]
    to_version: ClassVar[int]
    component: ClassVar[ComponentKind]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # 抽象 Migration を経由する中間基底 (abstract=True) は検査を
        # スキップ可能。abstractmethod が残っていれば通常 instantiate
        # できないので、ここでは class 変数の存在のみ強制する。
        missing: list[str] = []
        for attr in ("from_version", "to_version", "component"):
            if not hasattr(cls, attr):
                missing.append(attr)
        # ABC 自身を継承しただけの抽象サブクラスは `abstractmethod` が
        # 残るため最終具象クラスでのみ例外を送出する運用とする。
        if missing and not getattr(cls, "__abstractmethods__", frozenset()):
            raise TypeError(
                f"Migration subclass {cls.__name__} must define: "
                + ", ".join(missing),
            )

    @abstractmethod
    def dry_run(self, memory_dir: Path) -> MigrationPlan:
        """副作用なしで plan を返す.

        対象ファイル列挙 / 推定件数計算のみを行い、ファイル書換は禁止。
        """

    @abstractmethod
    def migrate(self, memory_dir: Path) -> MigrationResult:
        """in-place rewrite を実行する.

        呼び出し側 (`SchemaMigrator`) が対象ファイルを backup dir に
        退避した後に呼ばれる。本メソッド内で追加の backup を取る必要はない。
        """

    @abstractmethod
    def rollback(self, memory_dir: Path, backup_dir: Path) -> None:
        """backup_dir の内容で memory_dir を復元する.

        `SchemaMigrator` が migrate 失敗時に自動で呼ぶ。冪等であること。
        """


__all__ = [
    "ComponentKind",
    "Migration",
    "MigrationPlan",
    "MigrationResult",
]
