"""evorefmem CLI 実装モジュール

`scripts/evorefmem_cli.py` のエントリポイントから呼ばれる subcommand 群を
提供する。各 subcommand は副作用なしの ``plan`` 系関数 (dry-run 出力を返す)
と、``apply`` 系関数 (実際の書き換えを行う) のペアで構成される。

CLI 本体 (``scripts/evorefmem_cli.py``) は argparse の薄いラッパに留め、
本パッケージ配下のモジュールが実装責務を持つ。これにより subcommand 単位で
単体テストが可能になる。

## サブコマンド一覧

| サブコマンド | 用途 | 実装 |
|---|---|---|
| ``init`` | EvorefMem 初期化 (``init_evorefmem`` 委譲) | エントリ側 |
| ``inspect`` | fact 数 / 型別 / namespace 別 / scope 別の分布と snapshot 状態 | :mod:`.inspect_cmd` |
| ``migrate`` | SchemaMigrator 実行 / Migration 一覧表示 | :mod:`.migrate_cmd` |
| ``verify`` | supersession / 競合 / 埋め込み被覆 / claim の出所検査 | :mod:`.verify_cmd` |
| ``purge-private`` | private 由来のキュレーターファクトを取り下げる | :mod:`.purge_private_cmd` |
| ``export`` | semantic/ 全体を tar.gz バックアップ | :mod:`.export_import_cmd` |
| ``import`` | export からのリストア (既存データは退避) | :mod:`.export_import_cmd` |

``compact`` / ``rebuild-indices`` / ``migrate-embedding`` / ``reembed-facts``
は撤去した — 事象ログの畳み込み・転置索引・埋め込みはすべて sleep-time の
snapshot 生成が担う (c_16 §5.3 / §6)。

## 安全性

破壊的操作 (``migrate`` / ``import`` / ``purge-private``) はデフォルトで
dry-run。``--apply`` フラグで実行する。

多重起動防止のために :func:`acquire_cli_lock` / :func:`release_cli_lock` が
PID ベースの :data:`CLI_LOCK_PATH` を使う。
"""

from backend.free.memory.semantic.cli.lock import (
    CLI_LOCK_PATH,
    CliLockError,
    acquire_cli_lock,
    release_cli_lock,
)

__all__ = [
    "CLI_LOCK_PATH",
    "CliLockError",
    "acquire_cli_lock",
    "release_cli_lock",
]
