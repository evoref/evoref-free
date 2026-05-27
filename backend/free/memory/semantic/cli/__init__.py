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
| ``inspect`` | fact 数 / 型別分布 / index size / orphan 検出 / manifest 表示 | :mod:`.inspect_cmd` |
| ``migrate`` | SchemaMigrator 実行 / Migration 一覧表示 | :mod:`.migrate_cmd` |
| ``compact`` | facts.jsonl の last-write-wins 圧縮 | :mod:`.compact_cmd` |
| ``rebuild-indices`` | .idx 群を facts.jsonl から決定論的に再生成 | :mod:`.rebuild_indices_cmd` |
| ``verify`` | .idx ↔ facts.jsonl 整合性 + manifest 検証 + orphan 検出 | :mod:`.verify_cmd` |
| ``export`` | semantic/ 全体を tar.gz バックアップ | :mod:`.export_import_cmd` |
| ``import`` | export からのリストア (既存データは退避) | :mod:`.export_import_cmd` |
| ``migrate-embedding`` | swap_active_model_id を人手で駆動 | :mod:`.migrate_embedding_cmd` |

## 安全性

破壊的操作 (``migrate`` / ``compact`` / ``import`` / ``migrate-embedding``)
はデフォルトで dry-run。``--apply`` フラグで実行する。

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
