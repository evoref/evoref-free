#!/usr/bin/env python
"""EvorefMem 初期化スクリプト

本スクリプトは
``init`` / ``verify`` サブコマンドへの **薄い後方互換エイリアス** として残す.

新規利用は ``evorefmem_cli.py`` を直接呼ぶこと:

    # 推奨
    python scripts/evorefmem_cli.py init               # 初期化
    python scripts/evorefmem_cli.py verify             # 状態確認 (旧 --check の上位互換)

    # 旧来の使い方 (引き続き動作)
    python scripts/init_evorefmem.py            # init と等価
    python scripts/init_evorefmem.py --check    # verify (旧形式の出力を維持)

``--check`` を渡した場合は旧 ``init_evorefmem.py --check`` と同じ簡易出力
(memory_dir / expected version / actual version / status) を維持する。
詳細な orphan 検出 / .idx 整合性チェック等は ``evorefmem_cli.py verify``
を用いること。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# evorefmem_cli の compat 関数を再利用する
from scripts.evorefmem_cli import (  # noqa: E402
    _cmd_check_compat,
    _cmd_init,
    acquire_cli_lock,
    release_cli_lock,
    CliLockError,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="状態確認のみ実施 (副作用なし)。終了コードは一致時 0 / 不一致時 1。"
             "詳細チェックは `evorefmem_cli.py verify` を推奨。",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="多重起動防止ロックを取得しない (テスト用)",
    )
    args = parser.parse_args()

    acquired = False
    if not args.no_lock:
        try:
            acquire_cli_lock()
            acquired = True
        except CliLockError as exc:
            print(f"[init_evorefmem] {exc}", file=sys.stderr)
            return 2
    try:
        if args.check:
            return _cmd_check_compat(args)
        return _cmd_init(args)
    finally:
        if acquired:
            release_cli_lock()


if __name__ == "__main__":
    raise SystemExit(main())
