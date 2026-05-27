"""Windows ``os.replace`` 用 retry ポリシーの共通定義

``backend/free/rag/embedding_cache.py`` の :func:`_write_meta_atomic` が
Windows 上で 4 process × 20 init を含むフル 7629 test を 3 連続全緑で
通過した検証済リトライ仕様を集約する。retry 定数を 1 箇所にまとめることで
``AtomicWriter`` / 旧 ``_write_meta_atomic`` / 学習 / SemMem の全 atomic
書き込みで同じ品質の Windows flaky 対策を共有する。

``os.replace`` は POSIX / Windows 両方で原子的だが、Windows では宛先ファイルを
別プロセスが ``read_text`` で開いている瞬間に ``MoveFileExW`` 内部の
``DeleteFile(dst)`` が共有違反 (``ERROR_SHARING_VIOLATION`` = ``PermissionError``)
を起こすことがある。試行回数 / 累積待ち時間を以下のレンジに設定する:

- 試行回数: 10 (累積最大 ~5.575 秒 = 0.025 + 0.05 + 0.1 + 0.2 + 0.4 + 0.8 + 1.0 × 4)
- 1 ステップ上限: 1.0 秒

並行プロセス間で reader の close 待ちが連鎖してもこの枠内で確実に成功する。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# 検証済定数 (embedding_cache.py:60-62 由来)。
# 変更時は 4 process × 20 init の並行テストで再検証すること。
_REPLACE_RETRY_ATTEMPTS = 10
_REPLACE_RETRY_BASE_SEC = 0.025
_REPLACE_RETRY_MAX_SEC = 1.0


def _replace_with_retry(src: Path | str, dst: Path | str) -> int:
    """``os.replace(src, dst)`` を Windows ``PermissionError`` retry 付きで実行する。

    Returns:
        実際にかかったリトライ回数 (初回成功なら 0)。

    Raises:
        PermissionError: 全リトライ後も成功しなかった場合、最後の例外をそのまま投げる。
        OSError: ``PermissionError`` 以外の OS エラーは即座に投げる (リトライしない)。
    """
    last_err: PermissionError | None = None
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(src, dst)
            return attempt
        except PermissionError as e:
            last_err = e
            delay = min(
                _REPLACE_RETRY_BASE_SEC * (2 ** attempt),
                _REPLACE_RETRY_MAX_SEC,
            )
            time.sleep(delay)
    # 全試行失敗 — ここに来るのは last_err が必ず非 None
    assert last_err is not None
    raise last_err
