"""

``Action`` 列を実際には実行せず、観測 (``ActionResult``) として展開する
スタブ executor。受け入れ基準「dry-run executor で Action 列を表示するだけの
テストパス」を満たすために提供する。

実 executor は本モジュールを置き換える形で実装される
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from backend.free.harness.action import (
    Action,
    ActionResult,
    EditFileAction,
    NoopAction,
    RunCommandAction,
    SearchAction,
)


def dry_run_executor(actions: Iterable[Action]) -> list[ActionResult]:
    """``Action`` 列を順に「実行したことにする」スタブ executor。

    各 Action に対し ``success=True`` の ``ActionResult`` を生成する。
    出力欄には Action の概要文字列を入れ、呼び出し側 (テスト・harness.observe)
    が中身を確認できるようにする。

    本関数は副作用を一切持たない (ファイル IO もコマンド実行もしない)。
    """
    results: list[ActionResult] = []
    for action in actions:
        start = time.perf_counter()
        summary = _summarize(action)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            ActionResult(
                action=action,
                success=True,
                output=summary,
                duration_ms=elapsed_ms,
                metadata={"executor": "dry_run"},
            ),
        )
    return results


def _summarize(action: Action) -> str:
    """Action の概要文字列を返す (デバッグ表示・テスト用)。"""
    match action:
        case EditFileAction(path=path):
            return f"edit_file path={path!r}"
        case RunCommandAction(command=command):
            return "run_command " + " ".join(command)
        case SearchAction(query=query):
            return f"search query={query!r}"
        case NoopAction(reason=reason):
            return f"noop reason={reason!r}"
