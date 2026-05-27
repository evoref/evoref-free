"""

自律ループの「task → action 変換」中核制御層。

以下の責務を担う:

1. プロンプト合成: 対象 task + 関連 failure_pattern + 適用 policy + 関連 fewshot
   を SemMem から取得し、LLM 入力 messages を構築する
2. action 抽出: LLM 出力をパースし、実行可能な ``Action`` 列に落とす
3. 観測注入: 実行結果 (成功/失敗/差分/エラーログ) を次サイクルのコンテキストへ
   注入する
4. policy / fewshot 適用: ExplorationController の判定を反映する

実 executor (実際にコードを編集・コマンドを実行する層) はラルフループで
差し替え可能にするため、本パッケージは ``Action`` 表現と dry-run までを提供する。

公開 API:

- :class:`backend.free.harness.action.EditFileAction` /
  :class:`RunCommandAction` / :class:`SearchAction` / :class:`NoopAction`
- :class:`backend.free.harness.action.ActionResult`
- :class:`backend.free.harness.base.Harness` Protocol
- :class:`backend.free.harness.base.DefaultHarness` 既定実装
- :func:`backend.free.harness.parser.parse_actions`
- :func:`backend.free.harness.prompt_builder.build_messages`
- :func:`backend.free.harness.dry_run.dry_run_executor`
"""

from backend.free.harness.action import (
    Action,
    ActionResult,
    EditFileAction,
    NoopAction,
    RunCommandAction,
    SearchAction,
    action_from_dict,
    action_to_dict,
)
from backend.free.harness.base import DefaultHarness, Harness
from backend.free.harness.dry_run import dry_run_executor
from backend.free.harness.parser import ParseError, parse_actions
from backend.free.harness.prompt_builder import build_messages

__all__ = [
    "Action",
    "ActionResult",
    "DefaultHarness",
    "EditFileAction",
    "Harness",
    "NoopAction",
    "ParseError",
    "RunCommandAction",
    "SearchAction",
    "action_from_dict",
    "action_to_dict",
    "build_messages",
    "dry_run_executor",
    "parse_actions",
]
