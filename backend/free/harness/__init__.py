"""ハーネスの Action 表現 / 制作ステージポート

LLM / 生成工程が作る「ファイル編集 / コマンド実行 / 検索」を表す ``Action`` と
その実行結果 ``ActionResult``。実行は :class:`backend.free.loop.action_runner.ActionRunner`
が担う (クリエイトの staged パイプラインの test 工程が使う)。

``ProductionHarness`` (Phase 3a、f_03 §4.4) は create の制作 (staged /
longform) を meta の 1 ステージとして動かすための共通ポート。

公開 API:

- :class:`backend.free.harness.action.EditFileAction` /
  :class:`RunCommandAction` / :class:`SearchAction` / :class:`NoopAction`
- :class:`backend.free.harness.action.ActionResult`
- :class:`backend.free.harness.production.ProductionRequest` /
  :class:`ProductionEvent` / :class:`ProductionResult` / :class:`ProductionHarness`
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
from backend.free.harness.production import (
    ProductionEvent,
    ProductionHarness,
    ProductionRequest,
    ProductionResult,
)

__all__ = [
    "Action",
    "ActionResult",
    "EditFileAction",
    "NoopAction",
    "ProductionEvent",
    "ProductionHarness",
    "ProductionRequest",
    "ProductionResult",
    "RunCommandAction",
    "SearchAction",
    "action_from_dict",
    "action_to_dict",
]
