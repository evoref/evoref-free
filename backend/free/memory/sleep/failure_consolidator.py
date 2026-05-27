"""Step 13: failure_pattern 統合

``sleep_update.SleepTimeWorker._step13_consolidate_failure_patterns``
として実装された failure_pattern 統合ロジックを独立 module に切り出したもの。

対象は現在の project スコープのみ (global には failure_pattern は書き込まない
運用)。実際の統合処理は EvorefLoop pillar 内の
:func:`backend.free.loop.failure_note.consolidate_failure_patterns` に委譲する。
本 module は「sleep-time cycle の工程として Step 13 を呼び出す」という
オーケストレーションのみを担う。

Step 8 の直後に呼ぶことで、当該 sleep-time iteration で新たに抽出された
failure_pattern と、既にチャット応答パスの
``loop.write_failure_note`` で即時書き込みされた失敗レコードを同一 signature
でマージする。

本 module は EvorefMem pillar 内部扱いで、LoopFactView を経由して
``consolidate_failure_patterns`` を呼ぶ (Loop 所有の FactType への書込は
LoopFactView が唯一の境界)。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticFactStore

logger = get_logger("memory.sleep.failure_consolidator")


def consolidate_failure_patterns_for_project(
    store_provider: Callable[[str], "SemanticFactStore | None"],
    *,
    config: dict | None,
    current_project_id: str | None,
) -> dict[str, int]:
    """現在の project スコープで failure_pattern を統合する。

    Guards:

    - ``memory.facts.enable_extraction = False`` → no-op ``{}``
    - ``memory.facts.extract_from_mdp_trace = False`` → no-op ``{}``
    - ``store_provider`` が ``None`` / ``current_project_id`` 未設定 → no-op ``{}``
    - store 取得失敗 → warning ログ + ``{}``

    Args:
        store_provider: ``scope`` 文字列を受けて
            :class:`SemanticFactStore` (または ``None``) を返すコールバック。
        config: ``memory.facts`` を含む設定 dict。
        current_project_id: 現在のプロジェクト ID。未設定時は no-op。

    Returns:
        :class:`~backend.free.loop.failure_note.ConsolidationSummary` の
        ``as_dict()`` 互換形式。no-op 時は空 dict。
    """
    cfg_facts = (config or {}).get("memory", {}).get("facts", {}) or {}
    if not cfg_facts.get("enable_extraction", True):
        return {}
    if not cfg_facts.get("extract_from_mdp_trace", True):
        return {}
    if store_provider is None or not current_project_id:
        return {}

    try:
        project_store = store_provider(f"project:{current_project_id}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Step 13: failed to obtain project store: %s", exc,
        )
        return {}
    if project_store is None:
        return {}

    # SemanticFactStore 直参照を廃止し LoopFactView 経由に統一
    # Step 13 は project スコープの store を writeback に持つ view を使う。
    from backend.free.loop.failure_note import consolidate_failure_patterns
    from backend.free.memory.views.loop import LoopFactView

    view = LoopFactView(
        stores=[project_store], writeback_store=project_store,
    )
    project_scope = f"project:{current_project_id}"
    summary = consolidate_failure_patterns(view, scope=project_scope)
    return summary.as_dict()


__all__ = ["consolidate_failure_patterns_for_project"]
