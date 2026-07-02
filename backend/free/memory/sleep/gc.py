"""Step 9 GC: ``semmem_limits`` に基づく SemMem GC

``sleep_update.SleepTimeWorker._step9_run_semmem_gc`` として
実装された GC 実行ロジックを独立 module に切り出したもの。

削除候補の選定 (``select_gc_candidates``) 自体は
:mod:`backend.free.memory.semantic.gc` の純粋関数に委譲する。本 module は
config から対象 FactType / 上限値を読み取り、scope ごとの
:class:`~backend.free.memory.semantic.store.SemanticFactStore` に対して
``delete_fact`` を順次実行するオーケストレーションを担う。

本 module は EvorefMem pillar 内部扱いのため SemanticFactStore を直接参照する。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from backend.free.memory.semantic.gc import select_gc_candidates
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import SemanticFactStore

logger = get_logger("memory.sleep.gc")


def _target_scopes(current_project_id: str | None) -> list[str]:
    """GC 対象 scope のリストを返す (``global`` + current project)。"""
    scopes: list[str] = ["global"]
    if current_project_id:
        scopes.append(f"project:{current_project_id}")
    return scopes


def run_semmem_gc(
    store_provider: Callable[[str], "SemanticFactStore | None"],
    *,
    config: dict | None,
    current_project_id: str | None = None,
) -> dict[str, int]:
    """``semmem_limits`` 超過時に lowest_score 戦略で SemMem ファクトを削除する。

    対象ストアは ``store_provider`` 経由で取得した ``global`` と
    現在の ``project:<current_project_id>`` (両方)。pinned ファクトと
    既に supersede 済のファクトは削除対象外 (``select_gc_candidates`` が保証)。

    Args:
        store_provider: ``scope`` 文字列を受けて
            :class:`SemanticFactStore` (または ``None``) を返すコールバック。
        config: ``memory.semmem_limits`` を含む設定 dict。``enforcement`` が
            ``"hard"`` 以外の場合は no-op で ``{}`` を返す。
        current_project_id: 現在のプロジェクト ID。未設定時は global のみ。

    Returns:
        ``{"<scope>:<type>": deleted_count}`` の dict。何も削除されなかった
        組合せは省略。``config`` 未設定 / soft 時は ``{}``。
    """
    cfg_mem = (config or {}).get("memory", {}) or {}
    limits_cfg = cfg_mem.get("semmem_limits") or {}
    if not limits_cfg:
        return {}
    enforcement = limits_cfg.get("enforcement", "hard")
    if enforcement != "hard":
        return {}

    deleted_total: dict[str, int] = {}
    for scope in _target_scopes(current_project_id):
        try:
            store = store_provider(scope)
        except Exception as exc:
            logger.warning("Step 9 GC: failed to obtain store %s: %s", scope, exc)
            continue
        if store is None:
            continue
        for type_name, limit_raw in limits_cfg.items():
            if type_name in ("enforcement", "gc_strategy"):
                continue
            if not isinstance(limit_raw, int) or limit_raw <= 0:
                continue
            facts = store.search_by_type(type_name)
            if len(facts) <= limit_raw:
                continue
            candidates = select_gc_candidates(facts, max_count=limit_raw)
            deleted_for_type = 0
            for fid in candidates:
                try:
                    if store.delete_fact(fid):
                        deleted_for_type += 1
                except Exception as exc:
                    logger.warning(
                        "Step 9 GC: failed to delete %s in %s: %s",
                        fid, scope, exc,
                    )
            if deleted_for_type:
                key = f"{scope}:{type_name}"
                deleted_total[key] = deleted_for_type

    if deleted_total:
        logger.info("Step 9 GC: deleted %s", deleted_total)
    return deleted_total


__all__ = ["run_semmem_gc"]
