"""Step 10: 180 日無アクセスプロジェクトのアーカイブ

``sleep_update.SleepTimeWorker._step10_archive_inactive_projects``
として実装されたアーカイブロジックを独立 module に切り出したもの。

``LocalStateStore.propose_archives`` で候補を取得し、``state.projects[id].
archived = True`` を立てて再提案を防いだうえで、**そのプロジェクト scope の
ファクトを ``retract(reason="project_archived")`` で退役させる** (c_16 §3)。

## ディレクトリの物理移動はもう起きない

スコープはディレクトリではなく ``Evidence.scope`` フィールドになったため
(c_16 §4.2)、``semantic/projects/<id>/`` は存在しない。``archive_project``
呼び出しは残してあるが、これは **c_16 以前に作られた残骸を掃く経路**でしか
発火しない (実体が無ければ state のフラグだけ更新して ``None`` を返す)。
移動先を指していた ``memory.project.archive_dir`` は読み手が消えたので撤去し
(c_16 §8)、掃き先は ``<memory_dir>/semantic/archive/`` 固定にした。実際に
アーカイブされた中身は semantic ストアの事象ログ + snapshot に
``veracity=retracted`` として残る。

物理削除ではなく retract にするのは c_16 §3 の状態遷移規則 — 消すと監査で
辿れなくなる。物理 GC は snapshot 3 版後の保持方針の仕事。

本 module は EvorefMem pillar 内部扱い。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.memory.semantic.store import ScopedSemanticStore

logger = get_logger("memory.sleep.archive")

#: ``retract`` 事象に刻む理由 (c_16 §5.2)。
ARCHIVE_RETRACT_REASON = "project_archived"

#: scope 文字列 → :class:`ScopedSemanticStore` を返すプロバイダ。
#: ``sleep_update`` が持つ ``state.get_semantic_store`` と同じ面。
SemanticStoreProvider = Callable[[str], "ScopedSemanticStore | None"]


def retire_project_facts(
    store_provider: SemanticStoreProvider | None, project_id: str,
) -> int:
    """アーカイブしたプロジェクト scope のファクトを retract する。

    Args:
        store_provider: scope 文字列でスコープ束縛ビューを引くプロバイダ。
            ``None`` (配線されていない) なら何もしない。
        project_id: ``project:<id>`` の ``<id>``。

    Returns:
        retract した件数。

    ``ScopedSemanticStore`` は ``delete_fact`` (理由が ``"deleted"`` 固定) しか
    面に出していないので、理由を刻むために実体の :class:`SemanticStore` を
    ``.store`` 経由で取り出して ``retract_fact`` を呼ぶ。id 引きはスコープに
    依らないので、ビュー越しに選んだ id をそのまま渡してよい。
    """
    if store_provider is None:
        logger.warning(
            "Step 10: no semantic store provider wired; facts of %s stay live",
            project_id,
        )
        return 0
    scope = f"project:{project_id}"
    try:
        view: Any = store_provider(scope)
    except Exception as exc:
        logger.warning("Step 10: failed to open semantic store for %s: %s", scope, exc)
        return 0
    if view is None:
        return 0

    store = getattr(view, "store", view)
    retract = getattr(store, "retract_fact", None)
    if retract is None:
        logger.warning("Step 10: semantic store has no retract_fact; skipping %s", scope)
        return 0

    try:
        fact_ids = [fact.id for fact in view.all_facts(include_superseded=True)]
    except Exception as exc:
        logger.warning("Step 10: failed to list facts in %s: %s", scope, exc)
        return 0

    retired = 0
    for fact_id in fact_ids:
        try:
            if retract(fact_id, ARCHIVE_RETRACT_REASON):
                retired += 1
        except Exception as exc:
            logger.warning("Step 10: failed to retract %s: %s", fact_id, exc)
    if retired:
        logger.info("Step 10: retracted %d fact(s) in %s", retired, scope)
    return retired


def archive_inactive_projects(
    *,
    config: dict | None,
    store_invalidator: Callable[[str], None] | None = None,
    store_provider: SemanticStoreProvider | None = None,
) -> list[str]:
    """180 日無アクセスのプロジェクトをアーカイブする。

    処理手順:

    1. ``memory.project.auto_archive_inactive_days`` を読む。``<= 0`` の
       場合はアーカイブ無効として ``[]`` を返す。
    2. ``path_resolver`` から ``local_state_file`` / ``memory_dir`` を解決する。
    3. ``LocalStateStore.load`` / ``propose_archives`` で候補を取得する。
    4. 候補ごとに ``archive_project`` で ``archived=True`` を立てる
       (c_16 以前の ``semantic/projects/<id>/`` が残っていれば移動もする)。
    5. そのプロジェクト scope のファクトを ``retract`` する
       (:func:`retire_project_facts`)。
    6. ``store_invalidator(f'project:{pid}')`` でキャッシュ済ビューを破棄。
    7. 全候補処理後に ``LocalStateStore.save`` で state.json を再書き込み。

    Args:
        config: ``memory.project`` 配下の設定を含む設定 dict。
        store_invalidator: scope 文字列を受けてキャッシュ済のスコープ束縛
            ビューを破棄するコールバック。
        store_provider: scope 文字列で :class:`ScopedSemanticStore` を引く
            プロバイダ。``None`` だとファクトの retract は行われない。

    Returns:
        実際にアーカイブしたプロジェクト ID のリスト。

    Note:
        path_resolver や local_state_store が import できない / パス解決
        に失敗した場合は warning ログを残して ``[]`` を返す
        (sleep-time 全体は止めない)。
    """
    try:
        from backend.config import get_path_resolver
        from backend.free.memory.local_state_store import (
            LocalStateStore,
            archive_project,
            propose_archives,
        )
    except Exception as exc:
        logger.warning("Step 10: failed to import dependencies: %s", exc)
        return []

    cfg_mem = (config or {}).get("memory", {}) or {}
    proj_cfg = cfg_mem.get("project") or {}
    threshold = int(proj_cfg.get("auto_archive_inactive_days", 180))
    if threshold <= 0:
        logger.debug("Step 10: archival disabled (threshold=%s)", threshold)
        return []
    try:
        resolver = get_path_resolver()
        state_path = Path(resolver.resolve_local("local_state_file"))
        memory_dir = Path(resolver.resolve_local("memory_dir"))
    except Exception as exc:
        logger.warning("Step 10: failed to resolve paths: %s", exc)
        return []

    semantic_root = memory_dir / "semantic"
    # c_16 以前の残骸の掃き先。新規インストールでは src が存在しないので使われない。
    archive_dir = semantic_root / "archive"

    state = LocalStateStore.load(state_path)
    candidates = propose_archives(state, threshold_days=threshold)
    if not candidates:
        return []
    archived: list[str] = []
    retired_total = 0
    for pid in candidates:
        try:
            archive_project(
                state, pid,
                semantic_root=semantic_root,
                archive_dir=archive_dir,
            )
        except Exception as exc:
            logger.warning("Step 10: failed to archive %s: %s", pid, exc)
            continue
        retired_total += retire_project_facts(store_provider, pid)
        if store_invalidator is not None:
            try:
                store_invalidator(f"project:{pid}")
            except Exception as exc:
                logger.warning(
                    "Step 10: failed to invalidate cached store for %s: %s",
                    pid, exc,
                )
        archived.append(pid)

    if archived:
        try:
            LocalStateStore.save(state_path, state)
        except Exception as exc:
            logger.warning("Step 10: failed to save state.json: %s", exc)
        logger.info(
            "Step 10: archived %d inactive project(s) (%d fact(s) retracted): %s",
            len(archived), retired_total, archived,
        )
    return archived


__all__ = [
    "ARCHIVE_RETRACT_REASON",
    "archive_inactive_projects",
    "retire_project_facts",
]
