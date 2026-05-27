"""Step 10: 180 日無アクセスプロジェクトのアーカイブ

``sleep_update.SleepTimeWorker._step10_archive_inactive_projects``
として実装されたアーカイブロジックを独立 module に切り出したもの。

``LocalStateStore.propose_archives`` で候補を取得し、各プロジェクトの
``local/memory/semantic/projects/<id>/`` を config の ``project.archive_dir``
配下に物理移動する。``state.projects[id].archived = True`` を立てて再提案
を防ぐ。移動後に ``store_invalidator`` 経由で AppState のキャッシュを破棄する。

本 module は EvorefMem pillar 内部扱い。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from backend.log_config import get_logger

logger = get_logger("memory.sleep.archive")


def archive_inactive_projects(
    *,
    config: dict | None,
    store_invalidator: Callable[[str], None] | None = None,
) -> list[str]:
    """180 日無アクセスのプロジェクトを ``semantic/archive/`` に移動する。

    処理手順:

    1. ``memory.project.auto_archive_inactive_days`` を読む。``<= 0`` の
       場合はアーカイブ無効として ``[]`` を返す。
    2. ``path_resolver`` から ``local_state_file`` / ``memory_dir`` を解決する。
    3. ``LocalStateStore.load`` / ``propose_archives`` で候補を取得する。
    4. 候補ごとに ``archive_project`` を呼んで物理移動する。移動成功後に
       ``store_invalidator(f'project:{pid}')`` を呼び、キャッシュを破棄する。
    5. 全候補処理後に ``LocalStateStore.save`` で state.json を再書き込み。

    Args:
        config: ``memory.project`` 配下の設定を含む設定 dict。
        store_invalidator: scope 文字列を受けてキャッシュ済
            :class:`SemanticFactStore` を破棄するコールバック。

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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Step 10: failed to import dependencies: %s", exc)
        return []

    cfg_mem = (config or {}).get("memory", {}) or {}
    proj_cfg = cfg_mem.get("project") or {}
    threshold = int(proj_cfg.get("auto_archive_inactive_days", 180))
    if threshold <= 0:
        logger.debug("Step 10: archival disabled (threshold=%s)", threshold)
        return []
    archive_dir_str = (
        proj_cfg.get("archive_dir") or "local/memory/semantic/archive"
    )

    try:
        resolver = get_path_resolver()
        state_path = Path(resolver.resolve_local("local_state_file"))
        memory_dir = Path(resolver.resolve_local("memory_dir"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Step 10: failed to resolve paths: %s", exc)
        return []

    semantic_root = memory_dir / "semantic"
    archive_dir = Path(archive_dir_str)
    if not archive_dir.is_absolute():
        # resolver を介さない簡便フォールバック (通常使われない)。
        archive_dir = semantic_root.parent.parent.parent / archive_dir_str

    state = LocalStateStore.load(state_path)
    candidates = propose_archives(state, threshold_days=threshold)
    if not candidates:
        return []
    archived: list[str] = []
    for pid in candidates:
        try:
            archive_project(
                state, pid,
                semantic_root=semantic_root,
                archive_dir=archive_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Step 10: failed to archive %s: %s", pid, exc)
            continue
        if store_invalidator is not None:
            try:
                store_invalidator(f"project:{pid}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Step 10: failed to invalidate cached store for %s: %s",
                    pid, exc,
                )
        archived.append(pid)

    if archived:
        try:
            LocalStateStore.save(state_path, state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Step 10: failed to save state.json: %s", exc)
        logger.info(
            "Step 10: archived %d inactive projects: %s",
            len(archived), archived,
        )
    return archived


__all__ = ["archive_inactive_projects"]
