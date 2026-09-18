"""Step 5.89: staged クリエイトの run GC (f_10 §7 / c_05 §0.5.6)。

``local/create/<run_id>/`` (run.json + events.jsonl + WorkspaceManager 生成物)
は ``create.runs_keep`` (既定 20) を超えた古い run から ``ended_at`` 順に削除
する。走行中 (``activity_state == "running"``) は対象外。

実ロジックは :mod:`backend.free.loop.staged.run_record` (EvorefLoop pillar)。
本 module は EvorefMem pillar 内部扱いだが、実質は EvorefLoop 側のデータを
GC するオーケストレーション層 (``project_map.py`` と同じ形)。EvorefMem →
EvorefLoop の参照は ``backend/free/tests/test_pillar_boundary.py`` の
``LAZY_IMPORT_EXCEPTIONS`` に登録済み (``failure_consolidator.py`` の前例と同じ
理由: sleep-time の GC ステップが loop 側の公開データ層を関数内 lazy import
で読む)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.config import PathResolver

logger = get_logger("memory.sleep.create_runs")

#: ``create.runs_keep`` の出荷既定 (f_10 §7 / c_05 §0.5.6)。
DEFAULT_RUNS_KEEP = 20


def gc_staged_create_runs(*, config: dict, resolver: "PathResolver") -> int:
    """Step 5.89 本体。GC した run 数を返す。

    Args:
        config: アプリ全体設定 (``create.runs_keep`` を読む)。
        resolver: ``local_paths.create_workspace_dir`` の解決器。
    """
    create_cfg = (config.get("create") or {})
    keep = int(create_cfg.get("runs_keep", DEFAULT_RUNS_KEEP))
    try:
        from backend.free.loop.staged.run_record import gc_old_runs, stale_after_from_config
        create_dir = resolver.resolve_local("create_workspace_dir")
        removed = gc_old_runs(
            create_dir, keep=keep, stale_after_sec=stale_after_from_config(config),
        )
    except Exception as e:  # noqa: BLE001 — GC の失敗で Full を止めない
        logger.warning("Step 5.89: staged create run GC failed: %s", e)
        return 0
    return len(removed)


__all__ = ["DEFAULT_RUNS_KEEP", "gc_staged_create_runs"]
