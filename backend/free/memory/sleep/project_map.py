"""Step 5.87: ProjectMap (既存プロジェクトの code グラフ) の更新 (c_16 §4.4)。

file / class / function ノードと contains / imports / calls / inherits 辺を、
corpus の 1 パッケージ (root ごと) として tree-sitter による決定論抽出で持つ。
LLM を一切使わない層なので Step 5.85 (corpus 再構築) の直後・Step 5.9 (疑似
クエリ、LLM 生成) より前に置ける。応答パスは読むだけ、書き手は sleep-time
だけ (c_16 §2.1)。

本 module は EvorefMem pillar 内部扱いだが、実質は EvorefGen の
``backend.free.rag.projectmap`` を操作するオーケストレーション層 (旧
Step 5.9 の ``pseudo_query.py`` と同じ形)。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from backend.log_config import get_logger
from backend.trace_context import run_in_executor_with_context

if TYPE_CHECKING:
    from backend.config import PathResolver

logger = get_logger("memory.sleep.project_map")


def _pm_config(config: dict) -> dict:
    return ((config.get("rag") or {}).get("project_map") or {})


#: 静穏窓の既定 (秒)。5.9 (疑似クエリ) と同じ判定を使う (c_16 §4.4)。
DEFAULT_QUIET_SECONDS = 120.0

#: 構築を逃がす 1 スレッド executor (root ごとに直列)。
_PROJECT_MAP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="project-map")

#: 版を積まなかった / 走らなかったことを表す ``UpdateResult.update_kind``。
_NO_VERSION_UPDATE_KINDS = frozenset({"skip", "unavailable", "paused", "invalid"})

#: Step 5.87 (Full サイクル) と手動 API 実行が同じ root に同時に版を書くのを防ぐ
#: (版番号が active+1 で決まるので衝突する)。実行中のループから遅延生成する
#: (asyncio.Lock はイベントループ束縛が無い 3.10+ でもモジュール import 時点では
#: ループが無いことがあるため)。
_UPDATE_LOCK: asyncio.Lock | None = None


def _get_update_lock() -> asyncio.Lock:
    global _UPDATE_LOCK
    if _UPDATE_LOCK is None:
        _UPDATE_LOCK = asyncio.Lock()
    return _UPDATE_LOCK


def is_project_map_update_running() -> bool:
    """他の呼び手 (Step 5.87 / 手動 API) が既に :func:`update_project_map` を実行中か。

    ロック待ちで API リクエストを数分塞がないよう、呼出側はブロックする前に
    これで確認できる。
    """
    return _UPDATE_LOCK is not None and _UPDATE_LOCK.locked()


def quiet_seconds(config: dict) -> float:
    """``rag.project_map.update.quiet_seconds`` (c_16 §4.4)。"""
    try:
        update_cfg = _pm_config(config).get("update") or {}
        return float(update_cfg.get("quiet_seconds", DEFAULT_QUIET_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_QUIET_SECONDS


async def update_project_map(
    *,
    config: dict,
    resolver: "PathResolver",
    is_cancelled: Callable[[], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> int:
    """Step 5.87 本体。新しい版を積んだ root (パッケージ) の数を返す。

    Args:
        config: アプリ全体設定 (``rag.project_map`` を読む)。
        resolver: ``local_paths`` / プロジェクトルートの解決器。
        is_cancelled: ``True`` ならサイクル全体を打ち切る。
        should_pause: ``True`` を返したら root 境界で打ち切る (チャット開始
            への協調 yield)。残りは次サイクルで拾う。
    """
    cfg = _pm_config(config)
    if not bool(cfg.get("enabled", True)):
        return 0
    try:
        from backend.free.rag.evidence.config import merge_rag_evidence_config
        from backend.free.rag.projectmap import ProjectMapBuilder, is_available
    except ImportError as e:
        logger.warning("Step 5.87: ProjectMap module unavailable: %s", e)
        return 0
    if not is_available():
        logger.info("Step 5.87: no parser backend available, skipping code graph update")
        return 0

    async with _get_update_lock():
        # EvidenceStore (転置索引 / 保持方針) は corpus と同じ 1 枚の面を読む (c_16 §9)。
        # ``rag.project_map`` はその面の中から builder が引く。
        rag_config = merge_rag_evidence_config(config)
        roots = cfg.get("roots") or ["."]
        corpus_dir = resolver.resolve_corpus_dir()
        loop = asyncio.get_running_loop()
        updated = 0
        for root_rel in roots:
            if (is_cancelled and is_cancelled()) or (should_pause and should_pause()):
                logger.info("Step 5.87: paused/cancelled after %d root(s)", updated)
                break
            root = (resolver.root / root_rel).resolve()
            builder = ProjectMapBuilder(corpus_dir, root, rag_config=rag_config)

            def _run_in_worker(b: ProjectMapBuilder = builder) -> Any:
                # 走査 + tree-sitter 抽出 + 38k 件級の put は同期処理で、初回は 2 分級。
                # メインループで回すと SSE keepalive とチャットが止まるので、専用
                # スレッドの別ループで完走させる (builder は自前の EvidenceStore を
                # 持つので、ロックも他ストアと共有しない)。
                return asyncio.run(b.update(is_cancelled=is_cancelled, should_pause=should_pause))

            try:
                result = await run_in_executor_with_context(loop, _PROJECT_MAP_EXECUTOR, _run_in_worker)
            except Exception as e:  # noqa: BLE001 — 1 root の失敗で他 root を止めない
                logger.warning("Step 5.87: update failed for root %s: %s", root, e)
                continue
            finally:
                builder.close()
            logger.info(
                "Step 5.87: root=%s update_kind=%s nodes=%d edges=%d changed_files=%d",
                root, result.update_kind, result.nodes, result.edges, result.changed_files,
            )
            if result.update_kind not in _NO_VERSION_UPDATE_KINDS:
                updated += 1
        return updated


__all__ = ["is_project_map_update_running", "quiet_seconds", "update_project_map"]
