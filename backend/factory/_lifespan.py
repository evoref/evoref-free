"""FastAPI lifespan + shutdown ヘルパー

含まれるシンボル:

- :func:`_log_timings`                 : startup/shutdown timings サマリ INFO 出力
- ``_shutdown_*`` ヘルパー (9 種)        : level1 loop 停止 / 学習 cancel /
  WM flush / STM 永続化 / 経験バッファ / 学習済みパターン / アシスト経験 /
  LLMClient / Embedder / AssistClient / Pro
- :func:`_run_lifespan_startup`        : :func:`wire_pillars` への薄いエントリ
- :func:`_run_lifespan_shutdown`       : 全 shutdown ステップを timings 計測しつつ実行
- :func:`lifespan`                     : ``@asynccontextmanager`` 化された FastAPI lifespan

純粋な move であり、関数本体・引数・default 値は変更していない。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from backend.app_state import AppState
from backend.factory._pillar_wirer import (
    DevelopShutdownHook,
    ProShutdownHook,
    _LifespanContext,
    _timed,
    wire_pillars,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.learning.level0_instant import ExperienceBuffer
    from backend.free.learning.scheduler import LearningScheduler
    from backend.free.memory.scheduler import SleepTimeScheduler
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.memory.stores.working import WorkingMemory
    from backend.pro.assist_experience import AssistExperienceBuffer

logger = get_logger("factory.lifespan")


def _log_timings(label: str, timings: dict[str, float], total_ms: float) -> None:
    """タイミングサマリを INFO ログに出力"""
    summary = {k: round(v, 1) for k, v in timings.items()}
    logger.info("%s component timings (ms): %s — total: %.1f ms", label, summary, total_ms)


# ──────────────────────────────────────────────────────────────────────────
# Shutdown helpers
# ──────────────────────────────────────────────────────────────────────────


async def _shutdown_level1_loop(sleep_scheduler: "SleepTimeScheduler") -> None:
    """Level 1 常駐ループ停止（新規起動を防ぐ）"""
    try:
        await sleep_scheduler.stop_level1_loop()
    except Exception as e:
        logger.warning("Level 1 loop stop failed: %s", e)


async def _shutdown_learning_cancel(learning_scheduler: "LearningScheduler") -> None:
    """学習スケジューラの destructive cancel と完了待機"""
    try:
        learning_scheduler.cancel(graceful=False)
        # 走っている _task があれば完了を待つ（finally で session 保存される）
        task = getattr(learning_scheduler, "_task", None)
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.warning("Learning task await failed: %s", e)
    except Exception as e:
        logger.warning("Learning scheduler cancel failed: %s", e)


def _shutdown_wm_flush(wm: "WorkingMemory", stm: "ShortTermMemory") -> None:
    """WorkingMemory の残存ターンを STM にフラッシュ"""
    try:
        wm.clear()  # 全ターンを _evicted に移動
        evicted = wm.drain_evicted()
        for turn in evicted:
            stm.absorb(turn, wm.session_id)
        if evicted:
            logger.info("Flushed %d remaining turns to STM on shutdown", len(evicted))
    except Exception as e:
        logger.warning("WM flush to STM failed: %s", e)


def _shutdown_stm_save(stm: "ShortTermMemory", resolver: Any) -> None:
    """STM を永続化"""
    try:
        memory_dir = resolver.resolve_local("memory_dir")
        stm.save(memory_dir / "short_term_notes.json")
        logger.info("STM saved on shutdown: %d notes", len(stm.notes))
    except Exception as e:
        logger.warning("STM save on shutdown failed: %s", e)


def _shutdown_experience_save(exp_buf: "ExperienceBuffer", exp_file: Path) -> None:
    """経験バッファを保存"""
    try:
        exp_buf.save(exp_file)
        logger.info("Experience buffer saved: %d entries", len(exp_buf.entries))
    except Exception as e:
        logger.warning("Experience buffer save failed: %s", e)


def _shutdown_patterns_save(
    learned_patterns_store: "LearnedPatternStore", patterns_file: Path,
) -> None:
    """学習済みパターンを保存"""
    try:
        learned_patterns_store.save(patterns_file)
        logger.info("Learned patterns saved: %d patterns", learned_patterns_store.count)
    except Exception as e:
        logger.warning("Learned patterns save failed: %s", e)


async def _shutdown_assist_experience_save(
    assist_exp_buf: "AssistExperienceBuffer | None", resolver: Any,
) -> None:
    """アシストモデル経験バッファを保存"""
    if assist_exp_buf is None:
        return
    try:
        assist_exp_file = resolver.resolve_local("experience_assist_file")
        await assist_exp_buf.save(assist_exp_file)
        logger.info(
            "Assist experience buffer saved: %d entries",
            assist_exp_buf.count,
        )
    except Exception as e:
        logger.warning("Assist experience buffer save failed: %s", e)


async def _shutdown_llm_client(state: AppState) -> None:
    """LLMClient の HTTP クライアントを閉じる（LocalClient のみ）"""
    llm_c = state.llm_client
    if llm_c:
        try:
            await llm_c.aclose()
        except Exception as e:
            logger.warning("LLMClient close failed: %s", e)
        return
    # LLMClient がない場合は LocalClient を直接閉じる
    lc = state.local_client
    if lc:
        try:
            await lc.aclose()
        except Exception as e:
            logger.warning("LocalClient close failed: %s", e)


async def _shutdown_embedder(state: AppState) -> None:
    """埋め込みバックエンドを閉じる（キャッシュ永続化 + HTTP クライアント cleanup）"""
    emb = state.embedder
    if emb is None:
        return
    try:
        if hasattr(emb, "aclose"):
            await emb.aclose()
    except Exception as e:
        logger.warning("Embedder close failed: %s", e)


async def _shutdown_assist_client(state: AppState) -> None:
    """アシストモデルクライアントを閉じる（lazy-connect で後から設定された場合も含む）"""
    active_assist = state.assist_client
    if active_assist is None:
        return
    try:
        await active_assist.aclose()
    except Exception as e:
        logger.warning("AssistModelClient close failed: %s", e)


async def _shutdown_pro(pro_shutdown: ProShutdownHook, state: AppState) -> None:
    """Pro ライフサイクル シャットダウン（WidgetProxyManager 含む）"""
    if pro_shutdown is None:
        return
    try:
        await pro_shutdown(state)
    except Exception as e:
        logger.warning("Pro lifecycle shutdown failed: %s", e)


async def _shutdown_develop(
    develop_shutdown: DevelopShutdownHook, state: AppState,
) -> None:
    """Develop ライフサイクル シャットダウン (現状はスケルトン段階で no-op)"""
    if develop_shutdown is None:
        return
    try:
        await develop_shutdown(state)
    except Exception as e:  # noqa: BLE001
        logger.warning("Develop lifecycle shutdown failed: %s", e)


async def _shutdown_evolve_pipeline(
    log_ingestor: Any, policy_adjuster: Any, bridge_task: Any,
) -> None:
    """develop=evolve 用パイプラインの shutdown

    順序: bridge_task cancel → ingestor.stop (offset 永続化 + 残バッファ
    orphan flush) → adjuster.flush_all (閾値到達済の bucket を最終 emit)。

    bridge の ``finally`` 内で flush_all は呼ばれるため、ingestor.stop
    後の最終 flush は冗長 (二度呼んでも no-op)。安全側に明示する。
    """
    if bridge_task is not None:
        if not bridge_task.done():
            bridge_task.cancel()
        try:
            await bridge_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if log_ingestor is not None:
        try:
            await log_ingestor.stop()
        except Exception as e:  # noqa: BLE001
            logger.warning("LogIngestor.stop failed: %s", e)
    if policy_adjuster is not None:
        try:
            await policy_adjuster.flush_all()
        except Exception as e:  # noqa: BLE001
            logger.warning("PolicyAdjuster.flush_all failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────
# lifespan
# ──────────────────────────────────────────────────────────────────────────


async def _run_lifespan_startup(
    state: AppState, project_root: Path,
) -> tuple[_LifespanContext, dict[str, float]]:
    """起動シーケンス全体を実行する"""
    return await wire_pillars(state, project_root)


async def _run_lifespan_shutdown(
    state: AppState, ctx: _LifespanContext,
) -> dict[str, float]:
    """シャットダウンシーケンス全体を実行し、timings を返す。

    順序: stop_level1_loop (新規起動を防ぐ) → cancel (graceful=False) →
    in-flight タスクが finally で session を保存。
    """
    shutdown_timings: dict[str, float] = {}
    with _timed(shutdown_timings, "level1_loop_stop"):
        await _shutdown_level1_loop(ctx.sleep_scheduler)
    with _timed(shutdown_timings, "learning_cancel"):
        await _shutdown_learning_cancel(ctx.learning_scheduler)
    with _timed(shutdown_timings, "wm_flush"):
        _shutdown_wm_flush(ctx.wm, ctx.stm)
    with _timed(shutdown_timings, "stm_save"):
        _shutdown_stm_save(ctx.stm, ctx.resolver)
    with _timed(shutdown_timings, "experience_save"):
        _shutdown_experience_save(ctx.exp_buf, ctx.exp_file)
    with _timed(shutdown_timings, "patterns_save"):
        _shutdown_patterns_save(ctx.learned_patterns_store, ctx.patterns_file)
    with _timed(shutdown_timings, "assist_experience_save"):
        await _shutdown_assist_experience_save(ctx.assist_exp_buf, ctx.resolver)
    with _timed(shutdown_timings, "loop_run_cancel"):
        task = getattr(state, "loop_run_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
    # develop=evolve 時に起動した LogIngestor + PolicyAdjuster
    # bridge を安全停止する (loop_run_cancel の直後で、LLM client close より
    # 前に走らせる: bridge は LLM を直接呼ばないが、shutdown 順序を
    # 「pillar 内 → 外部 I/O」に揃えるため)。
    with _timed(shutdown_timings, "evolve_pipeline_shutdown"):
        await _shutdown_evolve_pipeline(
            ctx.log_ingestor, ctx.policy_adjuster, ctx.log_ingestor_bridge_task,
        )
    with _timed(shutdown_timings, "llm_client_close"):
        await _shutdown_llm_client(state)
    with _timed(shutdown_timings, "embedder_close"):
        await _shutdown_embedder(state)
    with _timed(shutdown_timings, "assist_client_close"):
        await _shutdown_assist_client(state)
    with _timed(shutdown_timings, "pro_shutdown"):
        await _shutdown_pro(ctx.pro_shutdown, state)
    with _timed(shutdown_timings, "develop_shutdown"):
        await _shutdown_develop(ctx.develop_shutdown, state)
    with _timed(shutdown_timings, "llama_manager_shutdown"):
        if state.llama_manager is not None:
            try:
                state.llama_manager.shutdown_all()
            except Exception as e:  # noqa: BLE001
                logger.warning("llama_manager shutdown_all failed: %s", e)
    return shutdown_timings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理

    起動・シャットダウンを `_run_lifespan_startup` / `_run_lifespan_shutdown`
    に委譲し、本関数はタイミング計測サマリと完了ログのみを担う。
    """
    project_root = Path(__file__).parent.parent.parent
    state = AppState()
    app.state.app_state = state

    startup_start = time.monotonic()
    ctx, timings = await _run_lifespan_startup(state, project_root)
    startup_total_ms = (time.monotonic() - startup_start) * 1000
    _log_timings("Startup", timings, startup_total_ms)
    from backend.edition import current_edition
    logger.info(
        "evoref backend started: instance=%s, edition=%s",
        ctx.instance_name, current_edition().name,
    )

    yield

    shutdown_start = time.monotonic()
    shutdown_timings = await _run_lifespan_shutdown(state, ctx)
    shutdown_total_ms = (time.monotonic() - shutdown_start) * 1000
    _log_timings("Shutdown", shutdown_timings, shutdown_total_ms)
    logger.info("evoref backend shutting down")
