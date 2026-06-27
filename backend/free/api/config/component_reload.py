"""設定変更時のコンポーネント再生成ヘルパー

config_api でセクション保存後、対象コンポーネントを再生成して
AppState に差し替える。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backend.config import get_config, get_path_resolver
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.app_state import AppState

logger = get_logger("api.component_reload")

# backend/free/api/config/component_reload.py から見て parents[4] がリポジトリルート。
# parents[3] (= backend/) を渡すと cache_dir 等の相対パスが backend/local/ に解決され、
# 正規の起動経路 (_pillar_wirer / component_rebind) とずれた迷子データを生む。
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


async def reload_embedder(state: AppState) -> None:
    """embedding セクション変更時に embedder を再生成して差し替える"""
    from backend.free.rag.embedding_factory import create_embedding_backend

    cfg = get_config()

    # 旧 embedder を閉じる
    old = state.embedder
    if old is not None and hasattr(old, "aclose"):
        try:
            await old.aclose()
        except Exception as e:
            logger.warning("Failed to close old embedder: %s", e)

    try:
        embedder = create_embedding_backend(
            cfg, _PROJECT_ROOT, debug_logger=state.debug_logger,
        )
        state.embedder = embedder
        logger.info(
            "Embedder reloaded: backend=%s, model=%s",
            embedder.backend_type(), embedder.model_name(),
        )
    except Exception as e:
        logger.error("Failed to reload embedder: %s", e)
        state.embedder = None

    # 次元整合性を再評価
    try:
        from backend.free.rag.dimension_check import check_embedding_dim_consistency
        check_embedding_dim_consistency(state)
    except Exception as e:
        logger.warning("Post-reload dimension check failed: %s", e)


async def reload_assist_model(state: AppState) -> None:
    """assist_model セクション変更時に assist_client を再生成して差し替える

    assist_model.enabled=false なら None を注入。health_check 失敗時も None。
    下流の各コンポーネントは `assist_client is None` をチェックしてフォールバック
    モードで動作するため、state 差し替えのみで切替が完結する。
    """
    from backend.free.llm.assist_client import AssistModelClient

    cfg = get_config()

    # 旧 client を閉じる
    old = state.assist_client
    if old is not None and hasattr(old, "aclose"):
        try:
            await old.aclose()
        except Exception as e:
            logger.warning("Failed to close old assist client: %s", e)

    assist_cfg = cfg.get("assist_model", {})
    if not assist_cfg.get("enabled", True):
        state.assist_client = None
        _sync_assist_client_downstream(state)
        # OFF にしたら llama-server プロセスも停止して VRAM を解放する
        _stop_assist_server_process(state, cfg)
        logger.info(
            "Assist model reloaded: disabled via config (enabled=false) — "
            "downstream features will use fallback modes"
        )
        return
    if not assist_cfg.get("local"):
        state.assist_client = None
        _sync_assist_client_downstream(state)
        logger.info(
            "Assist model reloaded: not configured (missing local section) — "
            "downstream features will use fallback modes"
        )
        return

    # ON にしたらプロセスが落ちていれば起動し直す (既に起動済みなら no-op)
    await _start_assist_server_process(cfg)

    try:
        client = AssistModelClient(cfg, debug_logger=state.debug_logger)
        if not await client.health_check():
            logger.warning(
                "Assist model reload: health check failed at %s — "
                "downstream features will use fallback modes",
                client.url,
            )
            await client.aclose()
            state.assist_client = None
            _sync_assist_client_downstream(state)
            return

        await client.update_params_from_server()
        state.assist_client = client
        _sync_assist_client_downstream(state)
        cc = client.concurrency
        logger.info(
            "Assist model reloaded: %s "
            "(concurrency realtime=%d, background=%d, learning=%d)",
            client.url,
            cc["realtime"], cc["background"], cc["learning"],
        )
    except Exception as e:
        logger.error("Failed to reload assist model: %s", e)
        state.assist_client = None
        _sync_assist_client_downstream(state)


def _stop_assist_server_process(state: AppState, cfg: dict) -> None:
    """assist の llama-server プロセスを停止して VRAM を解放する

    停止に失敗しても設定 OFF 自体は成立させたいので、例外は握りつぶして
    warning に留める (プロセス未起動・外部管理外でも no-op で安全)。
    """
    from backend.free.api.system.server_control import stop_server_process

    try:
        stopped, msg = stop_server_process("assist", cfg, state)
        logger.info("Assist server process stop on disable: %s (%s)", stopped, msg)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to stop assist server process on disable: %s", e)


async def _start_assist_server_process(cfg: dict) -> None:
    """assist の llama-server プロセスを起動する (既に起動済みなら no-op)

    起動に失敗しても後続の health_check 失敗でフォールバックに倒れるため、
    例外は握りつぶして warning に留める。
    """
    from backend.free.api.system.server_control import start_server_process

    try:
        started, msg = await start_server_process("assist", cfg)
        logger.info("Assist server process start on enable: %s (%s)", started, msg)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to start assist server process on enable: %s", e)


def _sync_assist_client_downstream(state: AppState) -> None:
    """assist_client の参照を下流コンポーネントに同期する

    コンストラクタで client を受け取って属性として保持しているコンポーネントを
    直接書き換えることで、ランタイムでの ON/OFF 切替を即座に反映する。
    該当コンポーネントは `assist_client is None` をチェックしてフォールバック
    モードで動作する契約なので、None 注入でも安全に動く。
    """
    client = state.assist_client

    # ToolCallJudge (backend/free/agent/tool_call_judge.py)
    judge = state.tool_call_judge
    if judge is not None and hasattr(judge, "_assist_client"):
        judge._assist_client = client

    # SleepScheduler の内部 worker が持つ client (backend/free/memory/scheduler.py)
    sched = state.sleep_scheduler
    if sched is not None:
        for attr in ("_assist_llm_client", "_assist_client", "assist_client"):
            if hasattr(sched, attr):
                setattr(sched, attr, client)
        worker = getattr(sched, "_worker", None) or getattr(sched, "worker", None)
        if worker is not None:
            for attr in ("_assist_llm_client", "_assist_client", "assist_client"):
                if hasattr(worker, attr):
                    setattr(worker, attr, client)

    # LearningScheduler 配下の CritiqueSynthesizer / PolicyParamEvolver 等
    ls = state.learning_scheduler
    if ls is not None:
        for attr_name in dir(ls):
            if attr_name.startswith("_"):
                attr = getattr(ls, attr_name, None)
                if attr is not None and hasattr(attr, "_assist_client"):
                    attr._assist_client = client

    # Loop driver / RalphExecutor
    driver = getattr(state, "loop_driver", None)
    if driver is not None:
        executor = getattr(driver, "_executor", None) or getattr(driver, "executor", None)
        if executor is not None and hasattr(executor, "_assist_client"):
            executor._assist_client = client


async def reload_prompt_manager(state: AppState) -> None:
    """instance セクション変更時に SystemPromptManager を再生成して差し替える"""
    from backend.free.agent.prompt_manager import SystemPromptManager

    cfg = get_config()
    resolver = get_path_resolver()
    # base システムプロンプトは (model×mode) パーティション配下 (resolve_learning)。
    prompt_dir = resolver.resolve_learning("prompts_dir")
    instance_name = cfg.get("instance", {}).get("name", "evoref")
    state.prompt_manager = SystemPromptManager(prompt_dir, instance_name=instance_name)
    logger.info("PromptManager reloaded: instance_name=%s", instance_name)
