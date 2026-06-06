"""ステータス API"""

import time

from fastapi import APIRouter, Depends

from backend.app_state import AppState, get_app_state
from backend.free.api.system._status_collectors import (
    compute_log_disk_usage_mb,
    count_recent_errors,
    extract_embedder_cache_hit_rate,
    extract_learning_brief,
    resolve_log_dir,
)
from backend.free.api.schemas import (
    CapabilityInfo,
    ComponentStatus,
    DebugStatusInfo,
    LlamaServerInfo,
    MemoryStats,
    ModelInfo,
    StatusResponse,
)
from backend.config import get_config, resolve_context_size
from backend.edition import current_edition
from backend.log_config import get_logger
from backend.version import get_runtime_version, get_version_info

logger = get_logger("api.status")

router = APIRouter(prefix="/api", tags=["status"])

_start_time = time.time()


async def _collect_component_statuses(
    cfg: dict, state: AppState, client: object | None, base_connected: bool,
) -> list[ComponentStatus]:
    """各コンポーネント (base / assist / embed / reranker) のステータスを収集する"""
    from pathlib import Path

    components: list[ComponentStatus] = []

    # base model
    base_model_path = cfg.get("model_paths", {}).get("base_model", "")
    base_name = Path(base_model_path).stem if base_model_path else ""
    if client and hasattr(client, "metadata") and client.metadata.model_id:
        base_name = client.metadata.model_id
    components.append(ComponentStatus(name=base_name, connected=base_connected))

    # assist model
    assist_model_path = cfg.get("model_paths", {}).get("assist_model", "")
    assist_name = Path(assist_model_path).stem if assist_model_path else ""
    assist_connected = False
    if state.assist_client is not None:
        try:
            assist_connected = await state.assist_client.health_check()
        except Exception:
            pass
    components.append(ComponentStatus(name=assist_name, connected=assist_connected))

    # embed model
    embed_cfg = cfg.get("embedding", {})
    embed_name = embed_cfg.get("model_name", "")
    embed_connected = False
    if state.embedder is not None and hasattr(state.embedder, "health_check"):
        try:
            embed_connected = await state.embedder.health_check()
        except Exception:
            pass
    components.append(ComponentStatus(name=embed_name, connected=embed_connected))

    # reranker
    reranker_cfg = cfg.get("reranker", {})
    reranker_name = reranker_cfg.get("model_name", "")
    reranker_connected = False
    if state.reranker is not None and hasattr(state.reranker, "health_check"):
        try:
            reranker_connected = await state.reranker.health_check()
        except Exception:
            pass
    components.append(ComponentStatus(name=reranker_name, connected=reranker_connected))

    return components


async def _try_lazy_connect(state: AppState, llama_url: str, llama_cfg: dict) -> bool:
    """local_client が未接続の場合、llama-server への遅延接続を試みる"""
    from backend.free.llm.local_client import LocalClient
    from backend.free.llm.model_metadata import fetch_model_metadata

    debug_logger = getattr(state, "debug_logger", None)
    try:
        metadata = await fetch_model_metadata(llama_url, debug_logger=debug_logger)
        from backend.config import resolve_client_reasoning, resolve_enable_thinking
        base_enable_thinking = resolve_enable_thinking(
            get_config(), "base",
            explicit=llama_cfg.get("enable_thinking"),
            chat_template=getattr(metadata, "chat_template", None),
        )
        think_budget, on_runaway = resolve_client_reasoning(get_config(), "base")
        client = LocalClient(
            llama_url,
            metadata,
            cache_prompt=llama_cfg.get("cache_prompt", True),
            slots=llama_cfg.get("slots", 1),
            enable_thinking=base_enable_thinking,
            debug_logger=debug_logger,
            client_think_budget=think_budget,
            on_runaway=on_runaway,
        )
        if await client.health_check():
            state.set_local_client(client)
            logger.info("llama-server lazy-connected: %s", llama_url)
            return True
    except Exception as e:
        logger.debug("llama-server lazy-connect failed: %s", e)
    return False


def _collect_capabilities(state: AppState) -> list[CapabilityInfo]:
    """base / assist クライアントの能力プローブ結果を Status 用に収集する (docs/c_15)。

    プローブ未完了 / 無効 (``capabilities is None``) の slot は含めない。
    """
    from backend.free.llm.capability import CapabilitySnapshot

    out: list[CapabilityInfo] = []
    for slot, client in (("base", state.local_client), ("assist", state.assist_client)):
        snap = getattr(client, "capabilities", None) if client is not None else None
        # 実 snapshot のみ採用 (MagicMock の自動属性 / None を弾く)
        if not isinstance(snap, CapabilitySnapshot):
            continue
        out.append(
            CapabilityInfo(
                slot=slot,
                model_id=snap.model_id or "",
                probed=snap.probed,
                effective_reasoning_mode=snap.effective_reasoning_mode,
                reasoning_separated=snap.reasoning_separated,
                emits_think_tags=snap.emits_think_tags,
                closes_think_tags=snap.closes_think_tags,
                json_schema_enforced=snap.json_schema_enforced,
                needs_lenient_json=snap.needs_lenient_json,
                probe_divergence=list(snap.probe_divergence or []),
                probed_at=snap.probed_at or "",
            ),
        )
    return out


def _collect_debug_info(cfg: dict, state: AppState) -> DebugStatusInfo:
    """デバッグセクションの詳細情報を収集する。

    ``cfg["debug"]`` セクションは廃止されたため、有効/無効と
    log_dir は ``state.develop_level`` から導出する。``log_dir`` は
    ``project_root/local/logs/debug`` に固定。

    純粋な計算 / パス解決 / ファイル I/O ベースの集計は
    `_status_collectors` に委譲し、本関数はオーケストレーションに専念する。
    """
    from pathlib import Path

    develop_level = getattr(state, "develop_level", "off")
    enabled = develop_level != "off"
    log_dir_str = "local/logs/debug/"

    # log_dir の絶対パスを解決
    # backend/free/api/system/status.py から 5 階層上が repo root
    project_root = Path(__file__).resolve().parents[4]
    log_dir = resolve_log_dir(log_dir_str, project_root)

    disk_usage_mb = compute_log_disk_usage_mb(log_dir)
    recent_errors_count = count_recent_errors(log_dir.parent / "backend.log")
    cache_hit_rate = extract_embedder_cache_hit_rate(state.embedder)

    # 直近リクエストの TTFT / tok/s
    metrics = state.last_request_metrics
    last_ttft_ms = metrics.ttft_ms
    last_tok_per_sec = metrics.tok_per_sec

    # 学習サイクル概要
    learning = extract_learning_brief(state.learning_scheduler)

    return DebugStatusInfo(
        enabled=enabled,
        log_dir=log_dir_str,
        disk_usage_mb=disk_usage_mb,
        recent_errors_count=recent_errors_count,
        cache_hit_rate=cache_hit_rate,
        last_ttft_ms=last_ttft_ms,
        last_tok_per_sec=last_tok_per_sec,
        learning=learning,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status(state: AppState = Depends(get_app_state)):
    """サーバー状態、インスタンス名、モデル情報、メモリ統計"""
    logger.debug("GET /api/status")
    cfg = get_config()
    llama_cfg = cfg.get("llama", {})
    instance_name = cfg.get("instance", {}).get("name", "evoref")

    # llama-server 接続チェック
    llama_host = llama_cfg.get("host", "127.0.0.1")
    llama_port = llama_cfg.get("port", 8080)
    llama_url = f"http://{llama_host}:{llama_port}"
    connected = False

    client = state.local_client
    if client:
        try:
            connected = await client.health_check()
        except Exception:
            connected = False

    # 未接続なら遅延接続を試行（auto-serve 時に llama-server が後から起動するケース）
    # 失敗後はバックオフし、サーバが死んでいる間も /api/status を高速化する
    from backend.free.api.system._lazy_connect import guarded_lazy_connect

    if not connected:
        connected = await guarded_lazy_connect(
            "local",
            lambda: _try_lazy_connect(state, llama_url, llama_cfg),
        )
        if connected:
            client = state.local_client

    # アシストモデルの遅延接続も試行（モデルロードが起動時に間に合わなかったケース）
    if state.assist_client is None:
        from backend.free.api.model.assist_model_api import _try_lazy_connect_assist
        await guarded_lazy_connect(
            "assist",
            lambda: _try_lazy_connect_assist(state, cfg),
        )

    # モデル情報
    model = None
    if connected and client:
        model = ModelInfo(
            name=client.metadata.model_id or None,
            chat_template=client.metadata.chat_template,
            has_system_role=client.metadata.has_system_role,
            context_size=resolve_context_size(cfg, "base"),
        )

    # --- 各コンポーネントのステータス ---
    components = await _collect_component_statuses(cfg, state, client, connected)

    # メモリ統計
    mem_sys = state.get_memory_system()
    memory = MemoryStats()
    if mem_sys:
        wm, stm, ltm = mem_sys
        memory = MemoryStats(
            working_turns=len(wm.turns),
            short_term_notes=len(stm.notes),
            long_term_chunks=ltm.vectors.count if ltm else 0,
        )

    status = "ok" if connected else "degraded"

    debug_info = _collect_debug_info(cfg, state)
    version_info = get_version_info()

    return StatusResponse(
        status=status,
        edition=current_edition().name.lower(),
        instance_name=instance_name,
        version=get_runtime_version(),
        free_version=version_info.free,
        pro_version=version_info.pro,
        schema_version=version_info.schema,
        uptime_seconds=time.time() - _start_time,
        llama_server=LlamaServerInfo(
            connected=connected,
            host=llama_host,
            port=llama_port,
        ),
        model=model,
        components=components,
        memory=memory,
        debug=debug_info,
        capabilities=_collect_capabilities(state),
    )
