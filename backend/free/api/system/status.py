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
from backend.free.core.system_info import (
    PROCESS_START_TIME as _PROCESS_START_TIME,
)
from backend.edition import current_edition
from backend.log_config import get_logger
from backend.version import get_runtime_version, get_version_info

logger = get_logger("api.status")

router = APIRouter(prefix="/api", tags=["status"])

# uptime の SSOT は ``core.system_info.PROCESS_START_TIME``。ここで独自に
# 起点を持つと、``evoref_runtime_info`` (チャットの自己構成回答) と
# ``/api/status`` の稼働時間が食い違う。
_start_time = _PROCESS_START_TIME


async def _collect_component_statuses(
    cfg: dict, state: AppState, client: object | None, base_connected: bool,
    *, serving_user: bool = False,
) -> list[ComponentStatus]:
    """各コンポーネント (base / embed) のステータスを収集する

    Args:
        serving_user: ベースモデルがチャット応答を生成中か。True のときは
            health check を投げない。ビジーな llama-server への probe は
            タイムアウトしやすく、**動いている最中に「未接続」と表示される**
            (2026-08-09 ライブ監査: チャット中だけ ``Health check: connection
            failed`` が 47 件、アイドル時は 0 件。手動 curl では base も embed も
            200 を返した)。ターンが進行中であること自体が接続済みの証拠で、
            これは ``_try_lazy_connect`` が既に採っている判断と同じ。
    """
    from pathlib import Path

    components: list[ComponentStatus] = []

    # base model
    base_model_path = cfg.get("model_paths", {}).get("base_model") or ""
    base_name = Path(base_model_path).stem if base_model_path else ""
    if client and hasattr(client, "metadata") and client.metadata.model_id:
        base_name = client.metadata.model_id
    components.append(ComponentStatus(name=base_name, connected=base_connected))

    # embed model
    embed_cfg = cfg.get("embedding", {})
    embed_name = embed_cfg.get("model_name") or ""
    embed_connected = False
    if state.embedder is not None and hasattr(state.embedder, "health_check"):
        if serving_user:
            # 進行中のターンは RAG (= 埋め込み) を既に通しているため接続済み。
            embed_connected = True
        else:
            try:
                embed_connected = await state.embedder.health_check()
            except Exception:
                pass
    components.append(ComponentStatus(name=embed_name, connected=embed_connected))

    return components


async def _try_lazy_connect(state: AppState, llama_url: str, llama_cfg: dict) -> bool:
    """local_client が未接続の場合、llama-server への遅延接続を試みる

    チャット生成中は試行しない。生成でビジーな llama-server への ``/props`` は
    ``RemoteProtocolError`` になりやすく、リトライ 3 回ぶんを空費したうえ
    「未接続」という誤った結論にも至りうる (2026-08-05 ライブ監査: 会話中に
    ``/props`` リトライが 3 回発生。実体はモデルが応答生成中だっただけ)。
    生成中であることは接続済みの証拠でもある。
    """
    from backend.free.llm.local_client import LocalClient
    from backend.free.llm.model_metadata import fetch_model_metadata

    llm_client = getattr(state, "llm_client", None)
    if llm_client is not None and getattr(llm_client, "is_serving_user", False):
        logger.debug("Skipping lazy-connect: base model is serving a chat turn")
        return False

    debug_logger = getattr(state, "debug_logger", None)
    try:
        metadata = await fetch_model_metadata(
            llama_url, debug_logger=debug_logger,
            purpose="lazy_reconnect/props",
        )
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
    """base クライアントの能力プローブ結果を Status 用に収集する (docs/c_15)。

    プローブ未完了 / 無効 (``capabilities is None``) の slot は含めない。
    """
    from backend.free.llm.capability import CapabilitySnapshot

    out: list[CapabilityInfo] = []
    for slot, client in (("base", state.local_client),):
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


def _collect_debug_info(cfg: dict, state: AppState) -> DebugStatusInfo:  # noqa: ARG001
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

    # 生成中は health check を投げない。ビジーな llama-server への probe は
    # タイムアウトしやすく、応答中なのに「未接続」と表示される
    # (_collect_component_statuses の serving_user 引数を参照)。
    serving_user = bool(
        getattr(getattr(state, "llm_client", None), "is_serving_user", False),
    )
    client = state.local_client
    if client:
        if serving_user:
            connected = True
        else:
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
    components = await _collect_component_statuses(
        cfg, state, client, connected, serving_user=serving_user,
    )

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

    cart_mgr = getattr(state, "cartridge_manager", None)
    cartridges_loaded = cart_mgr.loaded_count if cart_mgr is not None else 0
    return StatusResponse(
        status=status,
        edition=current_edition().name.lower(),
        instance_name=instance_name,
        version=get_runtime_version(),
        cartridges_loaded=cartridges_loaded,
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
