"""モデル情報 API + ベースモデル移行 API"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from backend.free.api.model._model_helpers import (
    build_model_detail_response,
    map_migration_history_items,
    migration_busy_error,
    migration_error,
    model_health_check_failed_error,
    model_reload_failed_error,
    resolve_lora_path,
)
from backend.free.api.schemas import (
    ComponentMigrateRequest,
    ComponentMigrateResponse,
    ComponentMigrationHistoryItem,
    ComponentMigrationHistoryResponse,
    ComponentRollbackRequest,
    ComponentRollbackResponse,
    MigrateDataSummary,
    MigrateRequest,
    MigrateResponse,
    MigrationHistoryResponse,
    ModelDetailResponse,
    ModelQualityResponse,
    ModelQualityRole,
    ModelStateResponse,
    ReloadResponse,
    RollbackRequest,
    RollbackResponse,
)
from backend.app_state import AppState, get_app_state
from backend.config import get_config, get_path_resolver, resolve_context_size
from backend.edition import get_pro_handler
from backend.i18n_helper import msg
from backend.log_config import get_logger

logger = get_logger("api.model")

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/info", response_model=ModelDetailResponse)
async def get_model_info(state: AppState = Depends(get_app_state)):
    """現在ロードされているモデルの情報"""
    logger.debug("GET /api/model/info")
    cfg = get_config()
    llama_cfg = cfg.get("llama", {})
    return build_model_detail_response(
        state.local_client, llama_cfg, resolve_context_size(cfg, "base"),
    )


# ────────────────────────────────────────────
# ベースモデル移行 API（§22.8）
# ────────────────────────────────────────────


def _get_model_state():
    """ModelState インスタンスを取得（オンデマンド生成）"""
    from backend.free.core.model_migration import ModelState

    cfg = get_config()
    resolver = get_path_resolver()
    state_dir = resolver.resolve_local("memory_dir")
    state_path = state_dir.parent / "model_state.json"
    ms = ModelState(state_path)
    if not ms.current_filename:
        ms.initialize_from_config(cfg)
    return ms


def _get_migrator(state: AppState):
    """ModelMigrator インスタンスを取得"""
    from backend.free.core.model_migration import ModelMigrator

    cfg = get_config()
    resolver = get_path_resolver()
    project_root = resolver.root

    model_state = _get_model_state()

    # 経験バッファ取得
    experience_buf = None
    fc = state.feedback_collector
    if fc:
        experience_buf = fc.buffer

    prompt_manager = state.prompt_manager
    learning_scheduler = state.learning_scheduler

    # EvalCoreManager（Pro 機能: Free では None）
    eval_core_mgr = None
    EvalCoreManager = get_pro_handler("eval_core_manager")
    if EvalCoreManager is not None:
        eval_core_path = resolver.resolve_local("eval_core_file")
        eval_core_mgr = EvalCoreManager(eval_core_path)

    # ShortTermMemory
    stm = None
    mem = state.get_memory_system()
    if mem:
        _, stm, _ = mem

    return ModelMigrator(
        config=cfg,
        project_root=project_root,
        model_state=model_state,
        experience_buf=experience_buf,
        prompt_manager=prompt_manager,
        eval_core_manager=eval_core_mgr,
        learning_scheduler=learning_scheduler,
        short_term_memory=stm,
    )


@router.post("/migrate", response_model=MigrateResponse)
async def migrate_model(req: MigrateRequest, state: AppState = Depends(get_app_state)):
    """ベースモデル移行を実行（§22.8.1）"""
    logger.info(
        "POST /api/model/migrate: new_model=%s, dry_run=%s, try_lora=%s",
        req.new_model_path, req.dry_run, req.try_lora,
    )

    from backend.free.core.model_migration import (
        MigrationBusyError,
        MigrationError,
    )

    try:
        migrator = _get_migrator(state)
        result = migrator.migrate(
            new_model_path=req.new_model_path,
            try_lora=req.try_lora,
            regenerate_context=req.regenerate_context,
            dry_run=req.dry_run,
        )
    except MigrationBusyError as e:
        raise migration_busy_error(str(e))
    except MigrationError as e:
        raise migration_error(str(e))

    return MigrateResponse(
        dry_run=result.dry_run,
        old_model=result.old_model,
        new_model=result.new_model,
        lora_action=result.lora_action,
        data_summary=MigrateDataSummary(**result.data_summary),
        calibration=None,
        recommendations=result.recommendations,
    )


@router.get("/state", response_model=ModelStateResponse)
async def get_model_state(state: AppState = Depends(get_app_state)):
    """model_state.json と config.yaml の整合性スナップショット"""
    logger.debug("GET /api/model/state")
    cfg = get_config()
    model_state = _get_model_state()

    config_base_model = cfg.get("model_paths", {}).get("base_model") or ""
    config_filename = Path(config_base_model).name if config_base_model else ""
    mismatch_info = state.model_state_mismatch or {}
    is_mismatch = bool(
        model_state.current_filename
        and config_filename
        and model_state.current_filename != config_filename
    )

    # 実際に serve 中のモデル。起動時検証 (_validate_served_model) が記録した
    # スナップショットを返す (再起動しない限り変わらない)。
    served_info = state.served_model_mismatch or {}

    return ModelStateResponse(
        current_filename=model_state.current_filename,
        config_filename=config_filename,
        config_base_model=config_base_model,
        config_mismatch=is_mismatch,
        lora_compatible=model_state.lora_compatible,
        strict_startup_check=bool(
            cfg.get("model_migration", {}).get("strict_startup_check", False),
        ),
        recommendation=mismatch_info.get("recommendation", "") if is_mismatch else "",
        served_filename=served_info.get("served_filename", ""),
        served_mismatch=bool(served_info),
        served_recommendation=served_info.get("recommendation", ""),
    )


@router.get("/quality", response_model=ModelQualityResponse)
async def get_model_quality(state: AppState = Depends(get_app_state)):
    """モデル切替時の出力品質プローブ結果を取得

    プローブはモデルが変わったときだけ走るため、まだ一度も切替が起きていない
    役割は含まれない (空リスト = 未検査であって合格ではない)。
    """
    logger.debug("GET /api/model/quality")
    store = state.model_quality
    if store is None:
        return ModelQualityResponse()
    roles = [ModelQualityRole(**r) for r in store.summary()]
    return ModelQualityResponse(
        roles=roles,
        degraded=any(not r.passed and not r.skipped_reason for r in roles),
    )


@router.get("/migration-history", response_model=MigrationHistoryResponse)
async def get_migration_history():
    """移行履歴を取得（§22.8.2）"""
    logger.debug("GET /api/model/migration-history")
    model_state = _get_model_state()

    # LoRA の存在確認
    cfg = get_config()
    project_root = Path(__file__).parent.parent.parent
    lora_path = resolve_lora_path(cfg.get("local_paths", {}), project_root)

    return MigrationHistoryResponse(
        current_model=model_state.current_filename,
        lora_available=lora_path.exists(),
        history=map_migration_history_items(model_state.migration_history),
    )


@router.post("/rollback", response_model=RollbackResponse)
async def rollback_model(req: RollbackRequest, state: AppState = Depends(get_app_state)):
    """直前の移行をロールバック（§22.8.3）"""
    logger.info(
        "POST /api/model/rollback: target=%s",
        req.target_model or "(previous)",
    )

    from backend.free.core.model_migration import MigrationError

    try:
        migrator = _get_migrator(state)
        result = migrator.rollback(target_model=req.target_model)
    except MigrationError as e:
        raise migration_error(str(e))

    return RollbackResponse(**result)


# ────────────────────────────────────────────
# コンポーネント (embedding) 移行 API
# ────────────────────────────────────────────

_VALID_COMPONENTS = ("embedding",)


def _validate_component(component: str) -> None:
    if component not in _VALID_COMPONENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown component: {component}. "
                   f"Expected one of {_VALID_COMPONENTS}",
        )


@router.post(
    "/{component}/migrate", response_model=ComponentMigrateResponse,
)
async def migrate_component(
    component: str,
    req: ComponentMigrateRequest,
    state: AppState = Depends(get_app_state),
):
    """embedding モデルを切り替える

    L2: `auto_restart=True` (既定) かつ LlamaProcessManager が当該
    コンポーネントを管理している場合、config.yaml 反映後に llama-server を
    再起動し、in-memory クライアントを差し替える。再起動失敗時は
    `rollback_component` で旧モデルへ自動巻き戻す。
    """
    _validate_component(component)
    logger.info(
        "POST /api/model/%s/migrate: new_model=%s, dry_run=%s, "
        "auto_restart=%s",
        component, req.new_model_path, req.dry_run, req.auto_restart,
    )

    from backend.free.core.model_migration import (
        MigrationBusyError,
        MigrationError,
    )

    try:
        migrator = _get_migrator(state)
        result = migrator.migrate_component(
            component=component,
            new_model_path=req.new_model_path,
            dry_run=req.dry_run,
        )
    except MigrationBusyError as e:
        raise migration_busy_error(str(e))
    except MigrationError as e:
        raise migration_error(str(e))

    # embedding モデルを実際に切り替えた場合、既存ベクトルは (dim 一致でも) stale。
    # reindex 要求マーカーを立て、dimension_check がモデル同一性ベースで mismatch を
    # 検知できるようにする (同 dim swap の「検知不発の罠」対策)。
    if (
        component == "embedding"
        and not req.dry_run
        and result.old_model != result.new_model
    ):
        from backend.free.memory.semantic.stale_guard import (
            set_semmem_reembed_required,
        )
        from backend.free.rag.dimension_check import set_embed_reindex_required

        set_embed_reindex_required(result.new_model)
        # SemMem fact 埋め込み (URL/コマンドリコール) も同 dim swap で stale になる。
        # 起動時 WARN で reembed-facts を案内するためのマーカーを立てる。
        set_semmem_reembed_required(result.new_model)

    restarted = False
    if (
        not req.dry_run
        and req.auto_restart
        and state.llama_manager is not None
        and state.llama_manager.is_managed(component)
    ):
        restarted = await _restart_and_rebind(
            component, state, migrator,
        )

    recommendations = result.recommendations
    if restarted:
        # llama-server の再起動+rebind は完了。ただし embedding 切替では RAG
        # reindex / SemMem reembed 手順が残るため、元リスト先頭の「再起動が必要」
        # だけ「完了」に置換し後続手順は残す (auto_restart 成功時に reindex/reembed
        # の案内が丸ごと消えていた不具合の修正)。
        recommendations = [
            f"{component} モデルの再起動とクライアント差し替えが完了しました",
            *result.recommendations[1:],
        ]

    return ComponentMigrateResponse(
        component=component,
        dry_run=result.dry_run,
        old_model=result.old_model,
        new_model=result.new_model,
        lora_action=result.lora_action,
        restarted=restarted,
        recommendations=recommendations,
    )


async def _restart_and_rebind(
    component: str, state: AppState, migrator,
) -> bool:
    """llama-server 再起動 + クライアント差し替え

    失敗時は rollback_component で旧モデルへ巻き戻し、500 エラーを返す。
    """
    from backend.free.core.component_rebind import rebind_component

    cfg = get_config()
    manager = state.llama_manager
    assert manager is not None

    try:
        manager.restart(component, cfg)
        await rebind_component(component, state, cfg)
        logger.info(
            "Component restart + rebind succeeded: %s", component,
        )
        return True
    except Exception as e:
        logger.error(
            "Component restart/rebind failed for %s: %s — rolling back",
            component, e,
        )
        try:
            migrator.rollback_component(component)
            cfg_after = get_config()
            try:
                manager.restart(component, cfg_after)
                from backend.free.core.component_rebind import rebind_component
                await rebind_component(component, state, cfg_after)
            except Exception as recover_err:
                logger.error(
                    "Rollback restart also failed for %s: %s",
                    component, recover_err,
                )
        except Exception as rb_err:
            logger.error(
                "rollback_component failed for %s: %s", component, rb_err,
            )
        raise HTTPException(
            status_code=500,
            detail=(
                f"{component} restart failed: {e}. "
                "Rollback attempted; check logs."
            ),
        )


@router.post(
    "/{component}/rollback", response_model=ComponentRollbackResponse,
)
async def rollback_component(
    component: str,
    req: ComponentRollbackRequest,
    state: AppState = Depends(get_app_state),
):
    """コンポーネントモデルをロールバック"""
    _validate_component(component)
    logger.info(
        "POST /api/model/%s/rollback: target=%s",
        component, req.target_model or "(previous)",
    )

    from backend.free.core.model_migration import MigrationError

    try:
        migrator = _get_migrator(state)
        result = migrator.rollback_component(
            component=component,
            target_model=req.target_model,
        )
    except MigrationError as e:
        raise migration_error(str(e))

    return ComponentRollbackResponse(
        component=component,
        rolled_back_to=result["rolled_back_to"],
        lora_restored=result["lora_restored"],
    )


@router.get(
    "/{component}/migration-history",
    response_model=ComponentMigrationHistoryResponse,
)
async def get_component_migration_history(component: str):
    """コンポーネント移行履歴を取得"""
    _validate_component(component)
    logger.debug("GET /api/model/%s/migration-history", component)
    model_state = _get_model_state()
    comp = model_state.get_component(component)
    return ComponentMigrationHistoryResponse(
        component=component,
        current_model=comp.current.filename,
        history=[
            ComponentMigrationHistoryItem(
                from_model=h.from_model,
                to_model=h.to_model,
                migrated_at=h.migrated_at,
            )
            for h in comp.history
        ],
    )


# ────────────────────────────────────────────
# llama-server プロセス管理 API
# ────────────────────────────────────────────

_ALL_PROCESS_COMPONENTS = ("base", "embedding")


def _validate_process_component(component: str) -> None:
    if component not in _ALL_PROCESS_COMPONENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown process component: {component}. "
                   f"Expected one of {_ALL_PROCESS_COMPONENTS}",
        )


@router.get("/process/status")
async def process_status(state: AppState = Depends(get_app_state)):
    """LlamaProcessManager の状態スナップショット"""
    if state.llama_manager is None:
        return {"managed": [], "components": {}}
    components: dict[str, dict] = {}
    for name in _ALL_PROCESS_COMPONENTS:
        entry = state.llama_manager.get_entry(name)
        if entry is None:
            components[name] = {"managed": False}
        else:
            components[name] = {
                "managed": True,
                "host": entry.host,
                "port": entry.port,
                "pid": entry.proc.pid,
                "alive": entry.proc.poll() is None,
            }
    return {
        "managed": state.llama_manager.list_managed(),
        "components": components,
    }


@router.post("/process/{component}/start")
async def process_start(
    component: str, state: AppState = Depends(get_app_state),
):
    """指定コンポーネントの llama-server を起動する"""
    _validate_process_component(component)
    if state.llama_manager is None:
        raise HTTPException(
            status_code=503,
            detail="LlamaProcessManager not initialized",
        )
    from backend.free.core.llama_process_manager import ProcessManagerError

    try:
        entry = state.llama_manager.start(component, get_config())
    except ProcessManagerError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "component": component,
        "started": True,
        "host": entry.host,
        "port": entry.port,
        "pid": entry.proc.pid,
    }


@router.post("/process/{component}/stop")
async def process_stop(
    component: str, state: AppState = Depends(get_app_state),
):
    """指定コンポーネントの llama-server を停止する"""
    _validate_process_component(component)
    if state.llama_manager is None:
        raise HTTPException(
            status_code=503,
            detail="LlamaProcessManager not initialized",
        )
    from backend.free.core.llama_process_manager import ProcessNotManagedError

    try:
        state.llama_manager.stop(component)
    except ProcessNotManagedError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"component": component, "stopped": True}


@router.post("/process/{component}/restart")
async def process_restart(
    component: str, state: AppState = Depends(get_app_state),
):
    """指定コンポーネントの llama-server を再起動する"""
    _validate_process_component(component)
    if state.llama_manager is None:
        raise HTTPException(
            status_code=503,
            detail="LlamaProcessManager not initialized",
        )
    from backend.free.core.llama_process_manager import (
        ProcessManagerError,
        ProcessNotManagedError,
    )

    try:
        entry = state.llama_manager.restart(component, get_config())
    except ProcessNotManagedError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ProcessManagerError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "component": component,
        "restarted": True,
        "host": entry.host,
        "port": entry.port,
        "pid": entry.proc.pid,
    }


@router.post("/reembed-facts")
async def reembed_facts(
    state: AppState = Depends(get_app_state),
    dry_run: bool = False,
):
    """SemMem fact (URL/コマンドリコール) の埋め込みを現在の Embedder で再構築する。

    埋め込みモデル切替後、SemMem fact ベクトルは旧モデル空間に取り残され
    ``search_by_embedding`` (URL/コマンドリコール) が空振りする。RAG reindex
    (``/api/rag/reindex``) は SemMem fact を対象外とするため本エンドポイントが
    補完する。2 経路:

    - **同一モデルでの再埋め込み**: 稼働中の live global store を in-place 更新
      するので mem_view も同一オブジェクトを参照しており **再起動不要**。
    - **モデル変更を伴う切替** (model_id 変化 or dim 変化): CLI ``reembed-facts
      --apply`` と同一の cross-model swap (``apply_reembed_swap``) を実行する。
      全 scope に新 model_id ディレクトリを作成 → manifest を atomic swap → 全
      scope の fact を新モデルで再埋め込み → stale マーカー解除。RAG Reindex
      ボタンがモデル変更込みで自己回復するのと対称。ただし稼働中バックエンドの
      Fact View は旧 store 参照を保持するため、**live リコールの反映には backend
      再起動が必須** で ``restart_required: True`` を返す (キャッシュ層は
      ``invalidate_semantic_stores`` で破棄するが View 再バインドは行わない)。
      なお再起動までの間に SemMem 書込み (SleepTimeWorker auto-merge / artifact
      fact / conflict resolution) が発生すると旧 model_id dir へ書かれ再起動後に
      孤立し得るため、cross-model 切替後は速やかに再起動すること (live View 再
      バインドによる完全な再起動レス化は今後の課題)。

    いずれの経路でも、embed サーバが新モデルへ未切替 (config と不一致) の場合は
    409 で拒否する (旧モデル空間で再embedしてマーカーまで消す順序依存の罠を防ぐ)。

    Query params:
        dry_run: True なら対象 fact 数だけ返して実行しない。
    """
    import shutil
    import time

    import numpy as np

    from backend.free.memory.semantic.cli._paths import cli_backup_root
    from backend.free.memory.semantic.cli.reembed_facts_cmd import (
        apply_reembed_swap,
        collect_reembed_targets,
    )
    from backend.free.memory.semantic.manifest import (
        load_manifest,
        normalize_embedding_model_id,
        update_manifest,
    )
    from backend.free.memory.semantic.stale_guard import (
        clear_semmem_reembed_required,
    )
    from backend.free.rag.dimension_check import embedder_config_mismatch
    from backend.utils import utc_now

    if state.embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")

    resolver = get_path_resolver()
    memory_dir = resolver.resolve_local("memory_dir")
    manifest = load_manifest(memory_dir)
    manifest_model_id = manifest.embedding.model_id if manifest else ""
    manifest_dim = manifest.embedding.dim if manifest else None

    try:
        expected_model_id = normalize_embedding_model_id(
            str(state.embedder.model_name() or ""),
        )
    except Exception:
        expected_model_id = ""
    try:
        embedder_dim = int(state.embedder.dim())
    except Exception:
        embedder_dim = manifest_dim or 0

    # モデル変更 (model_id 変化 or dim 変化) を検知。どちらかが変われば
    # cross-model swap (新 model_id dir + manifest swap) が必要。
    model_id_changed = bool(
        manifest_model_id
        and expected_model_id
        and manifest_model_id != expected_model_id
    )
    dim_changed = (
        manifest_dim is not None
        and embedder_dim > 0
        and manifest_dim != embedder_dim
    )

    if model_id_changed or dim_changed:
        # ── cross-model: 全 scope を新モデルで再embed + manifest swap ──
        targets = collect_reembed_targets(memory_dir, manifest)
        if dry_run:
            return {
                "dry_run": True,
                "fact_count": len(targets),
                "model_changed": True,
                "new_model_id": expected_model_id or None,
            }
        # embed サーバが新モデルへ未切替なら拒否 (順序依存の罠)。
        stale = embedder_config_mismatch(state)
        if stale is not None:
            raise HTTPException(
                status_code=409,
                detail=msg(
                    "error.rag.stale_embedder",
                    current=stale[0], expected=stale[1],
                ),
            )
        if not expected_model_id:
            raise HTTPException(
                status_code=500,
                detail="cannot resolve new embedding model_id from embedder",
            )
        objs = [t[3] for t in targets]
        t0 = time.monotonic()
        vectors = (
            await state.embedder.embed(objs, is_query=False) if objs else []
        )
        elapsed = time.monotonic() - t0
        # LlamaCppEmbedder.embed は常に L2 正規化して返すため normalized=True。
        # (旧 manifest の normalized を引き継ぐと、新モデルの実挙動とラベルが
        # 乖離しうる。CLI 側も normalized=True 既定。)
        try:
            reembedded, _backup = apply_reembed_swap(
                memory_dir,
                resolver.resolve_local("migration_archive_dir"),
                targets, vectors,
                new_model_id=expected_model_id, new_dim=embedder_dim,
                normalized=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        # 稼働中バックエンドのキャッシュ store を破棄 (次回 get で新 manifest 再ロード)。
        state.invalidate_semantic_stores()
        logger.info(
            "reembed-facts (cross-model): manifest %s -> %s, reembedded %d "
            "in %.2fs (backend restart REQUIRED for live recall)",
            manifest_model_id or "?", expected_model_id, reembedded, elapsed,
        )
        return {
            "dry_run": False,
            "reembedded": reembedded,
            "fact_count": reembedded,
            "elapsed_sec": round(elapsed, 2),
            "model_changed": True,
            "new_model_id": expected_model_id,
            "restart_required": True,
        }

    # ── same-model: live global store を in-place 更新 (再起動不要) ──
    store = state.get_semantic_store("global")
    live_facts = store.all_facts(include_superseded=False)
    # 本文は Step 8.8 と同じ ``fact.text`` (= statement or object)。``object``
    # を使うと再埋め込み済みと未処理で本文が食い違う (split-brain)。
    targets_g = [
        (f.id, (f.text or "").strip())
        for f in live_facts
        if f.embedding is not None and (f.text or "").strip()
    ]
    if dry_run:
        return {"dry_run": True, "fact_count": len(targets_g)}

    if not targets_g:
        return {"dry_run": False, "reembedded": 0, "fact_count": 0}

    # migrate 直後で embedder が旧モデルのまま実行すると、旧モデル空間で
    # 再埋め込みしてマーカーまでクリアしてしまう (順序依存の罠)。embed
    # サーバ再起動 + embedder reload が済むまで実行を拒否する。
    stale = embedder_config_mismatch(state)
    if stale is not None:
        raise HTTPException(
            status_code=409,
            detail=msg(
                "error.rag.stale_embedder", current=stale[0], expected=stale[1],
            ),
        )

    # 上書き前に現在の埋め込みベクトルを backup する (rollback 用)。
    try:
        backup_root = cli_backup_root(
            resolver.resolve_local("migration_archive_dir"), "reembed_facts_api",
        )
        emb_dir = store.root_dir / "embeddings"
        if emb_dir.exists():
            shutil.copytree(
                emb_dir, backup_root / "embeddings", dirs_exist_ok=True,
            )
    except Exception as exc:
        logger.warning("reembed-facts: backup skipped: %s", exc)

    objs = [t[1] for t in targets_g]
    t0 = time.monotonic()
    vectors = await state.embedder.embed(objs, is_query=False)
    if len(vectors) != len(objs):
        raise HTTPException(
            status_code=500,
            detail=f"embedder returned {len(vectors)} vectors for {len(objs)} texts",
        )
    for (fid, _), vec in zip(targets_g, vectors):
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != embedder_dim:
            raise HTTPException(
                status_code=500,
                detail=f"embedder vector dim {arr.shape[0]} != {embedder_dim}",
            )
        store.update_fact(fid, embedding=arr)
    elapsed = time.monotonic() - t0

    try:
        update_manifest(memory_dir, last_migrated_at=utc_now())
    except Exception as exc:
        logger.warning("reembed-facts: manifest stamp skipped: %s", exc)
    clear_semmem_reembed_required(memory_dir=memory_dir)

    logger.info(
        "reembed-facts: reembedded %d SemMem facts in %.2fs",
        len(targets_g), elapsed,
    )
    return {
        "dry_run": False,
        "reembedded": len(targets_g),
        "fact_count": len(targets_g),
        "elapsed_sec": round(elapsed, 2),
    }


@router.post("/reload", response_model=ReloadResponse)
async def reload_model(state: AppState = Depends(get_app_state)):
    """移行後の llama-server 再接続 + メタデータ更新"""
    logger.info("POST /api/model/reload")

    cfg = get_config()
    llama_cfg = cfg.get("llama", {})
    llama_host = llama_cfg.get("host", "127.0.0.1")
    llama_port = llama_cfg.get("port", 8080)
    llama_url = f"http://{llama_host}:{llama_port}"

    try:
        from backend.free.llm.local_client import LocalClient
        from backend.free.llm.model_metadata import fetch_model_metadata

        debug_logger = getattr(state, "debug_logger", None)
        metadata = await fetch_model_metadata(llama_url, debug_logger=debug_logger)
        from backend.config import resolve_enable_thinking
        base_enable_thinking = resolve_enable_thinking(
            cfg, "base",
            explicit=llama_cfg.get("enable_thinking"),
            chat_template=getattr(metadata, "chat_template", None),
        )
        client = LocalClient(
            llama_url,
            metadata,
            cache_prompt=llama_cfg.get("cache_prompt", True),
            slots=llama_cfg.get("slots", 1),
            enable_thinking=base_enable_thinking,
            debug_logger=debug_logger,
        )
        if not await client.health_check():
            raise model_health_check_failed_error()

        state.set_local_client(client)

        # model_state 更新
        model_state = _get_model_state()
        model_state.update_current(
            filename=Path(
                cfg.get("model_paths", {}).get("base_model") or ""
            ).name,
            chat_template_name=metadata.chat_template[:50] if metadata.chat_template else "",
            has_system_role=metadata.has_system_role,
        )
        model_state.save()

        logger.info(
            "Model reloaded: model_id=%s, has_system_role=%s",
            metadata.model_id, metadata.has_system_role,
        )

        return ReloadResponse(
            reloaded=True,
            model_id=metadata.model_id,
            chat_template=metadata.chat_template[:50] if metadata.chat_template else "",
            has_system_role=metadata.has_system_role,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model reload failed: %s", e)
        raise model_reload_failed_error(str(e))
