"""EvorefMem 初期化 + 起動時 SemMem bootstrap

含まれる関数:

- :func:`_init_memory` : ``WorkingMemoryRegistry`` / ``ShortTermMemory`` /
  ``LongTermMemory`` / ``VectorStore`` の初期化と SchemaMigrator 連鎖。
  EvorefMem スキーマバージョン検査 → in-place migration → destructive init
  fallback の判定を行う。
- :func:`bootstrap_loop_context_at_startup` : 起動時クリーンコンテキスト
  bootstrap。SemMem の active policy / pinned から Tier 1 素材を組み立て
  ``state.startup_bootstrap_result`` に保持する。
- :func:`apply_semmem_policy_overrides` : SemMem active policy ファクト
  → ``PolicyInterpreter`` 反映。

純粋な move であり、関数本体・引数・default 値は変更していない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.app_state import AppState
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.memory.stores.working import WorkingMemoryRegistry
    from backend.free.rag.vector_store import VectorStore

logger = get_logger("factory.memory_init")


def bootstrap_loop_context_at_startup(
    state: AppState,
    cfg: dict[str, Any],
    current_project_id: str | None,
    debug_logger: "DebugLogger | None" = None,
) -> None:
    """起動時クリーンコンテキスト bootstrap

    ``loop.enabled=True`` のとき、SemMem の active な ``policy`` ファクトと
    ``pinned`` ファクトのみから Tier 1 素材を組み立てて ``state`` に保持する。
    実際の注入は MemoryInjector / ループ本体接続後に行う。
    ``loop.enabled=False`` の場合や project_id 未解決の場合は no-op。

    起動時 (主導モード問わず — chat / create) に呼び出される。失敗しても
    アプリ起動を止めず WARN ログのみで握りつぶす (リスク対応:
    bootstrap が起動失敗の単一障害点にならないこと)。
    """
    loop_cfg = (cfg or {}).get("loop", {}) or {}
    if not loop_cfg.get("enabled", True):
        return
    if not current_project_id:
        logger.debug(
            "bootstrap_loop_context_at_startup: project_id unresolved, skipping",
        )
        return
    try:
        from backend.free.loop.bootstrap import bootstrap_project_context
        from backend.free.loop.driver import reopen_orphan_in_progress_tasks
        from backend.free.memory.views.loop import LoopFactView

        global_store = state.get_semantic_store("global")
        project_store = state.get_semantic_store(
            f"project:{current_project_id}",
        )
        learning_policy_cfg = ((cfg or {}).get("learning") or {}).get("policy") or {}
        min_conf = float(
            learning_policy_cfg.get("activation_min_confidence", 0.7),
        )
        # SemanticFactStore 直参照を廃止し LoopFactView 経由に統一
        view = LoopFactView(
            stores=[global_store, project_store],
            writeback_store=project_store,
        )
        # 前プロセスの強制終了で残った in_progress orphan task を
        # ``open`` に戻してから bootstrap する。
        try:
            reopened = reopen_orphan_in_progress_tasks(
                view, current_project_id,
            )
        except Exception as exc:
            logger.warning(
                "Loop startup orphan recovery failed for project=%s: %s",
                current_project_id, exc,
            )
            reopened = []
        if reopened:
            logger.info(
                "Loop startup orphan recovery: project=%s reopened=%d task_ids=%s",
                current_project_id,
                len(reopened),
                [v.task_id for v in reopened],
            )
        state.startup_orphan_tasks_reopened = [  # type: ignore[attr-defined]
            v.task_id for v in reopened
        ]
        result = bootstrap_project_context(
            view,
            project_id=current_project_id,
            policy_activation_min_confidence=min_conf,
        )
        # AppState に直接 attribute として保持 (LoopDriver 接続時に拾う)。
        state.startup_bootstrap_result = result  # type: ignore[attr-defined]
        logger.info(
            "Loop startup bootstrap: project=%s tier1=%d "
            "(policies=%d pinned_global=%d pinned_project=%d "
            "failure_candidates=%d skipped_below_confidence=%d)",
            current_project_id,
            result.total_tier1(),
            len(result.active_policies),
            len(result.pinned_global),
            len(result.pinned_project),
            len(result.candidate_failure_patterns),
            result.skipped_below_confidence,
        )
        # bootstrap スナップショットを memory.jsonl にも記録
        if debug_logger is not None:
            try:
                debug_logger.log_memory_state(
                    session_id="startup_bootstrap",
                    memory_dump={},
                    semmem_stats=result.as_dict(),
                )
            except Exception as exc:
                logger.debug(
                    "log_memory_state(startup_bootstrap) failed: %s", exc,
                )
    except Exception as exc:
        logger.warning(
            "Loop startup bootstrap failed for project=%s: %s",
            current_project_id, exc,
        )


def apply_semmem_policy_overrides(
    state: AppState,
    cfg: dict[str, Any],  # noqa: ARG001
    current_project_id: str | None,
) -> None:
    """SemMem 上の active policy ファクトを PolicyInterpreter に反映する

    メモリシステムと project_id 解決後に呼び出す。``policy_source`` が
    ``yaml`` の場合は何もしない (既存テスト保護)。``hybrid`` / ``semmem``
    の場合は ``[global, project:<id>]`` の順でストアを渡し、プロジェクト
    スコープが後勝ちで上書きする。
    """
    pi = state.policy_interpreter
    if pi is None:
        return
    if pi.policy_source == "yaml":
        return
    try:
        stores = [state.get_semantic_store("global")]
        if current_project_id:
            stores.append(state.get_semantic_store(f"project:{current_project_id}"))
        pi.set_semmem_stores(stores)
        logger.info(
            "PolicyInterpreter SemMem overrides applied (stores=%d, project_id=%s)",
            len(stores), current_project_id,
        )
    except Exception as exc:
        logger.warning(
            "Failed to apply SemMem policy overrides: %s "
            "(falling back to YAML-only values)", exc,
        )


def _init_memory(
    state: AppState, cfg: dict[str, Any], resolver: Any,
) -> tuple["WorkingMemoryRegistry", "ShortTermMemory", Any, "VectorStore | None"]:
    """6. メモリシステム初期化

    WM はセッション別 (:class:`WorkingMemoryRegistry`)。LRU 押し出し / 明示の
    セッション終了 / shutdown の転送は api 層の ``release_session_turns``
    (エコー落とし規則 + セッション蓄積の掃除) を注入して行う。
    """
    from backend.free.memory.stores.working import WorkingMemoryRegistry
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.memory.stores.long_term import LongTermMemory
    from backend.free.memory.init_evorefmem import (
        SCHEMA_VERSION as EVOREFMEM_SCHEMA_VERSION,
        initialize_evorefmem,
        needs_initialization,
        read_schema_version,
    )
    from backend.free.memory.migrations import (
        DEFAULT_MIGRATIONS,
        MigrationChainNotFoundError,
        MigrationError,
        SchemaMigrator,
    )
    from backend.free.memory.semantic.manifest import (
        ensure_manifest,
        normalize_embedding_model_id,
    )
    from backend.free.rag.vector_store import VectorStore

    # EvorefMem スキーマバージョン検査 & 自動初期化
    # - code 側の SCHEMA_VERSION を source of truth とする (cfg 値は参考情報)
    # - SchemaMigrator で in-place migration を試行 (v1 のみの現状は no-op)
    # - 未初期化 / 不一致の場合は destructive initialize_evorefmem を呼ぶ
    expected_version = EVOREFMEM_SCHEMA_VERSION
    cfg_version = cfg.get("memory", {}).get("schema_version")
    if cfg_version is not None and cfg_version != expected_version:
        logger.warning(
            "config.yaml memory.schema_version=%d does not match "
            "EvorefMem version=%d; using code version.",
            cfg_version, expected_version,
        )

    memory_dir = resolver.resolve_local("memory_dir")
    memory_dir.mkdir(parents=True, exist_ok=True)

    # SchemaMigrator で in-place migration を試行する
    # - 登録済 Migration で現在版から expected_version へ到達できる場合に実行
    # - v1 のみの現状は DEFAULT_MIGRATIONS=[] で常に no-op
    # - 連鎖解決できなかった場合は静かに fallback (destructive init 判定へ)
    current_version = read_schema_version(memory_dir)
    if (
        current_version is not None
        and current_version != expected_version
    ):
        migration_archive_dir_for_migrate = resolver.resolve_local(
            "migration_archive_dir",
        )
        migration_archive_dir_for_migrate.mkdir(parents=True, exist_ok=True)
        try:
            migrator = SchemaMigrator(
                migrations=list(DEFAULT_MIGRATIONS),
                migration_archive_dir=migration_archive_dir_for_migrate,
            )
            migrator.upgrade(
                memory_dir, current_version, expected_version,
            )
        except MigrationChainNotFoundError as e:
            logger.info(
                "SchemaMigrator has no chain for %d -> %d: %s "
                "(falling back to destructive init check)",
                current_version, expected_version, e,
            )
        except MigrationError as e:
            logger.warning(
                "SchemaMigrator upgrade failed (%d -> %d): %s "
                "(falling back to destructive init check)",
                current_version, expected_version, e,
            )

    if needs_initialization(memory_dir, expected_version):
        prompts_dir = resolver.resolve_local("prompts_dir")
        migration_archive_dir = resolver.resolve_local("migration_archive_dir")
        prompts_dir.mkdir(parents=True, exist_ok=True)
        migration_archive_dir.mkdir(parents=True, exist_ok=True)
        current = read_schema_version(memory_dir)
        logger.info(
            "EvorefMem auto-initialization: expected=%s actual=%s",
            expected_version, current,
        )
        try:
            result = initialize_evorefmem(
                memory_dir, prompts_dir, migration_archive_dir,
            )
            logger.info(
                "EvorefMem auto-initialized: backed_up=%d deleted=%d "
                "created=%d gc=%d",
                len(result.backed_up), len(result.deleted),
                len(result.created), len(result.gc_removed),
            )
        except Exception as e:
            logger.warning(
                "EvorefMem auto-initialization failed: %s — "
                "run `python scripts/init_evorefmem.py` manually", e,
            )

    # semantic/manifest.json を冪等に用意する
    # - destructive init 後 / legacy marker migration 後 / 通常起動の
    #   全経路で同じ呼び出しを通るよう、ここで無条件に ensure する
    # - model_id / dim は config.yaml::embedding から取得し、manifest
    #   未作成時のみ反映される (既存 manifest は優先されて上書きされない)
    try:
        emb_cfg = cfg.get("embedding", {}) or {}
        emb_model_name = emb_cfg.get("model_name")
        emb_dim = emb_cfg.get("dim")
        if emb_model_name and isinstance(emb_dim, int) and emb_dim >= 1:
            ensure_manifest(
                memory_dir,
                embedding_model_id=normalize_embedding_model_id(emb_model_name),
                embedding_dim=emb_dim,
            )
        else:
            logger.info(
                "semantic manifest ensure skipped: embedding config "
                "incomplete (model_name=%s, dim=%s)",
                emb_model_name, emb_dim,
            )
    except Exception as e:
        logger.warning(
            "semantic manifest ensure failed: %s — "
            "store will run without manifest-backed dim verification", e,
        )

    # EvorefMem トリガ辞書 (pin / fact / classify) の user override 配置先。
    # 同梱 default は ``backend/free/memory/_defaults/triggers/`` 配下。
    triggers_dir = resolver.resolve_local("triggers_dir")
    # note_builder のモジュールレベル default に設定し、以降に構築される
    # ChatNoteBuilder / CreateNoteBuilder (NoteBuilder singleton 経由 + sleep
    # extractors が fresh 構築するインスタンス) が同じ user override を拾うようにする。
    from backend.free.memory.notes.note_builder import set_default_triggers_dir
    set_default_triggers_dir(triggers_dir)
    stm = ShortTermMemory(cfg, triggers_dir=triggers_dir)
    from backend.free.api.chat.chat_recorder import release_session_turns

    wm = WorkingMemoryRegistry(
        cfg, drain_to=stm, drain_handler=release_session_turns,
    )

    # 他の永続ストア (vector store / experience / learned patterns) と同様、
    # 前回終了時のスナップショット (_lifespan._shutdown_stm_save が保存) を
    # 起動時に復元する。これが無いと restart 跨ぎでノートを失い、sleep-time の
    # URL/コマンド curation や fact 抽出が再起動前のノートを取りこぼす。
    # stm.load() はファイル未存在なら no-op。save とパスは完全一致させる。
    try:
        stm_path = resolver.resolve_local("memory_dir") / "short_term_notes.json"
        stm.load(stm_path)
        if stm.notes:
            logger.info("STM loaded on startup: %d notes", len(stm.notes))
        elif stm_path.exists():
            # 「ファイルはあるのに 0 件」は異常系。ここを黙って通すと STM 検索・
            # 閾値較正・記憶注入が揃って無言で死ぬ (2026-08-16 ライブ監査:
            # 40 ターン中 37 ターンで注入 0 件)。成功時しかログが無かったため
            # 起動ログからは正常と区別が付かなかった。
            logger.warning(
                "STM snapshot %s exists but yielded 0 notes; memory retrieval "
                "and threshold calibration will be inert this run", stm_path,
            )
    except Exception as e:
        logger.warning("STM load on startup skipped: %s", e)

    # ベクトルストア初期化（LTM 用）
    vs = None
    try:
        vectors_dir = resolver.resolve_local("vectors_dir")
        rag_cfg = cfg.get("rag", {})
        memmap_threshold = rag_cfg.get("memmap_threshold", 10000)
        vs = VectorStore(
            vectors_dir,
            memmap_threshold=memmap_threshold,
            quantization=str(rag_cfg.get("quantization", "int8")),
            debug_logger=state.debug_logger,
        )
        if vs.index_path.exists():
            vs.load()
            logger.info("Vector store loaded: %d vectors", vs.count)
        else:
            logger.info("Vector store initialized: empty (no existing index)")
        state.vector_store = vs
    except Exception as e:
        logger.warning("Vector store init skipped: %s", e)

    ltm = LongTermMemory(vs) if vs else None

    state.working_memory_registry = wm
    state.short_term_memory = stm
    state.long_term_memory = ltm
    logger.info("Memory system initialized")
    return wm, stm, ltm, vs
