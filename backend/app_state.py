"""アプリケーション状態コンテナ（依存性注入用）

グローバル変数ベースの _state.py を置き換え、FastAPI の Depends で
各エンドポイントに AppState を注入する。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import Request

# develop モードの 3 段階レベル。CLI フラグ
# `--develop=<level>` の SSOT として保持する。``"off"`` は通常起動
# (CLI フラグ未指定)、``"debug"`` は Free / Pro 共通、``"investigate"`` /
# ``"evolve"`` は Pro 限定。``EVOREF_DEVELOP_LEVEL`` 環境変数で
# サブプロセス (FastAPI バックエンド) へ伝搬する。型本体は
# ``backend.log_config`` で定義 (循環 import 回避)。
from backend.log_config import DevelopLevel

if TYPE_CHECKING:
    from collections.abc import Callable

    from backend.debug_logger import DebugLogger
    from backend.free.agent.assist_prompt_manager import AssistPromptManager
    from backend.free.core.llama_process_manager import LlamaProcessManager
    from backend.free.core.policy_interpreter import PolicyInterpreter
    from backend.free.agent.agent_tracer import AgentTracer
    from backend.free.agent.feedback import FeedbackCollector
    from backend.free.agent.file_manager import SessionFileManager
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.agent.prompt_manager import SystemPromptManager
    from backend.free.agent.reactive import ReactiveAgent
    from backend.free.agent.tool_call_judge import ToolCallJudge
    from backend.free.agent.tools_registry import ToolsRegistry
    from backend.free.learning.scheduler import LearningScheduler
    from backend.free.llm.assist_client import AssistModelClient
    from backend.free.llm.llm_client import LLMClient
    from backend.free.llm.local_client import LocalClient
    from backend.free.memory.stores.long_term import LongTermMemory
    from backend.free.memory.scheduler import SleepTimeScheduler
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.pipeline.rag_judge_assist_log import RagJudgeAssistLog
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.memory.stores.working import WorkingMemory
    from backend.free.memory.views.mem import MemFactView
    from backend.free.rag.assist_judge_tracker import AssistJudgeUsageTracker
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.lazy_contextual import LazyContextualPrefixService
    from backend.free.rag.cartridge_manager import CartridgeManager
    from backend.free.rag.vector_store import VectorStore
    from backend.free.themes.theme_service import ThemeManager
    from backend.pillars import (
        DevelopState,
        GenPillar,
        LearnPillar,
        LoopPillar,
        MemPillar,
        ProState,
    )
    from backend.pro.api.widgets import WidgetProxyManager
    from backend.pro.learning.cartridge_change_handler import CartridgeChangeHandler
    from backend.pro.terminal.session_manager import SessionManager as TerminalSessionManager

SESSION_TTL_SEC = 3600  # 異常終了セッションの自動クリーンアップ: 1時間


@dataclass
class LastRequestMetrics:
    """直近リクエストの生成メトリクス（デバッグオーバーレイ用）"""

    ttft_ms: float | None = None
    tok_per_sec: float | None = None
    updated_at: float = 0.0


@dataclass
class SessionInfo:
    """アクティブセッション情報"""
    session_id: str
    mode: str
    client_type: str  # "cli" | "web"
    registered_at: float = field(default_factory=time.time)


@dataclass
class AppState:
    """アプリケーション状態コンテナ

    lifespan で生成し app.state.app_state に格納する。
    各エンドポイントは Depends(get_app_state) で受け取る。
    """

    # ── モード状態 ──
    current_mode: str = "chat"

    # base 学習パーティションの active モデルスラグ (model_slug 済)。
    # partition 無効 / 未確定時は空文字。SemMem ``learn.*`` subject へのモデル次元
    # 注入元として学習コンポーネント構築・モデル切替 rebind から参照する。
    active_base_model_slug: str = ""

    # ── LLM クライアント ──
    local_client: LocalClient | None = None
    llm_client: LLMClient | None = None
    assist_client: AssistModelClient | None = None

    # ── メモリシステム ──
    working_memory: WorkingMemory | None = None
    short_term_memory: ShortTermMemory | None = None
    long_term_memory: LongTermMemory | None = None
    # SemanticFactStore: scope ("global" or "project:<id>") -> store
    _semantic_stores: dict[str, "SemanticFactStore"] = field(default_factory=dict)
    # 現在のプロジェクト ID: startup 時に project_resolver で
    # 解決し、`@self` 仮想カートリッジや SemMem ルックアップから参照する。
    # chat モードや解決失敗時は None。
    current_project_id: str | None = None

    # ── RAG ──
    vector_store: VectorStore | None = None
    embedder: EmbeddingBackend | None = None
    cartridge_manager: CartridgeManager | None = None
    # Lazy Contextual Retrieval — retrieval 時に on-demand で
    # プレフィックスを生成するサービス。``rag.contextual_prefix.mode=lazy``
    # の時のみ wire_pillars で構築・注入される。``None`` または
    # ``is_active=False`` の場合、search_pipeline 側は通知をスキップする。
    lazy_contextual: "LazyContextualPrefixService | None" = None
    # Self-RAG assist_judge のセッション / クエリ単位カウンタ
    # ``run_search_pipeline`` 経由で ``session_id`` と共に参照し、
    # ``rag.self_rag.assist_judge.max_per_session`` / ``max_per_query`` の
    # 上限判定とセッション切替時のリセット (``reset_session``) を担う。
    assist_judge_tracker: "AssistJudgeUsageTracker | None" = None
    # conflict_chat_judge のセッション単位発火カウンタ (RAG 用とは別インスタンス)。
    # ``maybe_resolve_pending_conflicts`` が ``memory.conflict.chat_review
    # .max_judge_per_session`` の上限判定とセッション切替リセットに使う。
    # ``check()`` は使わず ``record`` / ``get_session_count`` / ``reset_session``
    # のみ利用する。
    conflict_judge_tracker: "AssistJudgeUsageTracker | None" = None
    # RAG necessity/quality の embedding 決定論的リコール用リングバッファ。
    # SemMem ではない (チャット応答パスで直接 record するだけ)。sleep-time
    # Step 8.7 (rag_judge_curator) が drain して world_fact 化する。
    rag_judge_assist_log: "RagJudgeAssistLog | None" = None
    # URL/executable_command/RAG necessity・quality の embedding リコールで
    # 共有する global scope の MemFactView。``_init_tools`` のローカル変数を
    # 他の呼出元 (search_pipeline.py) でも再利用するため state に昇格。
    mem_view: "MemFactView | None" = None
    # 埋め込みモデルとストアの次元不一致フラグ
    embedding_dim_mismatch: bool = False
    # 不一致時の参考情報（fronend のバナー表示用）
    embedding_dim_stored: int | None = None
    embedding_dim_current: int | None = None

    # ── model_state.json / config.yaml 整合性 ──
    # 不一致検出時のみ dict を格納 (current_filename / config_filename / recommendation)
    model_state_mismatch: dict | None = None
    # component (assist/embed) の不一致。{config_key: {model_state, config}}
    component_state_mismatches: dict = field(default_factory=dict)
    # 出力品質プローブの記録 (QualityProbeStore)。モデル切替を検知した役割だけ
    # 検査され、結果は /api/status で参照される。型注釈を文字列にせず Any 相当で
    # 持つのは、AppState を EvorefGen 非依存に保つため (lazy import で構築)。
    model_quality: object | None = None
    _quality_probe_task: object | None = None

    # ── エージェント / プロンプト ──
    prompt_manager: SystemPromptManager | None = None
    assist_prompt_manager: AssistPromptManager | None = None
    feedback_collector: FeedbackCollector | None = None
    # assist 経験記録 closure (Pro/Develop 起動時のみ非 None)。primitive 5 引数
    # (action_type, input_context, output, outcome, mode) を受け、Pro の assist 経験
    # バッファへ記録する。mode は既定 "chat" (省略可)。factory 層で構築し、Free 側
    # (chat_service / judge) は Pro 型非依存で呼ぶ。
    assist_experience_recorder: "Callable[[str, str, str, float, str], None] | None" = None
    file_manager: SessionFileManager | None = None
    learned_patterns_store: LearnedPatternStore | None = None
    tools_registry: ToolsRegistry | None = None
    tool_call_judge: ToolCallJudge | None = None
    # Reactive 層 (挨拶/日時/キャッシュ即応) の常駐インスタンス。リクエスト毎に
    # 生成すると LRU キャッシュが温まらないため AppState に保持する。
    reactive_agent: "ReactiveAgent | None" = None
    agent_tracer: AgentTracer | None = None

    # ── スケジューラ ──
    sleep_scheduler: SleepTimeScheduler | None = None
    learning_scheduler: LearningScheduler | None = None

    # ── ポリシー ──
    policy_interpreter: PolicyInterpreter | None = None

    # ── llama-server プロセス管理 ──
    llama_manager: LlamaProcessManager | None = None

    # ── デバッグ / テーマ ──
    debug_logger: DebugLogger | None = None
    theme_manager: ThemeManager | None = None
    # develop モードレベル (SSOT)。CLI フラグ
    # `--develop=<level>` で起動時に解決し、log_config / DebugLogger /
    # 各 pillar が参照する。``"off"`` は通常起動。
    develop_level: DevelopLevel = "off"
    # 自己学習サイクルの無効化フラグ (SSOT)。CLI フラグ `--no-learning`
    # で起動時に True になり、``EVOREF_LEARNING_DISABLED=1`` 環境変数で
    # サブプロセスへ伝播する。True の場合、LearningScheduler tick /
    # Level 1 独立ループ / Level 2 runner 注入 / Pro assist 注入 /
    # SleepTimeWorker 学習関連書戻し / FeedbackCollector.record が
    # すべて no-op となり、SemMem の読み込みやチャット応答は通常通り。
    learning_disabled: bool = False

    # ── メトリクス ──
    last_request_metrics: LastRequestMetrics = field(
        default_factory=LastRequestMetrics,
    )

    # ── Pro ──
    widget_proxy_manager: WidgetProxyManager | None = None
    cartridge_change_handler: CartridgeChangeHandler | None = None
    # Pro Web ターミナル (PTY + WebSocket) のセッションマネージャ
    # `pro.terminal.enabled=true` のときのみ ``setup_pro_gen`` で構築される。
    terminal_session_manager: TerminalSessionManager | None = None

    # ── 4 pillar エントリポイント ──
    # wire_pillars() が依存順 (Gen → Mem → Loop → Learn) に構築して格納する。
    # 既存フィールド (local_client / working_memory / learning_scheduler 等) は
    # 移行期間中はそのまま共存し、新規コードは pillar 経由でアクセスすることが
    # 推奨される。``pro`` は Pro エディション起動時のみ非 None。
    gen: GenPillar | None = None
    mem: MemPillar | None = None
    loop: LoopPillar | None = None
    learn: LearnPillar | None = None
    pro: ProState | None = None
    # Develop 版補助 pillar 集約。Develop は Pro の上位互換 (Develop ⊇ Pro)
    # のため、Develop エディション起動時は ``pro`` と ``develop`` の双方が
    # 非 None になる。Free / Pro 版では ``None``。
    develop: DevelopState | None = None

    # develop=evolve 時のみ稼働する LogIngestor → PolicyAdjuster
    # ブリッジタスク。lifespan shutdown で cancel + flush_all される。
    log_ingestor_bridge_task: "Any | None" = None

    # ── セッションレジストリ ──
    _active_sessions: dict[str, SessionInfo] = field(default_factory=dict)

    # ── メモリシステムアクセス ──

    def get_memory_system(
        self,
    ) -> tuple[WorkingMemory, ShortTermMemory, LongTermMemory] | None:
        """メモリシステム 3 層を返す（未初期化時は None）"""
        if self.working_memory is None:
            return None
        return self.working_memory, self.short_term_memory, self.long_term_memory

    def get_semantic_store(self, scope: str = "global") -> "SemanticFactStore":
        """SemanticFactStore を scope ごとに lazy 取得する

        scope は ``"global"`` または ``"project:<id>"`` を受け付ける。
        初回呼び出し時にディスク (`<memory_dir>/semantic/...`) からロードし、
        以後同じインスタンスを返す。

        を読み込んで Store に注入し、``embeddings/<model_id>/vectors.npy``
        の次元と ``manifest.embedding.dim`` の照合を有効化する。manifest
        未作成 (=古い環境を事前初期化なしで読み込んだケース) では照合は
        自動的にスキップされる。
        """
        cached = self._semantic_stores.get(scope)
        if cached is not None:
            return cached
        from backend.config import get_path_resolver
        from backend.free.memory.semantic.manifest import load_manifest
        from backend.free.memory.semantic.store import SemanticFactStore

        memory_dir = get_path_resolver().resolve_local("memory_dir")
        semantic_root = memory_dir / "semantic"
        manifest = load_manifest(memory_dir)
        if scope == "global":
            store = SemanticFactStore.for_global(semantic_root, manifest=manifest)
        elif scope.startswith("project:"):
            project_id = scope.split(":", 1)[1]
            if not project_id:
                raise ValueError(f"invalid scope: {scope}")
            store = SemanticFactStore.for_project(
                semantic_root, project_id, manifest=manifest,
            )
        else:
            raise ValueError(
                f"unknown scope: {scope!r} (expected 'global' or 'project:<id>')",
            )
        self._semantic_stores[scope] = store
        return store

    def invalidate_semantic_stores(self) -> None:
        """キャッシュ済み SemanticFactStore を全て破棄する.

        embed モデルの cross-model 切替 (manifest の active model_id swap) 後に
        呼び、次回 :meth:`get_semantic_store` で新 manifest からロードし直させる。
        なお既に配線済みの Fact View (MemFactView / LearnFactView) は wire 時に
        掴んだ旧 store 参照を保持するため、chat リコール等の live 反映には別途
        backend 再起動が必要 (本メソッドはキャッシュ層のみ無効化する)。
        """
        self._semantic_stores.clear()

    # ── LLM クライアント設定 ──

    def set_local_client(self, client: LocalClient | None) -> None:
        """ローカルクライアントを設定し、LLMClient 側も同期する

        起動時に llama-server が未接続だった場合、LLMClient が未生成のまま
        lazy-connect が成功するケースがある。その場合は LLMClient を新規作成する。
        """
        self.local_client = client
        if client is None:
            return
        if self.llm_client is not None:
            self.llm_client.local = client
        else:
            from backend.free.llm.llm_client import LLMClient
            self.llm_client = LLMClient(local=client)

    def set_assist_client(self, client: AssistModelClient | None) -> None:
        """アシストモデルクライアントを設定し、下流コンポーネントを同期する.

        起動時 health check 失敗で None にステージされた後、lazy_connect 経由
        で本メソッドが呼ばれる。SleepTimeScheduler だけでなく、コンストラクタ
        で client を握っているコンポーネント (SleepTimeWorker / ToolCallJudge)
        の参照も差し替えないと、url_curator が永遠に degraded mode のままに
        なる。
        """
        self.assist_client = client
        if self.sleep_scheduler is not None and client is not None:
            self.sleep_scheduler.set_assist_llm_client(client)

        # SleepTimeWorker._assist_client を同期 (url_curator の入力)
        sched = self.sleep_scheduler
        if sched is not None:
            worker = getattr(sched, "_worker", None)
            if worker is not None and hasattr(worker, "_assist_client"):
                worker._assist_client = client

        # ToolCallJudge._assist_client を同期 (アシスト判定パス)
        judge = self.tool_call_judge
        if judge is not None and hasattr(judge, "_assist_client"):
            judge._assist_client = client

        # LearningScheduler.assist_llm_client を同期 (Level 1 プロンプト変異の
        # assist ルーティング #5)。lazy_connect で assist が後から起きた場合も縮退を防ぐ。
        ls = self.learning_scheduler
        if ls is not None and client is not None and hasattr(ls, "assist_llm_client"):
            ls.assist_llm_client = client

    # ── セッション管理 ──

    def register_session(
        self, session_id: str, mode: str, client_type: str = "cli",
    ) -> bool:
        """セッションを登録。重複がある場合は False を返す"""
        self._cleanup_expired_sessions()
        if session_id in self._active_sessions:
            return False
        self._active_sessions[session_id] = SessionInfo(
            session_id=session_id,
            mode=mode,
            client_type=client_type,
        )
        return True

    def unregister_session(self, session_id: str) -> bool:
        """セッションを解除。存在しない場合は False を返す"""
        return self._active_sessions.pop(session_id, None) is not None

    def get_active_sessions(self) -> list[SessionInfo]:
        """アクティブセッション一覧を返す（TTL クリーンアップ含む）"""
        self._cleanup_expired_sessions()
        return list(self._active_sessions.values())

    def _cleanup_expired_sessions(self) -> int:
        """TTL 超過セッションを削除し、削除件数を返す"""
        now = time.time()
        expired = [
            sid for sid, info in self._active_sessions.items()
            if now - info.registered_at > SESSION_TTL_SEC
        ]
        for sid in expired:
            del self._active_sessions[sid]
        return len(expired)


def get_app_state(request: Request) -> AppState:
    """FastAPI 依存性注入: リクエストから AppState を取得する"""
    return request.app.state.app_state
