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
from backend.log_config import DevelopLevel, get_logger

logger = get_logger("app_state")

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

    from backend.debug_logger import DebugLogger
    from backend.free.agent.aux_prompt_manager import AuxPromptManager
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
    from backend.free.llm.aux_client import AuxClient
    from backend.free.llm.llm_client import LLMClient
    from backend.free.llm.local_client import LocalClient
    from backend.free.memory.stores.long_term import LongTermMemory
    from backend.free.memory.scheduler import SleepTimeScheduler
    from backend.free.memory.semantic.store import SemanticFactStore
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.memory.stores.working import (
        WorkingMemory,
        WorkingMemoryRegistry,
    )
    from backend.free.rag.judge_usage_tracker import JudgeUsageTracker
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
    # 補助タスク (sleep-time / 学習 / 長文プラン / create の計画・仕様合成) を
    # ベースモデルの専有スロットで実行するクライアント。``local_client`` が
    # 未接続の間は ``None``。
    aux_client: AuxClient | None = None

    # ── メモリシステム ──
    # legacy: session_id を持たない読み手 (統計 API / テストの seam) 向け。
    # 本番では ``working_memory_registry`` が SSOT で、この属性は **最後に触った
    # セッション** の WM を返すプロパティになる (クラス定義直後で差し替え)。
    # 応答パスは必ず ``get_memory_system(session_id)`` を使うこと。
    working_memory: WorkingMemory | None = None
    # セッション別 WM の台帳 (``memory.working_max_sessions`` で LRU 上限)。
    working_memory_registry: WorkingMemoryRegistry | None = None
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
    # 内容精査ゲート (``ChunkContentGate``、``rag.self_rag.content_gate.*``) の
    # セッション / クエリ単位の aux 発火カウンタ。``run_search_pipeline`` 経由で
    # ``session_id`` と共に参照し、``max_per_session`` / ``max_per_query`` の
    # 上限判定とセッション切替時のリセット (``reset_session``) を担う。
    judge_tracker: "JudgeUsageTracker | None" = None
    # 競合確認のセッション単位カウンタ (RAG 用とは別インスタンス)。旧
    # ``conflict_chat_judge`` 経路は撤去済みで、現在は ``prepare_memory_context``
    # のセッション切替時 ``reset_session`` にしか使われない (予約)。
    conflict_judge_tracker: "JudgeUsageTracker | None" = None
    # 埋め込みモデルとストアの次元不一致フラグ
    embedding_dim_mismatch: bool = False
    # 不一致時の参考情報（fronend のバナー表示用）
    embedding_dim_stored: int | None = None
    embedding_dim_current: int | None = None

    # ── model_state.json / config.yaml 整合性 ──
    # 不一致検出時のみ dict を格納 (current_filename / config_filename / recommendation)
    model_state_mismatch: dict | None = None
    # component (embed) の不一致。{config_key: {model_state, config}}
    component_state_mismatches: dict = field(default_factory=dict)
    # llama-server が実際にロードしているモデル (/props) と config の不一致。
    # モデル移行は稼働中の llama-server を差し替えないため、再起動するまで
    # 別モデルが serve され続ける (2026-08-12 に 62.8 時間見逃した事象)。
    # {served_filename / expected_filename / recommendation}
    served_model_mismatch: dict | None = None
    # 出力品質プローブの記録 (QualityProbeStore)。モデル切替を検知した役割だけ
    # 検査され、結果は /api/status で参照される。型注釈を文字列にせず Any 相当で
    # 持つのは、AppState を EvorefGen 非依存に保つため (lazy import で構築)。
    model_quality: object | None = None
    _quality_probe_task: object | None = None

    # ── エージェント / プロンプト ──
    prompt_manager: SystemPromptManager | None = None
    aux_prompt_manager: AuxPromptManager | None = None
    feedback_collector: FeedbackCollector | None = None
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
    # Level 1 独立ループ / Level 2 runner 注入 / Pro 学習コンポーネント注入 /
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

    # 起動時に fire-and-forget で張る背景タスク (tool gate warmup /
    # fewshot 埋め込み backfill 等) の参照。イベントループは弱参照しか
    # 持たないので、参照を保持しないと途中で GC される。完了時に自ら抜ける。
    background_tasks: set["asyncio.Task[Any]"] = field(default_factory=set)

    # ── セッションレジストリ ──
    _active_sessions: dict[str, SessionInfo] = field(default_factory=dict)

    # max_tokens 到達で切れた応答の継続待ち (session_id -> TruncatedResponse)。
    # 次ターンの「続けて」を実際の継続生成へ繋ぐための観測事実で、
    # 読み書きは backend.free.api.chat._continuation に閉じる。
    truncated_responses: dict[str, "Any"] = field(default_factory=dict)

    # 直前ターンで生成した長文成果物 (session_id -> LastArtifact)。
    # 長文は履歴予算 (実測 1612 トークン) に入らず次ターンで消えるため、
    # 「いまの計画書は何章?」「保存して」に答える材料がどこにも残らない。
    # 読み書きは backend.free.api.chat._artifact に閉じる。
    last_artifacts: dict[str, "Any"] = field(default_factory=dict)

    # ── メモリシステムアクセス ──

    def get_memory_system(
        self, session_id: str | None = None,
    ) -> tuple[WorkingMemory, ShortTermMemory, LongTermMemory] | None:
        """メモリシステム 3 層を返す（未初期化時は None）

        ``session_id`` を渡すとそのセッションの WM (無ければ作る) を返す。
        応答パスの読み書きはこちら。省略時は legacy 動作で、最後に触った
        セッションの WM (台帳が空なら空の窓) — 統計 API 等、セッションを
        持たない読み手専用で、並行セッション下では「誰の窓か」が定まらない。
        """
        registry = self.working_memory_registry
        if registry is not None:
            wm = registry.get(session_id) if session_id else registry.current()
            return wm, self.short_term_memory, self.long_term_memory
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

        # ToolCallJudge._llm_client を同期 (文法制約 JSON のツール分類器が
        # ``generate_constrained`` を叩く)。モード切替の base 再起動でクライアント
        # が差し替わるため、ここで追随しないと judge が閉じた旧オブジェクトを
        # 掴んだままになる。
        judge = self.tool_call_judge
        if judge is not None and hasattr(judge, "_llm_client"):
            judge._llm_client = client

        # 補助クライアントは同じ LocalClient の上に立つ。base を差し替えたら
        # ここで追随させないと、閉じた旧クライアントを掴んだまま補助タスクが
        # 全滅する (モード切替で base を再起動するたびに再現する)。
        # ``rebind`` は較正値のモデルキーも再解決する (差し替えた base が別
        # モデルなら旧モデルの timeout 較正を引き継がない)。config が読めない
        # 経路 (テスト) では local の差し替えだけ行う。
        try:
            from backend.config import get_config
            cfg = get_config()
        except Exception:
            cfg = None
        if self.aux_client is not None:
            self.aux_client.rebind(client, cfg)
        else:
            # **未配線からの復帰**。起動時に llama-server がまだモデルをロード
            # 中だと ``_init_aux_client`` は ``None`` を配って終わり、以降
            # lazy-connect でチャットが復活しても補助タスクだけが縮退したまま
            # プロセス寿命を終える (2026-09-03 ライブ監査: 3.5 時間ぶん
            # MetaCognitive の計画生成と CritiqueSynthesizer が死んでいた)。
            # ここで作り直し、``None`` を掴んだ長命の保持者へ配り直す。
            self._wire_aux_client_late(client, cfg)

        # SleepTimeScheduler にも追随させる。Level 1 常駐ループは
        # ``_llm_client is None`` の tick を毎回 skip するため、起動時に
        # llama-server が未接続だと (wire 時の ``if state.llm_client:`` が
        # 通らず) lazy-connect 後もプロセスを再起動するまで Level 1 が
        # 永久に走らない。base 差し替え時に閉じた旧クライアントを掴み続ける
        # 問題も同じ経路で塞がる。
        if self.sleep_scheduler is not None:
            self.sleep_scheduler.set_llm_client(self.llm_client)

    def _wire_aux_client_late(self, client: LocalClient, cfg: dict | None) -> None:
        """起動時に作れなかった AuxClient を後から作って配り直す。

        ``state.aux_client`` を毎リクエスト読む保持者 (MetaCognitive / long-form /
        staged) は代入だけで復帰する。**構築時に一度だけ受け取る保持者**
        (CritiqueSynthesizer) は個別に差し替えないと ``None`` のまま残る。
        """
        if not hasattr(client, "generate_constrained"):
            return
        try:
            from backend.free.llm.aux_client import AuxClient

            self.aux_client = AuxClient(
                client, config=cfg, debug_logger=self.debug_logger,
            )
        except Exception as e:
            logger.warning("Late aux client wiring failed: %s", e)
            return
        logger.info("Aux client wired late after llama-server became reachable")

        scheduler = self.learning_scheduler
        synthesizer = getattr(scheduler, "_critique_synthesizer", None) if scheduler else None
        if synthesizer is not None and getattr(synthesizer, "_llm_client", None) is None:
            synthesizer._llm_client = self.aux_client
            logger.info("CritiqueSynthesizer re-bound to the late aux client")

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


def _legacy_working_memory(self: AppState) -> "WorkingMemory | None":
    registry = self.working_memory_registry
    if registry is not None:
        return registry.current()
    return self.__dict__.get("_working_memory_legacy")


def _set_legacy_working_memory(self: AppState, wm: "WorkingMemory | None") -> None:
    self.__dict__["_working_memory_legacy"] = wm


# dataclass の ``__init__`` は ``self.working_memory = ...`` を実行するので、
# フィールド宣言の既定値 (None) を dataclass に取り込ませた後でプロパティに
# 差し替える。コンストラクタ引数 ``working_memory=`` と代入はそのまま通り、
# 読み出しは台帳があれば最後に触ったセッションの WM を返す (legacy 読み手用)。
AppState.working_memory = property(_legacy_working_memory, _set_legacy_working_memory)  # type: ignore[assignment]

