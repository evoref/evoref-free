"""4 pillar エントリポイント dataclass 群

``wire_pillars()`` が 4 pillar (EvorefGen / EvorefMem / EvorefLoop / EvorefLearn)
を依存順に構築する際に各 pillar の「公開 entry point」を集約するためのコンテナ。
``AppState`` の ``gen`` / ``mem`` / ``loop`` / ``learn`` / ``pro`` 属性に格納される。

## 依存方向

``EvorefGen → EvorefMem ← EvorefLoop ← EvorefLearn``

- :class:`GenPillar`: 完全独立 (他 pillar に依存しない)
- :class:`MemPillar`: 最下流 (他 pillar に依存しない)
- :class:`LoopPillar`: :class:`MemPillar` + :class:`GenPillar` に依存
- :class:`LearnPillar`: :class:`MemPillar` + :class:`LoopPillar` + :class:`GenPillar` に依存

## Pro 拡張

:class:`ProState` は Pro 版で追加される補助 pillar (AuxComponents /
WidgetProxyManager 等) を集約する。Free 版では ``None``。

本モジュール自身は具象クラス依存を持たず、全属性を ``Any`` または
``TYPE_CHECKING`` 専用 import で宣言する (循環 import を避けるため)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.free.agent.aux_prompt_manager import AuxPromptManager
    from backend.free.agent.feedback import FeedbackCollector
    from backend.free.agent.learned_patterns import LearnedPatternStore
    from backend.free.agent.prompt_manager import SystemPromptManager
    from backend.free.learning.level0_instant import ExperienceBuffer
    from backend.free.learning.policy_adjuster import PolicyAdjuster
    from backend.free.learning.scheduler import LearningScheduler
    from backend.free.llm.aux_client import AuxClient
    from backend.free.llm.llm_client import LLMClient
    from backend.free.llm.local_client import LocalClient
    from backend.free.loop.driver import LoopDriver
    from backend.free.loop.log_ingestor import LogIngestor
    from backend.free.memory.stores.long_term import LongTermMemory
    from backend.free.memory.scheduler import SleepTimeScheduler
    from backend.free.memory.stores.short_term import ShortTermMemory
    from backend.free.memory.stores.working import WorkingMemory
    from backend.free.rag.cartridge_manager import CartridgeManager
    from backend.free.rag.embedding_backend import EmbeddingBackend
    from backend.free.rag.bm25_retriever import BM25Retriever
    from backend.free.rag.retriever import HybridRetriever
    from backend.free.rag.vector_store import VectorStore
    from backend.pro.learn_components import ProLearnComponents


@dataclass
class GenPillar:
    """EvorefGen pillar — LLM / RAG / 長文生成 (完全独立)。

    Attributes:
        local_client: ベースモデル (llama-server) クライアント。未接続時は ``None``。
        llm_client: LocalClient を束ねたファサード (chat_in_flight / is_serving_user を提供)。
        aux_client: 補助タスク LLM クライアント。未設定時は ``None``。
        embedder: 埋め込みバックエンド (llama-cpp server 経由)。未初期化時は ``None``。
        bm25_retriever: BM25 語彙検索索引。**全経路で共有する 1 インスタンス**
            (チャット応答経路の LTM ハイブリッド / sleep-time の索引更新 /
            hybrid_retriever)。
        hybrid_retriever: BM25 + Vector ハイブリッド検索器
            (ベンチマーク・長文生成の unit 検索)。
    """

    local_client: "LocalClient | None" = None
    llm_client: "LLMClient | None" = None
    aux_client: "AuxClient | None" = None
    embedder: "EmbeddingBackend | None" = None
    bm25_retriever: "BM25Retriever | None" = None
    hybrid_retriever: "HybridRetriever | None" = None


@dataclass
class MemPillar:
    """EvorefMem pillar — WM / STM / LTM + SemMem (最下流)。

    ``SemanticFactStore`` は ``AppState._semantic_stores`` に scope ごと
    lazy 生成される (他 pillar は Fact View 経由でのみアクセス)。
    """

    working_memory: "WorkingMemory"
    short_term_memory: "ShortTermMemory"
    long_term_memory: "LongTermMemory | None"
    vector_store: "VectorStore | None"
    cartridge_manager: "CartridgeManager | None"
    sleep_scheduler: "SleepTimeScheduler"
    current_project_id: str | None = None


@dataclass
class LoopPillar:
    """EvorefLoop pillar — 自律実行ループ / ハーネス / エージェント。

    ``loop.enabled=false`` の場合は ``driver=None`` で構築される。
    """

    driver: "LoopDriver | None" = None
    enabled: bool = True
    # develop=evolve 時のみ起動される decision/outcome JSONL の
    # tail-follow + JOIN コンポーネント。それ以外の develop_level では ``None``。
    log_ingestor: "LogIngestor | None" = None


@dataclass
class LearnPillar:
    """EvorefLearn pillar — 学習サイクル / 最適化 (最上流)。

    :class:`LearningScheduler` は Level 0.5 〜 Level 2 (Pro) の実行主体。
    """

    scheduler: "LearningScheduler"
    experience_buffer: "ExperienceBuffer"
    learned_patterns_store: "LearnedPatternStore"
    prompt_manager: "SystemPromptManager"
    aux_prompt_manager: "AuxPromptManager"
    feedback_collector: "FeedbackCollector"
    # develop=evolve 時のみ初期化される SemMem 書き戻し
    # コンポーネント。LogIngestor (Loop pillar) からの JoinedPair を消費する。
    policy_adjuster: "PolicyAdjuster | None" = None


@dataclass
class ProGenPillar:
    """Pro 版 EvorefGen 拡張。

    Pro Gen 拡張 (WidgetProxyManager / Web ターミナル / マルチローカルモデル
    切替など) の存在を示すマーカーとして機能する。
    """


@dataclass
class ProLearnPillar:
    """Pro 版 EvorefLearn 拡張。

    Attributes:
        components: :class:`LearnComponentsProtocol` 準拠の集約コンポーネント。

    補助タスク経験バッファは Free 側 (``AppState.aux_experience_buffer``) が
    所有する — アシストモデル撤去で判定自体がベースへ移り、Pro 限定である
    必要がなくなったため。
    """

    components: "ProLearnComponents | None" = None


@dataclass
class ProState:
    """Pro 版補助 pillar の集約。Free 版では ``AppState.pro=None``。"""

    gen: ProGenPillar | None = None
    learn: ProLearnPillar | None = None


@dataclass
class DevelopGenPillar:
    """Develop 版 EvorefGen 拡張のマーカー。

    現状はスケルトンのみ (シナリオハーネス等の Develop 限定 Gen 拡張は
    別タスクで実装)。:class:`ProGenPillar` と並列に存在し、
    Develop エディション ⊇ Pro エディションの上位互換階層を表す。
    """


@dataclass
class DevelopLearnPillar:
    """Develop 版 EvorefLearn 拡張のマーカー。

    現状はスケルトンのみ。シナリオベース自動学習ハーネスの上位レイヤー
    (`docs/e_04_user_feature_matrix.md` §14) は別タスクで実装する。
    """


@dataclass
class DevelopLoopPillar:
    """Develop 版 EvorefLoop 拡張のマーカー。

    現状はスケルトンのみ。シナリオランナー / 自動学習サイクル制御は
    別タスクで実装する。
    """


@dataclass
class DevelopState:
    """Develop 版補助 pillar の集約。Free / Pro 版では ``AppState.develop=None``。

    Develop は Pro の上位互換のため、Pro 拡張 (``AppState.pro``) と
    Develop 拡張 (``AppState.develop``) は同時に保持される設計
    (Develop エディション起動時は ``pro`` も ``develop`` も両方非 None になる)。
    """

    gen: DevelopGenPillar | None = None
    learn: DevelopLearnPillar | None = None
    loop: DevelopLoopPillar | None = None


__all__ = [
    "GenPillar",
    "MemPillar",
    "LoopPillar",
    "LearnPillar",
    "ProGenPillar",
    "ProLearnPillar",
    "ProState",
    "DevelopGenPillar",
    "DevelopLearnPillar",
    "DevelopLoopPillar",
    "DevelopState",
]
