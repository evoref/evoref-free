"""

EvorefLearn (learning / optimizer pillar) の Pro 拡張ポイントを Protocol
として固定する。Pro 版 (Free 版にない高度学習機能) は
``backend/pro/`` 配下で実装され、``wire_pillars`` /
``app_factory`` は edition を判定して以下いずれかを ``LearningScheduler``
に DI する:

- Free: :class:`NoopAssistComponents` (全 ``None`` を返すスタブ)
- Pro:  :class:`~backend.pro.assist_components.ProAssistComponents`
  (Level2Runner / AssistPromptEvolver / AssistExperienceBuffer / ...)

pillar 境界の不変条件により、本 Protocol ファイルは **Pro 具象クラスを
import してはならない** (Free → Pro 参照は依存方向違反)。従って、各
コンポーネントも **サブ Protocol として宣言** し、具象実装 (Free / Pro 側)
が duck typing でサブ Protocol を満たすよう設計する。

設計原則 (CLAUDE.md §8 / ``docs/p_01_advanced_learning.md`` §0.1):
- 最小 API 原則: 現在 ``LearningScheduler`` が Pro コンポーネントに対し
  呼び出しているメソッドだけを宣言
- ``@runtime_checkable``: isinstance チェックを可能にする
- Free / Pro 双方の実装を差し込み可能にする

## AssistComponents 集約

旧 ``backend.pro.inject_assist_components(scheduler, assist_experience_buf,
assist_version_manager, eval_assist_manager, assist_prompt_mgr,
assist_llm_client)`` の位置引数 6 本は
:class:`AssistComponentsProtocol` インスタンス 1 本に集約される。Protocol
には以下の 2 系統のフィールドが含まれる:

1. **運用コンポーネント** (Level 1 サイクル + アシストモデル実行で使う)
   - ``experience_buffer`` / ``version_manager`` / ``eval_manager`` /
     ``prompt_manager`` / ``assist_client``
2. **Level 2 拡張** (Pro 専用の LoRA 微調整 / プロンプト進化 / カートリッジ変更)
   - ``level2_runner`` / ``assist_prompt_evolver`` /
     ``experience_evaluator`` / ``cartridge_change_handler``

Free 版は全部 ``None`` を返す :class:`NoopAssistComponents` を渡し、Pro 版
は ``backend.pro.assist_components.ProAssistComponents`` を渡す。

``external_client`` プロパティは削除された。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from backend.free.llm.protocols import AssistModelClientProtocol


# ──────────────────────────────────────────────────────────────────────────
# 運用コンポーネント Protocol (Level 1 サイクル + アシストモデル実行)
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class AssistExperienceBufferProtocol(Protocol):
    """アシストモデル経験バッファ (§7.5.2) の抽象。

    Free 版では実装なし。Pro 版は
    ``backend.pro.assist_experience.AssistExperienceBuffer`` が本 Protocol
    を自然に満たす。``LearningScheduler`` からは ``get_filtered`` /
    ``append`` / ``count`` のみ使用されるため最小 API で宣言する。
    """

    @property
    def count(self) -> int:
        """エントリ総数を返す。"""
        ...

    def append(self, exp: Any) -> None:
        """経験エントリを追加する (同期)。"""
        ...

    def get_filtered(
        self, cartridge_ids: list[str] | None = None, mode: str | None = None,
    ) -> list[Any]:
        """カートリッジ ID / モードでフィルタ済のエントリリストを返す。"""
        ...


@runtime_checkable
class AssistVersionManagerProtocol(Protocol):
    """アシストモデル LoRA バージョン管理 (§7.5.3 / Pro 拡張)。

    Pro 版は ``backend.pro.version_manager.LoRAVersionManager`` を
    アシスト専用ディレクトリで初期化したインスタンスが自然に満たす。
    """

    def save_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        """LoRA スナップショットを保存する。"""
        ...

    def rollback(self, *args: Any, **kwargs: Any) -> Any:
        """直近または任意のバージョンへロールバックする。"""
        ...


@runtime_checkable
class AssistEvalManagerProtocol(Protocol):
    """アシストモデル評価マネージャ (§7.5.4) の抽象。

    Pro 版は ``backend.pro.eval_assist_manager.EvalAssistManager`` が
    自然に満たす。評価ベンチマークの load / save / get_score 等を
    ``LearningScheduler`` から呼び出す。
    """

    def load(self, *args: Any, **kwargs: Any) -> Any:
        """評価 set を読み込む。"""
        ...

    def save(self, *args: Any, **kwargs: Any) -> Any:
        """評価 set を保存する。"""
        ...


@runtime_checkable
class AssistPromptManagerProtocol(Protocol):
    """アシストモデルプロンプトマネージャ (§7.1.2) の抽象。

    Free 版・Pro 版共通の ``backend.free.agent.prompt_manager.AssistPromptManager``
    が本 Protocol を満たす想定。Free 版でも **インスタンス自体は存在**
    するが、AssistPromptEvolver が無効化されるため ``LearningScheduler``
    から見ると Protocol 経由で参照される必要はない (直接 ``None`` で可)。
    """

    def get_assist_prompt(self, task: str) -> str:
        """タスク別アシストプロンプトを返す。存在しなければ ``ValueError``。"""
        ...

    def update_assist_prompt(
        self,
        task: str,
        prompt: str,
        fitness: float,
    ) -> None:
        """タスク別アシストプロンプトを更新する。"""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Level 2 拡張 Protocol (Pro 専用)
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Level2RunnerProtocol(Protocol):
    """Level 2 SPSA LoRA 微調整ランナーの抽象。

    Free 版では実装なし (``None`` で注入)。Pro 版は
    ``backend.pro.learning.level2_trainer.Level2Runner`` が本 Protocol を
    自然に満たす。
    """

    def check_and_run(
        self,
        is_user_active: bool,
        **kwargs: Any,
    ) -> bool:
        """Level 2 トリガー判定と実行開始。学習を開始したら ``True``。"""
        ...


@runtime_checkable
class AssistPromptEvolverProtocol(Protocol):
    """アシストモデル用プロンプト進化 (§7.1.2) の抽象。

    Free 版では実装なし。Pro 版は
    ``backend.pro.assist_prompt_evolver.AssistPromptEvolver`` が実装する。
    """

    def evolve(self, *args: Any, **kwargs: Any) -> Any:
        """プロンプト進化 1 サイクルを実行する。"""
        ...


@runtime_checkable
class ExperienceEvaluatorProtocol(Protocol):
    """経験バッファ品質評価の抽象。

    Free 版では実装なし (全経験を通過)。Pro 版は Level 2 対象経験を
    フィルタリングする評価器を提供する。
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """経験 (失敗ケース等) を評価してフィルタリング結果を返す。"""
        ...


@runtime_checkable
class CartridgeChangeHandlerProtocol(Protocol):
    """カートリッジ変更ハンドラの抽象。

    Free 版では実装なし。Pro 版は
    ``backend.pro.learning.cartridge_change_handler.CartridgeChangeHandler``
    が実装し、カートリッジ編集を起点に Level 2 再学習を誘発する。
    """

    def on_cartridge_change(self, *args: Any, **kwargs: Any) -> Any:
        """カートリッジ変更イベントを受けて適切な後処理を行う。"""
        ...


# ──────────────────────────────────────────────────────────────────────────
# AssistComponentsProtocol (Level 1 サイクル + Level 2 拡張の集約)
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class AssistComponentsProtocol(Protocol):
    """Pro 拡張コンポーネント群を ``LearningScheduler`` に DI するための契約。

    vm, eval_mgr, prompt_mgr, ext_llm, assist_llm)`` の位置引数 7 本を
    本 Protocol インスタンス 1 本に集約した。``wire_pillars`` / ``app_factory``
    が edition を判定し、Free 版は :class:`NoopAssistComponents`、Pro 版
    は ``backend.pro.assist_components.ProAssistComponents`` を渡す。

    各コンポーネントはサブ Protocol で型付けされるため、Free 側の本ファイル
    から Pro 具象クラスへの import は不要 (pillar 境界を維持)。
    """

    # ── 運用コンポーネント ────────────────────────────────────────────

    @property
    def experience_buffer(self) -> AssistExperienceBufferProtocol | None:
        """Pro 版で AssistExperienceBuffer を返す。Free 版は常に ``None``。"""
        ...

    @property
    def version_manager(self) -> AssistVersionManagerProtocol | None:
        """Pro 版でアシスト用 LoRAVersionManager を返す。Free 版は ``None``。"""
        ...

    @property
    def eval_manager(self) -> AssistEvalManagerProtocol | None:
        """Pro 版で EvalAssistManager を返す。Free 版は ``None``。"""
        ...

    @property
    def prompt_manager(self) -> AssistPromptManagerProtocol | None:
        """Free / Pro 共通の AssistPromptManager を返す。未初期化時は ``None``。"""
        ...

    @property
    def assist_client(self) -> AssistModelClientProtocol | None:
        """アシストモデル用 LLM クライアント (全エディションで local llama-server)。"""
        ...

    # ── Level 2 拡張 (Pro 専用) ───────────────────────────────────────

    @property
    def level2_runner(self) -> Level2RunnerProtocol | None:
        """Pro 版で Level 2 ランナーを返す。Free 版は常に ``None``。"""
        ...

    @property
    def assist_prompt_evolver(self) -> AssistPromptEvolverProtocol | None:
        """Pro 版で AssistPromptEvolver を返す。Free 版は常に ``None``。"""
        ...

    @property
    def experience_evaluator(self) -> ExperienceEvaluatorProtocol | None:
        """Pro 版で ExperienceEvaluator を返す。Free 版は常に ``None``。"""
        ...

    @property
    def cartridge_change_handler(self) -> CartridgeChangeHandlerProtocol | None:
        """Pro 版で CartridgeChangeHandler を返す。Free 版は常に ``None``。"""
        ...


# ──────────────────────────────────────────────────────────────────────────
# NoopAssistComponents (Free 版の既定実装)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoopAssistComponents:
    """Free 版向けの :class:`AssistComponentsProtocol` 既定実装。

    Pro 拡張が存在しない環境で ``LearningScheduler.set_assist_components``
    に渡すと、全 Property が ``None`` を返して no-op となる
    ``app_factory`` は edition を判定し Pro が無効なら本クラスを生成する。

    :class:`AssistPromptManagerProtocol` のみは Free でも実体が存在しうる
    ため、呼び出し側が必要に応じて ``prompt_manager=...`` を渡せるよう
    dataclass の field として許可する (他は ``None`` 固定)。
    """

    prompt_manager: AssistPromptManagerProtocol | None = None

    @property
    def experience_buffer(self) -> AssistExperienceBufferProtocol | None:
        return None

    @property
    def version_manager(self) -> AssistVersionManagerProtocol | None:
        return None

    @property
    def eval_manager(self) -> AssistEvalManagerProtocol | None:
        return None

    @property
    def assist_client(self) -> AssistModelClientProtocol | None:
        return None

    @property
    def level2_runner(self) -> Level2RunnerProtocol | None:
        return None

    @property
    def assist_prompt_evolver(self) -> AssistPromptEvolverProtocol | None:
        return None

    @property
    def experience_evaluator(self) -> ExperienceEvaluatorProtocol | None:
        return None

    @property
    def cartridge_change_handler(self) -> CartridgeChangeHandlerProtocol | None:
        return None


__all__ = [
    "AssistComponentsProtocol",
    "AssistEvalManagerProtocol",
    "AssistExperienceBufferProtocol",
    "AssistPromptEvolverProtocol",
    "AssistPromptManagerProtocol",
    "AssistVersionManagerProtocol",
    "CartridgeChangeHandlerProtocol",
    "ExperienceEvaluatorProtocol",
    "Level2RunnerProtocol",
    "NoopAssistComponents",
]
