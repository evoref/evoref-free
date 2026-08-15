"""EvorefLearn の Pro 拡張ポイント Protocol

EvorefLearn (learning / optimizer pillar) の Pro 拡張ポイントを Protocol
として固定する。Pro 版 (Free 版にない高度学習機能) は
``backend/pro/`` 配下で実装され、``wire_pillars`` /
``app_factory`` は edition を判定して以下いずれかを ``LearningScheduler``
に DI する:

- Free: :class:`NoopLearnComponents` (全 ``None`` を返すスタブ)
- Pro:  :class:`~backend.pro.learn_components.ProLearnComponents`
  (Level2Runner / AuxPromptEvolver / LoRAVersionManager / ...)

pillar 境界の不変条件により、本 Protocol ファイルは **Pro 具象クラスを
import してはならない** (Free → Pro 参照は依存方向違反)。従って、各
コンポーネントも **サブ Protocol として宣言** し、具象実装 (Free / Pro 側)
が duck typing でサブ Protocol を満たすよう設計する。

設計原則 (CLAUDE.md §8 / ``docs/p_01_advanced_learning.md`` §0.1):
- 最小 API 原則: 現在 ``LearningScheduler`` が Pro コンポーネントに対し
  呼び出しているメソッドだけを宣言
- ``@runtime_checkable``: isinstance チェックを可能にする
- Free / Pro 双方の実装を差し込み可能にする

## LearnComponents 集約

Pro 拡張は :class:`LearnComponentsProtocol` インスタンス 1 本に集約して
注入する。Protocol には以下の 2 系統のフィールドが含まれる:

1. **運用コンポーネント** (Level 1 サイクルで使う)
   - ``prompt_manager`` / ``version_manager``
2. **Level 2 拡張** (Pro 専用の LoRA 微調整 / プロンプト進化 / カートリッジ変更)
   - ``level2_runner`` / ``aux_prompt_evolver`` / ``cartridge_change_handler``

Free 版は全部 ``None`` を返す :class:`NoopLearnComponents` を渡し、Pro 版
は ``backend.pro.learn_components.ProLearnComponents`` を渡す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ──────────────────────────────────────────────────────────────────────────
# 運用コンポーネント Protocol
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class LoRAVersionManagerProtocol(Protocol):
    """LoRA バージョン管理 (§7.5.3 / Pro 拡張)。

    Pro 版は ``backend.pro.version_manager.LoRAVersionManager`` が
    自然に満たす。
    """

    def save_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        """LoRA スナップショットを保存する。"""
        ...

    def rollback(self, *args: Any, **kwargs: Any) -> Any:
        """直近または任意のバージョンへロールバックする。"""
        ...


@runtime_checkable
class AuxPromptManagerProtocol(Protocol):
    """補助タスクプロンプトマネージャ (§7.1.2) の抽象。

    Free 版・Pro 版共通の ``backend.free.agent.aux_prompt_manager.AuxPromptManager``
    が本 Protocol を満たす。Free 版でも **インスタンス自体は存在** するが、
    ``AuxPromptEvolver`` が無効化されるため ``LearningScheduler`` から見ると
    Protocol 経由で参照される必要はない (直接 ``None`` で可)。
    """

    def get_aux_prompt(self, task: str) -> str:
        """タスク別プロンプトを返す。存在しなければ ``ValueError``。"""
        ...

    def update_aux_prompt(
        self,
        task: str,
        prompt: str,
        fitness: float,
    ) -> None:
        """タスク別プロンプトを更新する。"""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Level 2 拡張 Protocol (Pro 専用)
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Level2RunnerProtocol(Protocol):
    """Level 2 SPSA / cvector LoRA 微調整ランナーの抽象。

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
class AuxPromptEvolverProtocol(Protocol):
    """補助タスクプロンプト進化 (§7.1.2) の抽象。

    Free 版では実装なし。Pro 版は
    ``backend.pro.aux_prompt_evolver.AuxPromptEvolver`` が実装する。
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
# LearnComponentsProtocol (Level 1 サイクル + Level 2 拡張の集約)
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class LearnComponentsProtocol(Protocol):
    """Pro 拡張コンポーネント群を ``LearningScheduler`` に DI するための契約。

    ``wire_pillars`` / ``app_factory`` が edition を判定し、Free 版は
    :class:`NoopLearnComponents`、Pro 版は
    ``backend.pro.learn_components.ProLearnComponents`` を渡す。

    各コンポーネントはサブ Protocol で型付けされるため、Free 側の本ファイル
    から Pro 具象クラスへの import は不要 (pillar 境界を維持)。
    """

    # ── 運用コンポーネント ────────────────────────────────────────────

    @property
    def version_manager(self) -> LoRAVersionManagerProtocol | None:
        """Pro 版でベース LoRA の VersionManager を返す。Free 版は ``None``。"""
        ...

    @property
    def prompt_manager(self) -> AuxPromptManagerProtocol | None:
        """Free / Pro 共通の AuxPromptManager を返す。未初期化時は ``None``。"""
        ...

    # ── Level 2 拡張 (Pro 専用) ───────────────────────────────────────

    @property
    def level2_runner(self) -> Level2RunnerProtocol | None:
        """Pro 版で Level 2 ランナーを返す。Free 版は常に ``None``。"""
        ...

    @property
    def aux_prompt_evolver(self) -> AuxPromptEvolverProtocol | None:
        """Pro 版で AuxPromptEvolver を返す。Free 版は常に ``None``。"""
        ...

    @property
    def cartridge_change_handler(self) -> CartridgeChangeHandlerProtocol | None:
        """Pro 版で CartridgeChangeHandler を返す。Free 版は常に ``None``。"""
        ...


# ──────────────────────────────────────────────────────────────────────────
# NoopLearnComponents (Free 版の既定実装)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoopLearnComponents:
    """Free 版向けの :class:`LearnComponentsProtocol` 既定実装。

    Pro 拡張が存在しない環境で ``LearningScheduler.set_learn_components``
    に渡すと、全 Property が ``None`` を返して no-op となる。
    ``app_factory`` は edition を判定し Pro が無効なら本クラスを生成する。

    :class:`AuxPromptManagerProtocol` のみは Free でも実体が存在しうる
    ため、呼び出し側が必要に応じて ``prompt_manager=...`` を渡せるよう
    dataclass の field として許可する (他は ``None`` 固定)。
    """

    prompt_manager: AuxPromptManagerProtocol | None = None

    @property
    def version_manager(self) -> LoRAVersionManagerProtocol | None:
        return None

    @property
    def level2_runner(self) -> Level2RunnerProtocol | None:
        return None

    @property
    def aux_prompt_evolver(self) -> AuxPromptEvolverProtocol | None:
        return None

    @property
    def cartridge_change_handler(self) -> CartridgeChangeHandlerProtocol | None:
        return None


__all__ = [
    "AuxPromptEvolverProtocol",
    "AuxPromptManagerProtocol",
    "CartridgeChangeHandlerProtocol",
    "ExperienceEvaluatorProtocol",
    "LearnComponentsProtocol",
    "Level2RunnerProtocol",
    "LoRAVersionManagerProtocol",
    "NoopLearnComponents",
]
