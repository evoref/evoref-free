"""エンベッド検索指示プロンプトの進化

検索クエリに付加する instruction プロンプトを Darwinian Evolver で進化させる。
PromptEvolver を継承し、fitness を rag_top1_score ベースで計算する。
supports_instructions() が True のバックエンド（例: Qwen3-Embedding）でのみ有効。

保護セクション（<!-- PROTECTED --> マーカー）は PromptEvolver 基底クラスで
自動的に保護される。
"""

from __future__ import annotations

from backend.free.optimizer.prompt_evolver import EvolutionResult, PromptEvolver
from backend.log_config import get_logger

logger = get_logger("optimizer.embed_instruction_evolver")

# デフォルトの検索指示プロンプト
DEFAULT_EMBED_INSTRUCTION = """\
与えられたクエリに関連するドキュメントを検索してください。

<!-- PROTECTED -->
検索対象: ユーザーが保存したドキュメント、ノート、コード
<!-- /PROTECTED -->
"""

# 採用閾値（ベースモデルと同じ）
ADOPTION_THRESHOLD = 0.65


class EmbedInstructionEvolver(PromptEvolver):
    """エンベッド検索指示プロンプトの進化

    fitness は rag_top1_score（検索結果の上位1件のスコア）を使用する。
    リランカー未使用時のスコアを優先し、検索精度の改善を直接計測する。
    """

    def _calc_fitness(
        self,
        candidate: str,
        experiences: list[dict],
    ) -> float:
        """rag_top1_score ベースの適応度計算

        Args:
            candidate: プロンプト候補テキスト
            experiences: rag_top1_score を含む経験バッファエントリ

        Returns:
            適応度スコア (0.0〜1.0)
        """
        if not experiences:
            return 0.0

        scores: list[float] = []

        for exp in experiences:
            signals = exp.get("signals", {})
            top1 = signals.get("rag_top1_score")
            if top1 is None:
                continue
            scores.append(top1)

        if not scores:
            return 0.0

        base_score = sum(scores) / len(scores)

        # プロンプト候補の特徴による微調整
        prompt_len = len(candidate)
        bonus = 0.0
        if prompt_len < 10:
            bonus -= 0.02
        elif prompt_len > 500:
            bonus -= 0.01

        return max(0.0, min(1.0, base_score + bonus))

    def _extract_failure_hints(self, experiences: list[dict]) -> list[str]:
        """低スコアの検索結果からヒントを抽出"""
        hints = []
        for exp in experiences:
            signals = exp.get("signals", {})
            top1 = signals.get("rag_top1_score")
            if top1 is not None and top1 < 0.3:
                query = exp.get("query", "")[:80]
                hints.append(
                    f"Low search score ({top1:.2f}) for query: \"{query}\""
                )
        return hints[:5]

    async def evolve_instruction(
        self,
        experiences: list[dict],
        current_instruction: str,
        mutator_client,
        generations: int = 10,
        population_size: int = 5,
        seed: int | None = None,
    ) -> EvolutionResult:
        """検索指示プロンプトを進化させる

        Args:
            experiences: rag_top1_score を含む経験バッファ
            current_instruction: 現在の検索指示プロンプト
            mutator_client: 変異生成に使用する LLM
            generations: 世代数
            population_size: 集団サイズ
            seed: 乱数シード

        Returns:
            進化結果
        """
        # rag_top1_score がある経験のみ使用
        rag_exp = [
            e for e in experiences
            if e.get("signals", {}).get("rag_top1_score") is not None
        ]

        if not rag_exp:
            logger.info("No RAG experiences with top1_score, skipping evolution")
            return EvolutionResult(
                best_candidate=__import__(
                    "backend.free.optimizer.prompt_evolver",
                    fromlist=["PromptCandidate"],
                ).PromptCandidate(text=current_instruction),
                generations_run=0,
                initial_fitness=0.0,
                final_fitness=0.0,
            )

        logger.info(
            "Evolving embed instruction (%d RAG experiences, %d generations)",
            len(rag_exp), generations,
        )

        return await self._darwinian_evolve(
            current=current_instruction,
            experiences=rag_exp,
            llm_client=mutator_client,
            generations=generations,
            population_size=population_size,
            seed=seed,
        )
