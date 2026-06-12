"""エンベッド検索指示プロンプトの進化

検索クエリに付加する instruction プロンプトを Darwinian Evolver で進化させる。
PromptEvolver を継承し、fitness を rag_top1_score ベースで計算する。
supports_instructions() が True のバックエンド（例: Qwen3-Embedding）でのみ有効。

保護セクション（<!-- PROTECTED --> マーカー）は PromptEvolver 基底クラスで
自動的に保護される。
"""

from __future__ import annotations

from backend.free.optimizer.prompt_evolver import (
    EvolutionResult,
    PromptCandidate,
    PromptEvolver,
)
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

# 1 進化ランで実測評価に使うクエリ数の上限 (embed 呼出コスト抑制)。
MAX_EVAL_QUERIES = 8


class EmbedInstructionEvolver(PromptEvolver):
    """エンベッド検索指示プロンプトの進化

    ``embed_eval`` が注入されている場合は、候補 instruction で記録済みクエリを
    実際に再埋め込み・再検索して top1 cosine を実測した値を fitness に使う
    (候補テキストが fitness に反映され進化が機能する)。未注入時は記録済み
    rag_top1_score の平均ベース (``_calc_fitness``) へ degrade する。
    """

    def __init__(self, embed_eval=None) -> None:
        """
        Args:
            embed_eval: ``EmbedEvalProtocol`` 実装 (候補 instruction の実測評価器)。
                ``None`` なら記録済み rag_top1_score 平均へ degrade。
        """
        self._embed_eval = embed_eval
        # 1 ラン内の候補 text → fitness メモ (同一候補の再 embed を避ける)。
        self._fitness_memo: dict[str, float] = {}
        # 1 ラン内の実測評価に使うサンプリング済みクエリ。
        self._eval_queries: list[str] = []

    def set_embed_eval(self, embed_eval) -> None:
        """実測評価器を後注入する (wire_pillars からの後注入用)。"""
        self._embed_eval = embed_eval

    async def _score_candidate(
        self,
        candidate: str,
        experiences: list[dict],
    ) -> float:
        """候補 instruction の実測 fitness を返す (embed_eval 未注入時は degrade)。"""
        if self._embed_eval is None:
            return self._calc_fitness(candidate, experiences)
        if candidate in self._fitness_memo:
            return self._fitness_memo[candidate]

        measured = await self._embed_eval.score_candidate(candidate, self._eval_queries)
        if measured is None:
            # 実測不能 → 従来のシグナルベース fitness へ degrade
            fitness = self._calc_fitness(candidate, experiences)
        else:
            # 候補テキスト依存の微調整 (長さペナルティ) は実測値にも適用する
            prompt_len = len(candidate)
            bonus = 0.0
            if prompt_len < 10:
                bonus -= 0.02
            elif prompt_len > 500:
                bonus -= 0.01
            fitness = max(0.0, min(1.0, measured + bonus))
        self._fitness_memo[candidate] = fitness
        return fitness

    @staticmethod
    def _sample_eval_queries(experiences: list[dict]) -> list[str]:
        """実測評価に使うクエリを失敗寄り (低 top1) 優先で上位 N 抽出する。

        候補 instruction の弁別力が高いのは現状スコアが低いクエリなので、
        ``rag_top1_score`` 昇順でサンプリングする。
        """
        ranked = sorted(
            (
                (e.get("query", ""), e.get("signals", {}).get("rag_top1_score", 1.0))
                for e in experiences
            ),
            key=lambda t: t[1] if t[1] is not None else 1.0,
        )
        return [q for q, _ in ranked if q][:MAX_EVAL_QUERIES]

    def _calc_fitness(
        self,
        candidate: str,
        experiences: list[dict],
    ) -> float:
        """rag_top1_score 平均ベースの適応度計算 (embed_eval degrade 時のフォールバック)

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
                best_candidate=PromptCandidate(text=current_instruction),
                generations_run=0,
                initial_fitness=0.0,
                final_fitness=0.0,
            )

        # 実測評価用: ラン単位でメモをクリアし、失敗寄りクエリをサンプリング。
        self._fitness_memo.clear()
        self._eval_queries = self._sample_eval_queries(rag_exp)

        logger.info(
            "Evolving embed instruction (%d RAG experiences, %d generations, "
            "real_eval=%s, eval_queries=%d)",
            len(rag_exp), generations,
            self._embed_eval is not None, len(self._eval_queries),
        )

        return await self._darwinian_evolve(
            current=current_instruction,
            experiences=rag_exp,
            llm_client=mutator_client,
            generations=generations,
            population_size=population_size,
            seed=seed,
        )
