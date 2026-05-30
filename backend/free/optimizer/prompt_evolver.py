"""Darwinian 進化アルゴリズムによるシステムプロンプト最適化"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# 保護セクション検証・重複排除は agent.prompt_utils (EvorefLoop
# pillar の純粋 util) に集約済。Learn pillar 側はここから top-level import する。
from backend.free.agent.prompt_utils import (
    dedupe_paragraphs,
    extract_protected_sections,
    format_fewshot_section,
    restore_protected_sections,
    strip_orphan_protected_markers,
    text_contains_sentence,
    validate_protected_sections,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.learning.critique_synthesizer import CritiqueSynthesizer
    from backend.free.learning.fewshot_pool import FewShotPool
    from backend.free.learning.level1_session import Level1Session

logger = get_logger("optimizer.prompt_evolver")

# 突然変異リトライ上限
MAX_MUTATION_RETRIES = 3

# ── サーキットブレーカー閾値 ──
# 連続失敗回数がこの閾値以上で早期終了
CB_MAX_CONSECUTIVE_FAILURES: int = 3
# 失敗率がこの閾値を超えたら早期終了（最低 CB_MIN_ATTEMPTS 回試行後）
CB_FAILURE_RATE_THRESHOLD: float = 0.5
CB_MIN_ATTEMPTS: int = 4


@dataclass
class _CircuitBreaker:
    """進化ループ中の LLM 変異呼び出しサーキットブレーカー状態

    `_darwinian_evolve` から長大関数の状態管理を切り出すための内部用 dataclass。
    既存のロジック (連続失敗 / 失敗率閾値) は変えず、状態遷移をメソッドに集約する。
    """

    consecutive: int = 0
    total_failures: int = 0
    total_attempts: int = 0
    tripped: bool = False

    def record_attempt(self) -> None:
        """変異試行 1 回をカウントする (成否問わず先に呼ぶ)"""
        self.total_attempts += 1

    def record_failure(self) -> None:
        """変異が失敗した場合の状態遷移"""
        self.consecutive += 1
        self.total_failures += 1

    def record_success(self) -> None:
        """変異が成功した場合の連続失敗カウンタリセット"""
        self.consecutive = 0

    def should_stop(self) -> bool:
        """サーキットブレーカー判定。True なら早期終了すべき"""
        if self.consecutive >= CB_MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Circuit breaker (consecutive): stopping after %d "
                "consecutive mutation failures",
                self.consecutive,
            )
            return True
        if (
            self.total_attempts >= CB_MIN_ATTEMPTS
            and self.total_failures / self.total_attempts > CB_FAILURE_RATE_THRESHOLD
        ):
            logger.warning(
                "Circuit breaker (failure rate): stopping at %.0f%% "
                "failure rate (%d/%d attempts)",
                self.total_failures / self.total_attempts * 100,
                self.total_failures, self.total_attempts,
            )
            return True
        return False


@dataclass
class PromptCandidate:
    """プロンプト候補"""
    text: str
    fitness: float = 0.0
    generation: int = 0
    fewshot_ids: list[str] = field(default_factory=list)


@dataclass
class EvolutionResult:
    """進化結果"""
    best_candidate: PromptCandidate
    all_candidates: list[PromptCandidate] = field(default_factory=list)
    generations_run: int = 0
    initial_fitness: float = 0.0
    final_fitness: float = 0.0
    yielded: bool = False  # 協調 yield により途中終了したか


class PromptEvolver:
    """Darwinian 進化によるモード別システムプロンプト自動最適化

    Level 1 学習サイクルで使用。LLM によるプロンプト変異と
    経験バッファに基づく適応度計算でプロンプトを段階的に改善する。
    """

    def _calc_fitness(
        self,
        candidate: str,
        experiences: list[dict],
    ) -> float:
        """経験バッファに基づく適応度計算

        成功率ベースのスコアリング:
        - conversation_ended=True → 成功（ユーザーが満足して離脱）
        - rephrased_query=True → 失敗（ユーザーが言い直した）
        - user_correction is not None → 失敗（ユーザーが訂正した）

        候補テキストの特徴（長さ、キーワード含有）も考慮し、
        同一経験でも異なるプロンプトが異なるスコアを持つようにする。

        Args:
            candidate: プロンプト候補テキスト
            experiences: 経験バッファのエントリリスト

        Returns:
            適応度スコア (0.0〜1.0)
        """
        if not experiences:
            return 0.5  # デフォルト中立スコア

        total_weight = 0.0
        score = 0.0

        for exp in experiences:
            signals = exp.get("signals", {})
            weight = 1.0

            # 成功シグナル
            if signals.get("conversation_ended", False):
                score += weight * 1.0

            # 失敗シグナル
            if signals.get("rephrased_query", False):
                score -= weight * 0.5

            if signals.get("user_correction") is not None:
                score -= weight * 0.8

            # RAG 使用はニュートラル
            total_weight += weight

        if total_weight == 0:
            return 0.5

        base_score = (score / total_weight + 1) / 2

        # プロンプト候補の特徴によるボーナス/ペナルティ。
        # 1 回の進化ランでは experiences が固定なので base_score は全候補で同一に
        # なる。候補テキスト由来の差分をここで与えないと全候補が同 fitness に潰れ、
        # _run_one_generation の strict > 置換が一度も発火せず進化が no-op 化する
        # (initial==final が 10 世代続く)。base_score を支配項に保ちつつ、候補間を
        # 識別できる有界な重みを与えて選択圧を回復する (重みは base_score を覆さない)。
        candidate_lower = candidate.lower()
        bonus = 0.0

        # 失敗した経験のクエリ主要語を、その候補がどれだけカバーしているか
        # (カバー率)。候補ごとに値が変わるため候補間識別の主因。最大 +0.1。
        failure_keywords: set[str] = set()
        for exp in experiences:
            signals = exp.get("signals", {})
            if signals.get("rephrased_query") or signals.get("user_correction"):
                query = str(exp.get("query", "")).lower()
                failure_keywords.update(w for w in query.split() if len(w) >= 3)
        if failure_keywords:
            covered = sum(1 for w in failure_keywords if w in candidate_lower)
            bonus += 0.1 * (covered / len(failure_keywords))

        # プロンプト長の適切さ（短すぎず長すぎない）
        prompt_len = len(candidate)
        if prompt_len < 50:
            bonus -= 0.05
        elif prompt_len > 2000:
            bonus -= 0.02

        raw = base_score + bonus
        return max(0.0, min(1.0, raw))

    async def _mutate_prompt(
        self,
        current: str,
        failure_hints: list[str],
        llm_client,
    ) -> str | None:
        """LLM によるプロンプト変異

        失敗パターンを分析してプロンプトの改善点を生成する。

        Args:
            current: 現在のプロンプトテキスト
            failure_hints: 失敗ケースのヒント（ユーザーの訂正・言い直し等）
            llm_client: LLMClient / LocalClient インスタンス

        Returns:
            変異後のプロンプトテキスト。LLM エラー時は None
        """
        hints_text = "\n".join(f"- {h}" for h in failure_hints[:5])

        # 保護セクションがある場合は LLM にも指示
        has_protected = bool(extract_protected_sections(current))
        protected_instruction = (
            "\n\nIMPORTANT: Sections enclosed in <!-- PROTECTED --> ... <!-- /PROTECTED --> "
            "markers MUST be preserved exactly as-is. Do not modify, remove, or rewrite "
            "their content or the markers themselves."
        ) if has_protected else ""

        mutation_query = (
            "You are a prompt engineer. Improve the following system prompt "
            "based on the failure patterns. Make meaningful changes - "
            "add specific instructions, improve clarity, or restructure "
            "to better handle the failure cases.\n\n"
            f"Current system prompt:\n```\n{current}\n```\n\n"
            f"Failure patterns:\n{hints_text}\n\n"
            "Output ONLY the improved system prompt, nothing else. "
            "The improved prompt MUST be different from the current one."
            f"{protected_instruction}"
        )

        try:
            generate_kwargs: dict = {
                "messages": [{"role": "user", "content": mutation_query}],
                "stream": False,
                "temperature": 0.5,
                "max_tokens": 1024,
                "id_slot": getattr(llm_client, 'background_slot', -1),
            }
            result = await llm_client.generate(**generate_kwargs)
            mutated = result["choices"][0]["message"]["content"].strip()

            # コードブロックで囲まれている場合は除去
            if mutated.startswith("```"):
                lines = mutated.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                mutated = "\n".join(lines).strip()

            return mutated if mutated else None

        except Exception as e:
            logger.warning("Prompt mutation failed: %s: %s", type(e).__name__, e)
            return None

    def _create_rule_based_variation(self, current: str, failure_hints: list[str]) -> str:
        """LLM なしのルールベース変異（フォールバック）

        LLM が同一テキストを返した場合に多様性を確保するためのフォールバック。

        既に current に含まれているルールは追加候補から除外し
        全ルールが既出の場合は current をそのまま返す（同一文の無限累積を防ぐ）。
        """
        additions = [
            "When the user's intent is unclear, ask a clarifying question before responding.",
            "Provide concise, direct answers. Avoid unnecessary preamble.",
            "If the user rephrases their question, acknowledge the clarification and adjust your response.",
            "Break down complex topics into simple, digestible steps.",
            "Always consider the context of previous messages in the conversation.",
        ]

        # 失敗ヒントからキーワードを抽出して対応ルールを追加
        hint_additions = []
        for hint in failure_hints[:3]:
            if "rephrased" in hint.lower():
                hint_additions.append(
                    "Pay close attention to the user's exact wording and respond precisely to their question."
                )
            elif "corrected" in hint.lower():
                hint_additions.append(
                    "Double-check your responses for accuracy before presenting them."
                )

        # 既に current に含まれているルールは除外（重複追加防止）
        all_additions = hint_additions + additions
        novel_additions = [
            rule for rule in all_additions
            if not text_contains_sentence(current, rule)
        ]
        if not novel_additions:
            logger.debug(
                "Rule-based variation: all candidate rules already present in prompt, "
                "returning current unchanged",
            )
            return current

        # ランダムに 1-2 個のルールを追加
        rng = np.random.default_rng()
        n_add = int(rng.integers(1, min(3, len(novel_additions)) + 1))
        selected = rng.choice(len(novel_additions), size=n_add, replace=False)

        extra_rules = "\n".join(novel_additions[i] for i in selected)
        return f"{current}\n\n{extra_rules}"

    def _crossover(self, parent1: str, parent2: str) -> str:
        """2親からの交叉

        段落単位で交互に選択する。

        Args:
            parent1: 第1親プロンプト
            parent2: 第2親プロンプト

        Returns:
            交叉結果のプロンプト
        """
        paragraphs1 = [p.strip() for p in parent1.split("\n\n") if p.strip()]
        paragraphs2 = [p.strip() for p in parent2.split("\n\n") if p.strip()]

        if not paragraphs1:
            return parent2
        if not paragraphs2:
            return parent1

        result = []
        max_len = max(len(paragraphs1), len(paragraphs2))

        for i in range(max_len):
            if i % 2 == 0:
                if i < len(paragraphs1):
                    result.append(paragraphs1[i])
                elif i < len(paragraphs2):
                    result.append(paragraphs2[i])
            else:
                if i < len(paragraphs2):
                    result.append(paragraphs2[i])
                elif i < len(paragraphs1):
                    result.append(paragraphs1[i])

        # 同じ段落が両親由来で重複する場合があるため正規化重複除去
        return dedupe_paragraphs("\n\n".join(result))

    def _select(
        self,
        population: list[PromptCandidate],
        n: int,
        seed: int | None = None,
    ) -> list[PromptCandidate]:
        """トーナメント選択

        トーナメントサイズ 3 で n 個の親を選択する。

        Args:
            population: 集団
            n: 選択数
            seed: 乱数シード

        Returns:
            選択された候補リスト
        """
        rng = np.random.default_rng(seed)
        selected = []
        tournament_size = min(3, len(population))

        for _ in range(n):
            indices = rng.choice(len(population), size=tournament_size, replace=False)
            tournament = [population[i] for i in indices]
            winner = max(tournament, key=lambda c: c.fitness)
            selected.append(winner)

        return selected

    def _extract_failure_hints(self, experiences: list[dict]) -> list[str]:
        """経験バッファから失敗ヒントを抽出する"""
        hints = []
        for exp in experiences:
            signals = exp.get("signals", {})

            if signals.get("user_correction"):
                hints.append(
                    f"User corrected: query=\"{exp.get('query', '')[:80]}\" "
                    f"correction=\"{signals['user_correction'][:100]}\""
                )
            elif signals.get("rephrased_query"):
                hints.append(
                    f"User rephrased: query=\"{exp.get('query', '')[:80]}\""
                )

        return hints

    def _is_text_similar(self, text1: str, text2: str) -> bool:
        """2つのテキストが実質的に同一かどうかを判定"""
        # 正規化して比較
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()

        if t1 == t2:
            return True

        # 文字レベルの類似度（短いテキストの90%以上一致なら同一扱い）
        shorter = min(len(t1), len(t2))
        if shorter == 0:
            return True

        # 共通プレフィックス長で簡易判定
        common = 0
        for c1, c2 in zip(t1, t2):
            if c1 == c2:
                common += 1
            else:
                break

        return common / shorter > 0.9

    def _calc_fitness_with_fewshot(
        self,
        instruction: str,
        fewshot_ids: list[str],
        experiences: list[dict],
        fewshot_pool: FewShotPool | None,
        mode: str,
    ) -> float:
        """指示テキスト + Few-shot 例の結合テキストで fitness を計算する"""
        if fewshot_pool is None or not fewshot_ids:
            return self._calc_fitness(instruction, experiences)

        examples = fewshot_pool.get_by_ids(fewshot_ids, mode)
        combined = instruction + format_fewshot_section(examples)
        return self._calc_fitness(combined, experiences)

    async def _prepare_failure_hints(
        self,
        experiences: list[dict],
        critique_synthesizer: CritiqueSynthesizer | None,
    ) -> list[str]:
        """失敗ヒントを抽出し、批評フェーズ の改善ヒントを前置する"""
        failure_hints = self._extract_failure_hints(experiences)
        if critique_synthesizer is None:
            return failure_hints
        try:
            critique_result = await critique_synthesizer.critique(experiences)
            if critique_result.improvement_hints:
                # 批評ヒントを先頭に配置（変異プロンプトで優先的に使用される）
                failure_hints = critique_result.improvement_hints + failure_hints
                logger.info(
                    "Critique phase: %d hints added (source=%s)",
                    len(critique_result.improvement_hints),
                    critique_result.source,
                )
        except Exception as e:
            logger.warning("Critique phase failed, using raw hints: %s", e)
        return failure_hints

    def _init_fewshot_selection(
        self,
        fewshot_pool: FewShotPool | None,
        mode: str,
        seed: int | None,
    ) -> tuple[list[str], bool]:
        """Few-shot プール の初期選択。`(initial_ids, has_fewshot)` を返す"""
        has_fewshot = fewshot_pool is not None and fewshot_pool.count(mode) > 0
        if not has_fewshot:
            return [], False
        initial_selection = fewshot_pool.select(mode, seed=seed)
        initial_fewshot_ids = [ex.id for ex in initial_selection]
        logger.info(
            "Fewshot co-evolution enabled: %d candidates in pool, %d initially selected",
            fewshot_pool.count(mode),
            len(initial_fewshot_ids),
        )
        return initial_fewshot_ids, True

    async def _try_initial_mutation(
        self,
        current: str,
        failure_hints: list[str],
        llm_client,
        cb: _CircuitBreaker,
    ) -> tuple[str, bool]:
        """初期集団用に LLM 変異を最大 `MAX_MUTATION_RETRIES` 回試行する。

        多様性確保のため `_is_text_similar` を満たす変異が得られるまで再試行し、
        全試行が同一テキストだった場合はルールベース変異にフォールバックする。

        Returns:
            `(mutated_text, mutation_succeeded)` のタプル。
            `mutation_succeeded` は LLM が一度でも非 None を返したかを示す。
        """
        mutated_text = current
        mutation_succeeded = False
        for _retry in range(MAX_MUTATION_RETRIES):
            result = await self._mutate_prompt(
                current, failure_hints, llm_client,
            )
            cb.record_attempt()
            if result is None:
                cb.record_failure()
                if cb.should_stop():
                    break
                continue
            cb.record_success()
            mutation_succeeded = True
            mutated_text = result
            if not self._is_text_similar(mutated_text, current):
                break
        else:
            if mutation_succeeded:
                # LLM が同一テキストを返し続けた場合、ルールベースで多様性確保
                mutated_text = self._create_rule_based_variation(current, failure_hints)
                logger.debug(
                    "LLM mutation produced identical text %d times, "
                    "using rule-based variation",
                    MAX_MUTATION_RETRIES,
                )
        return mutated_text, mutation_succeeded

    async def _build_initial_population(
        self,
        *,
        current: str,
        failure_hints: list[str],
        llm_client,
        population_size: int,
        has_protected: bool,
        has_fewshot: bool,
        initial_fewshot_ids: list[str],
        fewshot_pool: FewShotPool | None,
        mode: str,
        experiences: list[dict],
        rng: np.random.Generator,
        cb: _CircuitBreaker,
    ) -> tuple[list[PromptCandidate], float]:
        """現在プロンプト + LLM 変異で初期集団を構築する。

        `(population, initial_fitness)` を返す。サーキットブレーカー発火時は
        `cb.tripped = True` をセットして部分集団を返す。
        """
        population = [PromptCandidate(
            text=current,
            generation=0,
            fewshot_ids=list(initial_fewshot_ids),
        )]
        initial_fitness = self._calc_fitness_with_fewshot(
            current, initial_fewshot_ids, experiences, fewshot_pool, mode,
        )
        population[0].fitness = initial_fitness

        for _ in range(population_size - 1):
            if cb.should_stop():
                cb.tripped = True
                break

            mutated_text, mutation_succeeded = await self._try_initial_mutation(
                current, failure_hints, llm_client, cb,
            )

            if cb.tripped or cb.should_stop():
                cb.tripped = True
                break

            # LLM 失敗時はルールベースフォールバック
            if not mutation_succeeded:
                mutated_text = self._create_rule_based_variation(current, failure_hints)

            # 段落レベル重複を正規化（LLM/ルールベース変異両方に適用）
            mutated_text = dedupe_paragraphs(mutated_text)
            # 孤児 PROTECTED マーカーを必ず除去
            mutated_text = strip_orphan_protected_markers(mutated_text)
            if has_protected:
                mutated_text = restore_protected_sections(current, mutated_text)

            # Few-shot 選択の変異
            cand_fewshot_ids = list(initial_fewshot_ids)
            if has_fewshot and fewshot_pool is not None:
                cand_fewshot_ids = fewshot_pool.mutate_selection(
                    cand_fewshot_ids, mode, seed=int(rng.integers(0, 2**31)),
                )

            candidate = PromptCandidate(
                text=mutated_text,
                generation=0,
                fewshot_ids=cand_fewshot_ids,
            )
            candidate.fitness = self._calc_fitness_with_fewshot(
                mutated_text, cand_fewshot_ids, experiences, fewshot_pool, mode,
            )
            population.append(candidate)

        return population, initial_fitness

    async def _run_one_generation(
        self,
        *,
        gen: int,
        current: str,
        population: list[PromptCandidate],
        failure_hints: list[str],
        llm_client,
        has_protected: bool,
        has_fewshot: bool,
        fewshot_pool: FewShotPool | None,
        mode: str,
        experiences: list[dict],
        seed: int | None,
        rng: np.random.Generator,
        cb: _CircuitBreaker,
    ) -> None:
        """1 世代分の親選択・交叉・変異・置換を実行する (population を in-place 更新)"""
        parents = self._select(population, n=2, seed=(seed + gen) if seed else None)
        child_text = self._crossover(parents[0].text, parents[1].text)

        # 変異（指示テキスト）
        if failure_hints:
            result = await self._mutate_prompt(
                child_text, failure_hints, llm_client,
            )
            cb.record_attempt()
            if result is None:
                cb.record_failure()
                # 変異失敗時は交叉結果をそのまま使用
            else:
                cb.record_success()
                child_text = result

        # 段落レベル重複を正規化
        child_text = dedupe_paragraphs(child_text)
        # 孤児 PROTECTED マーカーを必ず除去
        child_text = strip_orphan_protected_markers(child_text)
        if has_protected:
            child_text = restore_protected_sections(current, child_text)

        # Few-shot 交叉 + 変異
        child_fewshot_ids: list[str] = []
        if has_fewshot and fewshot_pool is not None:
            # 交叉: 両親の Few-shot を結合しランダムに max_examples 個選択
            merged = list(dict.fromkeys(
                parents[0].fewshot_ids + parents[1].fewshot_ids,
            ))
            max_ex = fewshot_pool.max_examples
            if len(merged) > max_ex:
                sel = rng.choice(len(merged), size=max_ex, replace=False)
                merged = [merged[i] for i in sel]
            # 変異: add/remove/swap
            child_fewshot_ids = fewshot_pool.mutate_selection(
                merged, mode, seed=int(rng.integers(0, 2**31)),
            )

        child = PromptCandidate(
            text=child_text,
            generation=gen,
            fewshot_ids=child_fewshot_ids,
        )
        child.fitness = self._calc_fitness_with_fewshot(
            child_text, child_fewshot_ids, experiences, fewshot_pool, mode,
        )

        # 最弱を置換
        population.sort(key=lambda c: c.fitness)
        if child.fitness > population[0].fitness:
            population[0] = child

    def _after_generation_callbacks(
        self,
        gen: int,
        generations: int,
        population: list[PromptCandidate],
        on_generation_complete: Callable[[int, PromptCandidate], None] | None,
        yield_check: Callable[[], bool] | None,
    ) -> bool:
        """世代終端の周期ログ + 進捗コールバック + 協調 yield 判定。

        Returns:
            yield が要求されたら True。例外は警告ログのみで握りつぶす。
        """
        if gen % 5 == 0:
            best = max(population, key=lambda c: c.fitness)
            logger.info(
                "Generation %d/%d: best fitness=%.4f",
                gen, generations, best.fitness,
            )
        if on_generation_complete is not None:
            try:
                best_so_far = max(population, key=lambda c: c.fitness)
                on_generation_complete(gen, best_so_far)
            except Exception as e:
                logger.warning("on_generation_complete raised: %s", e)
        if yield_check is None:
            return False
        try:
            should_yield = bool(yield_check())
        except Exception as e:
            logger.warning("yield_check raised: %s", e)
            return False
        if should_yield:
            logger.info(
                "Cooperative yield requested at generation %d/%d",
                gen, generations,
            )
            return True
        return False

    async def _run_evolution_loop(
        self,
        *,
        population: list[PromptCandidate],
        current: str,
        failure_hints: list[str],
        llm_client,
        generations: int,
        has_protected: bool,
        has_fewshot: bool,
        fewshot_pool: FewShotPool | None,
        mode: str,
        experiences: list[dict],
        seed: int | None,
        rng: np.random.Generator,
        cb: _CircuitBreaker,
        on_generation_complete: Callable[[int, PromptCandidate], None] | None,
        yield_check: Callable[[], bool] | None,
    ) -> tuple[int, bool]:
        """世代ループを実行する。`(generations_run, yielded)` を返す"""
        generations_run = 0
        yielded = False
        for gen in range(1, generations + 1):
            if cb.tripped or cb.should_stop():
                cb.tripped = True
                logger.info(
                    "Skipping remaining generations (%d/%d) due to circuit breaker",
                    gen, generations,
                )
                break

            await self._run_one_generation(
                gen=gen,
                current=current,
                population=population,
                failure_hints=failure_hints,
                llm_client=llm_client,
                has_protected=has_protected,
                has_fewshot=has_fewshot,
                fewshot_pool=fewshot_pool,
                mode=mode,
                experiences=experiences,
                seed=seed,
                rng=rng,
                cb=cb,
            )
            generations_run = gen

            if self._after_generation_callbacks(
                gen, generations, population, on_generation_complete, yield_check,
            ):
                yielded = True
                break
        return generations_run, yielded

    def _finalize_best_candidate(
        self,
        population: list[PromptCandidate],
        current: str,
        has_protected: bool,
    ) -> PromptCandidate:
        """最良候補に最終安全検証 (重複除去 + 保護セクション復元) を適用する"""
        best = max(population, key=lambda c: c.fitness)
        # 最終安全検証 - 段落レベルの重複を正規化
        best.text = dedupe_paragraphs(best.text)
        # 保護セクションの有無に関わらず孤児マーカーを必ず除去
        best.text = strip_orphan_protected_markers(best.text)
        # 最終安全検証: 最良候補の保護セクションを確認
        if has_protected and not validate_protected_sections(current, best.text):
            logger.warning("Best candidate lost protected sections, force-restoring")
            best.text = restore_protected_sections(current, best.text)
        if best.fewshot_ids:
            logger.info(
                "Best candidate uses %d fewshot examples: %s",
                len(best.fewshot_ids),
                best.fewshot_ids,
            )
        return best

    async def _darwinian_evolve(
        self,
        current: str,
        experiences: list[dict],
        llm_client,
        generations: int = 10,
        population_size: int = 5,
        seed: int | None = None,
        *,
        critique_synthesizer: CritiqueSynthesizer | None = None,
        fewshot_pool: FewShotPool | None = None,
        mode: str = "chat",
        yield_check: Callable[[], bool] | None = None,
        on_generation_complete: Callable[[int, PromptCandidate], None] | None = None,
    ) -> EvolutionResult:
        """進化ループ

        Args:
            current: 現在のプロンプト
            experiences: 経験バッファ
            llm_client: LLM クライアント
            generations: 世代数
            population_size: 集団サイズ
            seed: 乱数シード
            critique_synthesizer: 批評合成器。指定時は変異前に批評を実行
            fewshot_pool: Few-shot 候補プール。指定時は Few-shot も同時進化
            mode: 対象モード（Few-shot 選択のフィルタに使用）

        Returns:
            進化結果
        """
        failure_hints = await self._prepare_failure_hints(experiences, critique_synthesizer)
        has_protected = bool(extract_protected_sections(current))
        initial_fewshot_ids, has_fewshot = self._init_fewshot_selection(
            fewshot_pool, mode, seed,
        )

        cb = _CircuitBreaker()
        rng = np.random.default_rng(seed)

        population, initial_fitness = await self._build_initial_population(
            current=current,
            failure_hints=failure_hints,
            llm_client=llm_client,
            population_size=population_size,
            has_protected=has_protected,
            has_fewshot=has_fewshot,
            initial_fewshot_ids=initial_fewshot_ids,
            fewshot_pool=fewshot_pool,
            mode=mode,
            experiences=experiences,
            rng=rng,
            cb=cb,
        )

        unique_count = len({c.text for c in population})
        logger.info(
            "Initial population: %d candidates (%d unique), fitness range [%.4f, %.4f]",
            len(population),
            unique_count,
            min(c.fitness for c in population),
            max(c.fitness for c in population),
        )

        # 世代ループ
        generations_run, yielded = await self._run_evolution_loop(
            population=population,
            current=current,
            failure_hints=failure_hints,
            llm_client=llm_client,
            generations=generations,
            has_protected=has_protected,
            has_fewshot=has_fewshot,
            fewshot_pool=fewshot_pool,
            mode=mode,
            experiences=experiences,
            seed=seed,
            rng=rng,
            cb=cb,
            on_generation_complete=on_generation_complete,
            yield_check=yield_check,
        )

        best = self._finalize_best_candidate(population, current, has_protected)

        return EvolutionResult(
            best_candidate=best,
            all_candidates=sorted(population, key=lambda c: -c.fitness),
            generations_run=generations_run,
            initial_fitness=initial_fitness,
            final_fitness=best.fitness,
            yielded=yielded,
        )

    @staticmethod
    def _persist_session_safely(
        session: Level1Session | None,
        save_session: Callable[[Level1Session], None] | None,
    ) -> None:
        """`save_session` を try/except でラップして例外をログに変換する。"""
        if session is None or save_session is None:
            return
        try:
            save_session(session)
        except Exception as e:  # noqa: BLE001 — コールバック由来は何でも握る
            logger.warning("save_session raised: %s", e)

    def _handle_mode_yield(
        self,
        result: EvolutionResult,
        mode: str,
        generations: int,
        session: Level1Session | None,
        save_session: Callable[[Level1Session], None] | None,
    ) -> None:
        """モード進化が yield された場合の session 更新と永続化。"""
        if session is not None:
            session.yield_count += 1
            self._persist_session_safely(session, save_session)
        logger.info(
            "Yielded during mode %s at generation %d/%d",
            mode, result.generations_run, generations,
        )

    def _mark_mode_completed(
        self,
        mode: str,
        session: Level1Session | None,
        save_session: Callable[[Level1Session], None] | None,
    ) -> None:
        """モードを `completed_phases` に追加して session を永続化する。"""
        if session is None:
            return
        if mode not in session.completed_phases:
            session.completed_phases.append(mode)
        self._persist_session_safely(session, save_session)

    @staticmethod
    def _check_inter_mode_yield(
        yield_check: Callable[[], bool] | None,
    ) -> bool:
        """モード間で協調 yield が要求されていれば True。"""
        if yield_check is None:
            return False
        try:
            if yield_check():
                logger.info("Cooperative yield requested between modes")
                return True
        except Exception as e:  # noqa: BLE001 — コールバック由来は何でも握る
            logger.warning("yield_check raised between modes: %s", e)
        return False

    def _process_evolution_result(
        self,
        result: EvolutionResult,
        mode: str,
        generations: int,
        cross_mode_failures: int,
        session: Level1Session | None,
        save_session: Callable[[Level1Session], None] | None,
        yield_check: Callable[[], bool] | None,
    ) -> tuple[int, bool]:
        """進化結果を処理して `(new_cross_mode_failures, should_break)` を返す。

        - generations_run==0 ならクロスモード失敗カウンタを +1、それ以外は 0 リセット
        - yield された場合は `_handle_mode_yield` を呼んで break 指示
        - 完了時は `completed_phases` に追加 → モード間 yield チェック
        """
        new_failures = cross_mode_failures + 1 if result.generations_run == 0 else 0
        logger.info(
            "Mode %s evolution complete: fitness %.4f → %.4f "
            "(generations=%d/%d)",
            mode, result.initial_fitness, result.final_fitness,
            result.generations_run, generations,
        )
        if result.yielded:
            self._handle_mode_yield(
                result, mode, generations, session, save_session,
            )
            return new_failures, True
        self._mark_mode_completed(mode, session, save_session)
        if self._check_inter_mode_yield(yield_check):
            return new_failures, True
        return new_failures, False

    async def evolve_all_modes(
        self,
        experiences: list[dict],
        prompt_texts: dict[str, str],
        llm_client,
        generations: int = 10,
        population_size: int = 5,
        seed: int | None = None,
        *,
        critique_synthesizer: CritiqueSynthesizer | None = None,
        fewshot_pool: FewShotPool | None = None,
        session: Level1Session | None = None,
        yield_check: Callable[[], bool] | None = None,
        save_session: Callable[[Level1Session], None] | None = None,
    ) -> dict[str, EvolutionResult]:
        """全モードのプロンプトを進化させる

        Args:
            experiences: 経験バッファ全体
            prompt_texts: モード名 → 現在のプロンプトテキスト
            llm_client: LLM クライアント
            generations: 世代数
            population_size: 集団サイズ
            seed: 乱数シード
            critique_synthesizer: 批評合成器
            fewshot_pool: Few-shot 候補プール
            session: Level1Session（resume / 進捗保存用）
            yield_check: 各世代/モード終端で呼ばれる協調 yield 判定
            save_session: 各モード完了時に呼ばれる session 永続化コールバック

        Returns:
            モード名 → 進化結果
        """
        results: dict[str, EvolutionResult] = {}
        cross_mode_failures = 0
        completed_phases = list(session.completed_phases) if session else []

        for mode, prompt_text in prompt_texts.items():
            # 既に完了済みのモードはスキップ（resume 経路）
            if mode in completed_phases:
                logger.info(
                    "Mode %s already completed in active session, skipping",
                    mode,
                )
                continue

            # クロスモード・サーキットブレーカー
            if cross_mode_failures >= CB_MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Cross-mode circuit breaker: skipping mode %s "
                    "after %d consecutive mode failures",
                    mode, cross_mode_failures,
                )
                break

            # モード別経験フィルタ
            mode_exp = [e for e in experiences if e.get("mode") == mode]
            if not mode_exp:
                logger.info("No experiences for mode %s, skipping evolution", mode)
                continue

            logger.info(
                "Evolving %s prompt (%d experiences, %d generations)",
                mode, len(mode_exp), generations,
            )

            result = await self._darwinian_evolve(
                current=prompt_text,
                experiences=mode_exp,
                llm_client=llm_client,
                generations=generations,
                population_size=population_size,
                seed=seed,
                critique_synthesizer=critique_synthesizer,
                fewshot_pool=fewshot_pool,
                mode=mode,
                yield_check=yield_check,
            )
            results[mode] = result

            cross_mode_failures, should_break = self._process_evolution_result(
                result, mode, generations, cross_mode_failures,
                session, save_session, yield_check,
            )
            if should_break:
                break

        return results
