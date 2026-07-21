"""Darwinian 進化アルゴリズムによるシステムプロンプト最適化"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# 保護セクション検証・重複排除は agent.prompt_utils (EvorefLoop
# pillar の純粋 util) に集約済。Learn pillar 側はここから top-level import する。
from backend.free.agent.prompt_utils import (
    dedupe_paragraphs,
    extract_protected_sections,
    restore_protected_sections,
    strip_orphan_protected_markers,
    text_contains_sentence,
    validate_protected_sections,
)
from backend.log_config import get_logger
from backend.utils import estimate_tokens

if TYPE_CHECKING:
    from backend.free.learning.critique_synthesizer import CritiqueSynthesizer
    from backend.free.learning.level1_session import Level1Session

logger = get_logger("optimizer.prompt_evolver")

# 突然変異リトライ上限
MAX_MUTATION_RETRIES = 3

# プロンプト変異の生成トークン上限。システムプロンプト全文 (coding.md ~1350字) の
# 再生成 + reasoning モデルの思考漏れ分の途中切断を避けるため広めに取る。
_MUTATION_MAX_TOKENS = 2048

# assist context_size に対する安全マージン (chat template / 特殊トークン分)。
_CONTEXT_SAFETY_MARGIN = 256

# _is_text_similar の判定閾値。SequenceMatcher.ratio がこれ以上、かつ長さ差が
# _TEXT_LEN_DELTA 以内なら「実質同一」とみなす。長さ差は空白・句読点程度の自明な
# 差のみを許容する小さい値とし、ルール追記など内容が増える変異 (長文への短い
# 追記を含む) は非類似=採用に倒す (no-op 誤判定で全変異を棄却した旧バグの逆方向)。
_TEXT_SIMILARITY_THRESHOLD = 0.97
_TEXT_LEN_DELTA = 5

# ── サーキットブレーカー閾値 ──
# 連続失敗回数がこの閾値以上で早期終了
CB_MAX_CONSECUTIVE_FAILURES: int = 3
# 失敗率がこの閾値を超えたら早期終了（最低 CB_MIN_ATTEMPTS 回試行後）
CB_FAILURE_RATE_THRESHOLD: float = 0.5
CB_MIN_ATTEMPTS: int = 4

# reasoning モデルが content に吐く <think>...</think> ブロック (閉じたもの)
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
# 未閉鎖の <think> 開始タグ (暴走/打ち切りで </think> が無いケース)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """reasoning モデルが content に吐く ``<think>...</think>`` を除去する。

    Qwen3 / LFM2 等の reasoning ベース/アシストでプロンプトを変異させると、思考が
    そのまま変異後プロンプトに焼き込まれ system プロンプト (chat.md) を汚染する。
    閉じたブロックは除去し、未閉鎖 ``<think>`` (暴走/打ち切り) は以降をすべて思考と
    みなして破棄する (残った本文が空なら呼出側で無効判定させる)。
    """
    text = _THINK_TAG_RE.sub("", text)
    m = _THINK_OPEN_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


# fitness 識別用キーワード抽出: ASCII 語 (3 文字以上) + CJK 文字 bi-gram。
# ひらがなのみの bi-gram は機能語ノイズ (「して」「ください」等) のため除外し、
# 漢字・カタカナを含む内容語だけを残す。
_ASCII_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
# U+3040-30FF (ひらがな/カタカナ) U+3400-4DBF U+4E00-9FFF (漢字) U+F900-FAFF (互換漢字)
_CJK_RUN_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]+")
_HIRAGANA_ONLY_RE = re.compile(r"^[぀-ゟ]+$")


def _extract_query_terms(text: str) -> set[str]:
    """クエリから fitness のカバレッジ判定に使う語彙を抽出する。

    空白 split のみだと日本語クエリが 1 トークンに潰れ、候補プロンプトが
    クエリ全文を含まない限りカバー率が常に 0 になる (全候補が同 fitness に
    縮退して進化が no-op 化する)。BM25 の日本語トークナイズと同方針で
    CJK 連続文字列を文字 bi-gram に分解して識別力を確保する。
    """
    lowered = text.lower()
    terms = set(_ASCII_WORD_RE.findall(lowered))
    for run in _CJK_RUN_RE.findall(lowered):
        terms.update(
            bigram
            for bigram in (run[i:i + 2] for i in range(len(run) - 1))
            if not _HIRAGANA_ONLY_RE.match(bigram)
        )
    return terms


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
    """プロンプト候補 (instruction-only)。

    few-shot 例は推論時に FewShotPool.select_top_k が query 依存で動的選択する
    ため、進化では instruction テキストのみを最適化する (co-evolution 廃止)。
    """
    text: str
    fitness: float = 0.0
    generation: int = 0


@dataclass
class EvolutionResult:
    """進化結果"""
    best_candidate: PromptCandidate
    all_candidates: list[PromptCandidate] = field(default_factory=list)
    generations_run: int = 0
    initial_fitness: float = 0.0
    final_fitness: float = 0.0
    yielded: bool = False  # 協調 yield により途中終了したか
    # LLM 変異 (mutation) の試行/失敗回数。``learning_cycle_l1`` の outcome が
    # 「変異が systemic に失敗して何も学習できていない」ことを「改善なし (収束)」と
    # 区別できるよう、サーキットブレーカー集計を結果へ伝播する。
    mutation_attempts: int = 0
    mutation_failures: int = 0


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
        - long_form_used かつ long_form_success=False → 失敗（生成物が検証エラー）

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

            # 長文生成 (coding モードの成果物) の検証失敗も失敗シグナル。
            # coding モードは会話が単発で完結しがちで rephrase/user_correction が
            # 皆無になりやすく (2026-07-17 実データで 21 件中 0 件)、この信号が
            # ないと coding モードの経験が base_score に一切反映されない。
            if signals.get("long_form_used") and signals.get("long_form_success") is False:
                score -= weight * 0.5

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
        # rephrased_query/user_correction に加え、長文生成の検証失敗
        # (long_form_used かつ long_form_success=False) も失敗経験として含める
        # (coding モードは前者 2 シグナルがほぼ発生しないため、これが無いと
        # failure_keywords が常に空集合になり全候補の fitness が完全に一致する
        # = 進化が no-op 化する 2026-07-17 の実障害と同じ状態になる)。
        failure_keywords: set[str] = set()
        for exp in experiences:
            signals = exp.get("signals", {})
            is_long_form_failure = (
                signals.get("long_form_used")
                and signals.get("long_form_success") is False
            )
            if (
                signals.get("rephrased_query")
                or signals.get("user_correction")
                or is_long_form_failure
            ):
                failure_keywords.update(
                    _extract_query_terms(str(exp.get("query", ""))),
                )
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

    async def _score_candidate(
        self,
        candidate: str,
        experiences: list[dict],
    ) -> float:
        """候補の fitness を返す async seam。

        基底は同期 ``_calc_fitness`` をそのまま委譲する (挙動不変)。実測評価が
        必要なサブクラス (EmbedInstructionEvolver) はここを override して await
        ベースの実測 fitness に差し替える。進化ループ
        (``_build_initial_population`` / ``_run_one_generation``) は本メソッド
        経由で fitness を算出する。
        """
        return self._calc_fitness(candidate, experiences)

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

        messages = [{"role": "user", "content": mutation_query}]
        try:
            # 変異生成器が assist (AssistModelClient) か base (LocalClient) かで
            # generate のシグネチャが異なる: assist は purpose 必須・id_slot 無視、
            # base は purpose 非対応・id_slot 使用。低速 base のタイムアウトを避け
            # mutation は assist へ寄せる (learning スロット)。purpose は purpose
            # 監査 (test_assist_purpose_audit) が静的検出できるようリテラルで分岐する。
            if getattr(llm_client, "is_assist_client", False):
                context_size = getattr(llm_client, "context_size", 8192)
                input_tokens = estimate_tokens(mutation_query)
                if input_tokens + _MUTATION_MAX_TOKENS > context_size - _CONTEXT_SAFETY_MARGIN:
                    logger.warning(
                        "Prompt mutation skipped: prompt=%d tok + output budget=%d tok "
                        "exceeds assist context_size=%d",
                        input_tokens, _MUTATION_MAX_TOKENS, context_size,
                    )
                    return None
                result = await llm_client.generate(
                    messages=messages,
                    stream=False,
                    temperature=0.5,
                    max_tokens=_MUTATION_MAX_TOKENS,
                    purpose="prompt_evolution",
                )
            else:
                result = await llm_client.generate(
                    messages=messages,
                    stream=False,
                    temperature=0.5,
                    max_tokens=_MUTATION_MAX_TOKENS,
                    id_slot=getattr(llm_client, "background_slot", -1),
                )
            mutated = result["choices"][0]["message"]["content"].strip()

            # reasoning モデル (Qwen3 / LFM2 等) が content に吐く <think>...</think> を
            # 除去する。未除去だと思考が変異後プロンプトに焼き込まれ chat.md を汚染する。
            mutated = _strip_think_tags(mutated)

            # 応答全体が ``` でラップされている場合のみ外側フェンスを外す。
            # 全フェンス行を無差別除去すると、プロンプト本文中のコード例フェンス
            # (```python ... ```) まで剥がれてコード例が壊れるため、先頭フェンスと
            # 対応する末尾フェンスのペアだけを除去し本文中のフェンスは保護する。
            if mutated.startswith("```"):
                lines = mutated.split("\n")
                lines = lines[1:]  # 先頭フェンス行 (```lang 含む) を除去
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]  # 対応する末尾フェンスを除去
                mutated = "\n".join(lines).strip()

            # 思考が残る / 空になった場合は汚染とみなし無効化 (rule-based 変異へフォールバック)
            if not mutated or "<think>" in mutated.lower():
                logger.warning(
                    "Prompt mutation produced contaminated/empty output "
                    "(<think> residue or empty); rejecting for rule-based fallback",
                )
                return None

            return mutated

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
        """2つのテキストが実質的に同一かどうかを判定する。

        旧実装は共通プレフィックス率で判定していたため、LLM 変異の最頻形
        (末尾にルールを追記する) を prefix 一致率 1.0 で「同一」と誤判定し
        棄却していた (先頭を変える変異だけが生き残るバイアス)。長さが有意に
        変わる変異 (追記等) は非類似=採用とし、長さがほぼ同じ場合のみ
        ``difflib.SequenceMatcher.ratio`` の全体一致率で同一性を測る。
        """
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()

        if t1 == t2:
            return True
        if not t1 or not t2:
            return True
        # 長さが有意に変われば実質的な変更 (末尾追記など) → 非類似
        if abs(len(t1) - len(t2)) > _TEXT_LEN_DELTA:
            return False
        return difflib.SequenceMatcher(None, t1, t2).ratio() >= _TEXT_SIMILARITY_THRESHOLD

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
        experiences: list[dict],
        cb: _CircuitBreaker,
    ) -> tuple[list[PromptCandidate], float]:
        """現在プロンプト + LLM 変異で初期集団を構築する。

        `(population, initial_fitness)` を返す。サーキットブレーカー発火時は
        `cb.tripped = True` をセットして部分集団を返す。
        """
        population = [PromptCandidate(text=current, generation=0)]
        initial_fitness = await self._score_candidate(current, experiences)
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

            candidate = PromptCandidate(text=mutated_text, generation=0)
            candidate.fitness = await self._score_candidate(mutated_text, experiences)
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
        experiences: list[dict],
        seed: int | None,
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

        child = PromptCandidate(text=child_text, generation=gen)
        child.fitness = await self._score_candidate(child_text, experiences)

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
        experiences: list[dict],
        seed: int | None,
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
                experiences=experiences,
                seed=seed,
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
        yield_check: Callable[[], bool] | None = None,
        on_generation_complete: Callable[[int, PromptCandidate], None] | None = None,
    ) -> EvolutionResult:
        """進化ループ (instruction-only)

        Args:
            current: 現在のプロンプト
            experiences: 経験バッファ
            llm_client: LLM クライアント
            generations: 世代数
            population_size: 集団サイズ
            seed: 乱数シード
            critique_synthesizer: 批評合成器。指定時は変異前に批評を実行

        Returns:
            進化結果

        Note:
            few-shot 例は推論時に FewShotPool.select_top_k が query 依存で動的選択
            するため、進化は instruction テキストのみを最適化する (co-evolution 廃止)。
        """
        failure_hints = await self._prepare_failure_hints(experiences, critique_synthesizer)
        has_protected = bool(extract_protected_sections(current))

        cb = _CircuitBreaker()

        population, initial_fitness = await self._build_initial_population(
            current=current,
            failure_hints=failure_hints,
            llm_client=llm_client,
            population_size=population_size,
            has_protected=has_protected,
            experiences=experiences,
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
            experiences=experiences,
            seed=seed,
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
            mutation_attempts=cb.total_attempts,
            mutation_failures=cb.total_failures,
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
        except Exception as e:
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
        except Exception as e:
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
