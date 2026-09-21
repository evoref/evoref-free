"""Critique-Synthesis Loop: 変異前の失敗パターン批評

Darwinian Evolver の変異生成前に経験バッファの失敗パターンを分析し、
構造化された批評を変異ヒントとして提供する。

補助タスク接続時は LLM で深い分析を行い、
未接続時はルールベースのパターン分析にフォールバックする。

参考論文: PromptWizard (arXiv:2405.18369) の批評・合成ループ
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.free.learning.fitness import DEFECT_WEIGHTS, signal_is_defect
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger

logger = get_logger("learning.critique_synthesizer")

# ルールベース閾値
_CORRECTION_RATE_THRESHOLD = 0.3   # 訂正率がこれ以上なら「明確性の問題」
_CHANNEL_RATE_THRESHOLD = 0.3      # 支配的な失敗チャネルがこれ以上なら報告
_CONSECUTIVE_DECLINE_N = 3         # fitness 連続低下の検出閾値
_HIGH_AGENT_LOOPS = 3              # エージェントループ数の警告閾値

# システム能力と矛盾する改善ヒントの検出。「ローカルにアクセスできない前提で
# 振る舞え」系のヒントはファイル出力依頼の拒否/迂回を助長する
# (2026-07-15: この種のヒント経由で chat プロンプト v6 が配備された)。
_CAPABILITY_CONTRADICTION_RE = re.compile(
    r"cannot\s+(?:access|write|read)"
    r"|can'?t\s+(?:access|write|read)"
    r"|unable\s+to\s+(?:access|write|read)"
    r"|no\s+access\s+to\s+(?:the\s+)?(?:local|file)"
    r"|アクセスできない|書き込めない前提",
    re.IGNORECASE,
)


def _hint_contradicts_capabilities(hint: str) -> bool:
    """改善ヒントがシステム能力 (ローカル書込可) と矛盾するかを判定する。"""
    return bool(_CAPABILITY_CONTRADICTION_RE.search(hint))


def _is_failure(experience: dict) -> bool:
    """批評の母集団に入れる失敗か (語彙は共有の重み表が SSOT)。

    旧実装は ``rephrased_query`` / ``user_correction`` / ``turn_outcome`` の
    3 条件を手で並べていた。``rephrased_query`` は 2026-09-21 に重み表から
    外した字句判定 (実データ 161 件で発火 0 件)、``long_form`` の検証落ちは
    そもそも見ていなかった — create の失敗の大半がこれなのに、批評の
    母集団に 1 件も入らない状態だった (docs/f_04 §3.2.4)。
    """
    return any(
        signal_is_defect(experience.get("signals") or {}, key)
        for key in DEFECT_WEIGHTS
    )


#: 失敗チャネル → (パターン文, 改善ヒント)。``OUTCOME_REASON_CHANNELS`` が
#: 畳んだ 4 分類に一対一で対応する。旧実装の「言い直し率」分析は
#: ``rephrased_query`` の発火が 0 なので恒に不発だった。代わりに
#: **どのチャネルの失敗が支配的か** を見る (原因の別で処方が変わるため)。
_CHANNEL_CRITIQUE: dict[str, tuple[str, str]] = {
    "outcome_instruction": (
        "instruction violations dominate: explicit constraints in the request "
        "(character count, item count, output form) are not being honored",
        "Add instruction: 'When the request states an exact count, length or "
        "output form, satisfy it literally and verify before answering.'",
    ),
    "outcome_contradiction": (
        "self-contradictions dominate: the answer disagrees with its own "
        "calculations, with tool results, or with values already supplied",
        "Add instruction: 'Before concluding, re-read the numbers and tool "
        "results you were given and make sure the conclusion uses them.'",
    ),
    "outcome_broken_output": (
        "broken output dominates: responses are cut off or contain "
        "malformed / foreign-language fragments",
        "Keep responses within a length the model can finish; prefer a short "
        "complete answer over a long truncated one.",
    ),
    "outcome_execution": (
        "execution failures dominate: the requested action did not run or "
        "was routed to the wrong handler",
        "Add instruction: 'If an action cannot be completed, say so plainly "
        "instead of describing it as done.'",
    ),
    "long_form_failed": (
        "long-form deliverables fail validation: generated documents do not "
        "pass the structural checks",
        "Add instruction: 'For documents, follow the requested heading "
        "structure exactly and keep every section non-empty.'",
    ),
}


@dataclass
class CritiqueResult:
    """批評結果"""

    failure_patterns: list[str] = field(default_factory=list)
    """識別された失敗パターンの記述"""

    improvement_hints: list[str] = field(default_factory=list)
    """プロンプト変異に渡す具体的な改善ヒント"""

    summary: str = ""
    """批評の要約"""

    source: str = "rule_based"
    """批評の生成元: ``"llm"`` | ``"rule_based"``。

    ``"llm"`` は「LLM に生成させた」の意。批評はベースモデルで走る。
    過去の `learning_state.json` には旧値 ``"aux"`` が残るが、この値で分岐する
    コードは無い (ログ・デバッグ JSONL 用の記述ラベル)。
    """


def _analyze_correction_rate(
    failures: list[dict], total: int,
) -> tuple[str | None, str | None, int]:
    """訂正率を分析。`(pattern, hint, correction_count)` を返す。

    閾値未達 / total=0 の場合は `(None, None, count)`。
    """
    correction_count = sum(
        1 for f in failures
        if f.get("signals", {}).get("user_correction") is not None
    )
    if total == 0:
        return None, None, correction_count
    rate = correction_count / total
    if rate < _CORRECTION_RATE_THRESHOLD:
        return None, None, correction_count
    pattern = (
        f"High correction rate ({rate:.0%}): "
        "responses contain factual or interpretive errors"
    )
    hint = (
        "Add explicit instruction: 'When uncertain about facts or user intent, "
        "state your assumptions clearly before responding.'"
    )
    return pattern, hint, correction_count


def failure_channels(failures: list[dict]) -> dict[str, int]:
    """失敗チャネルごとの件数 (``_CHANNEL_CRITIQUE`` のキーのみ)。

    ``turn_outcome`` 由来の 4 チャネルは排他だが、``long_form_failed`` は
    それらと独立に立つので同じターンが 2 つ数えられうる。件数は「どの原因が
    支配的か」を見るためのもので、合計を失敗件数と一致させる意図はない。
    """
    counts: dict[str, int] = {}
    for f in failures:
        signals = f.get("signals") or {}
        for key in _CHANNEL_CRITIQUE:
            if signal_is_defect(signals, key):
                counts[key] = counts.get(key, 0) + 1
    return counts


def _analyze_failure_channels(
    failures: list[dict], total: int,
) -> tuple[str | None, str | None, dict[str, int]]:
    """支配的な失敗チャネルを分析。``(pattern, hint, counts)`` を返す。

    旧「言い直し率」分析の置き換え (2026-09-21)。``rephrased_query`` は
    重み表から外した字句判定で発火が 0 件だったため、この分析は恒に不発
    だった。検証器が既に **なぜ失敗したか** を持っているので、そちらを見る。
    """
    counts = failure_channels(failures)
    if total == 0 or not counts:
        return None, None, counts
    channel, count = max(counts.items(), key=lambda kv: kv[1])
    rate = count / total
    if rate < _CHANNEL_RATE_THRESHOLD:
        return None, None, counts
    description, hint = _CHANNEL_CRITIQUE[channel]
    return f"{rate:.0%} of turns — {description}", hint, counts


def _analyze_high_agent_loops(
    failures: list[dict],
) -> tuple[str | None, str | None]:
    """エージェントループ過多を検出。閾値未達なら `(None, None)`。"""
    high_loop_count = sum(
        1 for f in failures
        if f.get("signals", {}).get("agent_loops", 0) >= _HIGH_AGENT_LOOPS
    )
    if high_loop_count == 0:
        return None, None
    rate = high_loop_count / max(1, len(failures))
    if rate < 0.3:
        return None, None
    pattern = (
        f"Excessive agent loops in {rate:.0%} of failures: "
        "task decomposition may be too aggressive"
    )
    hint = (
        "Add instruction: 'Prefer direct answers over multi-step "
        "reasoning when the query is straightforward.'"
    )
    return pattern, hint


def _analyze_mode_concentration(
    failures: list[dict], all_experiences: list[dict],
) -> tuple[list[str], list[str]]:
    """モード別の失敗集中を検出。`(patterns, hints)` のリストを返す。"""
    mode_failures: dict[str, int] = {}
    mode_totals: dict[str, int] = {}
    for e in all_experiences:
        mode = e.get("mode", "chat")
        mode_totals[mode] = mode_totals.get(mode, 0) + 1
    for f in failures:
        mode = f.get("mode", "chat")
        mode_failures[mode] = mode_failures.get(mode, 0) + 1

    patterns: list[str] = []
    hints: list[str] = []
    for mode, fail_count in mode_failures.items():
        mode_total = mode_totals.get(mode, 1)
        rate = fail_count / mode_total
        if rate < 0.4:
            continue
        patterns.append(
            f"Failure concentrated in '{mode}' mode "
            f"({rate:.0%} failure rate)"
        )
        hints.append(
            f"Review and strengthen the '{mode}' mode system prompt: "
            "it may lack domain-specific instructions."
        )
    return patterns, hints


def _analyze_correction_content(
    failures: list[dict],
) -> tuple[str | None, str | None]:
    """訂正テキストの技術系比率を分析。半数超なら警告 (pattern, hint) を返す。"""
    corrections = [
        f.get("signals", {}).get("user_correction", "")
        for f in failures
        if f.get("signals", {}).get("user_correction")
    ]
    if not corrections:
        return None, None
    code_keywords = ("code", "error", "bug", "function", "class", "import")
    tech_corrections = sum(
        1 for c in corrections
        if any(kw in c.lower() for kw in code_keywords)
    )
    if tech_corrections <= len(corrections) * 0.5:
        return None, None
    pattern = (
        "Technical/code-related corrections dominate: "
        "coding instructions may be insufficient"
    )
    hint = (
        "Strengthen coding instructions: 'Always verify code syntax "
        "and logic before presenting. Include error handling.'"
    )
    return pattern, hint


def _format_critique_summary(
    failures: list[dict],
    total: int,
    correction_count: int,
    channel_counts: dict[str, int],
) -> str:
    """批評結果の summary 文字列を構築する。"""
    parts = [f"{len(failures)}/{total} failures analyzed"]
    if correction_count:
        parts.append(f"{correction_count} corrections")
    for channel, count in sorted(
        channel_counts.items(), key=lambda kv: (-kv[1], kv[0]),
    ):
        parts.append(f"{count} {channel.removeprefix('outcome_')}")
    return ", ".join(parts)


class CritiqueSynthesizer:
    """経験バッファの失敗パターンを批評し、変異ヒントを生成する

    LLM が利用可能な場合は構造化分析を行い、
    利用不可の場合はルールベースのパターン分析にフォールバックする。
    """

    def __init__(
        self,
        llm_client=None,
        debug_logger: DebugLogger | None = None,
    ) -> None:
        """初期化

        Args:
            llm_client: 批評合成に使う LLM クライアント (ベースモデルの
                ``AuxClient``)。``None`` ならルールベース専用。
            debug_logger: DebugLogger インスタンス（任意）
        """
        self._llm_client = llm_client
        self._debug_logger = debug_logger

    async def critique(self, experiences: list[dict]) -> CritiqueResult:
        """経験バッファの失敗パターンを批評する

        Args:
            experiences: 経験バッファのエントリリスト（dict 形式）

        Returns:
            構造化された批評結果
        """
        failures = [e for e in experiences if _is_failure(e)]

        if not failures:
            return CritiqueResult(
                summary="No failure patterns detected",
                source="rule_based",
            )

        t0 = time.monotonic()

        # LLM が利用可能かチェック
        if self._llm_client is not None:
            try:
                result = await self._llm_critique(failures, experiences)
                elapsed = time.monotonic() - t0
                logger.info(
                    "LLM critique completed: %d patterns, %d hints (%.2fs)",
                    len(result.failure_patterns),
                    len(result.improvement_hints),
                    elapsed,
                )
                self._log_critique(result, elapsed)
                return result
            except Exception as e:
                logger.warning(
                    "LLM critique failed, falling back to rule-based: %s: %s",
                    type(e).__name__, e,
                )

        # ルールベースフォールバック
        result = self._rule_based_critique(failures, experiences)
        elapsed = time.monotonic() - t0
        logger.info(
            "Rule-based critique completed: %d patterns, %d hints (%.2fs)",
            len(result.failure_patterns),
            len(result.improvement_hints),
            elapsed,
        )
        self._log_critique(result, elapsed)
        return result

    async def _llm_critique(
        self,
        failures: list[dict],
        all_experiences: list[dict],
    ) -> CritiqueResult:
        """補助タスクによる失敗パターン分析

        失敗経験を構造化して LLM に渡し、根本原因と改善提案を得る。
        """
        # 失敗経験のサマリを構築（最大10件）。
        #
        # **検証器が記録した理由をそのまま渡す** (2026-09-21)。旧実装は
        # "turn failed (deliverable not produced)" という原因を含まない一文
        # しか渡しておらず、批評 LLM は「何が悪かったか」を推測するしか
        # なかった。``turn_outcome_reason`` には
        # "length constraint: asked for exactly 100 chars but the answer is 71"
        # のような具体が既に入っている。
        failure_summaries = []
        for f in failures[:10]:
            signals = f.get("signals", {})
            entry = f"- query: \"{f.get('query', '')[:100]}\""
            if signals.get("user_correction"):
                entry += f", correction: \"{signals['user_correction'][:100]}\""
            if signals.get("agent_loops", 0) > 1:
                entry += f", agent_loops: {signals['agent_loops']}"
            if signals.get("turn_outcome") == "failed":
                reason = str(signals.get("turn_outcome_reason") or "").strip()
                entry += (
                    f", verifier flagged: {reason[:160]}" if reason
                    else ", turn failed (no reason recorded)"
                )
            if (
                signals.get("long_form_used")
                and signals.get("long_form_success") is False
            ):
                errors = signals.get("long_form_validation_errors") or 0
                entry += (
                    f", long-form document failed validation ({errors} error(s))"
                )
            failure_summaries.append(entry)

        total = len(all_experiences)
        failure_count = len(failures)
        correction_count = sum(
            1 for f in failures
            if f.get("signals", {}).get("user_correction") is not None
        )
        channel_counts = failure_channels(failures)
        channel_line = ", ".join(
            f"{count} {channel.removeprefix('outcome_')}"
            for channel, count in sorted(
                channel_counts.items(), key=lambda kv: (-kv[1], kv[0]),
            )
        ) or "no channel breakdown"

        prompt = (
            "You are analyzing failure patterns in an AI assistant's interactions "
            "to improve its system prompt.\n\n"
            "System capabilities (facts — do NOT suggest changes that contradict "
            "them): the assistant runs locally and CAN write files to local paths "
            "(write_file tool), read files, and run commands. Do NOT suggest that "
            "the assistant acknowledge inability to access the local system.\n\n"
            f"Statistics: {failure_count} failures out of {total} total interactions. "
            f"{correction_count} user corrections. "
            f"Failure causes by category: {channel_line}.\n\n"
            f"Failure examples:\n"
            + "\n".join(failure_summaries) + "\n\n"
            "Analyze the root causes of these failures and suggest specific "
            "improvements to the system prompt.\n\n"
            "Respond in JSON format:\n"
            "{\n"
            '  "failure_patterns": ["pattern1", "pattern2", ...],\n'
            '  "improvement_hints": ["hint1", "hint2", ...],\n'
            '  "summary": "brief summary"\n'
            "}\n\n"
            "Rules:\n"
            "- failure_patterns: 2-4 root cause descriptions (concise)\n"
            "- improvement_hints: 2-4 actionable prompt improvement suggestions\n"
            "- Each hint should be a specific instruction to add or modify\n"
            "- Focus on the most impactful changes"
        )

        data = await self._llm_client.generate_json(
            prompt,
            max_tokens=512,
            temperature=0.3,
            purpose="critique_synthesis",
        )

        if not data:
            raise ValueError("Critique model returned empty JSON")

        hints = [
            h for h in data.get("improvement_hints", [])
            if not _hint_contradicts_capabilities(str(h))
        ]
        return CritiqueResult(
            failure_patterns=data.get("failure_patterns", [])[:4],
            improvement_hints=hints[:4],
            summary=data.get("summary", ""),
            source="llm",
        )

    def _rule_based_critique(
        self,
        failures: list[dict],
        all_experiences: list[dict],
    ) -> CritiqueResult:
        """ルールベースの失敗パターン分析

        fitness 変化パターン・シグナル統計からヒューリスティックに批評を生成する。
        各分析セクションは module-level pure helper (`_analyze_*`) に委譲する。
        """
        total = len(all_experiences)
        if total == 0:
            return CritiqueResult(source="rule_based")

        patterns: list[str] = []
        hints: list[str] = []

        pat, hint, correction_count = _analyze_correction_rate(failures, total)
        if pat:
            patterns.append(pat)
            hints.append(hint)

        pat, hint, channel_counts = _analyze_failure_channels(failures, total)
        if pat:
            patterns.append(pat)
            hints.append(hint)

        pat, hint = _analyze_high_agent_loops(failures)
        if pat:
            patterns.append(pat)
            hints.append(hint)

        mode_pats, mode_hints = _analyze_mode_concentration(failures, all_experiences)
        patterns.extend(mode_pats)
        hints.extend(mode_hints)

        pat, hint = _analyze_correction_content(failures)
        if pat:
            patterns.append(pat)
            hints.append(hint)

        # パターンが検出されなかった場合の汎用ヒント
        if not patterns:
            patterns.append(
                f"Minor failure rate ({len(failures)}/{total}): "
                "no dominant pattern detected"
            )
            hints.append(
                "Consider adding more specific examples or constraints "
                "to the system prompt for edge cases."
            )

        return CritiqueResult(
            failure_patterns=patterns,
            improvement_hints=hints,
            summary=_format_critique_summary(
                failures, total, correction_count, channel_counts,
            ),
            source="rule_based",
        )

    async def synthesize_from_failure_cluster(
        self,
        cluster_facts: list[dict],
    ) -> CritiqueResult:
        """SemMem の ``failure_pattern`` ファクト群から共通失敗要因を合成する。

        疑似経験レコード (``signals.user_correction`` に failure summary、
        ``mode="create"``) に変換し、既存 ``critique()`` を再利用する。

        Args:
            cluster_facts: ``failure_pattern`` を 1 クラスタ分解して dict
                化したリスト。各要素は以下のキーを持つ:
                - ``subject``: ``loop.failure.<signature>``
                - ``object``: object JSON (failure_note.parse 済み dict でも可)
                - ``signature``: failure_signature (任意)

        Returns:
            ``CritiqueResult``。``source="llm"`` / ``"rule_based"`` の
            既存仕様を踏襲する。
        """
        if len(cluster_facts) < 2:
            return CritiqueResult(
                summary="Cluster too small for synthesis",
                source="rule_based",
            )

        pseudo_experiences: list[dict] = []
        for fact in cluster_facts:
            obj = fact.get("object", {})
            if isinstance(obj, str):
                try:
                    import json as _json
                    obj = _json.loads(obj)
                except (ValueError, TypeError):
                    obj = {"raw": obj}
            outcome = obj.get("outcome") or obj.get("error_type") or ""
            last_actions = obj.get("last_actions") or obj.get(
                "last_3_step_actions", []
            )
            last_actions_text = (
                ", ".join(str(a) for a in last_actions)
                if isinstance(last_actions, (list, tuple))
                else str(last_actions)
            )
            pseudo_experiences.append({
                "mode": "create",
                "query": str(obj.get("title") or fact.get("subject", ""))[:200],
                "signals": {
                    "user_correction": f"{outcome}: {last_actions_text}"[:300],
                    "agent_loops": int(obj.get("occurrences", 1)),
                },
            })
        result = await self.critique(pseudo_experiences)
        logger.info(
            "synthesize_from_failure_cluster: cluster=%d patterns=%d hints=%d",
            len(cluster_facts),
            len(result.failure_patterns),
            len(result.improvement_hints),
        )
        return result

    def _log_critique(self, result: CritiqueResult, elapsed: float) -> None:
        """デバッグログに批評結果を記録"""
        dl = self._debug_logger
        if dl:
            dl.log_learning_cycle(
                cycle_num=0,
                data={
                    "phase": "critique_synthesis",
                    "source": result.source,
                    "failure_patterns": result.failure_patterns,
                    "improvement_hints": result.improvement_hints,
                    "summary": result.summary,
                    "elapsed_sec": round(elapsed, 3),
                },
            )
