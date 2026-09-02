"""

ラルフループ が SemMem に書く ``progress_marker`` / ``failure_pattern`` /
``artifact`` を **Level 1 / Level 2 tick** の冒頭で走査し、以下の 3
経路で学習サイクル (FewShotPool / CritiqueSynthesizer / PolicyParamEvolver)
へ自動還流する。

経路 1: 成功 ``progress_marker`` + ``task`` + ``artifact`` → FewShotPool
経路 2: 連続 ``failure_pattern`` クラスタ → CritiqueSynthesizer →
        ``policy`` ファクト書き戻し
経路 3: ``progress_marker`` / ``failure_pattern`` 比率 → PolicyParamEvolver
        の fitness に実環境成功率項を加重合成

設計方針:
- LLM 不要。SemMem の type インデックス走査 + ルールベース集計のみ。
- 1 tick あたりの走査件数を ``max_scan_per_tick`` で上限制御 (暴走防止)。
- 循環検出: 同一 failure subject に対して短期間に複数回 critique を書き出す
  と log 警告 (policy → policy の自己ループ抑制)。
- 後方互換不要。
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from backend.free.memory.extractors.mdp_trace import LOOP_FAILURE_PREFIX
from backend.free.memory.types import make_fact
from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.learning.critique_synthesizer import CritiqueSynthesizer
    from backend.free.learning.fewshot_pool import FewShotPool
    from backend.free.memory.types import SemanticFact
    from backend.free.memory.views.learn import LearnFactView

logger = get_logger("learning.feedback_pipe")


# ──────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────

#: 経路 2 で書き戻す policy ファクトの subject prefix (レガシー形、モデル無し)。
#: owner pillar は EvorefLearn。partition 有効時は :func:`build_critique_policy_subject`
#: が ``learn.policy.<model>.<mode>.critique.<sig>`` を組む。
LEARN_CRITIQUE_POLICY_PREFIX = "learn.policy.critique."

#: critique policy の mode_origin (経路 2 はラルフループ由来なので create 固定)。
CRITIQUE_POLICY_MODE = "create"


def build_critique_policy_subject(base_model_id: str, signature: str) -> str:
    """critique policy の subject を組む。

    ``PolicyInterpreter._apply_semmem_overrides`` は partition 有効時
    ``learn.policy.<model>.`` prefix でしか policy を読まない。旧 subject
    ``learn.policy.critique.<sig>`` は <model>.<mode> セグメントを欠いていて
    **一度も読まれていなかった** (2026-09-02 監査 R-F1)。PolicyEvolver /
    FewShotPool と同じ「``base_model_id`` 空 = レガシー縮退」の規則に揃える。
    """
    if base_model_id:
        return (
            f"learn.policy.{base_model_id}.{CRITIQUE_POLICY_MODE}"
            f".critique.{signature}"
        )
    return f"{LEARN_CRITIQUE_POLICY_PREFIX}{signature}"

#: 経路 2 で書き戻す policy ファクトの predicate
CRITIQUE_POLICY_PREDICATE = "mitigation_for"

#: 経路 2 policy ファクトの初期 confidence (activation 閾値 0.7 より低く抑える)
CRITIQUE_POLICY_CONFIDENCE_INIT = 0.5


class FeedbackPipe:
    """品質ゲート結果 → 学習サイクル 還流パイプ

    :class:`~backend.free.memory.views.learn.LearnFactView` 経由に一本化。
    pillar 境界 (CLAUDE.md §8) を侵犯せず Learn の readers 範囲内で
    progress_marker / failure_pattern / artifact / task を読み、経路 2 の
    critique policy 書込は LearnFactView の owner チェックを通過する。

    Args:
        config: ``learning.feedback_pipe`` セクション dict
        learn_view: Learn pillar の Fact View (全 pillar 読取 + policy/fewshot
            書込)。``None`` 可 (bootstrap 完了前 / lifespan 後注入)。
        writeback_scope: 書込時の ``scope`` (``global`` / ``project:<id>``)
        fewshot_pool: 経路 1 の書き込み先 (任意)
        critique_synthesizer: 経路 2 の批評生成器 (任意)
        debug_logger: DebugLogger (任意)
    """

    def __init__(
        self,
        config: dict,
        *,
        learn_view: LearnFactView | None = None,
        writeback_scope: str = "global",
        fewshot_pool: FewShotPool | None = None,
        critique_synthesizer: CritiqueSynthesizer | None = None,
        debug_logger: DebugLogger | None = None,
    ) -> None:
        self._enabled: bool = bool(config.get("enabled", False))
        self._max_scan: int = int(config.get("max_scan_per_tick", 50))
        self._progress_lookback: int = int(
            config.get("progress_marker_lookback", 20),
        )
        self._failure_lookback: int = int(
            config.get("failure_pattern_lookback", 30),
        )
        self._failure_cluster_min_size: int = int(
            config.get("failure_cluster_min_size", 3),
        )
        self._weight_semmem_success: float = float(
            config.get("weight_semmem_success", 0.3),
        )

        self._learn_view: LearnFactView | None = learn_view
        self._writeback_scope: str = writeback_scope
        self._fewshot_pool = fewshot_pool
        self._critique = critique_synthesizer
        self._debug_logger = debug_logger
        # base 学習パーティションの active モデルスラグ。空 = レガシー subject。
        self._base_model_id: str = ""

        # 循環検出: 本 run 内で critique 書き戻しを試みた failure cluster
        # signature の集合。2 回目以降は skip + warn する。
        self._critiqued_signatures_this_run: set[str] = set()

    # ── アクセサ ─────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def weight_semmem_success(self) -> float:
        return self._weight_semmem_success

    def set_base_model_id(self, base_model_id: str) -> None:
        """base 学習パーティションの active モデルスラグを差し替える。

        以後の critique policy 書き戻しは ``learn.policy.<model>.create.critique.*``
        を subject にする (PolicyEvolver / FewShotPool と同じ契約)。空文字で
        レガシー縮退。
        """
        self._base_model_id = base_model_id or ""

    def set_components(
        self,
        *,
        fewshot_pool: FewShotPool | None = None,
        critique_synthesizer: CritiqueSynthesizer | None = None,
    ) -> None:
        """FewShotPool / CritiqueSynthesizer を後注入する"""
        if fewshot_pool is not None:
            self._fewshot_pool = fewshot_pool
        if critique_synthesizer is not None:
            self._critique = critique_synthesizer

    # ── 内部: LearnFactView 経由の走査ヘルパ ──────────────────────────

    def _collect_by_type(
        self, fact_type: str, limit: int,
    ) -> list[SemanticFact]:
        """LearnFactView 経由で ``fact_type`` を created_at 降順で最大 ``limit`` 件返す。"""
        if self._learn_view is None:
            return []
        remaining = min(limit, self._max_scan)
        try:
            return self._learn_view.list_facts_by_type(
                fact_type,
                limit=remaining,
                order="created_at_desc",
            )
        except (ValueError, AttributeError):
            return []

    @staticmethod
    def _parse_object_json(text: str) -> dict[str, Any]:
        if not text:
            return {}
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return {"raw": text}
        return data if isinstance(data, dict) else {}

    # ── 経路 1: progress_marker + artifact → FewShotPool ────────────

    def route_progress_to_fewshot(self) -> int:
        """経路 1: 成功 progress_marker + 関連 artifact → FewShotPool。

        Returns:
            採用された FewShotExample 数
        """
        if self._fewshot_pool is None:
            return 0
        markers = self._collect_by_type(
            "progress_marker", self._progress_lookback,
        )
        if not markers:
            return 0

        # 関連 artifact を task_id インデックスに落とし込む
        artifacts_by_task: dict[str, list[SemanticFact]] = defaultdict(list)
        for art in self._collect_by_type("artifact", self._max_scan):
            obj = self._parse_object_json(art.object)
            tid = obj.get("related_task_id")
            if tid:
                artifacts_by_task[str(tid)].append(art)

        # task ファクトを lookup 用に
        task_titles_by_id: dict[str, str] = {}
        for task_fact in self._collect_by_type("task", self._max_scan):
            obj = self._parse_object_json(task_fact.object)
            tid = obj.get("task_id")
            if tid:
                task_titles_by_id[str(tid)] = str(
                    obj.get("title") or obj.get("description") or "",
                )

        accepted = 0
        for marker in markers:
            m_obj = self._parse_object_json(marker.object)
            if m_obj.get("status") != "done":
                continue
            task_id = str(m_obj.get("task_id", ""))
            if not task_id:
                continue
            related_arts = artifacts_by_task.get(task_id, [])
            passed_arts = []
            for art in related_arts:
                a_obj = self._parse_object_json(art.object)
                if a_obj.get("gate_passed"):
                    passed_arts.append((art, a_obj))
            if not passed_arts:
                continue

            query = task_titles_by_id.get(task_id) or str(m_obj.get("title", ""))
            if not query:
                continue
            response_parts = [
                f"- {a_obj.get('file_path', '?')} "
                f"(+{a_obj.get('lines_added', 0)}/-{a_obj.get('lines_removed', 0)} "
                f"sha1={a_obj.get('diff_sha1', '')[:8]})"
                for _, a_obj in passed_arts[:5]
            ]
            response = "Applied edits:\n" + "\n".join(response_parts)

            ex = self._fewshot_pool.accept_from_artifact(
                query=query,
                response=response,
                mode="create",
                fitness=float(marker.confidence),
                added_at=utc_now(),
            )
            if ex is not None:
                accepted += 1

        if accepted and self._debug_logger is not None:
            self._debug_logger.log_learning_cycle(
                cycle_num=0,
                data={
                    "component": "feedback_pipe",
                    "feedback_route": 1,
                    "scanned_markers": len(markers),
                    "accepted_fewshot": accepted,
                },
            )
        return accepted

    # ── 経路 2: failure_pattern クラスタ → Critique → policy ───────────

    async def route_failures_to_critique(self) -> int:
        """経路 2: 連続 failure_pattern クラスタから critique を生成し policy 書き戻し。

        Returns:
            書き戻した policy ファクト数
        """
        if self._critique is None:
            return 0
        failures = self._collect_by_type(
            "failure_pattern", self._failure_lookback,
        )
        if len(failures) < self._failure_cluster_min_size:
            return 0

        # subject prefix でクラスタ化 (同一 failure_signature は prefix 一致)
        clusters: dict[str, list[SemanticFact]] = defaultdict(list)
        for fact in failures:
            if not fact.subject.startswith(LOOP_FAILURE_PREFIX):
                continue
            # signature (= subject のサフィックス) をキーに纏める
            signature = fact.subject[len(LOOP_FAILURE_PREFIX):]
            clusters[signature].append(fact)

        written = 0
        for signature, cluster_facts in clusters.items():
            if len(cluster_facts) < self._failure_cluster_min_size:
                continue
            if signature in self._critiqued_signatures_this_run:
                logger.warning(
                    "feedback_pipe: circular critique detected signature=%s "
                    "(skipping to avoid policy->policy self-loop)",
                    signature,
                )
                continue

            cluster_dicts = [
                {
                    "subject": f.subject,
                    "object": f.object,
                    "signature": signature,
                }
                for f in cluster_facts
            ]
            result = await self._critique.synthesize_from_failure_cluster(
                cluster_dicts,
            )
            if not result.improvement_hints:
                continue
            fact = self._writeback_critique_policy(signature, result)
            if fact is not None:
                written += 1
                self._critiqued_signatures_this_run.add(signature)

        if written and self._debug_logger is not None:
            self._debug_logger.log_learning_cycle(
                cycle_num=0,
                data={
                    "component": "feedback_pipe",
                    "feedback_route": 2,
                    "clusters_critiqued": written,
                    "failures_scanned": len(failures),
                },
            )
        return written

    def _writeback_critique_policy(
        self,
        signature: str,
        result: Any,
    ) -> SemanticFact | None:
        """critique 結果を ``policy`` ファクトとして書き戻す (LearnFactView 経由)。"""
        view = self._learn_view
        if view is None:
            return None
        subject = build_critique_policy_subject(self._base_model_id, signature)
        payload = json.dumps(
            {
                "failure_signature": signature,
                "improvement_hints": list(result.improvement_hints),
                "failure_patterns": list(result.failure_patterns),
                "summary": result.summary,
                "source": result.source,
            },
            ensure_ascii=False,
        )
        new_fact = make_fact(
            subject=subject,
            predicate=CRITIQUE_POLICY_PREDICATE,
            object_=payload,
            type="policy",
            scope=self._writeback_scope,
            mode_origin=CRITIQUE_POLICY_MODE,
            confidence=CRITIQUE_POLICY_CONFIDENCE_INIT,
            auto_evolved=True,
        )
        try:
            return view.add_policy_fact(new_fact)
        except ValueError as exc:
            logger.warning(
                "feedback_pipe: critique policy writeback failed subject=%s err=%s",
                subject, exc,
            )
            return None

    # ── 経路 3: SemMem 成功率プロバイダ ────────────────────────────────

    def compute_semmem_success_rate(
        self, domain: str, mode: str,
    ) -> float | None:
        """``(domain, mode)`` に対する実環境成功率を返す。

        現状の実装では domain / mode に依らず、プロジェクトスコープ全体の
        ``progress_marker`` 件数と ``failure_pattern`` 件数の比率を返す。
        対象ファクトが 0 件なら ``None`` を返す (fitness 補正しない)。
        """
        if not self._enabled or self._weight_semmem_success <= 0.0:
            return None
        if self._learn_view is None:
            return None
        try:
            prog_count = self._learn_view.count_facts_by_type("progress_marker")
            fail_count = self._learn_view.count_facts_by_type("failure_pattern")
        except (AttributeError, ValueError):
            return None
        total = prog_count + fail_count
        if total == 0:
            return None
        rate = prog_count / total
        if self._debug_logger is not None:
            self._debug_logger.log_learning_cycle(
                cycle_num=0,
                data={
                    "component": "feedback_pipe",
                    "feedback_route": 3,
                    "domain": domain,
                    "mode": mode,
                    "progress_count": prog_count,
                    "failure_count": fail_count,
                    "success_rate": rate,
                },
            )
        return rate

    # ── エントリポイント ──────────────────────────────────────────────

    async def run(self, *, cycle_num: int = 0) -> dict:
        """3 経路をまとめて実行する。

        経路 3 (fitness 補正) はプロバイダ参照型のため、ここでは実行せず
        ``PolicyParamEvolver.set_semmem_success_provider`` 経由で
        ``evolve()`` 時に自動反映される。

        Returns:
            ``{"enabled": bool, "accepted_fewshot": int,
              "clusters_critiqued": int, "elapsed_sec": float}``
        """
        if not self._enabled:
            return {"enabled": False, "skipped": True}

        self._critiqued_signatures_this_run.clear()
        t0 = time.monotonic()
        fewshot_added = self.route_progress_to_fewshot()
        critique_written = await self.route_failures_to_critique()
        elapsed = round(time.monotonic() - t0, 3)
        summary = {
            "enabled": True,
            "accepted_fewshot": fewshot_added,
            "clusters_critiqued": critique_written,
            "elapsed_sec": elapsed,
        }
        logger.info(
            "feedback_pipe.run: cycle=%d fewshot+=%d critique+=%d elapsed=%.3fs",
            cycle_num, fewshot_added, critique_written, elapsed,
        )
        return summary


__all__ = [
    "CRITIQUE_POLICY_CONFIDENCE_INIT",
    "CRITIQUE_POLICY_PREDICATE",
    "FeedbackPipe",
    "LEARN_CRITIQUE_POLICY_PREFIX",
]
