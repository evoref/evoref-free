"""FadeMem コンフリクト解決: 類似ノートの統合"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np

from backend.log_config import get_logger
from backend.free.llm.model_metadata import DEFAULT_PARAMS_B
from backend.free.memory.notes.note_evolver import compute_llm_call_interval
from backend.free.memory.stores.short_term import ShortTermMemory, MemoryNote

if TYPE_CHECKING:
    from backend.debug_logger import DebugLogger
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("memory.conflict_resolver")


class ConflictResolver:
    """類似度の高いノートを LLM で統合"""

    # サーキットブレーカー閾値（batch_size=5 に合わせた設定）
    _CB_MAX_CONSECUTIVE = 2
    _CB_MAX_FAILURE_RATE = 0.5
    _CB_MIN_ATTEMPTS = 3

    # 失敗ペアの quarantine: LLM マージに繰り返し失敗するノートを一定時間
    # conflict 検出から除外し、同一ペアを毎サイクル再試行して circuit breaker を
    # 起こし続ける livelock を断つ。低速 aux が一過性に遅いだけのケースに
    # 備え cooldown 経過後は再試行する (恒久 ban にはしない)。
    _FAIL_QUARANTINE_THRESHOLD = 3
    _COOLDOWN_SECONDS = 6 * 3600.0

    def __init__(
        self,
        config: dict,
        params_b: float = DEFAULT_PARAMS_B,
        policy: PolicyInterpreter | None = None,
        debug_logger: DebugLogger | None = None,
    ):
        mc = config.get("memory", {})
        # ポリシー優先、フォールバックは config
        if policy is not None:
            try:
                self.similarity_threshold = policy.get("memory", "conflict_similarity_threshold")
            except KeyError:
                self.similarity_threshold = mc.get("conflict_similarity_threshold", 0.85)
            try:
                self.batch_size = policy.get("memory", "conflict_batch_size")
            except KeyError:
                self.batch_size = mc.get("conflict_batch_size", 5)
        else:
            self.similarity_threshold: float = mc.get("conflict_similarity_threshold", 0.85)
            self.batch_size: int = mc.get("conflict_batch_size", 5)

        # LLM スキップ閾値 (min_merge_similarity / max_per_cycle)
        cr_cfg = mc.get("conflict_resolver", {}) or {}
        self.min_merge_similarity: float = float(
            cr_cfg.get("min_merge_similarity", 0.85)
        )
        self.max_per_cycle: int = int(cr_cfg.get("max_per_cycle", 5))

        # インターバル: 明示設定があればそのまま、なければモデルサイズから自動計算
        explicit = mc.get("llm_call_interval")
        base = mc.get("llm_call_base_interval", 1.0)
        if explicit is not None:
            self.llm_call_interval: float = float(explicit)
        else:
            self.llm_call_interval = compute_llm_call_interval(base, params_b)

        # memory.jsonl へ LLM 呼び出し/スキップ件数を記録する任意ロガー
        self._debug_logger: DebugLogger | None = debug_logger

        # 直近の detect_conflicts で cooldown により除外したペア数 (stats 用)
        self._last_cooldown_skipped: int = 0

    @staticmethod
    def _in_cooldown(note: MemoryNote, now: float) -> bool:
        """ノートが conflict quarantine cooldown 中か (float epoch 比較)。"""
        until = note.conflict_cooldown_until
        return until is not None and until > now

    def detect_conflicts(self, short_term: ShortTermMemory) -> list[tuple[str, str]]:
        """コサイン類似度で類似ノートペアを検出

        quarantine cooldown 中のノートを含むペアは除外する (livelock 防止)。

        Returns:
            list of (note_id_a, note_id_b) pairs
        """
        now = time.time()
        notes = [n for n in short_term.notes.values() if n.embedding is not None]
        if len(notes) < 2:
            self._last_cooldown_skipped = 0
            return []

        pairs: list[tuple[str, str]] = []
        seen: set[frozenset[str]] = set()
        cooldown_skipped = 0

        for i, a in enumerate(notes):
            for b in notes[i + 1:]:
                sim = float(np.dot(a.embedding, b.embedding))
                if sim < self.similarity_threshold:
                    continue
                key = frozenset((a.id, b.id))
                if key in seen:
                    continue
                seen.add(key)
                # 失敗が続いて quarantine 中のペアはスキップ
                if self._in_cooldown(a, now) or self._in_cooldown(b, now):
                    cooldown_skipped += 1
                    continue
                pairs.append((a.id, b.id))
                a.conflict_candidate = True
                a.conflict_partner_id = b.id
                b.conflict_candidate = True
                b.conflict_partner_id = a.id

        self._last_cooldown_skipped = cooldown_skipped
        logger.info(
            "Detected %d conflict pairs (threshold=%.2f, cooldown_skipped=%d)",
            len(pairs), self.similarity_threshold, cooldown_skipped,
        )
        return pairs

    async def resolve_conflicts(
        self,
        short_term: ShortTermMemory,
        llm_client,
    ) -> int:
        """類似ノートを LLM で統合

        サーキットブレーカーと事前ヘルスチェックにより、
        補助タスクがビジー/停止時の長時間ブロッキングを防止する。

        Returns:
            統合されたペア数
        """
        # まず conflict_candidate フラグが立っているペアを収集
        pairs = self.detect_conflicts(short_term)
        if not pairs:
            self._log_op_stats({
                "detected_pairs": 0,
                "low_sim_skipped": 0,
                "over_cap_skipped": 0,
                "cooldown_skipped": self._last_cooldown_skipped,
                "llm_calls": 0,
                "llm_merged": 0,
                "llm_failures": 0,
                "health_skipped": False,
                "min_merge_similarity": self.min_merge_similarity,
                "max_per_cycle": self.max_per_cycle,
                "batch_size": self.batch_size,
            })
            return 0

        # batch_size と max_per_cycle の小さい方を採用
        cycle_cap = min(self.batch_size, self.max_per_cycle)
        batch = pairs[:cycle_cap]
        over_cap_skipped = max(0, len(pairs) - cycle_cap)

        stats: dict = {
            "detected_pairs": len(pairs),
            "over_cap_skipped": over_cap_skipped,
            "cooldown_skipped": self._last_cooldown_skipped,
            "min_merge_similarity": self.min_merge_similarity,
            "max_per_cycle": self.max_per_cycle,
            "batch_size": self.batch_size,
        }

        if cycle_cap == 0:
            stats["low_sim_skipped"] = 0
            stats["llm_calls"] = 0
            stats["llm_merged"] = 0
            stats["llm_failures"] = 0
            stats["health_skipped"] = False
            self._log_op_stats(stats)
            return 0

        # 事前ヘルスチェック: 補助タスクが応答不能ならスキップ
        if hasattr(llm_client, "health_check"):
            healthy = await llm_client.health_check()
            if not healthy:
                logger.warning(
                    "Aux task unhealthy, skipping conflict resolution "
                    "(%d pairs detected)",
                    len(pairs),
                )
                stats["low_sim_skipped"] = 0
                stats["llm_calls"] = 0
                stats["llm_merged"] = 0
                stats["llm_failures"] = 0
                stats["health_skipped"] = True
                self._log_op_stats(stats)
                return 0

        resolved = 0
        consecutive_failures = 0
        total_failures = 0
        total_attempts = 0
        low_sim_skipped = 0

        for id_a, id_b in batch:
            # サーキットブレーカー: 連続失敗で残りをスキップ
            if consecutive_failures >= self._CB_MAX_CONSECUTIVE:
                remaining = len(batch) - total_attempts
                logger.warning(
                    "Circuit breaker (consecutive): skipping remaining %d pairs "
                    "after %d consecutive failures",
                    remaining, consecutive_failures,
                )
                break

            # サーキットブレーカー: 総失敗率超過で停止
            if (
                total_attempts >= self._CB_MIN_ATTEMPTS
                and total_failures / total_attempts > self._CB_MAX_FAILURE_RATE
            ):
                logger.warning(
                    "Circuit breaker (failure rate): stopping at %.0f%% failure rate "
                    "(%d/%d attempts)",
                    total_failures / total_attempts * 100,
                    total_failures, total_attempts,
                )
                break

            note_a = short_term.notes.get(id_a)
            note_b = short_term.notes.get(id_b)
            if note_a is None or note_b is None:
                continue

            # 実類似度が min_merge_similarity 未満なら LLM 呼び出しを
            # スキップし低優先度扱い。detect の閾値は広めに (0.85) 候補抽出に用い、
            # LLM 統合はより厳しい閾値で絞ることで無駄な aux 呼び出しを削減する。
            if note_a.embedding is not None and note_b.embedding is not None:
                sim = float(np.dot(note_a.embedding, note_b.embedding))
                if sim < self.min_merge_similarity:
                    low_sim_skipped += 1
                    if self._debug_logger is not None:
                        self._debug_logger.log_decision(
                            decision_point="conflict_resolution_path",
                            chosen="skip_low_sim",
                            candidates=["llm_merge", "skip_low_sim", "health_check_skip"],
                            reason="similarity_below_threshold",
                            context={
                                "sim": round(sim, 4),
                                "min_merge_similarity": self.min_merge_similarity,
                            },
                            scope="cycle",
                        )
                    continue

            total_attempts += 1
            merged_content = await self._merge_with_llm(note_a, note_b, llm_client)
            if merged_content:
                # note_a を統合結果で更新、note_b を削除
                note_a.content = merged_content
                note_a.conflict_candidate = False
                note_a.conflict_partner_id = None
                note_a.evolution_pending = True  # 再進化が必要
                note_a.embedding = None  # 再埋め込みが必要
                # 統合成功したので生存ノートの失敗マーカーをリセット
                note_a.conflict_fail_count = 0
                note_a.conflict_cooldown_until = None

                del short_term.notes[id_b]
                short_term._cache_dirty = True
                resolved += 1
                consecutive_failures = 0
                logger.info("Merged notes %s + %s", id_a, id_b)
            else:
                consecutive_failures += 1
                total_failures += 1
                self._mark_merge_failure(note_a, note_b)

            # llama-server の負荷軽減のためインターバルを挿入
            if self.llm_call_interval > 0:
                await asyncio.sleep(self.llm_call_interval)

        stats["low_sim_skipped"] = low_sim_skipped
        stats["llm_calls"] = total_attempts
        stats["llm_merged"] = resolved
        stats["llm_failures"] = total_failures
        stats["health_skipped"] = False
        self._log_op_stats(stats)
        return resolved

    def _log_op_stats(self, stats: dict) -> None:
        """memory.jsonl に conflict_resolve の LLM 判定削減統計を記録"""
        dl = self._debug_logger
        if dl is None:
            return
        try:
            dl.log_memory_op("conflict_resolve", stats)
        except Exception as exc:
            logger.warning("log_memory_op(conflict_resolve) failed: %s", exc)

    def _mark_merge_failure(self, note_a: MemoryNote, note_b: MemoryNote) -> None:
        """マージ失敗を両ノートに記録し、閾値到達で cooldown を設定する。

        同一ペアを毎サイクル再試行して circuit breaker を起こし続ける livelock を
        防ぐため、繰り返し失敗するノートを一定時間 conflict 検出から除外する。
        cooldown 経過後は再び検出対象に戻り、再失敗すれば即座に再 cooldown する
        (恒久 ban ではなくバックオフ)。
        """
        now = time.time()
        for note in (note_a, note_b):
            note.conflict_fail_count += 1
            if note.conflict_fail_count >= self._FAIL_QUARANTINE_THRESHOLD:
                note.conflict_cooldown_until = now + self._COOLDOWN_SECONDS

    async def _merge_with_llm(
        self,
        note_a: MemoryNote,
        note_b: MemoryNote,
        llm_client,
    ) -> str | None:
        """LLM を使ってノートを統合"""
        # コンテンツ長を制限（コンテキストウィンドウ超過防止）
        max_len = 800
        content_a = note_a.content[:max_len]
        content_b = note_b.content[:max_len]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory consolidation assistant. "
                    "Merge the following two memory notes into a single, "
                    "concise note that preserves all important information. "
                    "Output only the merged content, no explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Note A:\n{content_a}\n\n"
                    f"Note B:\n{content_b}"
                ),
            },
        ]

        try:
            # timeout は明示せず purpose="conflict_resolution" の解決に委ねる。
            # 明示 timeout は反応的自己較正 (_calibrated_timeouts) より優先されて
            # しまい、低速 aux (iGPU + 4B) で 15s 固定のまま ReadTimeout を連発し
            # SemMem auto-merge と学習を停滞させていた。purpose 既定 15s + 較正で
            # 実レイテンシに追従させる (背景処理なので per-attempt 延長の体感影響なし)。
            gen_kwargs: dict = {
                "stream": False,
                "max_tokens": 512,
                "id_slot": getattr(llm_client, "background_slot", -1),
            }

            result = await llm_client.generate(
                messages, purpose="conflict_resolution", **gen_kwargs,
            )
            content = result["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            logger.warning(
                "LLM merge failed for %s + %s: %s: %s",
                note_a.id, note_b.id, type(e).__name__, e,
            )
            return None
