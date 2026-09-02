"""FadeMem 適応的忘却スコア + メモリ Eviction"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.log_config import get_logger
from backend.policy_helpers import get_policy_value

if TYPE_CHECKING:
    from backend.free.core.policy_interpreter import PolicyInterpreter

logger = get_logger("memory.lightmem_scorer")


# タグ別半減期 (日単位) のデフォルト値
# `failure_pattern` は中期、`personal_fact` / `belief` は長期保持。
DEFAULT_HALF_LIFE_DAYS_BY_TAG: dict[str, float] = {
    "personal_fact": 90,
    "world_fact": 14,
    "preference": 60,
    "emotion": 30,
    "opinion": 21,
    "belief": 180,
    "decision": 60,
    "commitment": 30,
    "project": 9999,
    "create": 60,
    "task": 14,
    "create_task": 14,  # / D4: CreateExtractor 由来の task
    "policy": 9999,
    "fewshot": 9999,    # policy subtype から独立
    "failure_pattern": 60,
    "progress_marker": 9999,
    "artifact": 30,     # ラルフループの成果物トレース (中短期)
}


class FadeMemScorer:
    """FadeMem 適応的忘却スコア（arXiv:2601.18642）に基づく重要度計算"""

    def __init__(self, config: dict, policy: PolicyInterpreter | None = None):
        self._policy = policy
        mc = config.get("memory", {})
        # PolicyInterpreter 優先、フォールバックは config
        self.alpha: float = self._p("fade_alpha", mc.get("fade_alpha", 0.4))
        self.beta: float = self._p("fade_beta", mc.get("fade_beta", 0.3))
        self.gamma: float = self._p("fade_gamma", mc.get("fade_gamma", 0.3))
        self.threshold: float = self._p("fade_threshold", mc.get("fade_threshold", 0.15))
        decay_days = self._p("decay_days", mc.get("lightmem_decay_days", 7))
        self.default_half_life_days: float = float(decay_days)
        self.half_life: float = float(decay_days) * 86400

        # タグ別半減期 (秒単位) を構築
        # config.yaml > policy 上書き > DEFAULT_HALF_LIFE_DAYS_BY_TAG の優先順。
        tag_overrides = mc.get("half_life_days_by_tag", {})
        merged_days: dict[str, float] = dict(DEFAULT_HALF_LIFE_DAYS_BY_TAG)
        if isinstance(tag_overrides, dict):
            for k, v in tag_overrides.items():
                try:
                    merged_days[str(k)] = float(v)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid half_life_days_by_tag entry: %s=%r (skipped)", k, v
                    )
        self.half_life_days_by_tag: dict[str, float] = merged_days
        self.half_life_seconds_by_tag: dict[str, float] = {
            tag: days * 86400 for tag, days in merged_days.items()
        }

    def _p(self, key: str, default: int | float) -> int | float:
        """ポリシーからパラメータ取得（フォールバック付き）"""
        return get_policy_value(self._policy, "memory", key, default)

    def _select_half_life_seconds(self, item: Any) -> float:
        """対象アイテムのタグ/型から半減期 (秒) を選ぶ。

        タグ別半減期
        - ``SemanticFact`` (``type`` 属性) はその ``type`` を直接キーに引く。
        - ``MemoryNote`` (``tags`` 属性) はマッチするタグの中で最長の半減期を採用
          (保守側に倒し、長期保持タグが 1 つでも付いていれば優先する)。
        - 該当タグがなければデフォルト (``self.half_life``)。
        """
        # SemanticFact: type フィールドが FactType
        fact_type = getattr(item, "type", None)
        if isinstance(fact_type, str) and fact_type in self.half_life_seconds_by_tag:
            return self.half_life_seconds_by_tag[fact_type]

        # MemoryNote: tags リスト
        tags = getattr(item, "tags", None)
        if isinstance(tags, (list, tuple, set)) and tags:
            matched: list[float] = []
            for t in tags:
                hl = self.half_life_seconds_by_tag.get(str(t))
                if hl is not None:
                    matched.append(hl)
            if matched:
                # 最長を採用 (保守的: 長期保持タグを優先)
                return max(matched)

        return self.half_life

    def compute(self, note, query_vec: np.ndarray | None = None) -> float:
        """重要度 I(t) = α·relevance + β·freq_term + γ·recency

        各項は [0,1] に正規化済み。O(1)、インクリメンタル更新可能。

        ``note`` は ``MemoryNote`` でも ``SemanticFact`` でも
        受け取る。recency 項の半減期は :meth:`_select_half_life_seconds` で
        タグ/型から選ぶ。
        """
        now = time.time()

        # α: Relevance — 直近クエリとの意味的類似度
        embedding = getattr(note, "embedding", None)
        if query_vec is not None and embedding is not None:
            relevance = float(np.dot(embedding, query_vec))
            relevance = max(0.0, min(1.0, relevance))
        else:
            relevance = 0.5  # クエリなし時は中立値

        # β: Frequency — 飽和関数で過剰重み付けを防止
        access_count = int(getattr(note, "access_count", 0) or 0)
        freq_term = math.log(1 + access_count) / math.log(10)
        freq_term = min(1.0, freq_term)

        # γ: Recency — Ebbinghaus 指数減衰
        created_at = float(getattr(note, "created_at", 0.0) or 0.0)
        age = now - created_at
        half_life_sec = self._select_half_life_seconds(note)
        if half_life_sec > 0:
            recency = math.exp(-0.693 * age / half_life_sec)
        else:
            recency = 1.0

        importance = (
            self.alpha * relevance
            + self.beta * freq_term
            + self.gamma * recency
        )
        return importance


def can_fade(memory_id: str, experience_buf: dict | None, config: dict) -> bool:  # noqa: ARG001
    """FadeMem 削除実行前の安全チェック。
    experience.json と紐づくメモリは score に関わらず保持する。
    """
    if experience_buf is None:
        return True

    source_ids = experience_buf.get("source_memory_ids", [])
    pending_ids = experience_buf.get("pending_memory_ids", [])

    if memory_id in source_ids:
        return False
    if memory_id in pending_ids:
        return False

    return True


class MemoryEviction:
    """メモリ Eviction: FadeMem スコア最下位のノートを削除/LTM 移行"""

    EVICTION_RATIO = 0.2  # 上限到達時のデフォルト降格比率

    def __init__(self, policy: PolicyInterpreter | None = None):
        self._policy = policy

    #: 未抽出ノートの保護を降ろすノート数の上限 (``max_notes`` 倍)。
    #:
    #: Step 8 が一度も走らない構成 (extraction 無効 / Full が来ない) で STM が
    #: 無限に伸びるのを防ぐ安全弁。ここを超えたら保護を外し、従来どおり
    #: スコア下位から落とす。
    UNEXTRACTED_PROTECTION_CEILING = 2.0

    def evict(
        self,
        short_term,
        long_term,
        experience_buf: dict | None,
        scorer: FadeMemScorer,
        config: dict,
        unextracted_cutoff: float | None = None,
    ) -> int:
        """Eviction を実行。削除/降格したノート数を返す。

        ``unextracted_cutoff`` に「最後に Step 8 (ファクト抽出) が走った時刻」を
        渡すと、それより後に作られたノートを **降格対象から外す**。

        なぜ必要か: Full sleep-time は直前に ``snapshot_unabsorbed`` /
        ``drain_evicted_to_stm`` で会話ターンを STM へ流し込み、その STM ノートを
        Step 8 の入力にする。ところが Step 4 (本 eviction) は run_light の中に
        あり **Step 8 より先に走る**。新しいノートは ``access_count=0`` で
        frequency 項が 0 になるため FadeMem スコアが下位に固まり、
        「流し込んだ 20 件がそのまま下位 20% として降格される」形になっていた。

        実測 (2026-08-19 ライブ検証、chat 40 ターン / 4 セッション):

        - 17:25:14 に 20 ターンを STM へ吸収 → 17:25:15 に同じ id が
          ``Demoted note ... to LTM (score=0.431)`` で降格
        - セッション切替で吸収したテーマ B (16:55) / テーマ C (17:13) の
          ノートは、次の Light サイクルで降格され **1 件も残らなかった**
        - 結果 Step 8 の chat 抽出は **40 ターンで 1 ファクト**、訂正ターンは
          SemMem に一度も届かなかった

        降格自体は LTM への移動なのでエピソード記憶は失われないが、SemMem の
        入力は STM ノートだけなので、抽出前に落とすと意味記憶が育たない。

        ``None`` (既定) では従来どおり保護しない。
        """
        if len(short_term.notes) < short_term.max_notes:
            return 0
        protect_unextracted = (
            unextracted_cutoff is not None
            and len(short_term.notes)
            < short_term.max_notes * self.UNEXTRACTED_PROTECTION_CEILING
        )

        # ポリシー優先、フォールバックは config → デフォルト
        threshold = get_policy_value(
            self._policy, "memory", "fade_threshold",
            config.get("memory", {}).get("fade_threshold", 0.15),
        )
        eviction_ratio = get_policy_value(
            self._policy, "memory", "eviction_ratio", self.EVICTION_RATIO,
        )

        # FadeMem スコアで全ノートをソート
        scored = [(n, scorer.compute(n)) for n in short_term.notes.values()]
        scored.sort(key=lambda x: x[1])

        evict_count = int(len(scored) * eviction_ratio)
        removed = 0

        skipped_unextracted = 0
        for note, score in scored[:evict_count]:
            if not can_fade(note.id, experience_buf, config):
                continue
            # プライベートノートは LTM に昇格させず破棄のみ。pin より **先に**
            # 判定する — pin 保護を先に見ると pinned な private ノートが
            # セッション終了後も永久に残る (揮発する契約と矛盾する)。
            is_private = bool(getattr(note, "private", False))
            if not is_private:
                # pinned MemoryNote は常に保持
                if getattr(note, "pin_flag", False):
                    continue
                # Step 8 がまだ見ていないノートは SemMem の唯一の入力なので残す
                # (docstring の ``unextracted_cutoff`` 参照)。セッション別上限で
                # 見送られたノート (``extraction_deferred``) は作成時刻が cutoff
                # より前でも未消費なので同じく残す。
                if protect_unextracted and (
                    float(getattr(note, "created_at", 0.0) or 0.0)
                    > unextracted_cutoff
                    or getattr(note, "extraction_deferred", False)
                ):
                    skipped_unextracted += 1
                    continue

            if score < threshold or is_private:
                del short_term.notes[note.id]
                if is_private:
                    logger.info(
                        "Evicted private note %s (score=%.3f, no LTM promotion)",
                        note.id, score,
                    )
                else:
                    logger.info("Evicted note %s (score=%.3f < threshold)", note.id, score)
            else:
                long_term.absorb_from_short_term(note)
                del short_term.notes[note.id]
                logger.info("Demoted note %s to LTM (score=%.3f)", note.id, score)
            removed += 1

        if skipped_unextracted:
            logger.info(
                "Eviction: kept %d note(s) that Step 8 has not consumed yet "
                "(they are the only input for SemMem extraction)",
                skipped_unextracted,
            )
        if removed > 0:
            short_term.mark_dirty()

        return removed
