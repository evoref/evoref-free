"""

LogIngestor (`backend/free/loop/log_ingestor.py`) から流れてくる causal join
済 :class:`JoinedPair` を消費し、`(decision_point, chosen)` 単位で **失敗率 /
平均所要時間 / quality_signals** を集計する。閾値超過 (例: 失敗率 > 0.3 /
N >= 20) で SemMem に ``learned_failure_pattern`` ファクトを書き戻し、成功
率の高いパターンは ``learn.policy.runtime_observed.*`` namespace の policy
ファクトとして書き戻す。

責務範囲 (CLAUDE.md §8 — pillar 境界):

* :class:`backend.free.memory.views.learn.LearnFactView` 経由のみで SemMem
  に書き込む (4 pillar 境界遵守)
* PolicyParamEvolver (Level 2) との **二重書き込み回避**: 既存 namespace
  ``learn.policy.<mode>.<domain>.<param_path>`` とは別の
  ``learn.policy.runtime_observed.<decision_point>.<chosen>`` を採用する
  に書き込む。Loop owned の ``failure_pattern`` (quality_gate 由来) とは
  origin / namespace が物理的に分離される

Aggregation 方針:

* 集約 key = ``(decision_point, chosen)`` の 2 タプル (issue 案では reason
  も含むが、reason は自由文字列でカーディナリティが高くなるため除外)
* per-key カウンタ: ``samples`` / ``failures`` / ``duration_ms_sum`` /
  ``quality_signals`` (max retry / json_repair の件数等)
* flush タイミング: ``consume()`` 内で ``samples >= 20`` の bucket を判定し、
  failure_rate >= 0.3 → ``write_learned_failure_pattern``、success_rate >=
  0.7 → ``write_policy``。発火後は same bucket を flush 済としてカウンタ
  を半減させ、後続観測で confidence を漸近的に更新する
* ``flush_all()`` で残バッファを強制 flush (shutdown / cron 由来)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from backend.free.core.session_mode import (
    is_valid_session_mode,
    normalize_session_mode,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.loop.log_ingestor import JoinedPair
    from backend.free.memory.views.learn import LearnFactView

logger = get_logger("learning.policy_adjuster")


# ──────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────


DEFAULT_MIN_SAMPLES = 20
"""policy / failure_pattern を発火するための最小サンプル数。"""

DEFAULT_FAILURE_RATE_THRESHOLD = 0.3
"""``learned_failure_pattern`` を発火する failure_rate の下限 (>=)。"""

DEFAULT_SUCCESS_RATE_THRESHOLD = 0.7
"""``policy.runtime_observed`` を発火する success_rate の下限 (>=)。"""

DEFAULT_DECAY_FACTOR = 0.5
"""flush 後に bucket カウンタを残す割合 (0.5 = 半減)。これにより同一パター
ンの観測が継続しても線形増加せず、新たな観測との混合で confidence が
漸近的に再評価される。"""

LEARN_POLICY_RUNTIME_OBSERVED_PREFIX = "learn.policy.runtime_observed"
"""PolicyAdjuster が書き戻す policy ファクトの subject prefix。

PolicyParamEvolver (Level 2) が使う ``learn.policy.<mode>.<domain>.<param>``
方針)。"""

LEARN_FAILURE_PATTERN_PREFIX = "learn.failure_pattern"
"""PolicyAdjuster が書き戻す失敗パターンの subject prefix。

Loop owned の ``loop.failure.*`` (quality_gate 由来) とは prefix・FactType
共に独立しており、共存可能。"""

# subject 構成上のパート長制限 (validate_subject_namespace 互換)
_SUBJECT_PART_MAX = 64
_SUBJECT_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_SUBJECT_PART_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_\-]")


# ──────────────────────────────────────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _Bucket:
    """1 つの ``(decision_point, chosen)`` キーの集約状態。

    Attributes:
        decision_point: 観測 decision_point。
        chosen: 観測 chosen 値。
        samples: 観測数 (decision + outcome ペア)。
        failures: ``outcome.success == False`` だったペアの数。
        duration_ms_sum: ``outcome.duration_ms`` の合計 (測定可能だったペアのみ)。
        duration_count: duration を観測したペアの数 (平均算出用)。
        retry_max: ``quality_signals.retry_count`` の最大値。
        json_repair_count: ``quality_signals.json_repair_count > 0`` の件数。
        orphan_count: outcome 欠損 (orphan) の数。
        last_mode: 最後に観測した ``mode_origin``。
        last_trace_id: 最後に観測した trace_id (デバッグ用)。
        last_observed_ts: 最後に観測した monotonic 時刻 (idle 検出用)。
    """

    decision_point: str
    chosen: str
    samples: int = 0
    failures: int = 0
    duration_ms_sum: float = 0.0
    duration_count: int = 0
    retry_max: int = 0
    json_repair_count: int = 0
    orphan_count: int = 0
    last_mode: str = "chat"
    last_trace_id: str = ""
    last_observed_ts: float = 0.0

    @property
    def total_with_outcome(self) -> int:
        """outcome を伴う観測数 (orphan 除外)。"""
        return self.samples - self.orphan_count

    @property
    def failure_rate(self) -> float:
        """outcome 観測中に占める failure 割合。outcome 0 件のとき 0.0。"""
        n = self.total_with_outcome
        if n <= 0:
            return 0.0
        return self.failures / n

    @property
    def success_rate(self) -> float:
        """outcome 観測中に占める success 割合。outcome 0 件のとき 0.0。"""
        n = self.total_with_outcome
        if n <= 0:
            return 0.0
        return (n - self.failures) / n

    @property
    def avg_duration_ms(self) -> float | None:
        """duration を観測した平均 (ms)。観測 0 件のとき ``None``。"""
        if self.duration_count <= 0:
            return None
        return self.duration_ms_sum / self.duration_count


# ──────────────────────────────────────────────────────────────────────────
# subject 構築ヘルパ
# ──────────────────────────────────────────────────────────────────────────


def sanitize_subject_part(value: str, *, fallback: str = "x") -> str:
    """subject の 1 パートとして安全な文字列に正規化する。

    ``[A-Za-z0-9][A-Za-z0-9_\\-]*`` を満たすよう、許容外文字を ``_`` に置換し、
    先頭が非英数字の場合は接頭 ``p`` を付ける。長さは ``_SUBJECT_PART_MAX``
    に切り詰める。空入力時は ``fallback`` を返す。
    """
    if not value:
        return fallback
    sanitized = _SUBJECT_PART_SANITIZE_RE.sub("_", value)
    sanitized = sanitized[:_SUBJECT_PART_MAX]
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "p" + sanitized
        sanitized = sanitized[:_SUBJECT_PART_MAX]
    if not _SUBJECT_PART_RE.match(sanitized):
        return fallback
    return sanitized


def make_runtime_observed_subject(decision_point: str, chosen: str) -> str:
    """``learn.policy.runtime_observed.<decision_point>.<chosen>`` を構築する。

    PolicyParamEvolver の ``learn.policy.<mode>.<domain>.<param>`` とは
    namespace が物理的に分離されているため、二重書き込み判定が必要ない。
    """
    return ".".join([
        LEARN_POLICY_RUNTIME_OBSERVED_PREFIX,
        sanitize_subject_part(decision_point, fallback="dp"),
        sanitize_subject_part(chosen, fallback="ch"),
    ])


def make_failure_pattern_subject(decision_point: str, chosen: str) -> str:
    """``learn.failure_pattern.<decision_point>.<chosen>`` を構築する。"""
    return ".".join([
        LEARN_FAILURE_PATTERN_PREFIX,
        sanitize_subject_part(decision_point, fallback="dp"),
        sanitize_subject_part(chosen, fallback="ch"),
    ])


# ──────────────────────────────────────────────────────────────────────────
# PolicyAdjuster
# ──────────────────────────────────────────────────────────────────────────


class PolicyAdjuster:
    """LogIngestor 出力を消費し、SemMem に集約結果を書き戻す。

    使用例::

        adjuster = PolicyAdjuster(learn_view=...)
        async for pair in ingestor.stream_pairs():
            await adjuster.consume(pair)
        # shutdown 時 (or 定期 cron):
        await adjuster.flush_all()

    本クラスは Learn pillar 内に閉じ、SemMem 書込は ``learn_view``
    (LearnFactView) 経由のみ。LogIngestor (Loop pillar) からは
    :class:`~backend.free.loop.log_ingestor.JoinedPair` 型のみを参照する
    (Learn → Loop の依存は許可、CLAUDE.md §8)。
    """

    def __init__(
        self,
        learn_view: "LearnFactView",
        *,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        failure_rate_threshold: float = DEFAULT_FAILURE_RATE_THRESHOLD,
        success_rate_threshold: float = DEFAULT_SUCCESS_RATE_THRESHOLD,
        decay_factor: float = DEFAULT_DECAY_FACTOR,
        scope: str = "global",
    ) -> None:
        if min_samples <= 0:
            raise ValueError(f"min_samples must be positive, got {min_samples}")
        if not (0.0 <= failure_rate_threshold <= 1.0):
            raise ValueError(
                f"failure_rate_threshold must be in [0.0, 1.0], "
                f"got {failure_rate_threshold}",
            )
        if not (0.0 <= success_rate_threshold <= 1.0):
            raise ValueError(
                f"success_rate_threshold must be in [0.0, 1.0], "
                f"got {success_rate_threshold}",
            )
        if not (0.0 <= decay_factor < 1.0):
            raise ValueError(
                f"decay_factor must be in [0.0, 1.0), got {decay_factor}",
            )
        self.learn_view = learn_view
        self.min_samples = min_samples
        self.failure_rate_threshold = failure_rate_threshold
        self.success_rate_threshold = success_rate_threshold
        self.decay_factor = decay_factor
        self.scope = scope

        # (decision_point, chosen) -> _Bucket
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        # 観測カウンタ (テスト / デバッグ用)
        self._stats: dict[str, int] = {
            "consumed": 0,
            "orphans_seen": 0,
            "policy_emitted": 0,
            "failure_pattern_emitted": 0,
            "skipped_no_decision_point": 0,
        }

    # ------------------------------------------------------------------
    # 観測取得 (テスト / デバッグ用)
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def buckets(self) -> dict[tuple[str, str], _Bucket]:
        """**テスト用** バケット参照 (read-only)。"""
        return dict(self._buckets)

    # ------------------------------------------------------------------
    # 主 API: consume / flush_all
    # ------------------------------------------------------------------

    async def consume(self, pair: "JoinedPair") -> None:
        """1 件の :class:`JoinedPair` を吸収し、必要なら SemMem に書き戻す。

        ``decision_point`` を持たない (フィールド欠損 / 空文字列の) decision
        は skip。outcome が None (orphan) の場合は失敗扱いではなく ``orphan``
        として別カウントし、failure_rate には含めない (orphan は未確定の
        観測であって失敗ではないため)。
        """
        self._stats["consumed"] += 1
        decision = pair.decision
        decision_point = decision.get("decision_point") or ""
        chosen = decision.get("chosen") or ""
        if not decision_point or not chosen:
            self._stats["skipped_no_decision_point"] += 1
            return

        key = (decision_point, chosen)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(decision_point=decision_point, chosen=chosen)
            self._buckets[key] = bucket

        bucket.samples += 1
        bucket.last_observed_ts = time.monotonic()
        bucket.last_trace_id = pair.trace_id
        # mode_origin は decision の context に含まれていれば反映
        ctx = decision.get("context") if isinstance(decision.get("context"), dict) else {}
        mode = ctx.get("mode") if isinstance(ctx, dict) else None
        if is_valid_session_mode(mode):
            bucket.last_mode = mode

        if pair.is_orphan:
            bucket.orphan_count += 1
            self._stats["orphans_seen"] += 1
        else:
            outcome = pair.outcome or {}
            success = bool(outcome.get("success", True))
            if not success:
                bucket.failures += 1
            duration_ms = outcome.get("duration_ms")
            if isinstance(duration_ms, (int, float)):
                bucket.duration_ms_sum += float(duration_ms)
                bucket.duration_count += 1
            qs = outcome.get("quality_signals")
            if isinstance(qs, dict):
                retry = qs.get("retry_count")
                if isinstance(retry, (int, float)) and int(retry) > bucket.retry_max:
                    bucket.retry_max = int(retry)
                jr = qs.get("json_repair_count")
                if isinstance(jr, (int, float)) and int(jr) > 0:
                    bucket.json_repair_count += 1

        # 閾値判定 → 必要なら emit
        await self._maybe_emit(bucket)

    async def flush_all(self) -> None:
        """全 bucket を最低サンプル数到達条件で強制 flush する。

        shutdown 時 / 定期 cron で呼ばれることを想定。閾値未達の bucket は
        書込しないため、本メソッドは「閾値到達したまま flush 待ちだった
        bucket を確実に書き出す」役割を果たす。
        """
        for bucket in list(self._buckets.values()):
            await self._maybe_emit(bucket)

    # ------------------------------------------------------------------
    # 内部実装
    # ------------------------------------------------------------------

    async def _maybe_emit(self, bucket: _Bucket) -> None:
        """閾値判定 + emit + decay の一連処理。"""
        if bucket.samples < self.min_samples:
            return

        n_with_outcome = bucket.total_with_outcome
        if n_with_outcome <= 0:
            # outcome が一度も来ていない (orphan のみ)。書込判断不能。
            return

        emitted = False
        if bucket.failure_rate >= self.failure_rate_threshold:
            await self._emit_failure_pattern(bucket)
            emitted = True
        elif bucket.success_rate >= self.success_rate_threshold:
            await self._emit_policy(bucket)
            emitted = True

        if emitted:
            self._decay_bucket(bucket)

    async def _emit_failure_pattern(self, bucket: _Bucket) -> None:
        subject = make_failure_pattern_subject(bucket.decision_point, bucket.chosen)
        payload = self._build_payload(bucket, kind="failure")
        try:
            existing = self.learn_view.find_active_learned_failure_pattern_by_subject(
                subject,
            )
        except Exception as exc:
            logger.warning(
                "find_active_learned_failure_pattern failed for %s: %s",
                subject, exc,
            )
            existing = None
        try:
            new_fact = self.learn_view.write_learned_failure_pattern(
                subject=subject,
                predicate="indicates_failure",
                object_=payload,
                scope=self.scope,
                confidence=bucket.failure_rate,
                mode_origin=normalize_session_mode(bucket.last_mode),  # type: ignore[arg-type]
                trace_id=bucket.last_trace_id or None,
            )
        except Exception as exc:
            logger.warning(
                "write_learned_failure_pattern failed for %s: %s", subject, exc,
            )
            return
        if existing is not None:
            try:
                self.learn_view.supersede_learned_failure_pattern(
                    old_id=existing.id, new_id=new_fact.id,
                )
            except Exception as exc:
                logger.warning(
                    "supersede_learned_failure_pattern failed: %s", exc,
                )
        self._stats["failure_pattern_emitted"] += 1

    async def _emit_policy(self, bucket: _Bucket) -> None:
        subject = make_runtime_observed_subject(bucket.decision_point, bucket.chosen)
        payload = self._build_payload(bucket, kind="success")
        try:
            existing = self.learn_view.find_active_policy_by_subject(subject)
        except Exception as exc:
            logger.warning(
                "find_active_policy_by_subject failed for %s: %s", subject, exc,
            )
            existing = None
        try:
            new_fact = self.learn_view.write_policy(
                subject=subject,
                predicate="prefer",
                object_=payload,
                scope=self.scope,
                confidence=bucket.success_rate,
                mode_origin=normalize_session_mode(bucket.last_mode),  # type: ignore[arg-type]
                auto_evolved=True,
                trace_id=bucket.last_trace_id or None,
            )
        except Exception as exc:
            logger.warning("write_policy failed for %s: %s", subject, exc)
            return
        if existing is not None:
            try:
                self.learn_view.supersede_policy(
                    old_id=existing.id, new_id=new_fact.id,
                )
            except Exception as exc:
                logger.warning("supersede_policy failed: %s", exc)
        self._stats["policy_emitted"] += 1

    def _build_payload(self, bucket: _Bucket, *, kind: str) -> str:
        """SemanticFact.object に格納する JSON ペイロードを構築する。"""
        payload: dict[str, Any] = {
            "kind": kind,
            "decision_point": bucket.decision_point,
            "chosen": bucket.chosen,
            "samples": bucket.samples,
            "failures": bucket.failures,
            "orphans": bucket.orphan_count,
            "failure_rate": round(bucket.failure_rate, 4),
            "success_rate": round(bucket.success_rate, 4),
        }
        if bucket.avg_duration_ms is not None:
            payload["avg_duration_ms"] = round(bucket.avg_duration_ms, 3)
        quality_signals: dict[str, Any] = {}
        if bucket.retry_max > 0:
            quality_signals["retry_max"] = bucket.retry_max
        if bucket.json_repair_count > 0:
            quality_signals["json_repair_count"] = bucket.json_repair_count
        if quality_signals:
            payload["quality_signals"] = quality_signals
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _decay_bucket(self, bucket: _Bucket) -> None:
        """flush 後に bucket カウンタを decay_factor 倍に縮減する。

        後続観測との混合で confidence が漸近的に再評価され、同一パターンが
        繰り返し observed されても線形増加せず、新たな観測の重みが相対的
        に維持される。
        """
        f = self.decay_factor
        bucket.samples = int(bucket.samples * f)
        bucket.failures = int(bucket.failures * f)
        bucket.orphan_count = int(bucket.orphan_count * f)
        bucket.duration_ms_sum *= f
        bucket.duration_count = int(bucket.duration_count * f)
        bucket.json_repair_count = int(bucket.json_repair_count * f)
        # retry_max は max 値なので decay しない (履歴の最悪値を保持)


__all__ = [
    "DEFAULT_DECAY_FACTOR",
    "DEFAULT_FAILURE_RATE_THRESHOLD",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_SUCCESS_RATE_THRESHOLD",
    "LEARN_FAILURE_PATTERN_PREFIX",
    "LEARN_POLICY_RUNTIME_OBSERVED_PREFIX",
    "PolicyAdjuster",
    "make_failure_pattern_subject",
    "make_runtime_observed_subject",
    "sanitize_subject_part",
]
