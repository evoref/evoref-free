"""効果の死活監視 — 「正常に動いて、何もしていない」を見える化する台帳 (横断基盤)。

**なぜ要るか**: 記憶・学習の欠陥で最も多い型は、例外にならず、サイクルは成功で
終わり、WARNING が 1 行残るだけの「効果ゼロ」だった (2026-09-21 に監査記録
55 件を分類: 14 件 / 25% が最多、全件が手動監査かログの遡りで発見、テストでの
発見は 0 件)。例外の分類では拾えない — 14 件中、例外だったのは 2〜3 件だけ。

**設計** (SSOT: docs/c_07 §7.1):

- 段 (stage) ごとに「届いたか」「何件の効果を生んだか」「判定点がどちらに倒れたか」
  を数える。各段に手を入れず、結果が 1 箇所に集まる出口 (sleep-time の入口、
  Level 1 の後処理、Level 2 の記録、判定点の記録) で読む。
- 警告の状態は保存しない。読むたびに台帳から導出する (保存した状態と台帳が
  食い違う余地を作らない)。
- 警告は状態に入った時点で 1 回だけ WARNING を出す。同じ WARNING が 700 行
  並ぶとログを読まなくなる。
- 観測だけで、どの段の挙動も変えない。

``aux_telemetry`` と同じく横断基盤に置く (4 pillar のどこからでも書ける)。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.free.learning.json_state_store import JsonPayload, JsonStateStore
from backend.log_config import get_logger
from backend.utils import format_utc, utc_now_dt

logger = get_logger("liveness")

#: 以前届いていた段が、この回数だけ連続で未到達なら ``starved``。
STARVED_AFTER = 5
#: 入力 > 0 なのに効果 0 がこの回数続いたら ``stalled`` (段ごとの上書きは下)。
STALLED_AFTER = 5
#: 段の接頭辞ごとの ``stalled`` 閾値。Level 1 は収束すると正当に改善しなくなる
#: ので、1 回の空振りでは騒がない。
STALLED_AFTER_BY_PREFIX: Mapping[str, int] = {"learn.level1.": 10}
#: 連続失敗がこの回数に達したら ``failing``。
FAILING_AFTER = 3
#: 判定点の発火率を見る窓 (直近の判定件数)。
GATE_WINDOW = 200
#: この件数に満たない判定点は ``degenerate`` にしない (希少事象の判定点を誤報しない)。
GATE_MIN_DECISIONS = 100

FAILURE_CLASSES = ("transient", "unrecoverable")
_BAND_CODES = {"fire": "f", "abstain": "a", "skip": "s"}


@dataclass(frozen=True)
class LivenessAlert:
    """立っている警告 1 件。"""

    stage: str
    kind: str  # starved / stalled / degenerate / failing / emptied
    since: str  # ISO 8601 UTC ("" = 不明)
    detail: str  # 英語 (ログ用。UI は kind と数値から組み立てる)

    def as_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage, "kind": self.kind,
            "since": self.since, "detail": self.detail,
        }


def _stalled_after(stage: str) -> int:
    for prefix, threshold in STALLED_AFTER_BY_PREFIX.items():
        if stage.startswith(prefix):
            return threshold
    return STALLED_AFTER


class LivenessLedger(JsonStateStore):
    """段ごとの到達・効果・失敗・判定の台帳。

    段の記録は素の dict で持つ — 読めないキーも捨てずに次の保存へ書き戻す
    (c_05 §0.5: 未知キーを黙って落とさない)。
    """

    SCHEMA_VERSION = 1
    _state_logger = logger

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Callable[[], datetime] = utc_now_dt,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._clock = clock
        self._lock = threading.Lock()
        self._stages: dict[str, dict[str, Any]] = {}
        #: 既に WARNING を出した (stage, kind)。プロセス内だけで持つ — 再起動後に
        #: 1 回言い直すのは、見落とした人への再通知として妥当。
        self._announced: set[tuple[str, str]] = set()

    # ── 記録 ───────────────────────────────────────────────

    def _stage(self, stage: str) -> dict[str, Any]:
        rec = self._stages.get(stage)
        if rec is None:
            rec = self._stages[stage] = {}
        return rec

    def _now(self) -> str:
        return format_utc(self._clock())

    def record_run(
        self,
        stage: str,
        *,
        effect: int | None = None,
        input_count: int | None = None,
        reason: str | None = None,
    ) -> None:
        """段が実行された。``effect`` は生んだ件数、``input_count`` は入力件数。

        入力が 0 (または不明) で効果 0 は「暇なだけ」なので ``stalled`` へ数えない。
        ``reason`` は効果ゼロの理由 (``no_selection_pressure`` 等)。警告文に出す —
        「空回り」なのか「学ぶことが無い」のかを、警告を見た人が判断できるように。
        """
        with self._lock:
            rec = self._stage(stage)
            now = self._now()
            rec["runs"] = int(rec.get("runs", 0)) + 1
            rec["last_reached_at"] = now
            if reason:
                rec["last_reason"] = str(reason)
            else:
                rec.pop("last_reason", None)
            rec["missed_streak"] = 0
            rec.pop("missed_since", None)
            rec["error_streak"] = 0
            rec.pop("error_since", None)
            if effect is None:
                return
            rec["effect_total"] = int(rec.get("effect_total", 0)) + int(effect)
            if effect > 0:
                rec["last_effect_at"] = now
                rec["zero_effect_streak"] = 0
                rec.pop("zero_effect_since", None)
            elif input_count is not None and input_count > 0:
                streak = int(rec.get("zero_effect_streak", 0)) + 1
                rec["zero_effect_streak"] = streak
                rec["last_input"] = int(input_count)
                rec.setdefault("zero_effect_since", now)

    def record_missed(self, stage: str) -> None:
        """サイクルは走ったのに、この段まで届かなかった。

        一度も届いたことの無い段は数えない (期待される段の一覧を手で持たない
        ため — 手書きの一覧は腐る)。
        """
        with self._lock:
            rec = self._stages.get(stage)
            if rec is None or not rec.get("runs"):
                return
            rec["missed_streak"] = int(rec.get("missed_streak", 0)) + 1
            rec.setdefault("missed_since", self._now())

    def forget(self, stage: str) -> None:
        """段を台帳から消す (段そのものが無くなったとき)。"""
        with self._lock:
            self._stages.pop(stage, None)

    def record_error(
        self, stage: str, *, code: str, failure_class: str = "transient",
    ) -> None:
        """段が失敗した。``failure_class`` は再試行で直りうるか。"""
        if failure_class not in FAILURE_CLASSES:
            failure_class = "transient"
        with self._lock:
            rec = self._stage(stage)
            now = self._now()
            rec["error_streak"] = int(rec.get("error_streak", 0)) + 1
            rec.setdefault("error_since", now)
            rec["last_error"] = {"class": failure_class, "code": code, "at": now}

    def clear_errors(self, stage: str) -> None:
        """失敗が観測されなかった (成功を直接観測できない段の、連続失敗を切る)。"""
        with self._lock:
            rec = self._stages.get(stage)
            if rec is None:
                return
            rec["error_streak"] = 0
            rec.pop("error_since", None)

    def record_decision(self, stage: str, *, band: str, label: str) -> None:
        """判定点が 1 回判定した。窓は直近 ``GATE_WINDOW`` 件。

        恒真・恒偽は **選んだラベル** で見る (``band`` では見ない)。多クラスの
        判定点 (層の振り分け) は何かしらのラベルを必ず選ぶので、正常でも
        ``band`` は常に ``fire`` になる — 2026-09-21 の実機で最初の 3 ターンから
        ``fff`` だった。二値の判定点では「ラベルが 1 種類」は発火率 0% / 100%
        と同じ意味になる。``band`` は棄権・不発の件数を警告文に出すために残す。
        """
        code = _BAND_CODES.get(band)
        if code is None:
            return
        with self._lock:
            rec = self._stage(stage)
            window = str(rec.get("bands", "")) + code
            rec["bands"] = window[-GATE_WINDOW:]
            labels = list(rec.get("labels") or []) + [str(label)]
            rec["labels"] = labels[-GATE_WINDOW:]

    def record_size(self, stage: str, size: int) -> None:
        """ストアのレコード数。0 より大きい値から 0 へ落ちたら ``emptied``。"""
        with self._lock:
            rec = self._stage(stage)
            previous = rec.get("last_size")
            if size == 0 and isinstance(previous, int) and previous > 0:
                rec["emptied_at"] = self._now()
                rec["emptied_from"] = previous
            elif size > 0:
                rec.pop("emptied_at", None)
                rec.pop("emptied_from", None)
            rec["last_size"] = int(size)

    # ── 読み出し ─────────────────────────────────────────────

    def stages(self) -> dict[str, dict[str, Any]]:
        """段の記録のコピー (テスト / API 用)。"""
        with self._lock:
            return {name: dict(rec) for name, rec in self._stages.items()}

    def alerts(self) -> list[LivenessAlert]:
        """現在立っている警告。状態は保存せず、ここで台帳から導出する。"""
        with self._lock:
            snapshot = {name: dict(rec) for name, rec in self._stages.items()}
        found: list[LivenessAlert] = []
        for stage, rec in sorted(snapshot.items()):
            found.extend(_alerts_for(stage, rec))
        return found

    # ── 通知と永続化 ───────────────────────────────────────────

    def announce(self) -> list[LivenessAlert]:
        """新しく立った警告を WARNING で 1 回ずつ出し、消えた警告を INFO で知らせる。"""
        current = self.alerts()
        keys = {(a.stage, a.kind) for a in current}
        for alert in current:
            if (alert.stage, alert.kind) not in self._announced:
                logger.warning(
                    "Liveness: %s is %s (%s)", alert.stage, alert.kind, alert.detail,
                )
        for stage, kind in sorted(self._announced - keys):
            logger.info("Liveness: %s is no longer %s", stage, kind)
        self._announced = keys
        return current

    def flush(self) -> list[LivenessAlert]:
        """通知して、保存先があれば保存する。サイクル終端で呼ぶ。"""
        current = self.announce()
        if self._path is not None:
            self.save(self._path)
        return current

    def load_from_disk(self) -> None:
        if self._path is not None:
            self.load(self._path)

    def _to_payload(self) -> JsonPayload:
        with self._lock:
            return {"stages": {name: dict(rec) for name, rec in self._stages.items()}}

    def _from_payload(self, payload: JsonPayload) -> None:
        if not isinstance(payload, dict):
            raise TypeError(f"liveness.json must be a dict, got {type(payload).__name__}")
        stages = payload.get("stages", {})
        if not isinstance(stages, dict):
            raise TypeError("liveness.json 'stages' must be a dict")
        with self._lock:
            self._stages = {
                str(name): dict(rec)
                for name, rec in stages.items()
                if isinstance(rec, dict)
            }


def _alerts_for(stage: str, rec: Mapping[str, Any]) -> list[LivenessAlert]:
    out: list[LivenessAlert] = []

    missed = int(rec.get("missed_streak", 0))
    if missed >= STARVED_AFTER:
        out.append(LivenessAlert(
            stage, "starved", str(rec.get("missed_since", "")),
            f"not reached in the last {missed} cycles "
            f"(last reached {rec.get('last_reached_at', 'never')})",
        ))

    zero = int(rec.get("zero_effect_streak", 0))
    if zero >= _stalled_after(stage):
        out.append(LivenessAlert(
            stage, "stalled", str(rec.get("zero_effect_since", "")),
            f"no effect in {zero} consecutive runs with input "
            f"(last input {rec.get('last_input', '?')}, "
            f"last effect {rec.get('last_effect_at', 'never')}"
            + (f", reason {rec['last_reason']}" if rec.get("last_reason") else "")
            + ")",
        ))

    errors = int(rec.get("error_streak", 0))
    if errors >= FAILING_AFTER:
        last = rec.get("last_error") or {}
        out.append(LivenessAlert(
            stage, "failing", str(rec.get("error_since", "")),
            f"{errors} consecutive failures "
            f"(last: {last.get('code', '?')}, {last.get('class', '?')})",
        ))

    labels = rec.get("labels") or []
    if len(labels) >= GATE_MIN_DECISIONS and len(set(labels)) == 1:
        bands = str(rec.get("bands", ""))
        out.append(LivenessAlert(
            stage, "degenerate", "",
            f"always '{labels[0]}' in the last {len(labels)} decisions "
            f"(abstain {bands.count('a')}, skip {bands.count('s')})",
        ))

    if rec.get("emptied_at"):
        out.append(LivenessAlert(
            stage, "emptied", str(rec.get("emptied_at", "")),
            f"record count dropped from {rec.get('emptied_from', '?')} to 0",
        ))
    return out


# ── プロセス既定の台帳 ────────────────────────────────────────────
#
# 判定点 (core/predicate.py) のように AppState を持たないモジュールからも書ける
# よう、既定の台帳をモジュールに 1 つ置く。起動時に _pillar_wirer が保存先付きの
# 台帳へ差し替える。差し替え前 (テスト / 単体 import) はメモリ上だけで数える。

_default = LivenessLedger()


def ledger() -> LivenessLedger:
    """プロセス既定の台帳。"""
    return _default


def bind_ledger(instance: LivenessLedger) -> LivenessLedger:
    """既定の台帳を差し替える (起動時の配線 / テスト用)。前の台帳を返す。"""
    global _default
    previous, _default = _default, instance
    return previous


__all__ = [
    "FAILING_AFTER",
    "GATE_MIN_DECISIONS",
    "GATE_WINDOW",
    "STALLED_AFTER",
    "STARVED_AFTER",
    "LivenessAlert",
    "LivenessLedger",
    "bind_ledger",
    "ledger",
]
