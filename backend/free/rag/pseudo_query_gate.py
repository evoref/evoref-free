"""疑似クエリが元チャンクで答えられるかの判定点 ``pseudo_query_answerable`` (c_17 §3.15)。

sleep-time Step 5.9 (f_01 §6.4) が生成した問いと、その元チャンクの組をリランカー
(cross-encoder) で採点した logit から、問いを索引に **書いてよいか** を確認する。

- ``fire`` (``answerable``) — 確信をもって残す
- ``skip`` (``none``) — 確信をもって反対する (元チャンクが答えていない)。呼出側は書かない
- 棄権 — 判断できない (較正が無い / 採点が縮退 / 確信の持てない帯)。呼出側は現行どおり書く

閾値は **リランカーの model_key 単位の較正ファイル** (``cache/pseudo_query_gate.json``) から
しか来ない。静的な既定を持たないので、較正前は必ず棄権 = 現行と同じ (全部書く)。

較正 (:func:`compute_pseudo_query_gate_calibration`) は split conformal: 正例 (元チャンクが
答えている問い) の logit の下側分位を ``drop_below`` に置き、答えのある問いが落とされる確率を
α 以下に保つ。純関数で、較正の担当が材料 (``decision.jsonl`` の ``context.detail``) から呼ぶ。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Verdict,
    register_predicate,
)
from backend.io.format_registry import FormatSpec, register_format
from backend.io.versioned import VersionedPayloadFile
from backend.log_config import get_logger
from backend.utils import utc_now

logger = get_logger("rag.pseudo_query_gate")

#: 判定点の名前 (``decision.jsonl`` の ``decision_point``)。
PREDICATE_NAME = "pseudo_query_answerable"
#: 残す (発火) のラベル。反対は :data:`NEGATIVE_LABEL`。
ANSWERABLE_LABEL = "answerable"
#: 目標の誤棄却率 (答えのある問いを落とす確率の上限) の既定。
DEFAULT_ALPHA = 0.05

#: 棄権の理由 (``evidence``)。
REASON_UNCALIBRATED = "uncalibrated"
REASON_UNAVAILABLE = "rerank_unavailable"
REASON_UNCERTAIN = "uncertain"
#: 判定の材料が無いと分かっている棄権。死活監視に数えない (c_17 §2.1)。
LIVENESS_EXEMPT_REASONS = frozenset({REASON_UNCALIBRATED, REASON_UNAVAILABLE})

#: 較正ファイルの形式。材料 (ラベル付きの組) を集めるのが高価なので reset でも残す。
#: 書き手は較正、backend は起動時に読むだけ (``store/`` の外)。
PSEUDO_QUERY_GATE_FORMAT = register_format(FormatSpec(
    format_id="cache.pseudo_query_gate",
    version=1,
    klass="derived",
    writers=frozenset({"free"}),
    path_key="cache/pseudo_query_gate.json",
    retention="one entry per rerank model_key, rewritten by the calibration",
    keep_on_reset=True,
))


def conformal_lower(scores: Sequence[float], alpha: float) -> float | None:
    """昇順 ``⌊α (n+1)⌋`` 番目の値。``score < これ`` の確率は交換可能性の下で α 以下。

    ``⌊α (n+1)⌋ = 0`` (件数が足りず保証できない) なら ``None``。
    """
    s = sorted(float(v) for v in scores)
    j = math.floor(float(alpha) * (len(s) + 1) + 1e-9)
    if j < 1:
        return None
    return s[j - 1]


def conformal_upper(scores: Sequence[float], beta: float) -> float | None:
    """降順 ``⌊β (m+1)⌋`` 番目の値。``score > これ`` の確率は交換可能性の下で β 以下。"""
    s = sorted((float(v) for v in scores), reverse=True)
    j = math.floor(float(beta) * (len(s) + 1) + 1e-9)
    if j < 1:
        return None
    return s[j - 1]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class PseudoQueryGateCalibration:
    """リランカー 1 モデル分の較正 (``cache/pseudo_query_gate.json`` の 1 項目)。閾値は logit。"""

    model_key: str
    #: logit がこれ未満なら ``skip`` (書かない)。
    drop_below: float
    #: logit がこれを超えれば ``fire`` (``inf`` なら ``fire`` の帯は空)。
    keep_above: float
    alpha: float
    beta: float
    stats: Mapping[str, Any] = field(default_factory=dict)
    calibrated_at: str = ""
    source: str = ""

    def band_of(self, logit: float) -> str:
        """logit の帯 (``fire`` / ``abstain`` / ``skip``)。"""
        if logit < self.drop_below:
            return "skip"
        if logit > self.keep_above:
            return "fire"
        return "abstain"

    def to_payload(self) -> dict[str, Any]:
        return {
            "drop_below": float(self.drop_below),
            # JSON に inf は書けないので「fire の帯が空」は null で持つ。
            "keep_above": None if math.isinf(self.keep_above) else float(self.keep_above),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "stats": dict(self.stats),
            "calibrated_at": self.calibrated_at,
            "source": self.source,
        }

    @classmethod
    def from_payload(cls, model_key: str, entry: Any) -> PseudoQueryGateCalibration | None:
        """1 項目を読む。形が合わなければ ``None`` (検査を掛けない)。"""
        if not isinstance(entry, Mapping):
            return None
        try:
            raw_keep = entry.get("keep_above")
            cal = cls(
                model_key=model_key,
                drop_below=float(entry["drop_below"]),
                keep_above=math.inf if raw_keep is None else float(raw_keep),
                alpha=float(entry["alpha"]),
                beta=float(entry["beta"]),
                stats=dict(entry.get("stats") or {}),
                calibrated_at=str(entry.get("calibrated_at") or ""),
                source=str(entry.get("source") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(cal.drop_below) or cal.keep_above < cal.drop_below:
            return None
        return cal


# ── 判定点 ─────────────────────────────────────────────────────────


class _PseudoQueryStage:
    """段 (``Predicate`` プロトコル)。棄権の理由と数値の根拠を書き分けるため ``Verdict`` を直接返す。

    ``ctx``: ``logit`` (採点の logit、縮退したら ``None``) / ``calibration``
    (:class:`PseudoQueryGateCalibration` か ``None``) / ``package_id`` / ``target_id``
    (索引の ``attrs.target_id`` と同じ表記 = パッケージ内の chunk id) / ``position`` /
    ``pq_id`` (残したときの索引の行 id) / ``from_hint``。問いの本文は受け取らない (記録に載せない)。
    """

    name = f"{PREDICATE_NAME}_stage"

    def evaluate(self, text: str, ctx: Mapping[str, Any] | None = None) -> Verdict:  # noqa: ARG002
        c = ctx or {}
        calibration: PseudoQueryGateCalibration | None = c.get("calibration")
        detail: dict[str, Any] = {
            "calibrated": calibration is not None,
            "model_key": calibration.model_key if calibration is not None else "",
            "package_id": str(c.get("package_id") or ""),
            "target_id": str(c.get("target_id") or ""),
            "position": int(c.get("position") or 0),
            "pq_id": str(c.get("pq_id") or ""),
            "from_hint": bool(c.get("from_hint")),
        }
        raw = c.get("logit")
        if raw is None:
            return self._abstain(REASON_UNAVAILABLE, detail, 0.0)
        logit = float(raw)
        detail["logit"] = round(logit, 4)
        score = round(_sigmoid(logit), 4)
        if calibration is None:
            return self._abstain(REASON_UNCALIBRATED, detail, score)
        band = calibration.band_of(logit)
        if band == "skip":
            return Verdict(
                value=NEGATIVE_LABEL, score=score, band="skip", evidence="rerank_unanswerable",
                predicate=self.name, stage="rerank", detail=detail,
            )
        if band == "fire":
            return Verdict(
                value=ANSWERABLE_LABEL, score=score, band="fire", evidence="rerank_answerable",
                predicate=self.name, stage="rerank", detail=detail,
            )
        return self._abstain(REASON_UNCERTAIN, detail, score)

    def _abstain(self, reason: str, detail: Mapping[str, Any], score: float) -> Verdict:
        return Verdict(
            value=None, score=score, band="abstain", evidence=reason,
            predicate=self.name, stage="rerank", detail=dict(detail),
        )


_STAGE = _PseudoQueryStage()

#: プロセス共通の判定点。前段 (生成側の「断片だけで答えられる問い」の指示) を確認する
#: ``confirm`` だが、反対は棄権でなく段自身の ``skip`` で返す (棄権の縮退先が「書く」なので)。
predicate = register_predicate(
    CascadePredicate(
        PREDICATE_NAME,
        lexical=_STAGE,
        policy="confirm",
        candidates=[ANSWERABLE_LABEL, NEGATIVE_LABEL],
        scope="sleep",
        liveness_exempt=LIVENESS_EXEMPT_REASONS,
    ),
)


#: 束ねた ``DebugLogger`` (:func:`decisions_recorded` が見る)。
_bound: dict[str, Any] = {"logger": None}


def bind_debug_logger(debug_logger: Any) -> None:
    """配線時に既存の単一 ``DebugLogger`` を差し込む (新規生成はしない)。"""
    _bound["logger"] = debug_logger
    predicate.bind_debug_logger(debug_logger)


def decisions_recorded() -> bool:
    """判定の記録 (``decision.jsonl``) が書かれる構成か。

    較正前の採点は記録だけが目的 (較正の材料) なので、書かれない構成 (通常起動・``debug``)
    では採点を省く (c_16 §7.2.1 の第 5 段階 B)。``DebugLogger`` の ``enabled`` /
    ``log_decisions`` を見る (持たない記録器は書くとみなす)。
    """
    logger_ = _bound["logger"]
    if logger_ is None:
        return False
    return bool(getattr(logger_, "enabled", True)) and bool(getattr(logger_, "log_decisions", True))


def evaluate_pseudo_query(
    logit: float | None,
    calibration: PseudoQueryGateCalibration | None,
    *,
    package_id: str = "",
    target_id: str = "",
    position: int = 0,
    pq_id: str = "",
    from_hint: bool = False,
) -> Verdict:
    """判定点として評価する (``decision.jsonl`` へ記録する。問い 1 件に 1 回)。

    ``logit`` は (問い, 元チャンク) の再順位のスコア。採点が縮退したら ``None``。
    ``target_id`` は索引の ``attrs.target_id`` と同じ (パッケージ内の chunk id)、``pq_id`` は
    残したときの索引の行 id (``pseudo_query_id(target_id, position, 問い)``)。
    """
    return predicate.evaluate("", {
        "logit": logit,
        "calibration": calibration,
        "package_id": package_id,
        "target_id": target_id,
        "position": position,
        "pq_id": pq_id,
        "from_hint": from_hint,
    })


# ── 較正 (純関数) ────────────────────────────────────────────────


def compute_pseudo_query_gate_calibration(
    positives: Sequence[float],
    negatives: Sequence[float] = (),
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float | None = None,
    model_key: str = "",
    source: str = "",
) -> dict[str, Any]:
    """正例・負例の logit から較正を作る (純関数、c_17 §3.15)。

    Args:
        positives: 元チャンクが答えている問いの logit。
        negatives: 答えていない問いの logit (無ければ ``fire`` の帯は空)。
        alpha: 答えのある問いを落とす確率の上限 (``drop_below`` を決める)。
        beta: 答えの無い問いに ``fire`` を出す確率の上限 (``keep_above``)。``None`` は ``alpha``。

    Returns:
        ``{"ok", "reason", "calibration" (PseudoQueryGateCalibration | None), "stats"}``。
        ``ok`` が偽なら保存しない (理由: ``bad_alpha`` / ``too_few_positives`` / ``non_finite``)。
    """
    beta_v = float(alpha if beta is None else beta)
    fail = {"ok": False, "calibration": None, "stats": {}}
    if not (0.0 < float(alpha) < 1.0) or not (0.0 < beta_v < 1.0):
        return {**fail, "reason": "bad_alpha"}
    pos = [float(v) for v in positives]
    neg = [float(v) for v in negatives]
    if not all(math.isfinite(v) for v in (*pos, *neg)):
        return {**fail, "reason": "non_finite"}
    drop_below = conformal_lower(pos, alpha)
    if drop_below is None:
        return {**fail, "reason": "too_few_positives", "stats": {"n_positive": len(pos), "n_negative": len(neg)}}
    upper = conformal_upper(neg, beta_v) if neg else None
    keep_above = math.inf if upper is None else max(upper, drop_below)
    stats = {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "positive_drop_rate": round(sum(v < drop_below for v in pos) / len(pos), 4),
        "negative_drop_rate": round(sum(v < drop_below for v in neg) / len(neg), 4) if neg else None,
        "negative_keep_rate": round(sum(v > keep_above for v in neg) / len(neg), 4) if neg else None,
    }
    calibration = PseudoQueryGateCalibration(
        model_key=model_key,
        drop_below=float(drop_below),
        keep_above=float(keep_above),
        alpha=float(alpha),
        beta=beta_v,
        stats=stats,
        calibrated_at=utc_now(),
        source=source,
    )
    return {"ok": True, "reason": "", "calibration": calibration, "stats": stats}


# ── 較正ファイル ────────────────────────────────────────────────


def _calibration_file(path: Path | str) -> VersionedPayloadFile:
    return VersionedPayloadFile(
        PSEUDO_QUERY_GATE_FORMAT, path, component="pseudo_query_gate", state_logger=logger,
    )


def load_pseudo_query_gate_calibration(
    path: Path | str, model_key: str,
) -> PseudoQueryGateCalibration | None:
    """``model_key`` の項目を読む。無い / 読めない / 形が合わなければ ``None`` (検査を掛けない)。"""
    if not model_key:
        return None
    f = _calibration_file(path)
    if not f.load() or not isinstance(f.payload, dict):
        return None
    entry = f.payload.get(model_key)
    if entry is None:
        return None
    cal = PseudoQueryGateCalibration.from_payload(model_key, entry)
    if cal is None:
        logger.warning(
            "Pseudo-query gate calibration for %s is malformed; the gate stays off", model_key,
        )
    return cal


class CalibrationFileRefused(RuntimeError):
    """既存の較正ファイルが読めない (新しい版 / 壊れている) ので上書きしない。"""


#: 上書きしてよい既存ファイルの分類 (無い / 今の版 / 移行できた)。
_WRITABLE_STATUSES = frozenset({"absent", "current", "migrated"})


def save_pseudo_query_gate_calibration(
    path: Path | str, calibration: PseudoQueryGateCalibration,
) -> None:
    """``calibration.model_key`` の項目を書く (他のモデルの項目は残す)。

    既存のファイルが新しい版・壊れている等で読めなければ :class:`CalibrationFileRefused`。
    """
    if not calibration.model_key:
        raise ValueError("calibration has no model_key")
    f = _calibration_file(path)
    loaded = f.load()
    if not loaded and f.last_status not in _WRITABLE_STATUSES:
        raise CalibrationFileRefused(
            f"{path} is not readable by this version ({f.last_status}); refusing to overwrite it",
        )
    payload = f.payload if loaded and isinstance(f.payload, dict) else {}
    f.payload = {**payload, calibration.model_key: calibration.to_payload()}
    if not f.save():
        raise OSError(f"failed to save {path}")


__all__ = [
    "ANSWERABLE_LABEL",
    "DEFAULT_ALPHA",
    "LIVENESS_EXEMPT_REASONS",
    "PREDICATE_NAME",
    "PSEUDO_QUERY_GATE_FORMAT",
    "CalibrationFileRefused",
    "PseudoQueryGateCalibration",
    "bind_debug_logger",
    "compute_pseudo_query_gate_calibration",
    "conformal_lower",
    "conformal_upper",
    "decisions_recorded",
    "evaluate_pseudo_query",
    "load_pseudo_query_gate_calibration",
    "predicate",
    "save_pseudo_query_gate_calibration",
]
