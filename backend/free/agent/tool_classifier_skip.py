"""ツール分類器 (層 5.9) を省けるかの shadow 評価 — 挙動は変えず、記録だけする。

分類器は門 (``ToolCallJudge._gate_allows``) が開いたターンで撃たれ、1 回あたり
固定約 2 秒 + デコードを本生成の前に払う (2026-10-05 実測: deliberative の 39% で
1〜3 回、TTFT p50 は 0 回 6.1 秒 / 1 回 12.1 秒)。撃たれた回の多くの結論は
no_tool だった。そこで「分類器を呼ぶ前に、ツールは不要と強く言えるか」を
判定点 ``tool_classifier_skip`` として **影で** 評価する。

- 採択は常に規則 (= 現在の挙動、分類器を撃つ)。``policy="shadow"`` なので事例段の
  結論は ``decision.jsonl`` に並べて残すだけで、返り値は変わらない。
- 省けると言うのは、事例段 (``tool_classifier_skip_exemplars.jsonl`` の近傍投票) が
  ``none`` で決まり、**かつ** ターンに構造上のツールの手掛かりが 1 つも無いとき
  だけ。手掛かりは既存の抽出器の出力をそのまま読む (語彙を足さない)。
- 分類器の実際の結論 (``classifier_raw``) は呼出側が ``tool_call_decision`` の
  context に並べて載せる。省けると言った回に分類器がツールを選んだら取りこぼし
  (``scripts/bench/tool_gate_replay.py`` が数える)。

不変則 #15: 近道は現在の判定と証拠を上書きしない。昇格の条件 (取りこぼし 0 が
連続すること) と閾値の決め方は docs/f_03 §3.1.3 と docs/c_17 の登録節に書く。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backend.free.agent.tool_judge_signals import _query_has_tool_signal
from backend.free.core.date_math_cue import query_has_date_math_cue
from backend.free.core.intent_vocab import (
    find_file_reference_clauses,
    looks_like_numeric_question,
)
from backend.free.core.markdown_fence import FENCE_LINE_RE
from backend.free.core.predicate import (
    NEGATIVE_LABEL,
    CascadePredicate,
    Exemplar,
    ExemplarPredicate,
    LexicalPredicate,
    load_exemplars,
)
from backend.free.core.response_dates import mentions_literal_date
from backend.log_config import get_logger

logger = get_logger("agent.tool_classifier_skip")

#: 判定点の名前。``tool_call_decision`` / ``tool_gate`` とは別名にする (記録が混ざる)。
PREDICATE_NAME = "tool_classifier_skip"

#: 規則 (= 現在の挙動) のラベル。ツール要否ゲートと同じ語彙 (``tool`` / ``none``)。
TOOL_LABEL = "tool"

#: 同梱事例 (専用)。門 (``tool_gate_knn``) と層 5.95 の ``gate_verdict`` が読む
#: ``tool_gate_exemplars.jsonl`` とは分ける — 共有ファイルに none を足すと門の判定まで
#: 動く。ラベルは人が付けたものだけで、判定ログの出力からは採らない (不変則 #15)。
DEFAULT_EXEMPLARS_FILE = (
    Path(__file__).resolve().parent / "_defaults" / "tool_classifier_skip_exemplars.jsonl"
)

#: 近傍投票数と発火 (= ``none`` で決まる) に要る得票率。**保守的に「省かない」寄り**。
#:
#: ``none`` で決まるには ceil(5 × 0.8) = 4 票が要る。事例集合は門 (k=5 の多数決、
#: ``tool_gate_exemplars.jsonl``) と別なので、門が開いた回でも票は門から導けない。
#: 専用事例 (2026-10-10 初版 112 件、bge-m3) の LOO 格子: k=9/0.65 は none の
#: 適合率 0.907 (誤り 5 件すべて tool→none) で自己検査を割った。k=5/0.65・0.8 は
#: 判定分の正解率 0.987・棄権 37/112、k=7/0.8 は 1.0 だが棄権 62/112。棄権は
#: 分類器を撃つ (= 従来どおり) なので安全側だが、省ける回を取りこぼしすぎない
#: k=5 を採る。閾値を緩める判断は LOO (none の適合率) と再生測定 (取りこぼし 0)
#: の後に人が行う (docs/f_03 §3.1.3)。
DEFAULT_K = 5
DEFAULT_FIRE_RATIO = 0.8


def structural_tool_cues(query: str, recent_dialogue: str = "") -> tuple[str, ...]:
    """ターンに在る構造上のツールの手掛かりの名前 (純粋関数、既存の抽出器の出力)。

    1 つでもあれば省略ゲートは「省ける」と言わない。語彙は持たず、規則層・門・
    日付の層が既に使っている抽出器をそのまま読む。
    """
    text = query or ""
    cues: list[str] = []
    if _query_has_tool_signal(text):
        cues.append("tool_signal")
    if looks_like_numeric_question(text, recent_dialogue):
        cues.append("numeric_question")
    if find_file_reference_clauses(text):
        cues.append("file_reference")
    if any(FENCE_LINE_RE.match(line) for line in text.splitlines()):
        cues.append("code_fence")
    if mentions_literal_date(text):
        cues.append("literal_date")
    if query_has_date_math_cue(text):
        cues.append("date_math_cue")
    return tuple(cues)


class ToolClassifierSkipShadow:
    """分類器の省略ゲートを影で評価する。:meth:`observe` は判定を変えない。"""

    def __init__(
        self,
        embedder: Any,
        *,
        exemplars: Sequence[Exemplar] | None = None,
        k: int = DEFAULT_K,
        fire_ratio: float = DEFAULT_FIRE_RATIO,
        debug_logger: Any = None,
    ) -> None:
        records = (
            list(exemplars)
            if exemplars is not None
            else load_exemplars(DEFAULT_EXEMPLARS_FILE)
        )
        self._exemplar = ExemplarPredicate(
            PREDICATE_NAME,
            embedder,
            exemplars=records,
            k=k,
            mode="chat",
            fire_ratio=fire_ratio,
            # 害が出るのは「ツールが要るのに省く」(= none と決める) 向きだけ。反対向きの誤りは
            # 分類器を撃つだけで安全なので、全体の正解率でなく none の適合率で自己検査する。
            critical_label=NEGATIVE_LABEL,
        )
        self._cascade = CascadePredicate(
            PREDICATE_NAME,
            lexical=LexicalPredicate(
                f"{PREDICATE_NAME}_rule",
                lambda _text: TOOL_LABEL,
                evidence="gate_open",
            ),
            exemplar=self._exemplar,
            policy="shadow",
            debug_logger=debug_logger,
            candidates=[TOOL_LABEL, NEGATIVE_LABEL],
            scope="request",
        )

    def is_ready(self) -> bool:
        return self._exemplar.is_ready()

    def bind_debug_logger(self, debug_logger: Any) -> None:
        self._cascade.bind_debug_logger(debug_logger)

    def reset(self, embedder: Any = None) -> None:
        self._exemplar.reset(embedder)

    async def warmup(self) -> bool:
        return await self._exemplar.warmup()

    def calibration(self) -> dict[str, float | bool | int]:
        return self._exemplar.calibration

    def leave_one_out(self, *, with_errors: bool = False) -> dict[str, object]:
        return self._exemplar.leave_one_out(with_errors=with_errors)

    async def observe(
        self,
        query: str,
        *,
        recent_dialogue: str = "",
        query_vec: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """省けると言うかを評価し、``tool_call_decision`` の context に足す値を返す。

        未 warmup なら ``None`` (記録しない)。分類器を撃つかどうかには一切
        関与しない — 呼出側はこの返り値を記録にだけ使う。
        """
        if not self.is_ready():
            return None
        cues = structural_tool_cues(query, recent_dialogue)
        try:
            _chosen, shadow = await self._cascade.aevaluate_pair(
                query, None, query_vec=query_vec,
            )
        except Exception as e:  # pragma: no cover - 縮退で吸収する
            logger.info("Tool classifier skip shadow failed: %s", e)
            return None
        band = shadow.band if shadow is not None else "abstain"
        return {
            "skip_would": band == "skip" and not cues,
            "skip_band": band,
            "skip_score": round(float(shadow.score), 4) if shadow is not None else 0.0,
            "skip_evidence": shadow.evidence if shadow is not None else "",
            "skip_cues": list(cues),
        }
