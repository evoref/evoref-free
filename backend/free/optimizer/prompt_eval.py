"""base prompt 候補の実測評価 (PromptEvalProtocol) と評価ケースの選定

Level 1 phase1 (base prompt 進化) の **採用ゲート** に使う抽象。欠陥率 fitness
(:func:`backend.free.learning.fitness.defect_rate_fitness`) は経験集合から
計算されるため候補プロンプトに無反応で、候補間の差はキーワードカバー率の
タイブレーク (≤ ``COVERAGE_TIEBREAK_MAX``) しか無い (f_04 §8 禁則 7)。採用の
可否はその値ではなく、**現行と最良候補を同じ失敗ケースで実生成して比べた
実測値** で決める (f_04 §4.5)。

実体 (``backend.free.llm.prompt_candidate_eval.PromptCandidateEval``、Gen pillar)
は本 Protocol を明示継承せず構造的部分型で満たし、wire 時に注入する
(``EmbedEvalProtocol`` と同じ立て付け — Gen→Learn の越境を作らない)。
"""

from __future__ import annotations

import hashlib

from backend.free.learning.corrected_pairs import (
    response_honors_correction,
    strip_correction_preamble,
)
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: 失敗ケースの種別。judge へ渡すヒント文の出し分けに使う。
CASE_KIND_CORRECTION = "correction"
CASE_KIND_REPHRASE = "rephrase"
CASE_KIND_FAILED = "failed"


@dataclass(frozen=True)
class PromptEvalCase:
    """採用ゲートの評価ケース 1 件 (失敗した実ターンから作る)。

    Attributes:
        case_id: query から導出した安定 ID (同一 query は 1 ケースに畳む)。
        query: ユーザー発話 (そのまま user メッセージとして再生成に使う)。
        kind: :data:`CASE_KIND_*` のいずれか。
        hint: judge に渡す「何が期待されていたか」。correction ならユーザーの
            訂正文そのもの、それ以外は種別に応じた定型 (judge 側で解釈)。
    """

    case_id: str
    query: str
    kind: str
    hint: str = ""
    #: 訂正後にユーザーが受け入れた回答 (前置き剥がし済)。judge が「期待された
    #: 振る舞い」を訂正文の言い回しからでなく実際の正答から判定できる。
    #: 訂正ターンの回答が受諾だけ / 訂正を受け入れていない場合は空。
    reference: str = ""


@runtime_checkable
class PromptEvalProtocol(Protocol):
    """候補 system prompt で失敗ケースを再生成し、judge の採点を返す抽象。"""

    async def score_prompt(
        self, prompt_text: str, cases: list[PromptEvalCase],
    ) -> dict[str, float]:
        """``prompt_text`` を system prompt として各ケースを再生成・採点する。

        Returns:
            ``{case_id: score (0.0〜1.0)}``。生成 / 採点に失敗したケースは
            **含めない** (呼出側は現行・候補の両方で採点できたケースだけを
            比較する)。全滅なら空 dict。
        """
        ...


def _case_id(query: str) -> str:
    return hashlib.blake2b(
        query.strip().encode("utf-8"), digest_size=6,
    ).hexdigest()


def select_prompt_eval_cases(
    experiences: list[dict], mode: str, limit: int,
) -> list[PromptEvalCase]:
    """経験から採用ゲートの評価ケースを **新しい順に最大 ``limit`` 件** 選ぶ。

    失敗の証拠がある実ターンだけを使う:

    - ``user_correction`` が立っているエントリは **訂正発話そのもの** なので
      ケースにせず、その **直前の同モードのターン** (訂正された側) をケースに
      し、訂正文をヒントにする。
    - ``rephrased_query`` / ``turn_outcome == "failed"`` はそのターン自身。

    同一 query は最新 1 件に畳む。``limit <= 0`` なら空。
    """
    if limit <= 0 or not experiences:
        return []
    mode_exp = [e for e in experiences if e.get("mode") == mode]
    # 訂正された側を引くため、時系列順 (snapshot は append 順 = 時系列) を保つ
    picked: dict[str, PromptEvalCase] = {}
    prev: dict | None = None
    for exp in mode_exp:
        signals = exp.get("signals") or {}
        query = str(exp.get("query") or "").strip()
        correction = signals.get("user_correction")
        if correction:
            if prev is not None:
                pq = str(prev.get("query") or "").strip()
                if pq:
                    fixed = strip_correction_preamble(
                        str(exp.get("response_full") or exp.get("response_summary") or ""),
                    )
                    if not response_honors_correction(fixed, str(correction)):
                        fixed = ""
                    picked[_case_id(pq)] = PromptEvalCase(
                        case_id=_case_id(pq), query=pq,
                        kind=CASE_KIND_CORRECTION, hint=str(correction).strip(),
                        reference=fixed,
                    )
            prev = exp
            continue
        if query:
            if signals.get("turn_outcome") == "failed":
                picked[_case_id(query)] = PromptEvalCase(
                    case_id=_case_id(query), query=query, kind=CASE_KIND_FAILED,
                )
            elif signals.get("rephrased_query"):
                picked[_case_id(query)] = PromptEvalCase(
                    case_id=_case_id(query), query=query, kind=CASE_KIND_REPHRASE,
                )
        prev = exp
    # dict は挿入順 = 古い順。最新側から limit 件
    cases = list(picked.values())
    return cases[-limit:] if len(cases) > limit else cases


__all__ = [
    "CASE_KIND_CORRECTION",
    "CASE_KIND_FAILED",
    "CASE_KIND_REPHRASE",
    "PromptEvalCase",
    "PromptEvalProtocol",
    "select_prompt_eval_cases",
]
