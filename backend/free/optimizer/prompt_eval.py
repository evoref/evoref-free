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
    depends_on_context,
    resolve_corrected_turn,
    response_honors_correction,
    strip_correction_preamble,
)
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.free.learning.level0_instant import used_corpus_evidence
from backend.log_config import get_logger

logger = get_logger("optimizer.prompt_eval")

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

    - ``user_correction`` (= ``learning.correction_verifier`` が検証済みに
      昇格させた訂正だけが入る) が立っているエントリは **訂正発話そのもの**
      なのでケースにせず、**訂正が指すターン** (訂正された側) をケースにし、
      訂正文をヒントにする。字句止まりの ``correction_candidate`` はケースの
      根拠にしない (2026-09-08 監査 F-03: 偽陽性 2 件が唯一の評価ケースに
      なった)。
    - ``rephrased_query`` / ``turn_outcome == "failed"`` はそのターン自身。

    宛先の同定は :func:`~backend.free.learning.corrected_pairs.resolve_corrected_turn`
    (``signals.corrected_entry_id`` → 同一セッションの直前ターン → 双方に
    セッションが無い最古データのみ位置) に委ねる。経験バッファは **全セッション
    横断の 1 本** なので、以前のように「リスト上の直前エントリ」を無条件に
    訂正された側とみなすと、別会話のターンが評価ケースになる (訂正ペア側は
    2026-09-06 監査 F-01 で直したが、ここだけ位置頼みのまま残っていた)。
    宛先を解決できない訂正はケースを作らない — 誤ったケースは採用ゲートの
    測定そのものを狂わせるので、無いほうがましである。

    同一 query は最新 1 件に畳む。``limit <= 0`` なら空。
    """
    if limit <= 0 or not experiences:
        return []
    # モードで先に絞る。宛先解決もこの中だけを見るので、chat のゲートに
    # create のターンが混ざることはない。
    mode_exp = [e for e in experiences if e.get("mode") == mode]
    # 訂正された側を引くため、時系列順 (snapshot は append 順 = 時系列) を保つ
    picked: dict[str, PromptEvalCase] = {}
    grounded_dropped = 0
    for index, exp in enumerate(mode_exp):
        signals = exp.get("signals") or {}
        query = str(exp.get("query") or "").strip()
        correction = signals.get("user_correction")
        if correction:
            corrected = resolve_corrected_turn(mode_exp, index)
            pq = str((corrected or {}).get("query") or "").strip()
            # 文書チャンク ([参考情報]) を根拠に答えたターンは、system prompt
            # だけで再生成するゲートでは資料が無く、どの候補でも同じ点になる
            # (記憶依存の問いと同じ、f_04 §4.5)。疑似クエリ索引で注入率が
            # 上がったので、印 (gen_config.evidence_ids の corpus:) で外す。
            if pq and used_corpus_evidence(corrected or {}):
                grounded_dropped += 1
                continue
            if pq:
                correct_value = str(
                    signals.get("correction_correct_value") or "",
                ).strip()
                fixed = strip_correction_preamble(
                    str(exp.get("response_full") or exp.get("response_summary") or ""),
                )
                if not response_honors_correction(
                    fixed, str(correction), correct_value=correct_value,
                ):
                    fixed = ""
                # 採用ゲートはセッション snapshot (``compact_experience``、
                # 応答本文を持たない) から読むので ``fixed`` は実運用では常に
                # 空になる。検証器が逐語で取った ``correct_value`` は signals に
                # 残るため、応答本文が無いときはそれを参照値にする —
                # 「訂正後に受け入れられた値」として judge に渡す意味は同じ
                # (2026-09-08 監査 F-03 の追補)。
                picked[_case_id(pq)] = PromptEvalCase(
                    case_id=_case_id(pq), query=pq,
                    kind=CASE_KIND_CORRECTION, hint=str(correction).strip(),
                    reference=fixed or correct_value,
                )
            continue
        if query:
            if used_corpus_evidence(exp):
                grounded_dropped += 1
                continue
            if signals.get("turn_outcome") == "failed":
                picked[_case_id(query)] = PromptEvalCase(
                    case_id=_case_id(query), query=query, kind=CASE_KIND_FAILED,
                )
            elif signals.get("rephrased_query"):
                picked[_case_id(query)] = PromptEvalCase(
                    case_id=_case_id(query), query=query, kind=CASE_KIND_REPHRASE,
                )
    # 文脈 (直前ターン / 記憶 / ツール結果) を前提にした問いは、system prompt
    # だけで再生成するゲートでは **どの候補でも同じ点** になり、何も測れない。
    # 実測 (2026-09-10 ライブ監査 (i) I-17): 6/6 ケースが「私が今の会社に
    # 入った年は」「私の出身地は」型で、現行・候補とも全ケース 0.0 (1 件だけ
    # 「不明」と答えた候補が 0.9) → 0.150 → 0.150 で不採用。eval_core の
    # 足切りと同じ ``depends_on_context`` で落とす。
    cases = [c for c in picked.values() if not depends_on_context(c.query)]
    dropped = len(picked) - len(cases)
    if dropped or grounded_dropped:
        logger.info(
            "prompt eval: %d context-bound / %d corpus-grounded case(s) excluded "
            "from the adoption gate",
            dropped, grounded_dropped,
        )
    # dict は挿入順 = 古い順。最新側から limit 件
    return cases[-limit:] if len(cases) > limit else cases


__all__ = [
    "CASE_KIND_CORRECTION",
    "CASE_KIND_FAILED",
    "CASE_KIND_REPHRASE",
    "PromptEvalCase",
    "PromptEvalProtocol",
    "select_prompt_eval_cases",
]
