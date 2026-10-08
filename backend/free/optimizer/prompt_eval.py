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
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from backend.free.learning.case_filters import (
    has_failure_evidence,
    regeneration_mismatch_reason,
    replay_context,
)
from backend.log_config import get_logger

logger = get_logger("optimizer.prompt_eval")

#: 失敗ケースの種別。judge へ渡すヒント文の出し分けに使う。
CASE_KIND_CORRECTION = "correction"
CASE_KIND_REPHRASE = "rephrase"
CASE_KIND_FAILED = "failed"
#: ユーザーが 👎 を付けた実ターン (明示評価、2026-09-14)。任意の一言をヒントに。
CASE_KIND_USER_NEGATIVE = "user_negative"
#: 失敗の証拠が無い成功ターンからの標本 (一対比較ゲート専用、2026-09-14)。
CASE_KIND_SAMPLE = "sample"

#: 除外理由: 訂正の宛先ターンを解決できなかった (位置で代用しない)。
EXCLUDED_UNRESOLVED_CORRECTION = "unresolved_correction"
#: 除外理由: 問いが直前ターン・記憶・ツール結果を前提にしている。
EXCLUDED_DEPENDS_ON_CONTEXT = "depends_on_context"


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
    #: 事例のモード。欠陥計数の形式指定を create で数えない (失敗ラベルと同じ除外)。
    mode: str = "chat"
    #: 検証済みの訂正が示した正しい値 (``signals.correction_correct_value``、逐語
    #: span)。一対比較はこれを述べた応答を、欠陥数より先に勝たせる。検証済みで
    #: ない訂正候補からは入れない (不変則 #12)。
    expected_value: str = ""
    #: 検証済みの訂正が誤りとした値 (``signals.correction_wrong_claim``)。正しい値を
    #: 述べずにこれを言い直す応答は、欠陥数より先に負ける。
    wrong_value: str = ""
    #: ツール根拠のターンを再生するツール結果ブロック (``gen_config.tool_context``)。
    #: 再生成は user を ``query + tool_context`` にし、一対比較はこのブロックとの
    #: 矛盾 (calculate / 日付の結果の無視) を経験のラベルと同じ関数で数える。
    tool_context: str = ""
    #: 元のターンの calculate の式が会話から辿れなかった / 組み方に疑いがあった
    #: (``unexplained_numbers`` / ``expression_issues``)、または被演算子が問いの中
    #: だけと確かめられない (``calculate_history_operands`` が空でない / 無い)。
    #: このケースでは結果の無視を欠陥に数えない。
    calculate_unverified: bool = False
    #: 根拠になった経験の ID (メモリ上だけ。採用台帳 ``learning.prompt_adoption`` が
    #: 書く)。訂正は (宛先, 訂正した発話)、それ以外はそのターン自身。同じ
    #: ``case_id`` に畳んだターンは和集合 (古い順) — 再生成に使った厳密な
    #: ターンではなく根拠となった候補群。
    experience_ids: tuple[str, ...] = ()


def _merged_ids(
    picked: dict[str, PromptEvalCase], case_id: str, *ids: object,
) -> tuple[str, ...]:
    """``case_id`` に既に畳んだケースの ID と ``ids`` の和集合 (順序を保つ、空は落とす)。"""
    prev = picked.get(case_id)
    merged = list(prev.experience_ids) if prev is not None else []
    for value in ids:
        text = str(value or "")
        if text and text not in merged:
            merged.append(text)
    return tuple(merged)


@dataclass(frozen=True)
class EvalCaseSelection:
    """採用ゲートの評価ケースを失敗と標本に分けた選定結果。

    Attributes:
        failures: 失敗の証拠がある実ターンのケース (古い順、最大 ``limit`` 件)。
            一対比較の純勝ちはここでだけ数える。
        samples: 成功ターンからの標本 (古い順、最大 ``sample_cases`` 件)。
            失敗ケースの枠とは別枠 — 失敗が多くても無退行の検査が消えない。
        excluded: 失敗の証拠があったのにケースにしなかった件数 (理由 → 件数)。
            理由は ``case_filters`` の ``REASON_*`` と ``EXCLUDED_*``。
    """

    failures: list[PromptEvalCase] = field(default_factory=list)
    samples: list[PromptEvalCase] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)

    @property
    def cases(self) -> list[PromptEvalCase]:
        """標本を前、失敗を後ろに並べた 1 本のリスト (従来の戻り値の形)。"""
        return [*self.samples, *self.failures]


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


@runtime_checkable
class PromptPairEvalProtocol(Protocol):
    """現行と候補を **同じケースで一対比較** する評価器 (2026-09-14、f_04 §4.5)。

    絶対採点 (0〜1) は実データで 0.9 に張り付き、失敗ケースが無いと何も
    測れなかった。一対比較は「どちらが良いか」だけを judge に問うので判別力が
    高く、成功ターンの標本でも選択圧になる。
    """

    async def compare_prompts(
        self, current_text: str, candidate_text: str, cases: list[PromptEvalCase],
    ) -> dict[str, int]:
        """各ケースを両 prompt で再生成し、judge に比較させる。

        Returns:
            ``{case_id: +1 (候補が良い) | -1 (現行が良い) | 0 (同等)}``。生成 /
            判定に失敗したケースは **含めない**。
        """
        ...


def _case_id(query: str) -> str:
    return hashlib.blake2b(
        query.strip().encode("utf-8"), digest_size=6,
    ).hexdigest()


def _replay_fields(exp: dict) -> tuple[dict, str | None]:
    """ケースへ渡す再生の材料と、再生できない理由 (:mod:`~backend.free.learning.case_filters`)。"""
    mismatch = regeneration_mismatch_reason(exp)
    if mismatch is not None:
        return {}, mismatch
    context, mismatch = replay_context(exp)
    if mismatch is not None:
        return {}, mismatch
    if not context:
        return {}, None
    signals = exp.get("signals") or {}
    # 式の被演算子が今回の問いの中だけ (記録時の判定が空) のときだけ、結果を問いの
    # 計算として信頼できる。履歴の値で組んだ式の結果は誤りうる (実データ 3 問中 2 問で
    # 応答のほうが正しかった) — 旗の無い旧経験も信頼しない側に倒す (2026-10-08)。
    history_operands = signals.get("calculate_history_operands")
    return {
        "tool_context": context,
        "calculate_unverified": bool(
            signals.get("unexplained_numbers") or signals.get("expression_issues")
            or not isinstance(history_operands, list) or history_operands
        ),
    }, None


def select_prompt_eval_cases(
    experiences: list[dict], mode: str, limit: int, *, sample_cases: int = 0,
) -> list[PromptEvalCase]:
    """:func:`select_prompt_eval_case_sets` の結果を 1 本のリストで返す (互換)。

    標本が前、失敗ケースが後ろ。``limit`` は失敗ケースにだけ掛かり、標本は
    ``sample_cases`` の別枠 (以前は合わせて ``limit`` 件で、失敗が多いと
    標本が全部押し出されていた)。
    """
    return select_prompt_eval_case_sets(
        experiences, mode, limit, sample_cases=sample_cases,
    ).cases


def select_prompt_eval_case_sets(
    experiences: list[dict], mode: str, limit: int, *, sample_cases: int = 0,
) -> EvalCaseSelection:
    """経験から採用ゲートの評価ケースを選び、失敗・標本・除外件数に分けて返す。

    失敗ケースは **新しい順に最大 ``limit`` 件**、標本は別枠で最大
    ``sample_cases`` 件 (f_04 §10.1 #2 後半)。

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

    - ``user_negative`` (ユーザーの 👎、2026-09-14) はそのターン自身。本人が
      失敗と言った唯一の信号なので、失敗 / 言い直しより優先して残す。
    - ``sample_cases`` が 1 以上なら、失敗の証拠が無い **成功ターン** からも
      新しい順にその件数まで標本ケース (``CASE_KIND_SAMPLE``、ヒント無し) を
      足す。絶対採点では意味を持たないが、一対比較 (現行 vs 候補) の
      ゲートなら成功ターンでも選択圧になる (f_04 §4.5)。
    - 文書チャンクを根拠にしたターンと長文の経路を通ったターンは種別を問わず
      外す (:func:`~backend.free.learning.case_filters.
      regeneration_mismatch_reason`)。
    - ツール結果を根拠にしたターンは、記録したツール結果ブロックを添えて再生
      できるものだけケースにする (:func:`~backend.free.learning.case_filters.
      replay_context`、2026-10-08)。標本には使わない。

    同一 query は最新 1 件に畳む。``limit <= 0`` なら空。

    ``excluded`` は **失敗の証拠があった** ターン (訂正・👎・failed・言い直し)
    のうちケースにしなかった件数を理由別に数える。成功ターンの除外は数えない
    — 「失敗はあったが測れない」を「失敗が無い」と見分けるための内訳
    (f_04 §10.1 #4)。
    """
    if limit <= 0 or not experiences:
        return EvalCaseSelection()
    # モードで先に絞る。宛先解決もこの中だけを見るので、chat のゲートに
    # create のターンが混ざることはない。
    mode_exp = [e for e in experiences if e.get("mode") == mode]
    # 訂正された側を引くため、時系列順 (snapshot は append 順 = 時系列) を保つ
    picked: dict[str, PromptEvalCase] = {}
    #: 再生成で同じ入力にならないので外したターンの件数 (理由 → 件数)
    mismatched: dict[str, int] = {}
    for index, exp in enumerate(mode_exp):
        signals = exp.get("signals") or {}
        query = str(exp.get("query") or "").strip()
        correction = signals.get("user_correction")
        if correction:
            corrected = resolve_corrected_turn(mode_exp, index)
            pq = str((corrected or {}).get("query") or "").strip()
            # 文書チャンク ([参考情報]) やツール結果を根拠に答えたターン、長文の
            # 経路を通ったターンは、system prompt だけで短く再生成するゲートでは
            # 根拠も経路も無く、どの候補でも同じ点になる (記憶依存の問いと同じ、
            # f_04 §4.5)。判定は few-shot と同じ述語 (learning.case_filters)。
            if not pq:
                mismatched[EXCLUDED_UNRESOLVED_CORRECTION] = (
                    mismatched.get(EXCLUDED_UNRESOLVED_CORRECTION, 0) + 1
                )
                continue
            replay, mismatch = _replay_fields(corrected or {})
            if mismatch is not None:
                mismatched[mismatch] = mismatched.get(mismatch, 0) + 1
                continue
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
                experience_ids=_merged_ids(
                    picked, _case_id(pq), (corrected or {}).get("id"), exp.get("id"),
                ),
                kind=CASE_KIND_CORRECTION, hint=str(correction).strip(),
                reference=fixed or correct_value, mode=mode,
                # ``user_correction`` は検証済みの訂正にしか入らないので、
                # この分岐の span は検証器の門を通った値 (不変則 #12)。
                expected_value=correct_value,
                wrong_value=str(signals.get("correction_wrong_claim") or "").strip(),
                **replay,
            )
            continue
        if query and has_failure_evidence(signals):
            replay, mismatch = _replay_fields(exp)
            if mismatch is not None:
                # 長文の失敗 (検証落ち等) も、system prompt だけで短く再生成する
                # このゲートでは再現できず枠を無駄にする (docs/f_04 §2.5)。以前は
                # failed のときだけ外し、👎・言い直しの長文はケースにしていた。
                mismatched[mismatch] = mismatched.get(mismatch, 0) + 1
                continue
            ids = _merged_ids(picked, _case_id(query), exp.get("id"))
            if signals.get("user_negative") is True:
                picked[_case_id(query)] = PromptEvalCase(
                    case_id=_case_id(query), query=query,
                    kind=CASE_KIND_USER_NEGATIVE,
                    hint=str(signals.get("user_note") or "").strip(), mode=mode,
                    experience_ids=ids, **replay,
                )
            elif signals.get("turn_outcome") == "failed":
                picked[_case_id(query)] = PromptEvalCase(
                    case_id=_case_id(query), query=query, kind=CASE_KIND_FAILED, mode=mode,
                    experience_ids=ids, **replay,
                )
            elif signals.get("rephrased_query"):
                picked[_case_id(query)] = PromptEvalCase(
                    case_id=_case_id(query), query=query, kind=CASE_KIND_REPHRASE, mode=mode,
                    experience_ids=ids, **replay,
                )
    # 文脈 (直前ターン / 記憶 / ツール結果) を前提にした問いは、system prompt
    # だけで再生成するゲートでは **どの候補でも同じ点** になり、何も測れない。
    # 実測 (2026-09-10 ライブ監査 (i) I-17): 6/6 ケースが「私が今の会社に
    # 入った年は」「私の出身地は」型で、現行・候補とも全ケース 0.0 (1 件だけ
    # 「不明」と答えた候補が 0.9) → 0.150 → 0.150 で不採用。eval_core の
    # 足切りと同じ ``depends_on_context`` で落とす。ツール結果ブロックを添えて
    # 再生するケースは、道具語と被演算子の欠けでは落とさない (ブロックが運ぶ)。
    cases = [
        c for c in picked.values()
        if not depends_on_context(c.query, tool_supplied=bool(c.tool_context))
    ]
    dropped = len(picked) - len(cases)
    samples: list[PromptEvalCase] = []
    if sample_cases > 0:
        taken = {c.case_id for c in cases}
        for exp in reversed(mode_exp):
            if len(samples) >= sample_cases:
                break
            signals = exp.get("signals") or {}
            query = str(exp.get("query") or "").strip()
            if (
                not query or _case_id(query) in taken
                # snapshot で問いを切った行 (``compact_experience``) は、切れた
                # 問いを再生成しても元のターンの標本にならない。
                or exp.get("query_truncated")
                or signals.get("user_negative") is True
                or signals.get("user_correction")
                or signals.get("correction_candidate")
                or signals.get("turn_outcome") == "failed"
                # 標本にツールのターンは使わない (snapshot はブロックを残さない)
                or signals.get("tool_grounded")
                or regeneration_mismatch_reason(exp) is not None
                or depends_on_context(query)
            ):
                continue
            taken.add(_case_id(query))
            samples.append(PromptEvalCase(
                case_id=_case_id(query), query=query, kind=CASE_KIND_SAMPLE, mode=mode,
                experience_ids=_merged_ids({}, "", exp.get("id")),
            ))
        # 新しい順に集めたので古い順へ戻す (失敗ケースと同じ並び)
        samples.reverse()
    excluded = dict(mismatched)
    if dropped:
        excluded[EXCLUDED_DEPENDS_ON_CONTEXT] = (
            excluded.get(EXCLUDED_DEPENDS_ON_CONTEXT, 0) + dropped
        )
    if excluded:
        logger.info(
            "prompt eval: failure turns excluded from the adoption gate %s",
            dict(sorted(excluded.items())),
        )
    # dict は挿入順 = 古い順。失敗ケースは最新側から limit 件 (標本は別枠)
    return EvalCaseSelection(
        failures=cases[-limit:] if len(cases) > limit else cases,
        samples=samples,
        excluded=excluded,
    )


__all__ = [
    "CASE_KIND_CORRECTION",
    "CASE_KIND_FAILED",
    "CASE_KIND_REPHRASE",
    "CASE_KIND_SAMPLE",
    "CASE_KIND_USER_NEGATIVE",
    "EXCLUDED_DEPENDS_ON_CONTEXT",
    "EXCLUDED_UNRESOLVED_CORRECTION",
    "EvalCaseSelection",
    "PromptEvalCase",
    "PromptEvalProtocol",
    "PromptPairEvalProtocol",
    "select_prompt_eval_case_sets",
    "select_prompt_eval_cases",
]
