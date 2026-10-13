"""ReplayHarness — 回帰ケースを現行の資産で再生して合否を付ける (docs/f_04 §6.4 / §6.5)。

新しい生成経路・LLM judge・AuxClient の purpose は足さない。再生成と採点は採用ゲートの
評価器 (``llm.prompt_candidate_eval.PromptCandidateEval``、Gen pillar) の ``_regenerate`` /
``_rank`` をそのまま呼ぶ — 評価器は wire 時に注入され、本モジュールは Gen を import しない
(``PromptEvalProtocol`` と同じ duck typing)。再生成は ``generate(stream=False,
id_slot=background_slot)`` の既存の例外経路 (c_14 §7.2) で、チャットのスロットを塞がない。

単独のケースの合否 (§6.4):

- ``expected_value`` を持つ: 正しい値を述べ、誤値を言い直さず、欠陥 0
- ``consistency``: 再生成した応答が、直前の会話に対して整合性の検出を出さず、欠陥 0
- それ以外: 欠陥 0 (ツール結果との矛盾を含む)

ケースの役割 (guard / pressure / unknown) と ``flaky`` の遷移は :func:`record_result`、
採用ゲートの標本・失敗ケースへの足し方は :func:`gate_cases` / :func:`augment_selection`。
どれも純関数で、書き込みは呼出側 (Level 1 の ``phase8_growth_regression``) が学習ストアへ行う。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from backend.free.learning.conversation_consistency import find_inconsistencies
from backend.free.learning.corrected_pairs import depends_on_context
from backend.free.learning.growth.regression import (
    CASE_ID_PREFIX,
    SOURCE_CONSISTENCY,
    SOURCE_SEED,
    STATUS_ACTIVE,
    RegressionCase,
    RegressionCaseStatus,
    RegressionStatusFile,
)
from backend.log_config import get_logger

if TYPE_CHECKING:
    from backend.free.optimizer.prompt_eval import EvalCaseSelection, PromptEvalCase

logger = get_logger("learning.growth.replay")

RESULT_PASS = "pass"
RESULT_FAIL = "fail"

ROLE_GUARD = "guard"
ROLE_PRESSURE = "pressure"
ROLE_UNKNOWN = "unknown"

#: ``RegressionCaseStatus._extra`` に置く、直近の再生に使った資産束の digest。
#: ``flaky`` (同じ資産の版で合否が反転) の判定に使う。宣言フィールドにすると形式の版が
#: 動くので未知キーの通路で運ぶ (``learning_state`` の ``generation_adoptions`` と同じ)。
ASSET_DIGEST_KEY = "asset_digest"


def replay_eligible(case: RegressionCase) -> bool:
    """問い (+ 記録したツール結果 / 直前の会話) だけで再生して意味のあるケースか。

    採用ゲートの評価ケースと同じ ``depends_on_context`` で、直前ターン・記憶を前提にした
    問いを外す (不変則 #14 (a))。``consistency`` は直前の会話を自分で運ぶので外さない。
    """
    if case.source == SOURCE_CONSISTENCY:
        return bool(case.prior_turns)
    return not depends_on_context(case.query, tool_supplied=bool(case.tool_context))


def asset_digest(system_text: str, params: Mapping[str, Any] | None) -> str:
    """再生に使った資産束 (system の最終レンダ + sampling) の blake2b (16 hex)。"""
    payload = json.dumps(
        {"system": system_text, "params": dict(sorted((params or {}).items()))},
        ensure_ascii=False, sort_keys=True, default=str,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True)
class ReplayOutcome:
    """1 ケースの再生の結果。"""

    case_id: str
    passed: bool
    #: 評価器の順位キー ``(誤値, -正値, 欠陥数)``。
    rank: tuple[int, int, int]
    response: str


class ReplayHarness:
    """回帰ケースを 1 件ずつ再生成し、決定論で合否を付ける。

    Args:
        evaluator: ``_regenerate(prompt_text, query, *, params=, tool_context=, history=)`` と
            ``_rank(case, response)`` を持つ評価器 (``PromptCandidateEval``)。
    """

    def __init__(self, evaluator: Any) -> None:
        self._evaluator = evaluator

    @property
    def available(self) -> bool:
        """評価器が再生成と順位付けを持つか (持たなければ再生しない)。"""
        return callable(getattr(self._evaluator, "_regenerate", None)) and callable(
            getattr(self._evaluator, "_rank", None),
        )

    async def replay(
        self,
        case: RegressionCase,
        system_text: str,
        params: Mapping[str, Any] | None = None,
    ) -> ReplayOutcome | None:
        """``system_text`` / ``params`` でケースを再生成して採点する (生成できなければ ``None``)。"""
        kwargs: dict[str, Any] = {
            "params": dict(params) if params else None,
            "tool_context": case.tool_context or "",
        }
        history = _history_messages(case)
        if history:
            kwargs["history"] = history
        response = await self._evaluator._regenerate(system_text, case.query, **kwargs)
        if not response:
            return None
        rank = tuple(self._evaluator._rank(case, response))
        return ReplayOutcome(
            case_id=case.case_id, passed=self.judge(case, response, rank),
            rank=rank, response=response,  # type: ignore[arg-type]
        )

    @staticmethod
    def judge(case: RegressionCase, response: str, rank: tuple[int, ...]) -> bool:
        """順位キーと (``consistency`` なら) 整合性の検出から合否を決める (§6.4 の 3 通り)。"""
        wrong, negative_correct, defects = rank
        if defects:
            return False
        if case.source == SOURCE_CONSISTENCY:
            turns = [{"query": t.query, "response": t.response} for t in case.prior_turns]
            turns.append({"query": case.query, "response": response})
            # 直前の会話は再生で変わらないので、再生成したターンの検出だけを見る
            last = len(case.prior_turns)
            return not any(f.turn_index == last for f in find_inconsistencies(turns))
        if case.expected_value:
            return wrong == 0 and negative_correct < 0
        return True


def _history_messages(case: RegressionCase) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in case.prior_turns:
        messages.append({"role": "user", "content": turn.query})
        messages.append({"role": "assistant", "content": turn.response})
    return messages


# ── 役割と flaky ──


def record_result(
    previous: RegressionCaseStatus | None,
    *,
    passed: bool,
    asset: str,
    at: str,
    is_seed: bool,
    guard_min_pass_streak: int,
) -> RegressionCaseStatus:
    """1 回の再生の結果で状態を進める (純関数)。

    - 連続合否は **同じ資産束の中だけ** で数える (資産が変われば 0 から)。guard は
      「現行の資産で ``guard_min_pass_streak`` 回連続合格」(§6.1)。
    - ``flaky``: 同じ資産束で前回と合否が反転したら立てる。立っている間は guard にも
      pressure にも使わない。同じ結果が 2 回続いたら下ろす (§6.2)。
    - pressure は現行の資産で不合格のケース。種ケースは pressure にしない (unknown のまま)。
    """
    prev = previous or RegressionCaseStatus()
    result = RESULT_PASS if passed else RESULT_FAIL
    prev_asset = (prev._extra or {}).get(ASSET_DIGEST_KEY)
    same_asset = prev_asset == asset and prev.last_result is not None
    pass_streak = prev.pass_streak if same_asset else 0
    fail_streak = prev.fail_streak if same_asset else 0
    if passed:
        pass_streak, fail_streak = pass_streak + 1, 0
    else:
        pass_streak, fail_streak = 0, fail_streak + 1
    flaky = prev.flaky
    if same_asset and prev.last_result != result:
        flaky = True
    elif flaky and max(pass_streak, fail_streak) >= 2:
        flaky = False
    if flaky:
        role = ROLE_UNKNOWN
    elif pass_streak >= max(1, guard_min_pass_streak):
        role = ROLE_GUARD
    elif fail_streak >= 1 and not is_seed:
        role = ROLE_PRESSURE
    else:
        role = ROLE_UNKNOWN
    return replace(
        prev,
        role=role,  # type: ignore[arg-type]
        pass_streak=pass_streak,
        fail_streak=fail_streak,
        flaky=flaky,
        last_result=result,
        last_run_at=at,
        _extra={**(prev._extra or {}), ASSET_DIGEST_KEY: asset},
    )


def due_cases(
    cases: Iterable[RegressionCase],
    status: RegressionStatusFile,
    *,
    limit: int,
    prefer: frozenset[str] = frozenset(),
) -> list[RegressionCase]:
    """再生する順 (``last_run_at`` の古い順、未実行が先) に ``active`` で再生できるケースを選ぶ。

    同じ ``last_run_at`` は台帳の並び (追記順、種ケースは後ろ) を保つ。``prefer`` (採用した版の
    監視中の guard、f_04 §6.6) はそれより先に回す。
    """
    from backend.utils import parse_utc

    def _key(item: tuple[int, RegressionCase]) -> tuple[int, int, float, int]:
        index, case = item
        first = 0 if case.case_id in prefer else 1
        entry = status.cases.get(case.case_id)
        at = parse_utc(entry.last_run_at) if entry is not None and entry.last_run_at else None
        return (first, 0, 0.0, index) if at is None else (first, 1, at.timestamp(), index)

    eligible = [
        (i, c) for i, c in enumerate(cases)
        if c.status == STATUS_ACTIVE and replay_eligible(c)
    ]
    eligible.sort(key=_key)
    return [c for _, c in eligible[:max(0, limit)]]


@dataclass
class RoleCounts:
    """``regression_status`` の役割別の件数 (同じモード・``active`` のケースだけ)。"""

    guards: int = 0
    pressure: int = 0
    flaky: int = 0
    by_mode: dict[str, dict[str, int]] = field(default_factory=dict)


def count_roles(cases: Iterable[RegressionCase], status: RegressionStatusFile) -> RoleCounts:
    """``active`` なケースの guard / pressure / flaky の件数。"""
    counts = RoleCounts()
    for case in cases:
        if case.status != STATUS_ACTIVE:
            continue
        entry = status.cases.get(case.case_id)
        if entry is None:
            continue
        mode = counts.by_mode.setdefault(case.mode, {"guards": 0, "pressure": 0, "flaky": 0})
        if entry.flaky:
            counts.flaky += 1
            mode["flaky"] += 1
        elif entry.role == ROLE_GUARD:
            counts.guards += 1
            mode["guards"] += 1
        elif entry.role == ROLE_PRESSURE:
            counts.pressure += 1
            mode["pressure"] += 1
    return counts


# ── 採用ゲートへの組み込み (§6.5) ──


def gate_cases(
    cases: Iterable[RegressionCase],
    status: RegressionStatusFile,
    mode: str,
) -> tuple[list["PromptEvalCase"], list["PromptEvalCase"]]:
    """同じモードの ``(guard, pressure)`` を採用ゲートの評価ケースにして返す。

    ``flaky`` / ``quarantined`` / ``retired`` / 再生できない (``depends_on_context``) /
    ``PromptEvalCase`` で運べない (``consistency``) ケースは足さない。種ケースは guard 専用
    (pressure にしない — 利用者のデータでない失敗に最適化しない)。
    """
    guards: list[PromptEvalCase] = []
    pressure: list[PromptEvalCase] = []
    for case in cases:
        if case.mode != mode or case.status != STATUS_ACTIVE:
            continue
        entry = status.cases.get(case.case_id)
        if entry is None or entry.flaky or not replay_eligible(case):
            continue
        if entry.role == ROLE_GUARD:
            target = guards
        elif entry.role == ROLE_PRESSURE and case.source != SOURCE_SEED:
            target = pressure
        else:
            continue
        eval_case = case.to_eval_case()
        if eval_case is not None:
            target.append(eval_case)
    return guards, pressure


def augment_selection(
    selection: "EvalCaseSelection",
    *,
    guards: list["PromptEvalCase"],
    pressure: list["PromptEvalCase"],
    limit: int,
    max_guards: int,
) -> "EvalCaseSelection":
    """経験から選んだ評価ケースに guard (標本へ) と pressure (失敗ケースへ) を足す。

    - pressure は ``limit`` (``adoption_eval_cases``) の枠の空きにだけ足す (経験由来の失敗が先)。
    - guard は経験由来の標本の後ろに最大 ``max_guards`` 件 (標本の無退行検査がそのまま
      guard の退行を止める)。
    - 同じ問い (``prompt_eval._case_id``) は足さない。guard も pressure も無ければ ``selection``
      をそのまま返す (既存の採用の挙動を変えない)。
    """
    from backend.free.optimizer.prompt_eval import EvalCaseSelection, _case_id

    if not guards and not pressure:
        return selection
    taken = {_case_id(c.query) for c in (*selection.failures, *selection.samples)}
    failures = list(selection.failures)
    for case in pressure:
        if len(failures) >= limit:
            break
        key = _case_id(case.query)
        if key in taken:
            continue
        taken.add(key)
        failures.append(case)
    samples = list(selection.samples)
    added = 0
    for case in guards:
        if added >= max_guards:
            break
        key = _case_id(case.query)
        if key in taken:
            continue
        taken.add(key)
        samples.append(case)
        added += 1
    return EvalCaseSelection(failures=failures, samples=samples, excluded=dict(selection.excluded))


def guard_ids_of(cases: Iterable[Any]) -> frozenset[str]:
    """評価ケースのうち回帰台帳由来 (guard) のケースの ID (経験由来の ID は ``rgc_`` を持たない)。"""
    return frozenset(
        c.case_id for c in cases if str(getattr(c, "case_id", "")).startswith(CASE_ID_PREFIX)
    )


__all__ = [
    "ASSET_DIGEST_KEY",
    "RESULT_FAIL",
    "RESULT_PASS",
    "ROLE_GUARD",
    "ROLE_PRESSURE",
    "ROLE_UNKNOWN",
    "ReplayHarness",
    "ReplayOutcome",
    "RoleCounts",
    "asset_digest",
    "augment_selection",
    "count_roles",
    "due_cases",
    "gate_cases",
    "guard_ids_of",
    "record_result",
    "replay_eligible",
]
