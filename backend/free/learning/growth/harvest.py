"""sleep-time Step 13.6「回帰台帳の収穫」(docs/f_04 §6.3)。

経験バッファ (束ねたパーティションの経験だけ、Level 1 の入力と同じ) から回帰ケースの候補を
作り、重複の鍵で畳んで回帰台帳へ足す。判定はすべて既存の述語を呼ぶ (語彙を複製しない、
不変則 #14 (a)、c_17 §8):

- 訂正は ``learning.correction_verifier`` が昇格させた検証済みのもの (``user_correction`` +
  ``correction_verdict == "assistant"`` + ``correction_correct_value``) だけ。未検証の候補は
  使わない (不変則 #12)。検証待ちの候補は数えずに次の回へ持ち越す。
- 失敗は ``turn_outcome=failed`` のうち理由が ``fitness.OUTCOME_REASON_CHANNELS`` の決定論の
  検証器のもの (``outcome_channel`` が既知のチャネルを返すもの)。
- 再生できるかは ``case_filters`` の述語 (:func:`~.regression.replay_exclusion`)。近道がツールを
  選んだターンは外す (不変則 #15)。
- 会話の整合性は ``conversation_consistency.find_inconsistencies`` (同じセッションの中だけ)。
  同じ種類はセッションあたり 1 件、直前 4 ターン・計 2000 字の文脈で同じ検出が再現するものだけ。

書き先は学習ストア (回帰台帳と状態ファイル) で、Evidence Store には書かない (不変則 #2)。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from backend.free.learning.case_filters import REASON_TOOL_CONTEXT_TRUNCATED
from backend.free.learning.conversation_consistency import Finding, find_inconsistencies
from backend.free.learning.corrected_pairs import resolve_corrected_turn
from backend.free.learning.fitness import OUTCOME_CHANNEL_OTHER, outcome_channel
from backend.free.learning.growth.regression import (
    PRIOR_TURNS_MAX,
    PRIOR_TURNS_MAX_CHARS,
    SOURCE_CONSISTENCY,
    SOURCE_USER_NEGATIVE,
    SOURCE_VERIFIED_CORRECTION,
    SOURCE_VERIFIER_FAILURE,
    STATUS_QUARANTINED,
    PriorTurn,
    RegressionCase,
    RegressionLedger,
    is_model_origin,
    load_status,
    make_case,
    replay_exclusion,
    save_status,
)
from backend.free.learning.level0_instant import (
    TOOL_CONTEXT_CAP,
    entry_to_dict,
    teaches_via_verified_correction,
)
from backend.free.memory.protocols import GROWTH_HARVEST_KEYS
from backend.free.optimizer.prompt_eval import _replay_fields
from backend.io.id_registry import is_valid_id
from backend.io.readonly import DataReadonlyError, is_readonly
from backend.log_config import get_logger

logger = get_logger("learning.growth.harvest")

#: 昇格した訂正の判定 (``learning.correction_verifier._PROMOTED_TARGET`` と同じ値)。
_PROMOTED_VERDICT = "assistant"

#: 除外の理由 (``case_filters`` / ``regression`` の理由に足すもの)。
REJECT_UNVERIFIED_CORRECTION = "unverified_correction"
REJECT_CORRECTION_TURN_FAILED = "correction_turn_failed"
REJECT_NO_CORRECT_VALUE = "no_correct_value"
REJECT_TARGET_UNRESOLVED = "target_unresolved"
REJECT_NOT_DETERMINISTIC = "not_deterministic_verifier"
REJECT_CONTEXT_TOO_LONG = "context_too_long"


def empty_result() -> dict[str, int]:
    """Step 13.6 の結果辞書 (全キー 0)。"""
    return dict.fromkeys(GROWTH_HARVEST_KEYS, 0)


@dataclass
class Harvest:
    """1 回の収穫の候補と件数 (台帳へ足す前)。"""

    candidates: list[RegressionCase] = field(default_factory=list)
    #: 判定した候補の数 (源 × 経験、整合性の検出を含む)。検証待ちの訂正は含まない。
    input: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    #: 同じ呼出の中で畳んだ件数 (整合性の同じ種類・同じセッション)。
    duplicate: int = 0
    consistency_findings: int = 0
    #: 検証待ちで次の回へ持ち越した訂正の数。
    deferred: int = 0
    #: 経験 ID → この回に判定し終えた源。
    processed: dict[str, list[str]] = field(default_factory=dict)

    def mark(self, exp_id: str, source: str) -> None:
        done = self.processed.setdefault(exp_id, [])
        if source not in done:
            done.append(source)


def _signals(exp: Mapping[str, Any]) -> Mapping[str, Any]:
    sig = exp.get("signals")
    return sig if isinstance(sig, Mapping) else {}


def _replay(exp: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """再生の材料 (ツール結果ブロック・calculate の印) と、再生できない理由。"""
    reason = replay_exclusion(exp)
    if reason is not None:
        return {}, reason
    fields, reason = _replay_fields(dict(exp))
    if reason is not None:
        return {}, reason
    if len(str(fields.get("tool_context") or "")) > TOOL_CONTEXT_CAP:
        return {}, REASON_TOOL_CONTEXT_TRUNCATED
    return fields, None


def _from_correction(
    rows: list[dict], index: int, harvest: Harvest,
) -> RegressionCase | None:
    """訂正のターン ``rows[index]`` が指すターンをケースにする (除外は ``harvest`` に数える)。"""
    exp = rows[index]
    sig = _signals(exp)
    if not (sig.get("user_correction") and sig.get("correction_verdict") == _PROMOTED_VERDICT):
        harvest.rejected[REJECT_UNVERIFIED_CORRECTION] += 1
        return None
    if not teaches_via_verified_correction(sig):
        harvest.rejected[REJECT_CORRECTION_TURN_FAILED] += 1
        return None
    correct = str(sig.get("correction_correct_value") or "").strip()
    if not correct:
        harvest.rejected[REJECT_NO_CORRECT_VALUE] += 1
        return None
    target = resolve_corrected_turn(rows, index)
    if target is None:
        harvest.rejected[REJECT_TARGET_UNRESOLVED] += 1
        return None
    replay, reason = _replay(target)
    if reason is not None:
        harvest.rejected[reason] += 1
        return None
    return make_case(
        source=SOURCE_VERIFIED_CORRECTION,
        mode=str(target.get("mode") or exp.get("mode") or "chat"),
        query=str(target.get("query") or ""),
        expected_value=correct,
        wrong_value=str(sig.get("correction_wrong_claim") or "").strip(),
        hint=str(sig.get("user_correction") or "").strip(),
        experience_ids=(str(target.get("id") or ""), str(exp.get("id") or "")),
        session_id=str(target.get("session_id") or ""),
        **replay,
    )


def _from_turn(exp: Mapping[str, Any], source: str, hint: str, harvest: Harvest) -> RegressionCase | None:
    """そのターン自身をケースにする (👎 / 検証器の失敗)。"""
    replay, reason = _replay(exp)
    if reason is not None:
        harvest.rejected[reason] += 1
        return None
    sig = _signals(exp)
    quarantined = source == SOURCE_VERIFIER_FAILURE and is_model_origin(
        str(sig.get("turn_outcome_reason") or ""),
    )
    case = make_case(
        source=source,
        mode=str(exp.get("mode") or "chat"),
        query=str(exp.get("query") or ""),
        hint=hint,
        experience_ids=(str(exp.get("id") or ""),),
        session_id=str(exp.get("session_id") or ""),
        **replay,
    )
    if quarantined:
        case.status = STATUS_QUARANTINED
    return case


def _response(row: Mapping[str, Any]) -> str:
    return str(row.get("response_full") or row.get("response_summary") or "")


def _prior_turns(rows: list[dict], finding: Finding) -> list[PriorTurn] | None:
    """検出のターンの直前の会話 (≤ 4 ターン・計 2000 字)。

    古い側から字数の上限まで削り、残った会話 + 検出のターン (記録した応答) で同じ種類の
    検出が **再現する** ときだけ返す (根拠のターンが窓に収まらなければ ``None``)。再現しない
    文脈を持たせると、再生で何を応答しても合格になる。
    """
    at = finding.turn_index
    turns = [
        PriorTurn(query=str(r.get("query") or ""), response=_response(r))
        for r in rows[max(0, at - PRIOR_TURNS_MAX):at]
    ]
    while turns and sum(len(t.query) + len(t.response) for t in turns) > PRIOR_TURNS_MAX_CHARS:
        turns.pop(0)
    if not turns:
        return None
    replay = [{"query": t.query, "response": t.response} for t in turns]
    replay.append({"query": str(rows[at].get("query") or ""), "response": _response(rows[at])})
    reproduced = any(
        f.kind == finding.kind and f.turn_index == len(turns) for f in find_inconsistencies(replay)
    )
    return turns if reproduced else None


def _harvest_consistency(rows: list[dict], processed: Mapping[str, list[str]], harvest: Harvest) -> None:
    """同じモードの経験をセッションごとに並べて整合性の検出をケースにする。"""
    sessions: dict[str, list[dict]] = {}
    for exp in rows:
        sid = str(exp.get("session_id") or "")
        if sid:
            sessions.setdefault(sid, []).append(exp)
    for sid, srows in sessions.items():
        if len(srows) < 2:
            continue
        turns = [
            {
                "turn_id": str(r.get("id") or i), "session_id": sid,
                "query": str(r.get("query") or ""), "response": _response(r),
            }
            for i, r in enumerate(srows)
        ]
        kinds: set[str] = set()
        for finding in find_inconsistencies(turns):
            exp = srows[finding.turn_index]
            exp_id = str(exp.get("id") or "")
            if not exp_id or SOURCE_CONSISTENCY in processed.get(exp_id, ()):
                continue
            if SOURCE_CONSISTENCY in harvest.processed.get(exp_id, ()):
                continue  # 同じターンの 2 件目の検出 (1 ターン 1 件)
            harvest.consistency_findings += 1
            harvest.input += 1
            harvest.mark(exp_id, SOURCE_CONSISTENCY)
            if finding.kind in kinds:
                harvest.duplicate += 1  # 同じ種類はセッションあたり 1 件
                continue
            kinds.add(finding.kind)
            reason = replay_exclusion(exp)
            if reason is not None:
                harvest.rejected[reason] += 1
                continue
            prior = _prior_turns(srows, finding)
            if prior is None:
                harvest.rejected[REJECT_CONTEXT_TOO_LONG] += 1
                continue
            first = finding.turn_index - len(prior)
            harvest.candidates.append(make_case(
                source=SOURCE_CONSISTENCY,
                mode=str(exp.get("mode") or "chat"),
                query=str(exp.get("query") or ""),
                prior_turns=prior,
                hint=finding.kind,
                experience_ids=[str(r.get("id") or "") for r in srows[first:finding.turn_index + 1]],
                session_id=sid,
            ))


def collect_candidates(
    experiences: Iterable[Mapping[str, Any]],
    processed: Mapping[str, list[str]] | None = None,
    *,
    consistency: bool = True,
) -> Harvest:
    """経験 (時系列順の dict) から回帰ケースの候補を作る (純関数。台帳には触らない)。

    ``processed`` (経験 ID → 判定済みの源) にある組は判定し直さない。
    """
    done = processed or {}
    harvest = Harvest()
    by_mode: dict[str, list[dict]] = {}
    for exp in experiences:
        by_mode.setdefault(str(exp.get("mode") or ""), []).append(dict(exp))
    for rows in by_mode.values():
        for index, exp in enumerate(rows):
            exp_id = str(exp.get("id") or "")
            if not exp_id:
                continue
            seen = done.get(exp_id, ())
            sig = _signals(exp)
            if SOURCE_VERIFIED_CORRECTION not in seen and (
                sig.get("user_correction") or sig.get("correction_candidate")
            ):
                if not sig.get("correction_verified_at"):
                    harvest.deferred += 1  # 検証待ち (検証の後の回で判定する)
                else:
                    harvest.input += 1
                    harvest.mark(exp_id, SOURCE_VERIFIED_CORRECTION)
                    case = _from_correction(rows, index, harvest)
                    if case is not None:
                        harvest.candidates.append(case)
            if sig.get("user_negative") is True and SOURCE_USER_NEGATIVE not in seen:
                harvest.input += 1
                harvest.mark(exp_id, SOURCE_USER_NEGATIVE)
                case = _from_turn(exp, SOURCE_USER_NEGATIVE, str(sig.get("user_note") or "").strip(), harvest)
                if case is not None:
                    harvest.candidates.append(case)
            if sig.get("turn_outcome") == "failed" and SOURCE_VERIFIER_FAILURE not in seen:
                harvest.input += 1
                harvest.mark(exp_id, SOURCE_VERIFIER_FAILURE)
                channel = outcome_channel(sig)
                if channel is None or channel == OUTCOME_CHANNEL_OTHER:
                    harvest.rejected[REJECT_NOT_DETERMINISTIC] += 1
                else:
                    case = _from_turn(
                        exp, SOURCE_VERIFIER_FAILURE, str(sig.get("turn_outcome_reason") or "").strip(), harvest,
                    )
                    if case is not None:
                        harvest.candidates.append(case)
        if consistency:
            _harvest_consistency(rows, done, harvest)
    return harvest


def _merge_processed(
    previous: Mapping[str, list[str]], new: Mapping[str, list[str]], live_ids: set[str],
) -> dict[str, list[str]]:
    """判定済みの記録を足し合わせ、バッファから消えた経験の分を刈る。"""
    merged: dict[str, list[str]] = {}
    for source in (previous, new):
        for exp_id, sources in source.items():
            if exp_id not in live_ids:
                continue
            done = merged.setdefault(exp_id, [])
            done.extend(s for s in sources if s not in done)
    return merged


def _growth_settings(config: Mapping[str, Any] | None) -> Any:
    from backend.schemas.learning import GrowthConfig

    raw = ((config or {}).get("learning") or {}).get("growth") or {}
    return GrowthConfig.model_validate(raw)


class GrowthHarvester:
    """Step 13.6 の収穫器 (``memory.protocols.GrowthHarvesterProtocol`` を満たす)。

    ``learning.growth.enabled=false`` / readonly / 経験バッファ無しでは何もせず 0 を返す。
    ``--no-learning`` は SleepTimeWorker の側で呼ばない (同じく 0)。
    """

    def __init__(
        self,
        experience_buf: Any,
        config: Mapping[str, Any] | None,
        *,
        resolver: Callable[[], Any] | None = None,
        debug_logger: Any = None,
    ) -> None:
        self._buf = experience_buf
        self._config = config
        self._resolver = resolver
        self._debug_logger = debug_logger

    def _path_resolver(self) -> Any:
        if self._resolver is not None:
            return self._resolver()
        from backend.config import get_path_resolver

        return get_path_resolver()

    def _partition_key(self, resolver: Any) -> str:
        bound = getattr(self._buf, "bound_path", None)
        name = bound.parent.name if bound is not None else ""
        return name if is_valid_id(name, "mk_") else resolver.active_model_key

    def run(self) -> dict[str, int]:
        result = empty_result()
        settings = _growth_settings(self._config)
        if not settings.enabled:
            logger.debug("Step 13.6 (growth harvest) skipped: learning.growth.enabled=false")
            return result
        if self._buf is None:
            return result
        if is_readonly():
            logger.debug("Step 13.6 (growth harvest) skipped: data is read-only")
            return result
        resolver = self._path_resolver()
        rows = [entry_to_dict(e) for e in list(self._buf.partition_entries())]
        status_path = resolver.learning_path_for("regression_status_file", self._partition_key(resolver))
        status = load_status(status_path)
        harvest = collect_candidates(
            rows, status.harvested, consistency=settings.harvest.consistency,
        )
        ledger = RegressionLedger(
            resolver.resolve_local("regression_cases_file"),
            use_seed=settings.regression.use_seed,
            max_cases=settings.regression.max_cases,
        )
        try:
            existing = {
                (c.session_id, c.hint) for c in ledger.load() if c.source == SOURCE_CONSISTENCY
            }
            fresh: list[RegressionCase] = []
            duplicate = harvest.duplicate
            for case in harvest.candidates:
                if case.source == SOURCE_CONSISTENCY and (case.session_id, case.hint) in existing:
                    duplicate += 1  # このセッションの同じ種類は収穫済み
                    continue
                fresh.append(case)
            added, folded = ledger.add(fresh)
        except DataReadonlyError as e:
            logger.warning("Step 13.6 (growth harvest) skipped: the regression ledger is read-only (%s)", e)
            return result
        duplicate += folded
        live_ids = {str(rid) for r in rows if (rid := r.get("id"))}
        status.harvested = _merge_processed(status.harvested, harvest.processed, live_ids)
        save_status(status_path, status)

        rejected = sum(harvest.rejected.values())
        result.update({
            "growth_harvest_input": harvest.input,
            "growth_harvest": len(added) + duplicate,
            "growth_cases_added": len(added),
            "growth_cases_rejected": rejected,
            "growth_cases_duplicate": duplicate,
            "growth_consistency_findings": harvest.consistency_findings,
        })
        quarantined = sum(1 for c in added if c.status == STATUS_QUARANTINED)
        if added or rejected or duplicate:
            logger.info(
                "Step 13.6: harvested %d regression case(s) (quarantined %d, duplicate %d, rejected %d %s)",
                len(added), quarantined, duplicate, rejected, dict(sorted(harvest.rejected.items())),
            )
        if self._debug_logger is not None:
            try:
                self._debug_logger.log_memory_op("growth_harvest", {
                    **result,
                    "rejected_by_reason": dict(sorted(harvest.rejected.items())),
                    "quarantined": quarantined,
                    "deferred": harvest.deferred,
                    "sources_added": dict(Counter(c.source for c in added)),
                })
            except Exception as e:  # noqa: BLE001 - 観測の失敗で収穫を止めない
                logger.warning("DebugLogger.log_memory_op failed: %s", e)
        return result


__all__ = [
    "REJECT_CONTEXT_TOO_LONG",
    "REJECT_CORRECTION_TURN_FAILED",
    "REJECT_NOT_DETERMINISTIC",
    "REJECT_NO_CORRECT_VALUE",
    "REJECT_TARGET_UNRESOLVED",
    "REJECT_UNVERIFIED_CORRECTION",
    "GrowthHarvester",
    "Harvest",
    "collect_candidates",
    "empty_result",
]
