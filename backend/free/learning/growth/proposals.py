"""提案キュー (``learning.growth_proposal``) と ``GrowthPolicy`` (docs/f_04 §6.6)。

Level 1 の採用ゲートを通った候補 (プロンプト・生成パラメータ) を、自動採用が無効な間は
採用せず **提案** として積む。利用者が採用 / 却下する (API / CLI)。

- 提案は 1 件 1 ファイル (``store/learning/<mk>/growth/proposals/<gpp_id>.json``、封筒 json)。
  候補の本文を持つ。採用・却下・期限切れ・置き換えで成長の台帳 (``learning.growth_ledger``)
  に行を書いた **後に** ファイルを消す (候補の本文は採用されたものだけが ``history/`` /
  ``generation_deltas.json`` に残る)。
- 同じ (asset_kind, mode) に未決の提案があれば古い方を ``supersede`` にする。未決の提案は
  ``learning.growth.max_pending_proposals`` を超えたら古い順に、``proposal_ttl_days`` を
  過ぎたら ``expire`` (Level 1 の入口で掃く)。
- ``GrowthPolicy.auto_adopt_effective()``: 自動採用が効くのは 5 つの条件をすべて満たすとき
  (``auto_adopt`` / 学習の健康診断が ``continue`` / ``single_slot`` でない / guard が
  ``min_guards_for_auto`` 件以上 / 候補で guard 不合格 0)。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from backend.free.learning.growth.ledger import (
    ASSET_GENERATION_PARAMS,
    ASSET_PROMPT,
    LEDGER_FILE,
    PROPOSAL_ID_PREFIX,
    REASON_MAX_PENDING,
    REASON_SUPERSEDED,
    REASON_TTL,
    AssetKind,
    GrowthGate,
    GrowthLedger,
    GrowthLedgerRow,
)
from backend.free.learning.prompt_adoption_ledger import AdoptionCaseRef
from backend.io.codec import CodecError, codec_for, persisted
from backend.io.format_registry import FormatSpec, register_format
from backend.io.id_registry import new_id
from backend.io.versioned import VersionedPayloadFile
from backend.log_config import get_logger
from backend.utils import format_utc, parse_utc, utc_now_dt

logger = get_logger("learning.growth.proposals")

PROPOSALS_DIR = "proposals"

#: 止める基準 (§6.6 / §10): 自然な利用で判定が付いた (健康診断が ``hold`` でない) のに
#: pressure がこれ未満なら、提案を作らず guard だけ回す。
MIN_PRESSURE_FOR_PROPOSALS = 5

#: ``auto_adopt_effective`` が返す理由 (満たさなかった最初の条件)。
REASON_AUTO_ADOPT_OFF = "auto_adopt_off"
REASON_GROWTH_DISABLED = "growth_disabled"
REASON_HEALTH_PREFIX = "health_"
REASON_SINGLE_SLOT = "single_slot"
REASON_INSUFFICIENT_GUARDS = "insufficient_guards"
REASON_CANDIDATE_GUARD_FAILED = "candidate_guard_failed"


# ── 形式 ──


@persisted()
@dataclass(kw_only=True)
class ProposalCandidate:
    """候補の本文 (プロンプトは ``prompt_text``、生成パラメータは ``deltas``)。"""

    prompt_text: str = ""
    deltas: dict[str, float] = field(default_factory=dict)
    _extra: dict[str, Any] | None = None


@persisted()
@dataclass(kw_only=True)
class GrowthProposal:
    """``proposals/<gpp_id>.json`` のペイロード。"""

    proposal_id: str
    asset_kind: AssetKind = ASSET_PROMPT
    mode: str = ""
    #: 提案時点の現行版 (成長の台帳の ``parent_version`` と同じ)。採用時に現行と違えば 409。
    parent_version: str = ""
    candidate: ProposalCandidate = field(default_factory=ProposalCandidate)
    #: 採用ゲートの結果 (成長の台帳の ``gate`` と同じ形)。
    measure: GrowthGate | None = None
    created_at: str = ""
    expires_at: str = ""
    candidate_digest: str = ""
    #: 採用ゲートの評価ケースへの参照 (問いの本文は持たない。採用台帳へそのまま渡す)。
    cases: list[AdoptionCaseRef] = field(default_factory=list)
    #: 採用後監視の基準値から除く問いの ID (``prompt_eval._case_id``)。
    case_ids: list[str] = field(default_factory=list)
    #: 標本に入った回帰台帳の guard (採用後の guard 監視の対象)。
    guard_ids: list[str] = field(default_factory=list)
    #: 採用時に ``update_evolved`` へ渡す fitness (プロンプトだけ)。
    fitness: float | None = None
    level1_session_id: str | None = None
    _extra: dict[str, Any] | None = None

    @property
    def regression_case_ids(self) -> list[str]:
        from backend.free.learning.growth.regression import CASE_ID_PREFIX

        return [c.case_id for c in self.cases if c.case_id.startswith(CASE_ID_PREFIX)]

    @property
    def experience_ids(self) -> list[str]:
        seen: list[str] = []
        for ref in self.cases:
            for exp_id in ref.experience_ids:
                if exp_id not in seen:
                    seen.append(exp_id)
        return seen


GROWTH_PROPOSAL_FORMAT = register_format(FormatSpec(
    format_id="learning.growth_proposal",
    version=1,
    klass="sot",
    writers=frozenset({"free"}),
    path_key=f"store/learning/<mk>/growth/{PROPOSALS_DIR}/<proposal>.json",
    retention=(
        "pending only: deleted after the ledger row of adopt/reject/expire/supersede; "
        "learning.growth.max_pending_proposals (oldest expire first) and proposal_ttl_days"
    ),
    export=True,
    enums={"asset_kind": "closed"},
    records=(GrowthProposal,),
))

_PROPOSAL_CODEC = codec_for(GrowthProposal)


def _proposal_file(path: Path) -> VersionedPayloadFile:
    return VersionedPayloadFile(
        GROWTH_PROPOSAL_FORMAT, path, component="GrowthProposals", state_logger=logger,
        decode=_PROPOSAL_CODEC.decode, encode=_PROPOSAL_CODEC.encode,
    )


def load_proposal(path: Path | str) -> GrowthProposal | None:
    """提案ファイルを 1 件読む (無い・読めない・版が新しければ ``None``)。"""
    f = _proposal_file(Path(path))
    if f.load() and isinstance(f.payload, GrowthProposal):
        return f.payload
    return None


def save_proposal(path: Path | str, proposal: GrowthProposal) -> bool:
    """提案ファイルを書く (原子的。書けたら ``True``)。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    f = _proposal_file(target)
    f.payload = proposal
    return f.save()


def candidate_digest(candidate: ProposalCandidate) -> str:
    """候補本文の blake2b (16 hex)。台帳は本文の代わりにこれを持つ。"""
    payload = json.dumps(
        {"prompt_text": candidate.prompt_text, "deltas": dict(sorted(candidate.deltas.items()))},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def make_proposal(
    *,
    asset_kind: AssetKind,
    mode: str,
    parent_version: str,
    candidate: ProposalCandidate,
    measure: GrowthGate | None,
    ttl_days: int,
    cases: Iterable[AdoptionCaseRef] = (),
    case_ids: Iterable[str] = (),
    guard_ids: Iterable[str] = (),
    fitness: float | None = None,
    level1_session_id: str | None = None,
) -> GrowthProposal:
    """新しい提案を作る (ID を発番し、期限と候補の digest を付ける)。"""
    now = utc_now_dt()
    return GrowthProposal(
        proposal_id=new_id(PROPOSAL_ID_PREFIX),
        asset_kind=asset_kind,
        mode=mode,
        parent_version=str(parent_version),
        candidate=candidate,
        measure=measure,
        created_at=format_utc(now),
        expires_at=format_utc(now + timedelta(days=max(1, int(ttl_days)))),
        candidate_digest=candidate_digest(candidate),
        cases=list(cases),
        case_ids=list(case_ids),
        guard_ids=sorted(set(guard_ids)),
        fitness=fitness,
        level1_session_id=level1_session_id,
    )


# ── キュー ──


class ProposalQueue:
    """未決の提案 (``growth/proposals/``) と成長の台帳 (``growth/ledger.jsonl``) を束ねる。

    Args:
        growth_dir: パーティションの ``growth/`` (``store/learning/<mk>/growth``)。
    """

    def __init__(self, growth_dir: Path | str) -> None:
        self.growth_dir = Path(growth_dir)
        self.dir = self.growth_dir / PROPOSALS_DIR
        self.ledger = GrowthLedger(self.growth_dir / LEDGER_FILE)

    def _path(self, proposal_id: str) -> Path:
        return self.dir / f"{proposal_id}.json"

    # ── 読む ──

    def get(self, proposal_id: str) -> GrowthProposal | None:
        """未決の提案を 1 件 (無い・読めない・ID の形が違えば ``None``)。"""
        if not proposal_id.startswith(PROPOSAL_ID_PREFIX) or "/" in proposal_id or "\\" in proposal_id:
            return None
        return load_proposal(self._path(proposal_id))

    def pending(self) -> list[GrowthProposal]:
        """未決の提案を作った順に (読めないファイルは飛ばして WARNING)。"""
        if not self.dir.is_dir():
            return []
        out: list[GrowthProposal] = []
        skipped = 0
        for path in sorted(self.dir.glob(f"{PROPOSAL_ID_PREFIX}*.json")):
            try:
                proposal = self.get(path.stem)
            except CodecError:
                proposal = None
            if proposal is None:
                skipped += 1
                continue
            out.append(proposal)
        if skipped:
            logger.warning("Skipped %d unreadable growth proposal file(s) in %s", skipped, self.dir)
        out.sort(key=lambda p: _time_key(p.created_at))
        return out

    # ── 書く ──

    def _ledger_row(self, proposal: GrowthProposal, op: str, **fields: Any) -> GrowthLedgerRow:
        return self.ledger.append(
            proposal_id=proposal.proposal_id,
            op=op,
            asset_kind=proposal.asset_kind,
            mode=proposal.mode,
            parent_version=proposal.parent_version,
            candidate_digest=proposal.candidate_digest,
            gate=proposal.measure,
            regression_case_ids=proposal.regression_case_ids,
            experience_ids=proposal.experience_ids,
            level1_session_id=proposal.level1_session_id,
            **fields,
        )

    def propose(
        self, proposal: GrowthProposal, *, max_pending: int, reason: str = "",
    ) -> tuple[GrowthLedgerRow, list[str]]:
        """提案ファイルと台帳の ``propose`` 行を書く。``(propose 行, 期限切れ・置き換えにした ID)``。

        同じ (asset_kind, mode) の未決の提案は ``supersede``、件数が ``max_pending`` を超えたら
        古い順に ``expire`` (``reason=max_pending``)。
        """
        closed: list[str] = []
        for old in self.pending():
            if (old.asset_kind, old.mode) == (proposal.asset_kind, proposal.mode):
                self.resolve(old, "supersede", reason=REASON_SUPERSEDED)
                closed.append(old.proposal_id)
        if not save_proposal(self._path(proposal.proposal_id), proposal):
            raise OSError(f"failed to write growth proposal {proposal.proposal_id}")
        row = self._ledger_row(proposal, "propose", reason=reason)
        closed.extend(self.enforce_max_pending(max_pending))
        return row, closed

    def resolve(
        self,
        proposal: GrowthProposal,
        op: str,
        *,
        decided_by: str | None = None,
        reason: str = "",
        adoption_id: str | None = None,
    ) -> GrowthLedgerRow:
        """提案を閉じる: 台帳に行を書いて **から** ファイルを消す (書けなければ消さない)。"""
        row = self._ledger_row(
            proposal, op, decided_by=decided_by, reason=reason, adoption_id=adoption_id,
        )
        try:
            self._path(proposal.proposal_id).unlink()
        except FileNotFoundError:
            pass
        return row

    def enforce_max_pending(self, max_pending: int) -> list[str]:
        """未決が ``max_pending`` を超えた分を古い順に ``expire`` する。"""
        pending = self.pending()
        excess = len(pending) - max(1, int(max_pending))
        expired: list[str] = []
        for proposal in pending[:max(0, excess)]:
            self.resolve(proposal, "expire", reason=REASON_MAX_PENDING)
            expired.append(proposal.proposal_id)
        return expired

    def sweep(self, *, max_pending: int) -> list[str]:
        """期限 (``expires_at``) を過ぎた提案と上限超過を ``expire`` する。閉じた ID を返す。"""
        now = utc_now_dt()
        expired: list[str] = []
        for proposal in self.pending():
            at = parse_utc(proposal.expires_at)
            if at is not None and at <= now:
                self.resolve(proposal, "expire", reason=REASON_TTL)
                expired.append(proposal.proposal_id)
        expired.extend(self.enforce_max_pending(max_pending))
        if expired:
            logger.info("Growth proposals expired: %d (%s)", len(expired), ", ".join(expired))
        return expired


def _time_key(value: str) -> float:
    at = parse_utc(value)
    return float("-inf") if at is None else at.timestamp()


# ── GrowthPolicy ──


@dataclass(frozen=True)
class AutoAdoptStatus:
    """``{configured, effective, reason}`` (``reason`` は満たさなかった最初の条件、効けば空)。"""

    configured: bool
    effective: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"configured": self.configured, "effective": self.effective, "reason": self.reason}


class GrowthPolicy:
    """自動採用が効くか (§6.6 の 5 条件) と、提案を作ってよいか (止める基準)。

    Args:
        settings: ``learning.growth`` (``GrowthConfig``)。
        health: 学習の健康診断 (``health_report.health_snapshot`` の形) を返す関数。読めなければ ``None``。
            自動採用の設定が真のときと提案の直前だけ呼ぶ (経験の全走査なので)。
        single_slot: ``llama.slots < 2`` が確定しているか。
        guard_count: モード → そのモードの guard 件数。
    """

    def __init__(
        self,
        settings: Any,
        *,
        health: Callable[[], dict[str, Any] | None],
        single_slot: Callable[[], bool],
        guard_count: Callable[[str], int],
    ) -> None:
        self.settings = settings
        self._health = health
        self._single_slot = single_slot
        self._guard_count = guard_count

    def auto_adopt_effective(
        self,
        mode: str | None = None,
        *,
        guards_failed: int | None = None,
        modes: Iterable[str] = (),
    ) -> AutoAdoptStatus:
        """自動採用が効くか。``mode`` が無ければ ``modes`` のどれかで guard が足りるかを見る
        (``GET /api/learning/growth`` の表示)。``guards_failed`` は候補の採用ゲートの結果。
        """
        configured = bool(self.settings.auto_adopt)
        if not configured:
            return AutoAdoptStatus(False, False, REASON_AUTO_ADOPT_OFF)
        if not self.settings.enabled:
            return AutoAdoptStatus(True, False, REASON_GROWTH_DISABLED)
        verdict = str((self._health() or {}).get("verdict") or "unavailable")
        if verdict != "continue":
            return AutoAdoptStatus(True, False, f"{REASON_HEALTH_PREFIX}{verdict}")
        if self._single_slot():
            return AutoAdoptStatus(True, False, REASON_SINGLE_SLOT)
        need = int(self.settings.min_guards_for_auto)
        targets = [mode] if mode is not None else list(modes)
        if need > 0 and not any(self._guard_count(m) >= need for m in targets):
            return AutoAdoptStatus(True, False, REASON_INSUFFICIENT_GUARDS)
        if guards_failed:
            return AutoAdoptStatus(True, False, REASON_CANDIDATE_GUARD_FAILED)
        return AutoAdoptStatus(True, True, "")

    def should_propose(self, pressure: Callable[[], int]) -> bool:
        """止める基準: 自然な利用で判定が付いたのに pressure が少なければ提案を作らない。

        ``pressure`` (そのモードの pressure 件数) は判定が付いたときだけ数える。
        """
        verdict = str((self._health() or {}).get("verdict") or "hold")
        return verdict == "hold" or pressure() >= MIN_PRESSURE_FOR_PROPOSALS


__all__ = [
    "ASSET_GENERATION_PARAMS",
    "ASSET_PROMPT",
    "GROWTH_PROPOSAL_FORMAT",
    "MIN_PRESSURE_FOR_PROPOSALS",
    "PROPOSALS_DIR",
    "REASON_AUTO_ADOPT_OFF",
    "REASON_CANDIDATE_GUARD_FAILED",
    "REASON_GROWTH_DISABLED",
    "REASON_HEALTH_PREFIX",
    "REASON_INSUFFICIENT_GUARDS",
    "REASON_SINGLE_SLOT",
    "AutoAdoptStatus",
    "GrowthPolicy",
    "GrowthProposal",
    "ProposalCandidate",
    "ProposalQueue",
    "candidate_digest",
    "load_proposal",
    "make_proposal",
    "save_proposal",
]
