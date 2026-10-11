"""Level 2 (成長) の提案・採用・巻き戻し・監視 (docs/f_04 §6.6 / §6.8)。

``LearningScheduler`` に束ねた 1 パーティション (``store/learning/<mk>/``) の上で動く。

- **提案**: Level 1 の採用ゲートを通った候補を、``GrowthPolicy`` が自動採用を認めなければ
  提案キューへ積む (:meth:`GrowthService.propose`)。
- **利用者の採用・却下・巻き戻し** (API / CLI): 既存の経路 (``update_evolved`` /
  ``restore_deltas`` → 採用台帳 → 採用後監視) をそのまま使い、成長の台帳に行を書く。
- **採用した版の監視**: 採用時に標本へ入った guard を ``phase8_growth_regression`` が採用後の
  資産で回し、採用時より不合格が増えたら、自動採用した版は巻き戻し (``decided_by=monitor``)、
  利用者が採用した版は ``rollback_suggested`` を立てて WARNING を 1 回出す (自動では戻さない)。
  guard の連続合格は資産が変わると 0 から数え直す (PR4) ので、監視は役割 (guard) ではなく
  **採用後の資産での直近の合否** (``last_result`` と ``asset_digest``) で数える。

採用の記録は ``learning_state`` の ``_extra["growth_adoptions"]`` に置く (宣言フィールドに
すると形式の版が動くので、``generation_adoptions`` と同じ未知キーの通路で運ぶ)。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.free.learning.growth.ledger import (
    ASSET_GENERATION_PARAMS,
    ASSET_PROMPT,
    PROPOSABLE_KINDS,
    REASON_GUARD_REGRESSION,
    REASON_USER,
    GrowthGate,
    GrowthLedgerRow,
    encode_gate,
    encode_row,
    gate_from_verdict,
)
from backend.free.learning.growth.proposals import (
    AutoAdoptStatus,
    GrowthPolicy,
    GrowthProposal,
    ProposalCandidate,
    ProposalQueue,
    make_proposal,
)
from backend.log_config import get_logger
from backend.utils import utc_now

if TYPE_CHECKING:
    from backend.free.learning.scheduler import LearningScheduler

logger = get_logger("learning.growth.service")

#: ``learning_state._extra`` の中の採用の記録 (``"<asset_kind>:<mode>"`` → 記録)。
GROWTH_ADOPTIONS_KEY = "growth_adoptions"

#: 成長の台帳に書く巻き戻しの理由 (欠陥率の採用後監視、f_04 §4.5 / §4.7)。
REASON_DEFECT_RATE = "defect_rate_regression"
#: 409 の理由。
CONFLICT_STALE_PARENT = "stale_parent"
CONFLICT_LEARNING_RUNNING = "learning_running"
CONFLICT_IDENTICAL = "identical_after_normalization"
CONFLICT_NO_TARGET = "no_rollback_target"
#: ``recent`` に載せる台帳の末尾の行数。
RECENT_ROWS = 20


class GrowthError(Exception):
    """API / CLI へ返す失敗 (``status`` は HTTP の状態、``reason`` は機械可読の理由)。"""

    status = 400

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


class GrowthNotFound(GrowthError):
    status = 404


class GrowthConflict(GrowthError):
    status = 409


class GrowthReadonly(GrowthError):
    status = 423


class GrowthUnavailable(GrowthError):
    status = 503


class GrowthPolicyDenied(GrowthError):
    """組織ポリシーパックで固定された資産 (docs/p_09 §4)。"""

    status = 403

    def __init__(self, reason: str, detail: str = "", context: dict[str, Any] | None = None) -> None:
        super().__init__(reason, detail)
        self.context = dict(context or {})


#: 自動採用を止めた理由 (組織ポリシーパックで固定、docs/p_09 §4)。
REASON_POLICY_PINNED = "policy_pinned"


def adoption_key(asset_kind: str, mode: str) -> str:
    return f"{asset_kind}:{mode}"


class GrowthService:
    """1 パーティションの成長層の操作 (``LearningScheduler`` の状態と部品を使う)。"""

    def __init__(self, scheduler: "LearningScheduler") -> None:
        self.s = scheduler

    # ── 置き場と部品 ──

    @property
    def growth_dir(self) -> Path:
        """``store/learning/<mk>/growth`` (プロンプトの置き場の隣、束ねたパーティション)。"""
        return Path(self.s.prompt_manager.prompt_dir).parent / "growth"

    def queue(self) -> ProposalQueue:
        return ProposalQueue(self.growth_dir)

    def settings(self) -> Any:
        return self.s._growth_settings()

    def policy(self) -> GrowthPolicy:
        return GrowthPolicy(
            self.settings(),
            health=self.s._growth_health,
            single_slot=lambda: bool(self.s.single_slot),
            guard_count=self._guard_count,
        )

    def _cases_and_status(self) -> tuple[list[Any], Any]:
        from backend.free.learning.growth.regression import load_status

        settings = self.settings()
        return (
            self.s._regression_ledger(settings).cases(),
            load_status(self.s._regression_status_path()),
        )

    def _role_counts(self) -> Any:
        from backend.free.learning.growth.replay import count_roles

        cases, status = self._cases_and_status()
        return count_roles(cases, status)

    def _guard_count(self, mode: str) -> int:
        return int(self._role_counts().by_mode.get(mode, {}).get("guards", 0))

    def _pressure_count(self, mode: str) -> int:
        return int(self._role_counts().by_mode.get(mode, {}).get("pressure", 0))

    def current_version(self, asset_kind: str, mode: str) -> str:
        """その資産の現行版 (プロンプトは ``meta.version``、生成パラメータは ``delta_version``)。"""
        if asset_kind == ASSET_PROMPT:
            return str(self.s.prompt_manager.get_meta(mode).version)
        if asset_kind == ASSET_GENERATION_PARAMS:
            from backend.free.learning.generation_delta_store import delta_version

            evolver = self._evolver()
            return delta_version(evolver.persisted_deltas(mode))
        raise GrowthNotFound("unknown_asset_kind", f"asset_kind {asset_kind!r} cannot be adopted")

    def _evolver(self) -> Any:
        evolver = self.s._generation_param_evolver
        if evolver is None:
            raise GrowthUnavailable("generation_evolver_unavailable")
        return evolver

    # ── 採用の記録 (learning_state._extra) ──

    def adoptions(self) -> dict[str, dict[str, Any]]:
        records = (self.s._state_extra or {}).get(GROWTH_ADOPTIONS_KEY)
        return dict(records) if isinstance(records, dict) else {}

    def _store_adoptions(self, records: dict[str, dict[str, Any]]) -> None:
        extra = dict(self.s._state_extra or {})
        if records:
            extra[GROWTH_ADOPTIONS_KEY] = records
        else:
            extra.pop(GROWTH_ADOPTIONS_KEY, None)
        self.s._state_extra = extra or None

    def adoption_of(self, asset_kind: str, mode: str) -> dict[str, Any] | None:
        rec = self.adoptions().get(adoption_key(asset_kind, mode))
        return rec if isinstance(rec, dict) else None

    def track_adoption(
        self,
        asset_kind: str,
        mode: str,
        *,
        decided_by: str,
        adopted_version: str,
        rollback_to: Any,
        guard_ids: Iterable[str] = (),
        guards_failed: int | None = None,
        proposal_id: str = "",
    ) -> None:
        """採用を記録する (guard 監視と巻き戻し先)。保存は呼出側 (``_save_state``)。"""
        records = self.adoptions()
        ids = sorted(set(guard_ids))
        records[adoption_key(asset_kind, mode)] = {
            "decided_by": decided_by,
            "proposal_id": proposal_id,
            "adopted_version": str(adopted_version),
            "rollback_to": rollback_to,
            "guard_ids": ids,
            "guards_failed": int(guards_failed or 0),
            "adopted_at": utc_now(),
            "monitoring": bool(ids),
            "rollback_suggested": False,
            "suggestion_reason": "",
        }
        self._store_adoptions(records)

    def _drop_adoption(self, asset_kind: str, mode: str) -> None:
        records = self.adoptions()
        if records.pop(adoption_key(asset_kind, mode), None) is not None:
            self._store_adoptions(records)

    def decided_by_user(self, asset_kind: str, mode: str, adopted_version: str) -> bool:
        """現行の版を利用者が採用したか (欠陥率の採用後監視が巻き戻さず提案に留める)。"""
        rec = self.adoption_of(asset_kind, mode)
        return bool(
            rec and rec.get("decided_by") == "user"
            and str(rec.get("adopted_version") or "") == str(adopted_version)
        )

    def suggest_rollback(self, asset_kind: str, mode: str, reason: str) -> None:
        """利用者が採用した版に巻き戻しを提案する (WARNING は 1 回だけ)。"""
        records = self.adoptions()
        rec = records.get(adoption_key(asset_kind, mode))
        if not isinstance(rec, dict) or rec.get("rollback_suggested"):
            return
        rec = {**rec, "rollback_suggested": True, "suggestion_reason": reason}
        records[adoption_key(asset_kind, mode)] = rec
        self._store_adoptions(records)
        logger.warning(
            "Growth: the user-adopted %s for %s (v%s) regressed (%s); not rolled back "
            "automatically — roll it back with `evoref growth rollback --kind %s --mode %s`",
            asset_kind, mode, rec.get("adopted_version"), reason, asset_kind, mode,
        )

    # ── 台帳 ──

    def append_row(self, **fields: Any) -> GrowthLedgerRow | None:
        """成長の台帳へ 1 行 (書けなくても採用・巻き戻しは取り消さず WARNING)。"""
        try:
            return self.queue().ledger.append(**fields)
        except Exception as exc:  # noqa: BLE001 - 台帳の失敗で資産の変更を巻き戻さない
            logger.warning(
                "Growth: failed to append the growth ledger (%s %s %s): %s",
                fields.get("op"), fields.get("asset_kind"), fields.get("mode"), exc,
            )
            return None

    # ── 提案 (Level 1) ──

    def auto_adopt(
        self, mode: str, verdict: dict[str, Any], *, asset_kind: str = ASSET_PROMPT,
    ) -> AutoAdoptStatus:
        """採用ゲートを通った候補を自動採用してよいか。

        組織ポリシーパックがその資産を固定していれば、自動採用が効く条件を満たしていても
        採用せず提案にする (``policy_pinned``。提案は組織へ集めるバンドルの材料、p_09 §5)。
        """
        guards_failed = verdict.get("guards_failed")
        status = self.policy().auto_adopt_effective(
            mode, guards_failed=guards_failed if isinstance(guards_failed, int) else None,
        )
        if status.effective and _policy_denial(asset_kind, mode) is not None:
            return AutoAdoptStatus(status.configured, False, REASON_POLICY_PINNED)
        return status

    def propose(
        self,
        asset_kind: str,
        mode: str,
        *,
        candidate: ProposalCandidate,
        verdict: dict[str, Any],
        parent_version: str,
        cases: Iterable[Any] = (),
        case_ids: Iterable[str] = (),
        guard_ids: Iterable[str] = (),
        fitness: float | None = None,
        level1_session_id: str | None = None,
    ) -> GrowthProposal | None:
        """提案ファイルと台帳の ``propose`` 行を書く。止める基準で作らなければ ``None``。"""
        settings = self.settings()
        if not self.policy().should_propose(lambda: self._pressure_count(mode)):
            logger.info(
                "Growth: %s proposal for %s withheld (natural use reached a verdict but "
                "pressure is low); guards keep running", asset_kind, mode,
            )
            return None
        refs = list(cases)
        from backend.free.learning.prompt_adoption_ledger import case_digest

        proposal = make_proposal(
            asset_kind=asset_kind,  # type: ignore[arg-type]
            mode=mode,
            parent_version=parent_version,
            candidate=candidate,
            measure=gate_from_verdict(verdict, case_digest=case_digest([r.case_id for r in refs])),
            ttl_days=settings.proposal_ttl_days,
            cases=refs,
            case_ids=case_ids,
            guard_ids=guard_ids,
            fitness=fitness,
            level1_session_id=level1_session_id,
        )
        row, closed = self.queue().propose(
            proposal, max_pending=settings.max_pending_proposals,
            reason=str(verdict.get("reason") or ""),
        )
        logger.info(
            "Growth: proposed %s for %s (%s, parent v%s, row %s%s)",
            asset_kind, mode, proposal.proposal_id, parent_version, row.row_id,
            f"; closed {', '.join(closed)}" if closed else "",
        )
        return proposal

    def record_auto_adoption(
        self,
        asset_kind: str,
        mode: str,
        *,
        verdict: dict[str, Any],
        parent_version: str,
        adopted_version: str,
        rollback_to: Any,
        candidate: ProposalCandidate,
        cases: Iterable[Any] = (),
        guard_ids: Iterable[str] = (),
        adoption_id: str | None = None,
        level1_session_id: str | None = None,
    ) -> None:
        """自動採用 (``decided_by=auto``) を台帳と監視の記録に残す。"""
        from backend.free.learning.growth.proposals import candidate_digest
        from backend.free.learning.growth.regression import CASE_ID_PREFIX
        from backend.free.learning.prompt_adoption_ledger import case_digest

        refs = list(cases)
        self.append_row(
            op="adopt", asset_kind=asset_kind, mode=mode, parent_version=str(parent_version),
            candidate_digest=candidate_digest(candidate), decided_by="auto",
            gate=gate_from_verdict(verdict, case_digest=case_digest([r.case_id for r in refs])),
            reason=str(verdict.get("reason") or ""), adoption_id=adoption_id,
            regression_case_ids=[r.case_id for r in refs if r.case_id.startswith(CASE_ID_PREFIX)],
            experience_ids=_experience_ids(refs), level1_session_id=level1_session_id,
        )
        guards_failed = verdict.get("guards_failed")
        self.track_adoption(
            asset_kind, mode, decided_by="auto", adopted_version=adopted_version,
            rollback_to=rollback_to, guard_ids=guard_ids,
            guards_failed=guards_failed if isinstance(guards_failed, int) else 0,
        )

    def sweep(self) -> int:
        """期限切れ・上限超過の提案を ``expire`` する (Level 1 の入口)。閉じた件数。"""
        try:
            return len(self.queue().sweep(max_pending=self.settings().max_pending_proposals))
        except Exception as exc:  # noqa: BLE001 - 掃除の失敗で学習を止めない
            logger.warning("Growth: proposal sweep failed: %r", exc)
            return 0

    # ── 利用者の操作 (API / CLI) ──

    @staticmethod
    def _guard_policy(asset_kind: str, mode: str) -> None:
        denial = _policy_denial(asset_kind, mode)
        if denial is not None:
            raise GrowthPolicyDenied(denial.reason, denial.detail, denial.context())

    def _guard_write(self) -> None:
        from backend.io.readonly import is_readonly, readonly_reason

        if is_readonly():
            raise GrowthReadonly("data_readonly", str(readonly_reason() or "read-only"))
        if self.s.running:
            raise GrowthConflict(CONFLICT_LEARNING_RUNNING, "Level 1 is running")

    def get_proposal(self, proposal_id: str) -> GrowthProposal:
        proposal = self.queue().get(proposal_id)
        if proposal is None:
            raise GrowthNotFound("proposal_not_found", f"no pending proposal {proposal_id}")
        return proposal

    def adopt(self, proposal_id: str) -> dict[str, Any]:
        """利用者の採用。``parent_version`` が現行と違えば 409 (``stale_parent``)。"""
        self._guard_write()
        proposal = self.get_proposal(proposal_id)
        kind, mode = proposal.asset_kind, proposal.mode
        if kind not in PROPOSABLE_KINDS:
            raise GrowthConflict("unsupported_asset_kind", kind)
        self._guard_policy(kind, mode)
        current = self.current_version(kind, mode)
        if current != proposal.parent_version:
            raise GrowthConflict(
                CONFLICT_STALE_PARENT,
                f"proposal was made on v{proposal.parent_version} but the current is v{current}",
            )
        experiences = self.s._get_filtered_experiences()
        adoption_id: str | None = None
        if kind == ASSET_PROMPT:
            pm = self.s.prompt_manager
            rollback_to = int(pm.get_meta(mode).version)
            written = pm.update_evolved(
                mode, proposal.candidate.prompt_text, float(proposal.fitness or 0.0),
            )
            if not written:
                raise GrowthConflict(CONFLICT_IDENTICAL, "the candidate equals the current prompt")
            self.s._record_prompt_adoption(mode, rollback_to, experiences, case_ids=proposal.case_ids)
            m = proposal.measure
            record = self.s._append_adoption_ledger(
                op="adopt", mode=mode, version=pm.get_meta(mode).version,
                parent_version=rollback_to, level1_session_id=proposal.level1_session_id,
                cases=list(proposal.cases),
                wins=m.wins if m else None, losses=m.losses if m else None,
                ties=m.ties if m else None,
            )
            adoption_id = getattr(record, "adoption_id", None)
            adopted = str(pm.get_meta(mode).version)
            rollback_target: Any = rollback_to
        else:
            evolver = self._evolver()
            previous = evolver.persisted_deltas(mode)
            deltas = dict(proposal.candidate.deltas)
            evolver.restore_deltas(mode, deltas)
            self.s._record_generation_adoption(
                mode, previous, deltas, experiences, case_ids=proposal.case_ids,
            )
            adopted = self.current_version(kind, mode)
            rollback_target = previous
        row = self.queue().resolve(
            proposal, "adopt", decided_by="user", reason=REASON_USER, adoption_id=adoption_id,
        )
        guards_failed = proposal.measure.guards_failed if proposal.measure else 0
        self.track_adoption(
            kind, mode, decided_by="user", adopted_version=adopted, rollback_to=rollback_target,
            guard_ids=proposal.guard_ids, guards_failed=guards_failed,
            proposal_id=proposal.proposal_id,
        )
        self.s._save_state()
        if kind == ASSET_PROMPT and self.s.has_active_session():
            # 継続中の session の候補は採用前の本文から作ったもの
            self.s.discard_active_session()
        logger.info(
            "Growth: user adopted %s for %s (%s): v%s -> v%s",
            kind, mode, proposal.proposal_id, proposal.parent_version, adopted,
        )
        return {
            "proposal_id": proposal.proposal_id, "asset_kind": kind, "mode": mode,
            "version": adopted, "row_id": row.row_id, "adoption_id": adoption_id,
        }

    def reject(self, proposal_id: str) -> dict[str, Any]:
        """利用者の却下 (台帳に ``reject`` を書いて提案を消す)。"""
        from backend.io.readonly import is_readonly, readonly_reason

        if is_readonly():
            raise GrowthReadonly("data_readonly", str(readonly_reason() or "read-only"))
        proposal = self.get_proposal(proposal_id)
        row = self.queue().resolve(proposal, "reject", decided_by="user", reason=REASON_USER)
        logger.info("Growth: user rejected %s (%s %s)", proposal_id, proposal.asset_kind, proposal.mode)
        return {
            "proposal_id": proposal_id, "asset_kind": proposal.asset_kind,
            "mode": proposal.mode, "row_id": row.row_id,
        }

    def rollback(self, asset_kind: str, mode: str) -> dict[str, Any]:
        """利用者の巻き戻し (直前の版へ、``decided_by=user``)。"""
        self._guard_write()
        self._guard_policy(asset_kind, mode)
        return self._rollback(asset_kind, mode, decided_by="user", reason=REASON_USER)

    def _rollback(
        self, asset_kind: str, mode: str, *, decided_by: str, reason: str,
    ) -> dict[str, Any]:
        if asset_kind not in PROPOSABLE_KINDS:
            raise GrowthNotFound("unknown_asset_kind", asset_kind)
        current = self.current_version(asset_kind, mode)
        rec = self.adoption_of(asset_kind, mode)
        tracked = rec is not None and str(rec.get("adopted_version") or "") == current
        adoption_id: str | None = None
        if asset_kind == ASSET_PROMPT:
            pm = self.s.prompt_manager
            if tracked:
                target = int(rec["rollback_to"])  # type: ignore[index]
            else:
                parent = getattr(pm.get_meta(mode), "parent_version", None)
                if not isinstance(parent, int):
                    raise GrowthConflict(CONFLICT_NO_TARGET, f"no previous {asset_kind} version for {mode}")
                target = parent
            pm.rollback(mode, target)
            record = self.s._append_adoption_ledger(
                op="rollback", mode=mode, version=pm.get_meta(mode).version, parent_version=target,
            )
            adoption_id = getattr(record, "adoption_id", None)
            restored: Any = target
        else:
            evolver = self._evolver()
            if tracked:
                target_deltas = rec.get("rollback_to")  # type: ignore[union-attr]
            else:
                gen = self.s._generation_adoptions().get(mode)
                target_deltas = (
                    gen.get("rollback_to")
                    if isinstance(gen, dict) and str(gen.get("adopted_version") or "") == current else None
                )
            if not isinstance(target_deltas, dict):
                raise GrowthConflict(CONFLICT_NO_TARGET, f"no previous {asset_kind} for {mode}")
            evolver.restore_deltas(mode, dict(target_deltas))
            restored = dict(target_deltas)
        row = self.append_row(
            op="rollback", proposal_id=str((rec or {}).get("proposal_id") or "") if tracked else "",
            asset_kind=asset_kind, mode=mode, parent_version=current,
            decided_by=decided_by, reason=reason, adoption_id=adoption_id,
        )
        self._drop_adoption(asset_kind, mode)
        self.s._save_state()
        if asset_kind == ASSET_PROMPT:
            if self.s.running:
                # Level 1 の段 (phase8) からの巻き戻し: 実行中の session の候補は巻き戻した版から
                # 作ったものなので、この回の finalize で採用も提案もさせない
                self.s._growth_rolled_back_modes.add(mode)
            elif self.s.has_active_session():
                self.s.discard_active_session()
        new_version = self.current_version(asset_kind, mode)
        log = logger.warning if decided_by == "monitor" else logger.info
        log(
            "Growth: %s for %s rolled back by %s (%s): v%s -> v%s",
            asset_kind, mode, decided_by, reason, current, new_version,
        )
        return {
            "asset_kind": asset_kind, "mode": mode, "from_version": current,
            "version": new_version, "restored": restored,
            "row_id": row.row_id if row else None, "adoption_id": adoption_id,
        }

    def retire_case(self, case_id: str) -> dict[str, Any]:
        """回帰ケースを ``retired`` にする (誤った訂正から収穫したケースを利用者が外す)。"""
        from backend.free.learning.growth.regression import STATUS_RETIRED
        from backend.io.readonly import is_readonly, readonly_reason

        if is_readonly():
            raise GrowthReadonly("data_readonly", str(readonly_reason() or "read-only"))
        ledger = self.s._regression_ledger(self.settings())
        current = next((c for c in ledger.load() if c.case_id == case_id), None)
        if current is None:
            raise GrowthNotFound("case_not_found", f"no regression case {case_id} in the ledger")
        changed = ledger.retire(case_id)
        logger.info("Growth: regression case %s retired by the user (changed=%s)", case_id, changed)
        return {"case_id": case_id, "status": STATUS_RETIRED, "changed": bool(changed)}

    # ── 監視 (phase8 の後) ──

    def monitor_guards(self, cases: Iterable[Any], status: Any) -> dict[str, int]:
        """採用した版の guard を採用後の資産の合否で数え、退行していれば巻き戻す / 提案する。

        Returns:
            ``{"growth_rollbacks": 自動で巻き戻した数, "growth_rollback_suggestions": 提案した数}``
        """
        from backend.free.learning.growth.replay import ASSET_DIGEST_KEY, RESULT_FAIL
        from backend.free.learning.growth.regression import STATUS_ACTIVE

        out = {"growth_rollbacks": 0, "growth_rollback_suggestions": 0}
        records = self.adoptions()
        if not records:
            return out
        active = {c.case_id for c in cases if c.status == STATUS_ACTIVE}
        min_streak = int(self.settings().regression.guard_min_pass_streak)
        for key, rec in list(records.items()):
            if not isinstance(rec, dict) or not rec.get("monitoring"):
                continue
            kind, _, mode = key.partition(":")
            try:
                current = self.current_version(kind, mode)
            except GrowthError:
                continue
            if current != str(rec.get("adopted_version") or ""):
                # 別の採用・手編集・巻き戻しで版が変わった: 監視の対象ではない
                self._drop_adoption(kind, mode)
                continue
            digest = self.s._current_replay_assets(mode)[2]
            guard_ids = [g for g in rec.get("guard_ids") or [] if g in active]
            failed = passed = 0
            for gid in guard_ids:
                entry = status.cases.get(gid)
                if entry is None or (entry._extra or {}).get(ASSET_DIGEST_KEY) != digest:
                    continue
                if entry.last_result == RESULT_FAIL:
                    failed += 1
                elif entry.pass_streak >= min_streak:
                    passed += 1
            if failed > int(rec.get("guards_failed") or 0):
                if rec.get("decided_by") == "auto":
                    self._rollback(kind, mode, decided_by="monitor", reason=REASON_GUARD_REGRESSION)
                    out["growth_rollbacks"] += 1
                elif not rec.get("rollback_suggested"):
                    self.suggest_rollback(kind, mode, REASON_GUARD_REGRESSION)
                    out["growth_rollback_suggestions"] += 1
                continue
            if passed == len(guard_ids):
                # 採用後の資産で guard がそろって合格し直した: 監視を終える (巻き戻し先は残す)
                records = self.adoptions()
                if key in records:
                    records[key] = {**records[key], "monitoring": False}
                    self._store_adoptions(records)
        return out

    def monitored_guard_ids(self) -> frozenset[str]:
        """監視中の採用の guard (``phase8`` が先に回す)。"""
        ids: set[str] = set()
        for rec in self.adoptions().values():
            if isinstance(rec, dict) and rec.get("monitoring"):
                ids.update(str(g) for g in rec.get("guard_ids") or [])
        return frozenset(ids)

    # ── 観測 (API / CLI) ──

    def proposal_summary(self, proposal: GrowthProposal, *, with_text: bool = False) -> dict[str, Any]:
        try:
            current = self.current_version(proposal.asset_kind, proposal.mode)
        except GrowthError:
            current = ""
        out: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "asset_kind": proposal.asset_kind,
            "mode": proposal.mode,
            "parent_version": proposal.parent_version,
            "current_version": current,
            "stale": bool(current) and current != proposal.parent_version,
            "created_at": proposal.created_at,
            "expires_at": proposal.expires_at,
            "measure": encode_gate(proposal.measure),
            "candidate_deltas": dict(proposal.candidate.deltas),
            "regression_case_ids": proposal.regression_case_ids,
            "guard_ids": list(proposal.guard_ids),
        }
        if with_text:
            out["candidate_text"] = proposal.candidate.prompt_text
            out["current_text"] = ""
            out["current_deltas"] = {}
            try:
                if proposal.asset_kind == ASSET_PROMPT:
                    out["current_text"] = self.s.prompt_manager.get_raw_prompt(proposal.mode)
                else:
                    out["current_deltas"] = dict(self._evolver().persisted_deltas(proposal.mode))
            except (GrowthError, ValueError):
                pass
        return out

    def regression_summary(self) -> dict[str, Any]:
        from backend.free.learning.growth.regression import (
            STATUS_ACTIVE,
            STATUS_QUARANTINED,
            STATUS_RETIRED,
        )
        from backend.free.learning.growth.replay import count_roles

        cases, status = self._cases_and_status()
        counts = count_roles(cases, status)
        by_source: dict[str, int] = {}
        for case in cases:
            if case.status == STATUS_ACTIVE:
                by_source[case.source] = by_source.get(case.source, 0) + 1
        runs = [s.last_run_at for s in status.cases.values() if s.last_run_at]
        return {
            "active": sum(1 for c in cases if c.status == STATUS_ACTIVE),
            "guards": counts.guards,
            "pressure": counts.pressure,
            "flaky": counts.flaky,
            "quarantined": sum(1 for c in cases if c.status == STATUS_QUARANTINED),
            "retired": sum(1 for c in cases if c.status == STATUS_RETIRED),
            "by_source": by_source,
            "by_mode": counts.by_mode,
            "last_run_at": max(runs, key=_time_key) if runs else None,
        }

    def auto_adopt_status(self) -> AutoAdoptStatus:
        return self.policy().auto_adopt_effective(modes=self.s.MODES)

    def adoption_summaries(self) -> list[dict[str, Any]]:
        out = []
        for key, rec in sorted(self.adoptions().items()):
            if not isinstance(rec, dict):
                continue
            kind, _, mode = key.partition(":")
            out.append({
                "asset_kind": kind,
                "mode": mode,
                "decided_by": str(rec.get("decided_by") or ""),
                "proposal_id": str(rec.get("proposal_id") or ""),
                "adopted_version": str(rec.get("adopted_version") or ""),
                "adopted_at": rec.get("adopted_at"),
                "monitoring": bool(rec.get("monitoring")),
                "rollback_suggested": bool(rec.get("rollback_suggested")),
                "suggestion_reason": str(rec.get("suggestion_reason") or ""),
            })
        return out

    def overview(self) -> dict[str, Any]:
        """``GET /api/learning/growth`` の中身。"""
        queue = self.queue()
        return {
            "auto_adopt": self.auto_adopt_status().as_dict(),
            "proposals": [self.proposal_summary(p) for p in queue.pending()],
            "recent": [encode_row(r) for r in queue.ledger.tail(RECENT_ROWS)],
            "regression": self.regression_summary(),
            "adoptions": self.adoption_summaries(),
        }

    def summary(self) -> dict[str, Any]:
        """``/api/learning/status`` に載せる要約。"""
        status = self.auto_adopt_status()
        counts = self._role_counts()
        return {
            "auto_adopt_configured": status.configured,
            "auto_adopt_effective": status.effective,
            "auto_adopt_reason": status.reason,
            "pending_proposals": len(self.queue().pending()),
            "guards": counts.guards,
            "pressure": counts.pressure,
            "rollback_suggested": sum(1 for a in self.adoption_summaries() if a["rollback_suggested"]),
        }


def _policy_denial(asset_kind: str, mode: str) -> Any:
    """組織ポリシーパックの判定 (Pro の登録が無ければ ``None``、docs/p_09 §2)。"""
    from backend.free.core.org_policy import policy_denial

    return policy_denial(asset_kind, mode=mode)


def _experience_ids(refs: Iterable[Any]) -> list[str]:
    seen: list[str] = []
    for ref in refs:
        for exp_id in getattr(ref, "experience_ids", ()) or ():
            if exp_id not in seen:
                seen.append(exp_id)
    return seen


def _time_key(value: str) -> float:
    from backend.utils import parse_utc

    at = parse_utc(value)
    return float("-inf") if at is None else at.timestamp()


__all__ = [
    "CONFLICT_IDENTICAL",
    "CONFLICT_LEARNING_RUNNING",
    "CONFLICT_NO_TARGET",
    "CONFLICT_STALE_PARENT",
    "GROWTH_ADOPTIONS_KEY",
    "REASON_DEFECT_RATE",
    "REASON_POLICY_PINNED",
    "GrowthConflict",
    "GrowthError",
    "GrowthNotFound",
    "GrowthPolicyDenied",
    "GrowthReadonly",
    "GrowthService",
    "GrowthUnavailable",
    "GrowthGate",
    "adoption_key",
]
