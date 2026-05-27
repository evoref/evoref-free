"""

EvorefMem 統合仕様 における sleep-time **Step 6** の SemMem 対応分
``backend/free/memory/pipeline/conflict_resolver.py`` は ShortTermMemory (FadeMem)
向けで、本モジュールは ``SemanticFactStore`` 上で同 ``(subject, predicate)``
を持つ複数ファクトの競合を解消する。

設計仕様:

1. ``project_tag_always_manual: true`` を基本とする。
   ``project`` / ``policy`` タグの競合は ``review_status="pending"`` に
   振り分け、UI で人手解決する
2. 例外として、``auto_for_evolved_policies: true`` かつ winner が
   ``auto_evolved=True`` の ``policy`` ファクトの場合のみ自動マージする
   (PolicyEvolver 由来の進化結果)。
3. ``default_mode: auto`` ではそれ以外のタグは原則 newest-wins で
   supersede するが、微妙ケース (同 source / ``confirm_window_hours``
   以内) は pending に振り分ける。
4. ``default_mode: manual`` では全件 pending。
5. pinned ファクトを含む競合は常に pending (誤って自動消失するのを防ぐ)。

ファイル出力 (1 ストアあたり)::

    <store.root_dir>/
    ├── conflicts.jsonl            # 追記式: pending 状態の競合エントリ
    └── conflicts_resolved.jsonl   # 追記式: 自動解決された競合エントリ

各行スキーマ::

    {
      "ts": float,
      "scope": str,
      "subject": str,
      "predicate": str,
      "type": str,
      "winner_id": str,
      "loser_ids": [str, ...],
      "decision": "auto" | "pending",
      "reason": str
    }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from backend.free.memory.semantic.store import SemanticFactStore
from backend.free.memory.types import SemanticFact
from backend.log_config import get_logger

logger = get_logger("memory.semantic.conflict")


CONFLICTS_PENDING_FILENAME = "conflicts.jsonl"
CONFLICTS_RESOLVED_FILENAME = "conflicts_resolved.jsonl"

# 自動解決を許可しない (基本) タグ集合
_MANUAL_BASE_TAGS: frozenset[str] = frozenset({"project", "policy"})

Decision = Literal["auto", "pending"]


@dataclass(frozen=True)
class ConflictDecision:
    """1 グループの解決判断結果"""

    decision: Decision
    reason: str
    winner: SemanticFact
    losers: tuple[SemanticFact, ...]


class SemanticConflictResolver:
    """1 ``SemanticFactStore`` 内のファクト競合を検出し解消するリゾルバ。

    インスタンスは 1 sleep-time サイクル内で使い捨てる前提とし、内部状態は
    持たない。``resolve()`` を呼び出すと検出 → 判定 → 適用 → ファイル記録
    までを 1 パスで行う。
    """

    def __init__(
        self,
        store: SemanticFactStore,
        config: dict | None = None,
        *,
        now_provider=None,
    ) -> None:
        self.store = store
        cfg = ((config or {}).get("memory", {}) or {}).get("conflict", {}) or {}
        self.default_mode: str = cfg.get("default_mode", "auto")
        self.confirm_window_sec: float = (
            float(cfg.get("confirm_window_hours", 1.0)) * 3600.0
        )
        self.project_tag_always_manual: bool = bool(
            cfg.get("project_tag_always_manual", True),
        )
        self.auto_for_evolved_policies: bool = bool(
            cfg.get("auto_for_evolved_policies", True),
        )
        self._now_provider = now_provider or time.time

    # ── public API ────────────────────────────────────────────────────

    def resolve(self) -> dict[str, int]:
        """ストア内の競合を検出 → 判定 → 適用する。

        Returns:
            ``{"detected", "auto_resolved", "pending", "groups"}`` のサマリ。
        """
        result = {
            "detected": 0,
            "auto_resolved": 0,
            "pending": 0,
            "groups": 0,
        }
        groups = self._detect_groups()
        for facts in groups:
            result["groups"] += 1
            result["detected"] += len(facts)
            decision = self._decide(facts)
            self._apply(decision, result)
        if result["groups"]:
            logger.info(
                "SemMem conflict resolution: groups=%d auto=%d pending=%d "
                "(scope=%s)",
                result["groups"],
                result["auto_resolved"],
                result["pending"],
                self._infer_scope(),
            )
        return result

    # ── 検出 ─────────────────────────────────────────────────────────

    def _detect_groups(self) -> list[list[SemanticFact]]:
        """同 ``(subject, predicate)`` で異なる ``object`` を持つ active ファクト群を抽出する。

        ``review_status == "pending"`` のファクトは前サイクルで既に pending
        と判定済みなので再処理しない (二重通知防止)。
        """
        active = self.store.all_facts(include_superseded=False)
        buckets: dict[tuple[str, str], list[SemanticFact]] = {}
        for f in active:
            if f.review_status == "pending":
                continue
            buckets.setdefault((f.subject, f.predicate), []).append(f)
        groups: list[list[SemanticFact]] = []
        for facts in buckets.values():
            if len(facts) < 2:
                continue
            objects = {f.object for f in facts}
            if len(objects) < 2:
                continue
            facts.sort(key=lambda x: x.created_at)
            groups.append(facts)
        return groups

    # ── 判定 ─────────────────────────────────────────────────────────

    def _decide(self, facts: list[SemanticFact]) -> ConflictDecision:
        """1 グループの自動/手動判定を返す。``facts`` は created_at 昇順。"""
        winner = facts[-1]
        losers = tuple(facts[:-1])

        if any(f.pinned for f in facts):
            return ConflictDecision(
                "pending", "pinned_present", winner, losers,
            )

        if self.default_mode == "manual":
            return ConflictDecision(
                "pending", "default_manual", winner, losers,
            )

        types_in_group = {f.type for f in facts}
        manual_tag_hit = types_in_group & _MANUAL_BASE_TAGS
        if manual_tag_hit and self.project_tag_always_manual:
            if (
                self.auto_for_evolved_policies
                and winner.type == "policy"
                and winner.auto_evolved
                and all(f.type == "policy" for f in facts)
            ):
                return ConflictDecision(
                    "auto", "auto_evolved_policy", winner, losers,
                )
            return ConflictDecision(
                "pending",
                "project_tag_manual" if "project" in manual_tag_hit
                else "policy_tag_manual",
                winner, losers,
            )

        if self._is_borderline(winner, losers):
            return ConflictDecision(
                "pending", "confirm_window", winner, losers,
            )

        return ConflictDecision("auto", "newest_wins", winner, losers)

    def _is_borderline(
        self,
        winner: SemanticFact,
        losers: tuple[SemanticFact, ...],
    ) -> bool:
        """微妙ケース (同 source または ``confirm_window_hours`` 以内) 判定。

        - 同 source: provenance の ``session_id`` または ``note_id`` が重複
        - 時間: winner と任意 loser の created_at 差が窓以下
        """
        win_sessions, win_notes = _provenance_keys(winner)
        for loser in losers:
            if abs(winner.created_at - loser.created_at) <= self.confirm_window_sec:
                return True
            l_sessions, l_notes = _provenance_keys(loser)
            if win_sessions & l_sessions or win_notes & l_notes:
                return True
        return False

    # ── 適用 ─────────────────────────────────────────────────────────

    def _apply(self, decision: ConflictDecision, result: dict[str, int]) -> None:
        if decision.decision == "auto":
            self._apply_auto(decision)
            result["auto_resolved"] += len(decision.losers)
        else:
            self._apply_pending(decision)
            result["pending"] += 1 + len(decision.losers)

    def _apply_auto(self, decision: ConflictDecision) -> None:
        winner = decision.winner
        for loser in decision.losers:
            try:
                self.store.supersede(loser.id, winner.id)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "supersede failed for %s -> %s: %s",
                    loser.id, winner.id, exc,
                )
                continue
        # 解決 mark を winner にも残す
        try:
            self.store.update_fact(
                winner.id,
                review_status="resolved_keep_new",
            )
        except KeyError:
            pass
        self._write_jsonl(CONFLICTS_RESOLVED_FILENAME, decision)

    def _apply_pending(self, decision: ConflictDecision) -> None:
        for fact in (decision.winner, *decision.losers):
            try:
                self.store.update_fact(
                    fact.id,
                    requires_user_review=True,
                    review_status="pending",
                )
            except KeyError:
                continue
        self._write_jsonl(CONFLICTS_PENDING_FILENAME, decision)

    # ── 永続化 ────────────────────────────────────────────────────────

    def _write_jsonl(self, filename: str, decision: ConflictDecision) -> None:
        path = self.store.root_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "ts": float(self._now_provider()),
            "scope": self._infer_scope(),
            "subject": decision.winner.subject,
            "predicate": decision.winner.predicate,
            "type": decision.winner.type,
            "winner_id": decision.winner.id,
            "loser_ids": [f.id for f in decision.losers],
            "decision": decision.decision,
            "reason": decision.reason,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _infer_scope(self) -> str:
        """ストアの ``root_dir`` 構造から scope 文字列を推定する。

        ``<semantic_root>/global`` or ``<semantic_root>/projects/<id>``
        を仮定する。それ以外はディレクトリ名をそのまま返す。
        """
        root = self.store.root_dir
        if root.name == "global":
            return "global"
        if root.parent.name == "projects":
            return f"project:{root.name}"
        return root.name


def _provenance_keys(
    fact: SemanticFact,
) -> tuple[set[str], set[str]]:
    """``(session_ids, note_ids)`` のタプルを返す。``None`` は除外する。"""
    sessions: set[str] = set()
    notes: set[str] = set()
    for p in fact.provenances:
        if p.session_id:
            sessions.add(p.session_id)
        if p.note_id:
            notes.add(p.note_id)
    return sessions, notes


def resolve_semmem_conflicts(
    stores: Iterable[SemanticFactStore],
    config: dict | None = None,
) -> dict[str, int]:
    """複数ストアに対して :class:`SemanticConflictResolver` を順次適用する。

    sleep-time Step 6 のヘルパ。``stores`` は ``[global_store, project_store]``
    を想定するが任意個に対応する。集計サマリを返す。
    """
    total = {"detected": 0, "auto_resolved": 0, "pending": 0, "groups": 0}
    for store in stores:
        if store is None:
            continue
        sub = SemanticConflictResolver(store, config).resolve()
        for k, v in sub.items():
            total[k] = total.get(k, 0) + v
    return total
